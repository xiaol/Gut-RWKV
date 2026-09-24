import math
from pathlib import Path

import torch
from torch import nn
from .backbone.rwkv7 import Rwkv7State, WorldTokenizer, load_rwkv7

from .schema import Question, serialize


class LowRankLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device)
        self.up = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, inputs):
        return self.base(inputs) + self.up(self.down(inputs.float())).to(inputs.dtype)


class DecisionModel(nn.Module):
    def __init__(self, backbone, tokenizer, rank=0, head_size=128, max_tokens=2048, candidate_batch=16,
                 state_tuning=False):
        super().__init__()
        if rank < 0 or head_size < 1 or max_tokens < 1 or candidate_batch < 1:
            raise ValueError("Invalid model dimensions or limits")
        if state_tuning and rank:
            raise ValueError("Choose state tuning or LoRA, not both")
        backbone.requires_grad_(False)
        backbone.head = nn.Identity()
        self.backbone = backbone
        self.initial_wkv = None
        if state_tuning:
            config = backbone.config
            self.initial_wkv = nn.Parameter(torch.zeros(config.num_hidden_layers, 1, config.num_heads,
                                                        config.head_size, config.head_size,
                                                        device=backbone.device, dtype=torch.float32))
        self.tokenizer = tokenizer
        self.head = nn.Sequential(nn.LayerNorm(backbone.config.hidden_size),
                                  nn.Linear(backbone.config.hidden_size, head_size), nn.GELU(),
                                  nn.Linear(head_size, 1, bias=False)).to(backbone.device)
        if rank:
            for block in backbone.blocks:
                for name in ("receptance", "key", "value", "output"):
                    setattr(block.att, name, LowRankLinear(getattr(block.att, name), rank))
                for name in ("key", "value"):
                    setattr(block.ffn, name, LowRankLinear(getattr(block.ffn, name), rank))
        self.settings = {"rank": rank, "head_size": head_size, "max_tokens": max_tokens,
                         "candidate_batch": candidate_batch}
        if state_tuning:
            self.settings["state_tuning"] = True
        self.temperature = 1.0
        self.trained_steps = 0

    @classmethod
    def from_base(cls, model_path, vocab_path, device="cpu", dtype=torch.float32, **settings):
        return cls(load_rwkv7(str(model_path), device=device, dtype=dtype), WorldTokenizer(str(vocab_path)), **settings)

    def _tokens(self, text):
        return self.tokenizer.encode(text)

    def _forward(self, rows, state=None):
        if not rows or any(not row for row in rows):
            raise ValueError("Token rows must be nonempty")
        lengths = torch.tensor([len(row) for row in rows], device=self.backbone.device)
        tokens = torch.zeros(len(rows), int(lengths.max()), dtype=torch.long, device=self.backbone.device)
        for index, row in enumerate(rows):
            tokens[index, :len(row)] = torch.tensor(row, device=tokens.device)
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None] < lengths[:, None]
        if state is None and self.initial_wkv is not None:
            zeros = self.backbone.zero_state(len(rows))
            state = Rwkv7State(zeros.att_x, self.initial_wkv.expand(-1, len(rows), -1, -1, -1).contiguous(),
                               zeros.ffn_x)
        output = self.backbone(input_ids=tokens, attention_mask=mask, state=state)
        hidden = output.logits[torch.arange(len(rows), device=tokens.device), lengths - 1]
        return hidden, output.state

    def _encode(self, state, question):
        prefix = self._tokens("State: " + serialize(state))
        question_tokens = self._tokens(question.prefix())
        candidates = [self._tokens(candidate) for candidate in question.candidates()]
        if max(len(prefix) + len(question_tokens) + len(candidate) for candidate in candidates) > self.settings["max_tokens"]:
            raise ValueError("State + question + candidate exceeds max_tokens; no truncation is applied")
        return prefix, question_tokens, candidates

    def logits(self, state, question, use_cache=False, prefix_state=None):
        prefix, question_tokens, candidates = self._encode(state, question)
        if use_cache and torch.is_grad_enabled():
            raise ValueError("Cached inference requires no_grad; training uses complete sequences")
        if use_cache:
            if prefix_state is None:
                _, prefix_state = self._forward([prefix])
            _, question_state = self._forward([question_tokens], prefix_state)
        scores = []
        batch_size = self.settings["candidate_batch"]
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start:start + batch_size]
            if use_cache:
                indices = torch.zeros(len(batch), dtype=torch.long, device=self.backbone.device)
                hidden, _ = self._forward(batch, question_state.index(indices))
            else:
                hidden, _ = self._forward([prefix + question_tokens + candidate for candidate in batch])
            scores.append(self.head(hidden.float()).squeeze(-1))
        return torch.cat(scores)

    @torch.no_grad()
    def system_one(self, state, questions):
        if not self.trained_steps:
            raise RuntimeError("Decision head is untrained; train or load an adapter before inference")
        if not isinstance(state, (str, dict, list)):
            raise ValueError("state must be a string, object or array")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        parsed = {name: Question.parse(value) for name, value in questions.items()}
        encoded = [self._encode(state, question) for question in parsed.values()]
        self.eval()
        _, prefix_state = self._forward([encoded[0][0]])
        scores = [[] for _ in encoded]
        batch_size = self.settings["candidate_batch"]
        for start in range(0, len(encoded), batch_size):
            group = encoded[start:start + batch_size]
            indices = torch.zeros(len(group), dtype=torch.long, device=self.backbone.device)
            _, question_states = self._forward([item[1] for item in group], prefix_state.index(indices))
            branches = [(question_index, candidate) for question_index, item in enumerate(group)
                        for candidate in item[2]]
            for offset in range(0, len(branches), batch_size):
                batch = branches[offset:offset + batch_size]
                indices = torch.tensor([item[0] for item in batch], device=self.backbone.device)
                hidden, _ = self._forward([item[1] for item in batch], question_states.index(indices))
                batch_scores = self.head(hidden.float()).squeeze(-1)
                for (question_index, _), score in zip(batch, batch_scores):
                    scores[start + question_index].append(score)
        answers = {}
        for (name, question), question_scores in zip(parsed.items(), scores):
            probabilities = (torch.stack(question_scores) / self.temperature).softmax(-1)
            answers[name] = question.answer(probabilities.tolist())
        return {"model": "gut-rwkv", "answers": answers, "generated_tokens": 0}

    def save_adapter(self, path, metadata=None):
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        weights = {name: tensor.detach().cpu() for name, tensor in self.state_dict().items() if name in names}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "settings": self.settings, "weights": weights,
                    "temperature": self.temperature, "trained_steps": self.trained_steps,
                    "metadata": metadata or {}}, path)

    def load_adapter(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["settings"] != self.settings:
            raise ValueError("Adapter format/settings do not match the model")
        expected = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(payload["weights"]) != expected:
            raise ValueError("Adapter parameter names do not match the model")
        self.load_state_dict(payload["weights"], strict=False)
        self.temperature = payload["temperature"]
        self.trained_steps = payload["trained_steps"]
        return payload["metadata"]
