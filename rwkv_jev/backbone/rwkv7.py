"""Pure-PyTorch RWKV-7 ("Goose") language model with an explicit recurrent state.

Why a from-scratch implementation: the receiver in a *state* bridge is an RNN whose entire
memory of the prompt is a per-layer state (a time-shift vector and an ``H x N x N`` WKV
matrix per layer).  The bridge writes into that state, so the model has to expose it as a
first-class input and output.  This file follows the reference ``rwkv_v7_demo.py`` from
RWKV-LM exactly (same parameter names and shapes, so official ``.pth`` checkpoints load
directly) and needs no CUDA kernels: the WKV recurrence is a plain fp32 loop over time.

Padding: everything is *right*-padded.  Pad positions are made exactly inert (zero k, v, a, b
and a decay of 1), so the state after the batch is each row's state at its own last real
token.  Left padding would decay and pollute the state before the first real token.

Also included: the RWKV "World" trie tokenizer, wrapped in a minimal Hugging Face-like API.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- tokenizer


class WorldTokenizer:
    """RWKV World tokenizer (greedy longest match over a byte trie).  Vocabulary file:
    one line per token, ``<id> <python literal> <byte length>`` (``rwkv_vocab_v20230424.txt``)."""

    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<eos>"
    padding_side = "right"

    def __init__(self, vocab_path: str, chat_prefix: str = "", chat_template_kind: str = "world"):
        self.idx2token: dict[int, bytes] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx = int(line[: line.index(" ")])
                lit = eval(line[line.index(" ") : line.rindex(" ")])  # noqa: S307 - trusted vocab file
                tok = lit.encode("utf-8") if isinstance(lit, str) else lit
                self.idx2token[idx] = tok
        self.vocab_size = max(self.idx2token) + 1
        self.trie: dict = {}
        for idx, tok in self.idx2token.items():
            node = self.trie
            for b in tok:
                node = node.setdefault(b, {})
            node[-1] = idx  # terminal marker
        self.chat_prefix = chat_prefix
        self.chat_template_kind = chat_template_kind
        self._cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        if text in self._cache:
            return list(self._cache[text])
        data = text.encode("utf-8")
        out, i, n = [], 0, len(data)
        while i < n:
            node, j, last_idx, last_j = self.trie, i, None, i
            while j < n and data[j] in node:
                node = node[data[j]]
                j += 1
                if -1 in node:
                    last_idx, last_j = node[-1], j
            if last_idx is None:  # cannot happen with a byte-complete vocab; skip the byte
                i += 1
                continue
            out.append(last_idx)
            i = last_j
        if len(self._cache) < 200_000:
            self._cache[text] = list(out)
        return out

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]
        if skip_special_tokens:
            ids = [i for i in ids if i != 0]
        return b"".join(self.idx2token.get(i, b"") for i in ids).decode("utf-8", errors="replace")

    def batch_decode(self, seqs, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(s, skip_special_tokens) for s in seqs]

    def __call__(self, texts, add_special_tokens: bool = False, padding: bool = False, truncation: bool = False, max_length: int | None = None, return_tensors: str | None = None, **_):
        single = isinstance(texts, str)
        seqs = [self.encode(t) for t in ([texts] if single else texts)]
        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]
        if return_tensors == "pt" or padding:
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), L), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[i, : len(s)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": seqs[0] if single else seqs}

    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = True, **_) -> str:
        """World-model chat format: ``User: ...\\n\\nAssistant:`` (+ optional prefix such as
        ``" <think>\\n</think>\\n"`` to skip RWKV7-G1 reasoning)."""
        parts = []
        for m in messages:
            role = "User" if m["role"] == "user" else ("Assistant" if m["role"] == "assistant" else "System")
            parts.append(f"{role}: {m['content'].strip()}")
        text = "\n\n".join(parts)
        if add_generation_prompt:
            text += "\n\nAssistant:" + self.chat_prefix
        return text


# ---------------------------------------------------------------- model


@dataclass
class Rwkv7State:
    """Per-layer recurrent state.  ``att_x``/``ffn_x``: [L, B, C] last inputs of the time-shift;
    ``wkv``: [L, B, H, N, N] the WKV matrices."""

    att_x: torch.Tensor
    wkv: torch.Tensor
    ffn_x: torch.Tensor

    @staticmethod
    def zeros(L: int, B: int, C: int, H: int, N: int, device, dtype=torch.float32) -> "Rwkv7State":
        return Rwkv7State(torch.zeros(L, B, C, device=device, dtype=dtype), torch.zeros(L, B, H, N, N, device=device, dtype=torch.float32), torch.zeros(L, B, C, device=device, dtype=dtype))

    def index(self, idx) -> "Rwkv7State":
        return Rwkv7State(self.att_x[:, idx], self.wkv[:, idx], self.ffn_x[:, idx])


def _time_shift(x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """x: [B,T,C], x_prev: [B,C] -> the previous token's input at every position."""
    return torch.cat([x_prev.unsqueeze(1).to(x.dtype), x[:, :-1]], dim=1)


def _last_valid(x: torch.Tensor, lengths: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
    """x: [B,T,C]; returns x at each row's last real token (or ``prev`` for empty rows)."""
    idx = (lengths - 1).clamp(min=0)
    last = x.gather(1, idx.view(-1, 1, 1).expand(-1, 1, x.shape[-1])).squeeze(1)
    return torch.where((lengths > 0).unsqueeze(-1), last, prev.to(x.dtype))


def _wkv7_loop(r, w, k, v, a, b, state):
    with torch.autocast(device_type=r.device.type, enabled=False):  # the recurrence stays fp32
        w = torch.exp(-torch.exp(w.float()))
        r, k, v, a, b = (t.float() for t in (r, k, v, a, b))
        state = state.float()
        outs = []
        for t in range(r.shape[1]):
            state = state * w[:, t, :, None, :] + state @ a[:, t, :, :, None] @ b[:, t, :, None, :] + v[:, t, :, :, None] @ k[:, t, :, None, :]
            outs.append((state @ r[:, t, :, :, None]).squeeze(-1))
        return torch.stack(outs, 1), state


USE_CUDA_KERNEL = os.environ.get("STATE_BRIDGE_WKV7_TORCH", "0") != "1"


def wkv7(r, w, k, v, a, b, state):
    """WKV-7 recurrence.  r,w,k,v,a,b: [B,T,H,N] (w pre-exponent); state: [B,H,N,N].
    Returns (out [B,T,H,N], final state).

    On CUDA with ``T % 16 == 0`` and a compilable toolchain this runs the fused kernel in
    ``wkv7_kernel`` (with gradients into the initial state).  Otherwise the reference fp32
    Python loop is used; under autograd that loop is gradient-checkpointed, because saving
    every step's state for a 600-token batch would need tens of GB."""
    if USE_CUDA_KERNEL and r.is_cuda and r.shape[1] % 16 == 0:
        from . import wkv7_kernel

        if wkv7_kernel.available(r.shape[-1]):
            return wkv7_kernel.wkv7_cuda(r, w, k, v, a, b, state)
    if torch.is_grad_enabled() and r.shape[1] > 1:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(_wkv7_loop, r, w, k, v, a, b, state, use_reentrant=False)
    return _wkv7_loop(r, w, k, v, a, b, state)


def _ln(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Normalisation layers are kept in fp32; cast around them."""
    return norm(x.float()).to(x.dtype)


class TimeMix(nn.Module):
    def __init__(self, C: int, H: int, N: int, layer_id: int, d_decay: int, d_aaa: int, d_mv: int, d_gate: int):
        super().__init__()
        self.C, self.H, self.N, self.layer_id = C, H, N, layer_id
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "k_k", "k_a"):
            setattr(self, name, nn.Parameter(torch.zeros(1, 1, C)))
        self.w1 = nn.Parameter(torch.zeros(C, d_decay)); self.w2 = nn.Parameter(torch.zeros(d_decay, C))
        self.a1 = nn.Parameter(torch.zeros(C, d_aaa)); self.a2 = nn.Parameter(torch.zeros(d_aaa, C))
        if layer_id > 0:
            self.v0 = nn.Parameter(torch.zeros(1, 1, C)); self.v1 = nn.Parameter(torch.zeros(C, d_mv)); self.v2 = nn.Parameter(torch.zeros(d_mv, C))
        self.g1 = nn.Parameter(torch.zeros(C, d_gate)); self.g2 = nn.Parameter(torch.zeros(d_gate, C))
        self.r_k = nn.Parameter(torch.zeros(H, N))
        self.receptance = nn.Linear(C, C, bias=False); self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False); self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first, x_prev, wkv_state, mask, lengths):
        B, T, C = x.shape
        H, N = self.H, self.N
        xx = _time_shift(x, x_prev) - x
        xr, xw, xk, xv, xa, xg = (x + xx * p for p in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g))
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = F.normalize((k * self.k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        # make padded positions inert: no write, no decay
        m = mask.unsqueeze(-1).to(x.dtype)
        w = torch.where(mask.unsqueeze(-1), w, torch.full_like(w, -1e4))
        out, new_state = wkv7(
            (r * m).view(B, T, H, N), w.view(B, T, H, N), (k * m).view(B, T, H, N), (v * m).view(B, T, H, N),
            (-kk * m).view(B, T, H, N), (kk * a * m).view(B, T, H, N), wkv_state,
        )
        out = _ln(self.ln_x, out.to(x.dtype).view(B * T, C)).view(B, T, C)
        out = out + ((r * k * self.r_k.view(1, 1, C)).view(B, T, H, N).sum(-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, C)
        return self.output(out * g), v_first, _last_valid(x, lengths, x_prev), new_state


class ChannelMix(nn.Module):
    def __init__(self, C: int, dim_ffn: int):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(1, 1, C))
        self.key = nn.Linear(C, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, C, bias=False)

    def forward(self, x, x_prev, lengths):
        xx = _time_shift(x, x_prev) - x
        k = torch.relu(self.key(x + xx * self.x_k)) ** 2
        return self.value(k), _last_valid(x, lengths, x_prev)


class Block(nn.Module):
    def __init__(self, cfg, layer_id: int):
        super().__init__()
        C = cfg.hidden_size
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(C)
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.att = TimeMix(C, cfg.num_heads, cfg.head_size, layer_id, cfg.d_decay, cfg.d_aaa, cfg.d_mv, cfg.d_gate)
        self.ffn = ChannelMix(C, cfg.dim_ffn)

    def forward(self, x, v_first, att_x, wkv, ffn_x, mask, lengths):
        if self.layer_id == 0:
            x = _ln(self.ln0, x)
        dx, v_first, att_x, wkv = self.att(_ln(self.ln1, x), v_first, att_x, wkv, mask, lengths)
        x = x + dx
        dx, ffn_x = self.ffn(_ln(self.ln2, x), ffn_x, lengths)
        return x + dx, v_first, att_x, wkv, ffn_x


class Rwkv7ForCausalLM(nn.Module):
    """RWKV-7 LM with a Hugging Face-like ``forward`` and an explicit ``state`` argument."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        self.config = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(cfg.hidden_size)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=[0])
        self.grad_checkpoint = True

    # -- HF-ish accessors used elsewhere in the package
    def get_input_embeddings(self):
        return self.emb

    @property
    def lm_head(self):
        return self.head

    @property
    def device(self):
        return self.emb.weight.device

    def zero_state(self, B: int) -> Rwkv7State:
        c = self.config
        return Rwkv7State.zeros(c.num_hidden_layers, B, c.hidden_size, c.num_heads, c.head_size, self.device, self.emb.weight.dtype)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, state: Rwkv7State | None = None, labels=None, output_hidden_states=False, **_):
        x = self.emb(input_ids) if inputs_embeds is None else inputs_embeds.to(self.emb.weight.dtype)
        B, T, C = x.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=x.device)
        # the fused kernel wants T % 16 == 0: append inert pad positions, cut them off at the end
        T_in = T
        if USE_CUDA_KERNEL and x.is_cuda and T > 1 and T % 16:
            pad = 16 - T % 16
            x = F.pad(x, (0, 0, 0, pad))
            attention_mask = F.pad(attention_mask, (0, pad))
            if labels is not None:
                labels = F.pad(labels, (0, pad), value=-100)
            T = T + pad
        mask = attention_mask.bool().to(x.device)
        lengths = mask.long().sum(1)
        if state is None:
            state = self.zero_state(B)
        hidden = [x] if output_hidden_states else None
        v_first = None
        att_x, wkv, ffn_x = [], [], []
        # per-block gradient checkpointing: a frozen receiver only needs gradients w.r.t. its
        # inputs / initial state, and recomputing each block in the backward pass cuts activation
        # memory by roughly the number of intermediate tensors per block (~20x)
        ckpt = self.grad_checkpoint and torch.is_grad_enabled() and T > 1
        for i, blk in enumerate(self.blocks):
            if ckpt:
                from torch.utils.checkpoint import checkpoint

                x, v_first, ax, s, fx = checkpoint(blk, x, v_first, state.att_x[i], state.wkv[i], state.ffn_x[i], mask, lengths, use_reentrant=False)
            else:
                x, v_first, ax, s, fx = blk(x, v_first, state.att_x[i], state.wkv[i], state.ffn_x[i], mask, lengths)
            att_x.append(ax); wkv.append(s); ffn_x.append(fx)
            if output_hidden_states:
                hidden.append(x)
        h = _ln(self.ln_out, x)
        if T != T_in:
            h = h[:, :T_in]
            if labels is not None:
                labels = labels[:, :T_in]
            if output_hidden_states:
                hidden = [hh[:, :T_in] for hh in hidden]
        loss, logits = None, None
        if labels is not None:
            # Memory-efficient loss: the head is applied only at labelled positions, in chunks, under
            # gradient checkpointing.  Full [B, T, 65536] logits in fp32 plus their gradient were the
            # largest tensors of a training step; logits are not returned in this mode.
            loss = self._chunked_ce(h, labels)
        else:
            logits = self.head(h)
        new_state = Rwkv7State(torch.stack(att_x), torch.stack(wkv), torch.stack(ffn_x))
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=tuple(hidden) if hidden else None, state=new_state, past_key_values=None)

    def _chunked_ce(self, h: torch.Tensor, labels: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        y = labels[:, 1:]
        keep = y != -100
        hs, ys = h[:, :-1][keep], y[keep]
        if hs.shape[0] == 0:
            return (h.sum() * 0.0).float()

        def ce(a, b):
            return F.cross_entropy(self.head(a).float(), b, reduction="sum")

        total = 0.0
        for i in range(0, hs.shape[0], chunk):
            if torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                total = total + checkpoint(ce, hs[i : i + chunk], ys[i : i + chunk], use_reentrant=False)
            else:
                total = total + ce(hs[i : i + chunk], ys[i : i + chunk])
        return total / hs.shape[0]

    @torch.no_grad()
    def generate_greedy(self, tokenizer: WorldTokenizer, prompt_ids: list[list[int]], max_new_tokens: int, state: Rwkv7State | None = None, inputs_embeds: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> list[str]:
        """Right-padded prefill (from ``state`` if given), then token-by-token decoding.

        A row stops at EOS (token 0), when it starts hallucinating the next turn
        (``\\n\\nUser:``), or at a blank line once a ``\\boxed{}`` answer has been written.  World
        models end a turn with a blank line, but blank lines also occur inside a solution, so a
        bare ``\\n\\n`` is not a stop by itself."""
        dev = self.device
        if inputs_embeds is None:
            B = len(prompt_ids)
            L = max(len(p) for p in prompt_ids)
            ids = torch.zeros(B, L, dtype=torch.long)
            mask = torch.zeros(B, L, dtype=torch.long)
            for i, p in enumerate(prompt_ids):
                ids[i, : len(p)] = torch.tensor(p); mask[i, : len(p)] = 1
            out = self.forward(input_ids=ids.to(dev), attention_mask=mask.to(dev), state=state)
            lengths = mask.sum(1).to(dev)
        else:
            B = inputs_embeds.shape[0]
            out = self.forward(inputs_embeds=inputs_embeds.to(dev), attention_mask=attention_mask.to(dev), state=state)
            lengths = attention_mask.sum(1).to(dev)
        last_logits = out.logits.gather(1, (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.logits.shape[-1])).squeeze(1)
        state = out.state
        nxt = last_logits.argmax(-1)
        finished = [False] * B
        boxed = [False] * B
        generated = [[] for _ in range(B)]
        for _ in range(max_new_tokens):
            toks = nxt.tolist()
            for i in range(B):
                if finished[i]:
                    continue
                t = toks[i]
                if t == 0:
                    finished[i] = True
                    continue
                generated[i].append(t)
                tail = tokenizer.decode(generated[i][-8:])
                if not boxed[i] and "\\boxed{" in tail:
                    boxed[i] = True
                if tail.endswith("\n\nUser:") or (boxed[i] and tail.endswith("\n\n")):
                    finished[i] = True
            if all(finished):
                break
            fin = torch.tensor(finished, device=dev)
            nxt = torch.where(fin, torch.zeros_like(nxt), nxt)
            out = self.forward(input_ids=nxt.view(B, 1), state=state)
            state = out.state
            nxt = out.logits[:, -1].argmax(-1)
        texts = []
        for g in generated:
            t = tokenizer.decode(g)
            if t.endswith("\n\nUser:"):
                t = t[: -len("\n\nUser:")]
            texts.append(t.rstrip())
        return texts


def _lora_dims(sd: dict, prefix: str) -> tuple[int, int, int, int]:
    d_decay = sd[f"{prefix}att.w1"].shape[1]
    d_aaa = sd[f"{prefix}att.a1"].shape[1]
    d_gate = sd[f"{prefix}att.g1"].shape[1]
    return d_decay, d_aaa, d_gate


def load_rwkv7(path: str, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16) -> Rwkv7ForCausalLM:
    """Load an official RWKV-7 ``.pth`` checkpoint (BlinkDL naming) into ``Rwkv7ForCausalLM``."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    V, C = sd["emb.weight"].shape
    N = sd["blocks.0.att.r_k"].shape[1]
    H = C // N
    d_decay, d_aaa, d_gate = _lora_dims(sd, "blocks.0.")
    d_mv = sd["blocks.1.att.v1"].shape[1] if n_layer > 1 else 32
    cfg = SimpleNamespace(
        model_type="rwkv7", hidden_size=C, num_hidden_layers=n_layer, vocab_size=V, head_size=N, num_heads=H,
        dim_ffn=sd["blocks.0.ffn.key.weight"].shape[0], d_decay=d_decay, d_aaa=d_aaa, d_mv=d_mv, d_gate=d_gate,
        num_attention_heads=H, num_key_value_heads=H, head_dim=N, tie_word_embeddings=False,
    )
    model = Rwkv7ForCausalLM(cfg)
    # official checkpoints carry unused v0/v1/v2 on layer 0 (the reference forward never reads them)
    sd = {k: v for k, v in sd.items() if k not in ("blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"RWKV-7 checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.to(device=device, dtype=dtype)
    # keep normalisation layers in fp32 for stability (the WKV loop is fp32 regardless)
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)):
            mod.float()
    model.eval()
    model.requires_grad_(False)
    return model


def find_vocab(model_path: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    p = Path(model_path)
    for cand in [p.parent / "rwkv_vocab_v20230424.txt", p.parent.parent / "rwkv_vocab_v20230424.txt", Path(__file__).resolve().parent.parent / "assets" / "rwkv_vocab_v20230424.txt"]:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("RWKV World vocab file rwkv_vocab_v20230424.txt not found; set models.receiver.tokenizer")
