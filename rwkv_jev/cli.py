import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as functional

from .model import DecisionModel
from .schema import Question


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_data(path, with_sources=False):
    records = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("state"), (str, dict, list)) or not record.get("questions"):
                raise ValueError(f"Invalid state/questions at line {line_number}")
            for value in record["questions"].values():
                question = Question.parse(value)
                item = (record["state"], question, question.target(value["label"]))
                if with_sources:
                    item += (value.get("src", record.get("_meta", {}).get("source")),)
                records.append(item)
    if not records:
        raise ValueError("Dataset contains no labeled questions")
    return records


def training_order(records, generator, sampling="natural", sources=None):
    if sampling not in ("natural", "noul-balanced"):
        raise ValueError("Unknown sampling mode")
    order = list(range(len(records)))
    generator.shuffle(order)
    if sampling == "natural":
        return order
    if sources is None or len(sources) != len(records):
        raise ValueError("Balanced sampling requires a source for each question")
    pools = defaultdict(lambda: defaultdict(list))
    for index, (_, question, target) in enumerate(records):
        if question.kind == "noul":
            if not isinstance(sources[index], str) or not sources[index]:
                raise ValueError("Balanced Noul questions require source metadata")
            pools[sources[index]][target].append(index)
    if any(set(pool) != {0, 1} for pool in pools.values()):
        raise ValueError("Each balanced Noul source must contain both labels")
    next_labels = {source: generator.randrange(2) for source in pools}
    for position, index in enumerate(order):
        if records[index][1].kind == "noul":
            source = sources[index]
            target = next_labels[source]
            order[position] = generator.choice(pools[source][target])
            next_labels[source] = 1 - target
    return order


def proper_score_reward(logits, target, spherical_weight=0.5, rps_weight=0.5,
                        ordered=True, stable_log=False):
    if not 0 <= target < logits.shape[-1]:
        raise ValueError("Target is outside the candidate distribution")
    if not all(math.isfinite(value) and value >= 0 for value in (spherical_weight, rps_weight)):
        raise ValueError("Proper-score weights must be finite and nonnegative")
    probabilities = logits.float().softmax(-1)
    log_reward = (functional.log_softmax(logits.float(), dim=-1)[..., target] if stable_log else
                  probabilities[..., target].clamp_min(1e-30).log())
    spherical_reward = probabilities[..., target] / probabilities.square().sum(-1).sqrt().clamp_min(1e-30)
    if not ordered or probabilities.shape[-1] == 1:
        rps_reward = torch.zeros_like(log_reward)
    else:
        cumulative = probabilities.cumsum(-1)[..., :-1]
        target_cumulative = torch.arange(probabilities.shape[-1] - 1, device=logits.device) >= target
        rps_reward = -(cumulative - target_cumulative.to(probabilities.dtype)).square().mean(-1)
    return log_reward + spherical_weight * spherical_reward + rps_weight * rps_reward


def gaussian_policy_loss(logits, target, question_kind, estimator="reinforce", exploration_std=0.05,
                         policy_samples=32, spherical_weight=0.5, rps_weight=0.5):
    if logits.ndim != 1 or question_kind not in ("choice", "noul", "score"):
        raise ValueError("Gaussian policy requires one logit vector and a valid question kind")
    if estimator not in ("reinforce", "pathwise"):
        raise ValueError("Unknown Gaussian policy estimator")
    if not math.isfinite(exploration_std) or exploration_std <= 0:
        raise ValueError("Policy exploration standard deviation must be finite and positive")
    if not isinstance(policy_samples, int) or policy_samples < 2 or policy_samples % 2:
        raise ValueError("Policy samples must be an even integer of at least two")
    noise = torch.randn(policy_samples // 2, logits.numel(), device=logits.device, dtype=torch.float32)
    noise = torch.cat((noise, -noise), dim=0)
    mean = logits.float()
    sampled_logits = (mean if estimator == "pathwise" else mean.detach()) + exploration_std * noise
    rewards = proper_score_reward(sampled_logits, target, spherical_weight, rps_weight,
                                  ordered=question_kind == "score", stable_log=True)
    if estimator == "pathwise":
        return -rewards.mean(), rewards.detach().mean()
    baseline = proper_score_reward(mean.detach(), target, spherical_weight, rps_weight,
                                   ordered=question_kind == "score", stable_log=True)
    advantages = (rewards - baseline).detach()
    log_density = -0.5 * ((sampled_logits.detach() - mean) / exploration_std).square().sum(-1)
    return -(advantages * log_density).mean(), rewards.detach().mean()


def objective_loss(logits, target, objective="cross-entropy", exploration_std=0.05,
                   spherical_weight=0.5, rps_weight=0.5, baseline=None, question_kind=None,
                   policy_samples=32):
    if objective == "cross-entropy":
        return functional.cross_entropy(logits[None], torch.tensor([target], device=logits.device)), None
    if objective in ("reinforce", "pathwise"):
        return gaussian_policy_loss(logits, target, question_kind, objective, exploration_std,
                                    policy_samples, spherical_weight, rps_weight)
    if objective not in ("proper-score", "rlcd"):
        raise ValueError("Unknown training objective")
    if objective == "rlcd":
        if not math.isfinite(exploration_std) or exploration_std <= 0:
            raise ValueError("RLCD exploration standard deviation must be finite and positive")
        logits = logits + torch.randn_like(logits) * exploration_std
    reward = proper_score_reward(logits, target, spherical_weight, rps_weight)
    if baseline is None:
        return -reward, reward.detach()
    return -(reward - baseline.detach()), reward.detach()


def train(model, records, epochs, learning_rate, seed, max_steps=0, checkpoint_every=0, checkpoint_callback=None,
          state_learning_rate=None, state_max_norm=None, sampling="natural", sources=None,
          objective="cross-entropy", exploration_std=0.05, spherical_weight=0.5, rps_weight=0.5,
          baseline_decay=0.95, policy_samples=32):
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or max_steps < 0:
        raise ValueError("Invalid training settings")
    if checkpoint_every < 0 or (checkpoint_every and checkpoint_callback is None):
        raise ValueError("Checkpoint interval requires a callback and must be nonnegative")
    if not records:
        raise ValueError("Training records must be nonempty")
    for value in (state_learning_rate, state_max_norm):
        if value is not None and (not math.isfinite(value) or value <= 0 or model.initial_wkv is None):
            raise ValueError("State controls require state tuning and finite positive values")
    if objective not in ("cross-entropy", "proper-score", "rlcd", "reinforce", "pathwise"):
        raise ValueError("Unknown training objective")
    if objective in ("rlcd", "reinforce", "pathwise") and (not math.isfinite(exploration_std) or exploration_std <= 0):
        raise ValueError("Policy exploration standard deviation must be finite and positive")
    if not isinstance(policy_samples, int) or policy_samples < 2 or policy_samples % 2:
        raise ValueError("Policy samples must be an even integer of at least two")
    if not math.isfinite(baseline_decay) or not 0 <= baseline_decay < 1:
        raise ValueError("Baseline decay must be finite and in [0, 1)")
    generator = random.Random(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if state_learning_rate is None:
        groups = parameters
    else:
        groups = [{"params": [parameter for parameter in parameters if parameter is not model.initial_wkv]},
                  {"params": [model.initial_wkv], "lr": state_learning_rate}]
    optimizer = torch.optim.AdamW(groups, lr=learning_rate)
    model.train()
    losses = []
    reward_baseline = None
    for epoch in range(epochs):
        order = training_order(records, generator, sampling, sources)
        for position, index in enumerate(order):
            state, question, target = records[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(state, question)
            loss, reward = objective_loss(logits, target, objective, exploration_std,
                                          spherical_weight, rps_weight, reward_baseline,
                                          question.kind, policy_samples)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            if reward is not None:
                reward_baseline = reward if reward_baseline is None else (
                    baseline_decay * reward_baseline + (1 - baseline_decay) * reward)
            if state_max_norm is not None:
                with torch.no_grad():
                    norm = torch.linalg.vector_norm(model.initial_wkv)
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite initial state norm")
                    model.initial_wkv.mul_((state_max_norm / norm.clamp_min(1e-30)).clamp(max=1.0))
            model.trained_steps += 1
            losses.append(float(loss.detach()))
            if len(losses) == 1 or len(losses) % 10 == 0:
                progress = {"step": model.trained_steps, "epoch": epoch + 1, "loss": losses[-1]}
                if reward is not None:
                    progress["reward"] = float(reward)
                    progress["reward_baseline"] = float(reward_baseline)
                if model.initial_wkv is not None:
                    progress["initial_state_norm"] = float(model.initial_wkv.detach().norm())
                print(json.dumps(progress), flush=True)
            if checkpoint_every and (model.trained_steps % checkpoint_every == 0 or
                                     position == len(order) - 1 or (max_steps and len(losses) >= max_steps)):
                checkpoint_callback(model, epoch + 1)
            if max_steps and len(losses) >= max_steps:
                return losses
    return losses


@torch.no_grad()
def evaluate(model, records):
    model.eval()
    outcomes = []
    for state, question, target in records:
        probabilities = (model.logits(state, question, use_cache=True) / model.temperature).softmax(-1)
        expected = torch.zeros_like(probabilities)
        expected[target] = 1
        outcomes.append({"type": question.kind, "correct": int(probabilities.argmax()) == target,
                         "nll": -math.log(max(float(probabilities[target]), 1e-30)),
                         "brier": float(((probabilities - expected) ** 2).sum()),
                         "probability": float(probabilities.max())})
    result = summarize(outcomes)
    result["by_type"] = {kind: summarize([row for row in outcomes if row["type"] == kind])
                         for kind in sorted({row["type"] for row in outcomes})}
    return result


def summarize(outcomes):
    count = len(outcomes)
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    calibration_error = 0.0
    for index in range(10):
        bucket = [row for row in outcomes if min(9, int(row["probability"] * 10)) == index]
        if bucket:
            calibration_error += abs(sum(row["probability"] - row["correct"] for row in bucket)) / count
    return {"questions": count, "accuracy": sum(row["correct"] for row in outcomes) / count,
            "nll": sum(row["nll"] for row in outcomes) / count,
            "brier": sum(row["brier"] for row in outcomes) / count, "ece_10_bins": calibration_error}


def main():
    parser = argparse.ArgumentParser(description="Train and run RWKV-7 typed decision adapters")
    parser.add_argument("command", choices=("train", "predict", "evaluate", "serve"))
    parser.add_argument("--base", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data")
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--state-tuning", action="store_true", help="Train initial WKV matrices and decision head; no LoRA")
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--candidate-batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Save adapter snapshots every N updates and at epoch ends; 0 disables snapshots")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--state-lr", type=float, help="Initial WKV learning rate; defaults to --lr")
    parser.add_argument("--state-max-norm", type=float, help="Project initial WKV to this global Frobenius norm after each update")
    parser.add_argument("--sampling", choices=("natural", "noul-balanced"), default="natural",
                        help="Balance Noul labels within each source, preserving source/type positions")
    parser.add_argument("--objective", choices=("cross-entropy", "proper-score", "rlcd", "reinforce", "pathwise"), default="cross-entropy",
                        help="reinforce/pathwise compare Gaussian policy gradients; rlcd is the legacy noisy objective")
    parser.add_argument("--exploration-std", type=float, default=0.05,
                        help="Gaussian logit exploration standard deviation")
    parser.add_argument("--policy-samples", type=int, default=32,
                        help="Even antithetic sample count for reinforce/pathwise, without extra backbone forwards")
    parser.add_argument("--spherical-weight", type=float, default=0.5,
                        help="Weight of the spherical proper-score reward")
    parser.add_argument("--rps-weight", type=float, default=0.5,
                        help="Weight of the ranked probability proper-score reward")
    parser.add_argument("--baseline-decay", type=float, default=0.95,
                        help="EMA decay for centering logged proper-score rewards; does not change gradients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command in ("train", "evaluate") and not args.data:
        parser.error("--data is required for train/evaluate")
    if args.command == "predict" and not args.request:
        parser.error("--request is required for predict")
    if args.command == "train" and Path(args.adapter).exists():
        parser.error("Adapter already exists; choose a new output path")
    if args.state_tuning and args.rank:
        parser.error("--state-tuning requires --rank 0")
    for value in (args.state_lr, args.state_max_norm):
        if value is not None and (args.command != "train" or not args.state_tuning or
                                  not math.isfinite(value) or value <= 0):
            parser.error("State optimizer controls require train --state-tuning and finite positive values")
    if args.sampling != "natural" and args.command != "train":
        parser.error("--sampling requires train")
    if args.objective != "cross-entropy" and args.command != "train":
        parser.error("--objective requires train")
    if args.objective in ("rlcd", "reinforce", "pathwise") and (not math.isfinite(args.exploration_std) or args.exploration_std <= 0):
        parser.error("Policy exploration standard deviation must be finite and positive")
    if args.policy_samples < 2 or args.policy_samples % 2:
        parser.error("Policy samples must be an even integer of at least two")
    for value in (args.spherical_weight, args.rps_weight):
        if not math.isfinite(value) or value < 0:
            parser.error("Proper-score weights must be finite and nonnegative")
    if not math.isfinite(args.baseline_decay) or not 0 <= args.baseline_decay < 1:
        parser.error("Baseline decay must be finite and in [0, 1)")
    if args.checkpoint_every < 0 or (args.checkpoint_every and args.command != "train"):
        parser.error("--checkpoint-every requires train and a nonnegative interval")
    checkpoint_dir = Path(args.adapter).parent / (Path(args.adapter).stem + "-checkpoints")
    if args.command == "train" and args.checkpoint_every and checkpoint_dir.exists():
        parser.error("Checkpoint directory already exists; choose a new adapter path")
    torch.manual_seed(args.seed)
    base_hash, vocab_hash = file_hash(args.base), file_hash(args.vocab)
    settings = {"rank": args.rank, "head_size": args.head_size, "max_tokens": args.max_tokens,
                "candidate_batch": args.candidate_batch}
    if args.state_tuning:
        settings["state_tuning"] = True
    if args.command != "train":
        payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
        settings = payload["settings"]
        if payload["metadata"].get("base_sha256") != base_hash or payload["metadata"].get("vocab_sha256") != vocab_hash:
            parser.error("Base weights/tokenizer do not match this adapter")
        if not payload["trained_steps"]:
            parser.error("Adapter has no training steps")
        if args.command == "evaluate" and file_hash(args.data) == payload["metadata"].get("data_sha256"):
            parser.error("Evaluation data is identical to training data; supply a held-out file")
    model = DecisionModel.from_base(args.base, args.vocab, device=args.device, dtype=getattr(torch, args.dtype), **settings)
    if args.command == "train":
        sourced_records = read_data(args.data, with_sources=True)
        records = [row[:3] for row in sourced_records]
        sources = [row[3] for row in sourced_records]
        for state, question, _ in records:
            model._encode(state, question)
        metadata = {"base_sha256": base_hash, "vocab_sha256": vocab_hash,
                    "data_sha256": file_hash(args.data), "seed": args.seed,
                    "lr": args.lr, "dtype": args.dtype, "epochs": args.epochs,
                    "head_initialization": "before_lora", "base": str(Path(args.base).resolve()),
                    "state_lr": args.state_lr, "state_max_norm": args.state_max_norm,
                    "sampling": args.sampling, "max_steps": args.max_steps,
                    "objective": args.objective, "exploration_std": args.exploration_std,
                    "spherical_weight": args.spherical_weight, "rps_weight": args.rps_weight,
                    "baseline_decay": args.baseline_decay, "policy_samples": args.policy_samples,
                    "reward_definition": ("score-only-rps-stable-log" if args.objective in ("reinforce", "pathwise")
                                          else "legacy"),
                    "training_cli_sha256": file_hash(__file__)}

        def save_checkpoint(current_model, epoch):
            current_model.save_adapter(checkpoint_dir / f"step-{current_model.trained_steps}.pt",
                                       {**metadata, "epoch": epoch, "snapshot_only": True})

        losses = train(model, records, args.epochs, args.lr, args.seed, args.max_steps,
                       args.checkpoint_every, save_checkpoint, args.state_lr, args.state_max_norm,
                       args.sampling, sources, args.objective, args.exploration_std,
                       args.spherical_weight, args.rps_weight, args.baseline_decay, args.policy_samples)
        model.save_adapter(args.adapter, metadata)
        result = {"adapter": args.adapter, "steps": model.trained_steps, "final_training_loss": losses[-1]}
        if model.initial_wkv is not None:
            result["initial_state_norm"] = float(model.initial_wkv.detach().norm())
    else:
        model.load_adapter(args.adapter)
        if args.command == "serve":
            from .server import serve

            serve(model, args.host, args.port)
            return
        elif args.command == "evaluate":
            result = evaluate(model, read_data(args.data))
        else:
            request = json.loads(Path(args.request).read_text(encoding="utf-8"))
            result = model.system_one(request["state"], request["questions"])
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
