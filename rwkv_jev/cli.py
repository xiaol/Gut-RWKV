import argparse
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


def read_data(path):
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
                records.append((record["state"], question, question.target(value["label"])))
    if not records:
        raise ValueError("Dataset contains no labeled questions")
    return records


def train(model, records, epochs, learning_rate, seed, max_steps=0):
    if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0 or max_steps < 0:
        raise ValueError("Invalid training settings")
    generator = random.Random(seed)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)
    model.train()
    losses = []
    for epoch in range(epochs):
        order = list(range(len(records)))
        generator.shuffle(order)
        for index in order:
            state, question, target = records[index]
            optimizer.zero_grad(set_to_none=True)
            logits = model.logits(state, question)
            loss = functional.cross_entropy(logits[None], torch.tensor([target], device=logits.device))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            model.trained_steps += 1
            losses.append(float(loss.detach()))
            if len(losses) == 1 or len(losses) % 10 == 0:
                print(json.dumps({"step": model.trained_steps, "epoch": epoch + 1, "loss": losses[-1]}), flush=True)
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
    parser.add_argument("--lr", type=float, default=1e-4)
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
        records = read_data(args.data)
        for state, question, _ in records:
            model._encode(state, question)
        losses = train(model, records, args.epochs, args.lr, args.seed, args.max_steps)
        model.save_adapter(args.adapter, {"base_sha256": base_hash, "vocab_sha256": vocab_hash,
                                         "data_sha256": file_hash(args.data), "seed": args.seed,
                                         "lr": args.lr, "dtype": args.dtype,
                                         "base": str(Path(args.base).resolve())})
        result = {"adapter": args.adapter, "steps": model.trained_steps, "final_training_loss": losses[-1]}
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
