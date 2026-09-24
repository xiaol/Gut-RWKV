import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from rwkv_jev.cli import file_hash, summarize
from rwkv_jev.model import DecisionModel
from rwkv_jev.schema import Question, serialize


def read_questions(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("_meta", {})
            for name, definition in record["questions"].items():
                question = Question.parse(definition)
                rows.append({"state": record["state"], "question": question,
                             "target": question.target(definition["label"]),
                             "source": definition.get("src", metadata.get("source", "unknown")),
                             "group": metadata.get("group_id"),
                             "instance_key": serialize([metadata, name, record["state"], definition]),
                             "id": f"{line_number}:{name}"})
    if not rows:
        raise ValueError("Dataset contains no labeled questions")
    return rows


def question_key(row):
    question = row["question"]
    return serialize([row["state"], question.kind, question.instructions,
                      sorted(zip(question.names, question.descriptions))])


def data_audit(training, development):
    training_states = {serialize(row["state"]) for row in training}
    training_groups = {(row["source"], row["group"]) for row in training if row["group"] is not None}
    training_keys = {question_key(row) for row in training}
    labels = defaultdict(set)
    for row in training + development:
        labels[question_key(row)].add(row["question"].names[row["target"]])
    return {
        "training_questions": len(training), "development_questions": len(development),
        "training_by_source": dict(Counter(row["source"] for row in training)),
        "development_by_source": dict(Counter(row["source"] for row in development)),
        "development_questions_with_training_state": sum(serialize(row["state"]) in training_states for row in development),
        "development_questions_with_training_group": sum((row["source"], row["group"]) in training_groups for row in development if row["group"] is not None),
        "development_questions_with_training_question": sum(question_key(row) in training_keys for row in development),
        "conflicting_question_keys_across_splits": sum(len(values) > 1 for values in labels.values()),
        "scope": "Exact serialized states, source/group IDs, and semantic question keys; no fuzzy overlap detection.",
    }


@torch.no_grad()
def audit_model(model, rows, subset_keys, parity_per_source):
    outcomes, parity = [], []
    counts = Counter()
    remaining_subset = Counter(subset_keys)
    for index, row in enumerate(rows):
        question = row["question"]
        probabilities = (model.logits(row["state"], question, use_cache=True) / model.temperature).softmax(-1)
        expected = torch.zeros_like(probabilities)
        expected[row["target"]] = 1
        prediction = int(probabilities.argmax())
        instance_key = row["instance_key"]
        in_subset = remaining_subset[instance_key] > 0
        if in_subset:
            remaining_subset[instance_key] -= 1
        outcome = {"id": row["id"], "source": row["source"], "type": question.kind,
                   "target": row["target"], "prediction": prediction,
                   "correct": prediction == row["target"],
                   "nll": float(-probabilities[row["target"]].clamp_min(1e-30).log()),
                   "brier": float(((probabilities - expected) ** 2).sum()),
                   "probability": float(probabilities.max()),
                   "uniform_accuracy": 1 / len(question.names),
                   "pilot_subset": in_subset}
        outcomes.append(outcome)
        if counts[row["source"]] < parity_per_source:
            complete = (model.logits(row["state"], question) / model.temperature).softmax(-1)
            parity.append({"id": row["id"], "source": row["source"],
                           "max_probability_difference": float((complete - probabilities).abs().max()),
                           "argmax_agrees": int(complete.argmax()) == prediction})
            counts[row["source"]] += 1
        if (index + 1) % 100 == 0:
            print(json.dumps({"evaluated": index + 1, "total": len(rows)}), flush=True)
    result = summarize(outcomes)
    for field in ("source", "type"):
        result[f"by_{field}"] = {}
        for value in sorted({row[field] for row in outcomes}):
            group = [row for row in outcomes if row[field] == value]
            result[f"by_{field}"][value] = {**summarize(group),
                "uniform_accuracy": sum(row["uniform_accuracy"] for row in group) / len(group)}
    subset = [row for row in outcomes if row["pilot_subset"]]
    if any(remaining_subset.values()):
        raise ValueError("Some requested subset instances are absent from the evaluation file")
    result["pilot_subset"] = summarize(subset) if subset else None
    result["cache_parity"] = {
        "questions": len(parity),
        "argmax_disagreements": sum(not row["argmax_agrees"] for row in parity),
        "max_probability_difference": max((row["max_probability_difference"] for row in parity), default=None),
        "probes": parity,
    }
    return result, outcomes


def main():
    parser = argparse.ArgumentParser(description="Audit matched development performance and exact data overlap")
    parser.add_argument("--base", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-per-source", type=int, default=2)
    args = parser.parse_args()
    if args.parity_per_source < 0:
        parser.error("--parity-per-source must be nonnegative")
    training, development = read_questions(args.train), read_questions(args.data)
    audit = data_audit(training, development)
    subset_keys = Counter(row["instance_key"] for row in read_questions(args.subset))
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    identities = {name: file_hash(getattr(args, name)) for name in ("base", "vocab", "adapter", "train", "data", "subset")}
    for name in ("base", "vocab"):
        if payload["metadata"][f"{name}_sha256"] != identities[name]:
            raise ValueError(f"Adapter {name} hash mismatch")
    if payload["metadata"]["data_sha256"] != identities["train"]:
        raise ValueError("Supply this adapter's actual training data")
    if identities["train"] == identities["data"]:
        raise ValueError("Training and evaluation files are identical")
    model = DecisionModel.from_base(args.base, args.vocab, device=args.device,
                                   dtype=torch.bfloat16, **payload["settings"])
    model.load_adapter(args.adapter)
    model.eval()
    metrics, outcomes = audit_model(model, development, subset_keys, args.parity_per_source)
    result = {"adapter": args.adapter, "sha256": identities, "trained_steps": model.trained_steps,
              "data_audit": audit, "metrics": metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".outcomes.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in outcomes), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
