import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from rwkv_jev import DecisionModel, Question
from rwkv_jev.cli import file_hash
from rwkv_jev.schema import serialize


@torch.no_grad()
def serial_questions(model, state, questions):
    _, prefix = model._forward([model._tokens("State: " + serialize(state))])
    return {name: (model.logits(state, Question.parse(question), use_cache=True, prefix_state=prefix)
                   / model.temperature).softmax(-1).tolist() for name, question in questions.items()}


def main():
    parser = argparse.ArgumentParser(description="Measure within-request branching, not concurrent clients")
    parser.add_argument("--base", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(count < 1 for count in args.counts):
        parser.error("Counts and repeats must be positive")
    payload = torch.load(args.adapter, map_location="cpu", weights_only=True)
    if payload["metadata"]["base_sha256"] != file_hash(args.base) or payload["metadata"]["vocab_sha256"] != file_hash(args.vocab):
        parser.error("Adapter base/tokenizer hash mismatch")
    model = DecisionModel.from_base(args.base, args.vocab, device=args.device, dtype=torch.bfloat16, **payload["settings"])
    model.load_adapter(args.adapter)
    model.eval()
    state = "I was charged twice. Please refund the extra payment."
    definition = {"type": "choice", "instructions": "Route this request", "criteria": {
        "billing": "Payments and refunds", "technical": "Software bugs", "sales": "New purchases"}}
    results = []
    for count in args.counts:
        questions = {f"question_{index}": definition for index in range(count)}
        timings, outputs = {}, {}
        for mode, operation in (("serial_shared_prefix", serial_questions),
                                ("batched_shared_prefix", lambda model, state, questions: model.system_one(state, questions))):
            operation(model, state, questions)
            elapsed = []
            for _ in range(args.repeats):
                if model.backbone.device.type == "cuda":
                    torch.cuda.synchronize(model.backbone.device)
                started = time.perf_counter()
                outputs[mode] = operation(model, state, questions)
                if model.backbone.device.type == "cuda":
                    torch.cuda.synchronize(model.backbone.device)
                elapsed.append((time.perf_counter() - started) * 1000)
            timings[mode] = {"milliseconds": elapsed, "median_ms": statistics.median(elapsed),
                             "decisions_per_second": count * 1000 / statistics.median(elapsed)}
        difference = max(abs(serial - parallel) for name in questions
                         for serial, parallel in zip(outputs["serial_shared_prefix"][name],
                                                    outputs["batched_shared_prefix"]["answers"][name]["probabilities"].values()))
        results.append({"questions": count, "max_probability_difference": difference, "timings": timings,
                        "speedup": timings["serial_shared_prefix"]["median_ms"] / timings["batched_shared_prefix"]["median_ms"]})
        print(json.dumps(results[-1]), flush=True)
    model_state = model.backbone.zero_state(1)
    state_bytes = sum(value.numel() * value.element_size() for value in (model_state.att_x, model_state.wkv, model_state.ffn_x))
    report = {"device": torch.cuda.get_device_name(model.backbone.device) if model.backbone.device.type == "cuda" else "cpu",
              "dtype": "bfloat16", "adapter_sha256": file_hash(args.adapter),
              "state_bytes_per_branch": state_bytes, "candidate_batch": model.settings["candidate_batch"],
              "workload": "Repeated identical three-option questions; warm model; no cross-request cache",
              "scope": "Within-request branch batching, not concurrent client serving", "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
