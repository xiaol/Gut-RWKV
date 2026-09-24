import argparse
import json
from pathlib import Path

import torch

from rwkv_jev.cli import evaluate, file_hash, read_data
from rwkv_jev.model import DecisionModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate compatible adapter snapshots with one base-model load")
    parser.add_argument("--base", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ablate-state", action="store_true")
    parser.add_argument("--head-from", help="Exploratory ablation: replace each adapter's head with this adapter's head")
    args = parser.parse_args()
    hashes = {name: file_hash(getattr(args, name)) for name in ("base", "vocab", "data")}
    records = read_data(args.data)
    donor = None
    if args.head_from:
        donor = torch.load(args.head_from, map_location="cpu", weights_only=True)
        for name in ("base", "vocab"):
            if donor["metadata"][f"{name}_sha256"] != hashes[name]:
                raise ValueError(f"Donor {name} hash mismatch")
        if donor["metadata"]["data_sha256"] == hashes["data"]:
            raise ValueError("Evaluation file matches donor training file")
    model = None
    results = []
    for adapter in args.adapters:
        payload = torch.load(adapter, map_location="cpu", weights_only=True)
        for name in ("base", "vocab"):
            if payload["metadata"][f"{name}_sha256"] != hashes[name]:
                raise ValueError(f"Adapter {name} hash mismatch")
        if payload["metadata"]["data_sha256"] == hashes["data"]:
            raise ValueError("Evaluation file matches training file")
        if not payload["trained_steps"]:
            raise ValueError("Adapter is untrained")
        if model is None:
            model = DecisionModel.from_base(args.base, args.vocab, device=args.device,
                                           dtype=torch.bfloat16, **payload["settings"])
        model.load_adapter(adapter)
        if donor is not None:
            model.head.load_state_dict({name.removeprefix("head."): value for name, value in donor["weights"].items()
                                        if name.startswith("head.")})
        row = {"adapter": adapter, "sha256": file_hash(adapter), "steps": model.trained_steps,
               "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
               "adapter_bytes": Path(adapter).stat().st_size, "metrics": evaluate(model, records)}
        if donor is not None:
            row["head_from"] = {"adapter": args.head_from, "sha256": file_hash(args.head_from)}
        if args.ablate_state:
            if model.initial_wkv is None:
                raise ValueError("--ablate-state requires a state-tuned adapter")
            row["initial_state_norm"] = float(model.initial_wkv.detach().norm())
            with torch.no_grad():
                model.initial_wkv.zero_()
            row["zero_state_metrics"] = evaluate(model, records)
        results.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sha256": hashes, "results": results}, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
