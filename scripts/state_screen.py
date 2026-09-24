import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from rwkv_jev.cli import file_hash, read_data, training_order


ARMS = {
    "control": [],
    "lower_state_lr": ["--state-lr", "1e-5"],
    "state_norm_cap": ["--state-max-norm", "6"],
    "balanced_noul": ["--sampling", "noul-balanced"],
}

POLICY_ARMS = {
    name: ["--state-lr", "1e-5", "--objective", name, "--policy-samples", "32",
           "--exploration-std", "0.05", "--spherical-weight", "0.5", "--rps-weight", "0.5"]
    for name in ("pathwise", "reinforce")
}


def idle_gpu(minimum_free_mib=30000, preferred_uuid=None):
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"], text=True)
    applications = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    occupied = {line.strip() for line in applications.splitlines() if line.strip()}
    for identifier, free_memory in csv.reader(io.StringIO(devices)):
        identifier = identifier.strip()
        if (preferred_uuid is None or identifier == preferred_uuid) and identifier not in occupied and int(free_memory.strip()) >= minimum_free_mib:
            return identifier
    return None


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def status(output, phase, **details):
    value = {"phase": phase, "pid": os.getpid(), "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    write_json(output / "status.json", value)
    print(json.dumps(value), flush=True)


def build_commands(root, args, name):
    arms = POLICY_ARMS if getattr(args, "suite", "state") == "policy" else ARMS
    directory = args.output / name
    shared = ["--base", str(args.base), "--vocab", str(args.vocab), "--adapter", str(directory / "adapter.pt")]
    training = [sys.executable, "-m", "rwkv_jev.cli", "train", *shared,
                "--data", str(args.train), "--state-tuning", "--epochs", "1",
                "--max-steps", str(args.steps), "--lr", "1e-4", "--seed", str(args.seed),
                "--checkpoint-every", str(args.checkpoint_every),
                "--output", str(directory / "train-summary.json"), *arms[name]]
    evaluation = [sys.executable, str(root / "scripts" / "audit_decisions.py"), *shared,
                  "--train", str(args.train), "--data", str(args.development),
                  "--subset", str(args.subset), "--parity-per-source", "2",
                  "--output", str(directory / "development.json")]
    return training, evaluation


def wait_for_gpu(output, wait, name, gpu_uuid=None):
    announced = False
    while True:
        if (output / "STOP").exists():
            status(output, "cancelled", arm=name)
            return None
        device = idle_gpu(preferred_uuid=gpu_uuid)
        if device:
            return device
        if not announced:
            status(output, "waiting_for_gpu", arm=name, minimum_free_mib=30000)
            announced = True
        if not wait:
            raise RuntimeError("No idle GPU with 30,000 MiB free; use --wait-for-gpu to queue")
        time.sleep(30)


def verify_inputs(plan):
    for filename, expected in {**plan["input_sha256"], **plan["code_sha256"]}.items():
        if file_hash(filename) != expected:
            raise RuntimeError(f"Queued experiment input changed: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Run matched state-tuning or policy-gradient experiments on an idle GPU")
    for name in ("base", "vocab", "train", "development", "subset", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--checkpoint-every", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--suite", choices=("state", "policy"), default="state")
    parser.add_argument("--arm", choices=tuple(ARMS) + tuple(POLICY_ARMS), help="Run only this intervention; otherwise run the suite sequentially")
    parser.add_argument("--gpu-uuid", help="Wait specifically for this GPU UUID")
    args = parser.parse_args()
    arms = POLICY_ARMS if args.suite == "policy" else ARMS
    if args.arm and args.arm not in arms:
        parser.error("Selected arm does not belong to this suite")
    root = Path(__file__).resolve().parents[1]
    for name in ("base", "vocab", "train", "development", "subset", "output"):
        setattr(args, name, getattr(args, name).resolve())
    sourced = read_data(args.train, with_sources=True)
    records, sources = [row[:3] for row in sourced], [row[3] for row in sourced]
    if not 1 <= args.steps <= len(records) or args.checkpoint_every < 1:
        parser.error("Steps must fit in one pass through the training pool; checkpoint interval must be positive")
    inputs = {str(getattr(args, name)): file_hash(getattr(args, name))
              for name in ("base", "vocab", "train", "development", "subset")}
    if inputs[str(args.train)] == inputs[str(args.development)]:
        parser.error("Training and development files must differ")
    paths = list((root / "rwkv_jev").rglob("*")) + list((root / "scripts").glob("*.py"))
    code = {str(path): file_hash(path) for path in sorted(paths)
            if path.is_file() and path.suffix in {".py", ".cu", ".cpp"}}
    plan = {"suite": args.suite, "seed": args.seed, "steps": args.steps, "head_lr": 1e-4, "gpu_uuid": args.gpu_uuid,
            "checkpoint_every": args.checkpoint_every, "input_sha256": inputs, "code_sha256": code,
            "protocol": "Fresh initializations; fixed final-step evaluation on development; one seed; no JevBench test access",
            "arms": {}}
    for name in arms:
        if args.arm and name != args.arm:
            continue
        sampling = "noul-balanced" if name == "balanced_noul" else "natural"
        indices = training_order(records, random.Random(args.seed), sampling, sources)[:args.steps]
        label_counts = defaultdict(Counter)
        for index in indices:
            if records[index][1].kind == "noul":
                label_counts[sources[index]][str(records[index][2])] += 1
        training, evaluation = build_commands(root, args, name)
        plan["arms"][name] = {"train": training, "evaluate": evaluation,
                              "unique_training_questions": len(set(indices)),
                              "source_counts": dict(Counter(sources[index] for index in indices)),
                              "noul_label_counts": dict(label_counts)}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    status(args.output, "prepared")
    try:
        summaries = {}
        for name, arm in plan["arms"].items():
            directory = args.output / name
            directory.mkdir()
            for phase in ("train", "evaluate"):
                device = wait_for_gpu(args.output, args.wait_for_gpu, name, args.gpu_uuid)
                if device is None:
                    return
                verify_inputs(plan)
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": device, "OMP_NUM_THREADS": "4"}
                status(args.output, phase, arm=name, gpu=device)
                with (directory / f"{phase}.log").open("w") as log:
                    subprocess.run(arm[phase], cwd=root, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            report = json.loads((directory / "development.json").read_text())
            outcomes = [json.loads(line) for line in (directory / "development.outcomes.jsonl").read_text().splitlines()]
            summaries[name] = {"metrics": report["metrics"], "sha256": report["sha256"],
                               "noul_predictions": dict(Counter(str(row["prediction"]) for row in outcomes if row["type"] == "noul")),
                               "trained_steps": report["trained_steps"]}
            write_json(args.output / "results.json", summaries)
        status(args.output, "complete", results=str(args.output / "results.json"))
    except Exception as error:
        status(args.output, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
