import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import urllib.request


EXPECTED = {
    "train": "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad",
    "development": "8d5765d7aec4d08c61854f4664ca79ec2ba44ed092e9b967eaf61ed86c496d9c",
}


def main():
    parser = argparse.ArgumentParser(description="Prepare a small hashed Kev decision-v7 pilot")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    parser.add_argument("--output", default="runs/pilot-data")
    parser.add_argument("--download-dir", help="Directory containing upstream-train/development.jsonl")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    generator = random.Random(42)
    manifest = {"upstream": "jaredpalmer/kev-suites/v7/decision-v7", "seed": 42,
                "sources": ["agnews", "boolq", "sst5"], "files": {}}
    seen_groups, seen_states = set(), set()
    for split, limit in (("train", 64), ("development", 16)):
        url = args.endpoint.rstrip("/") + f"/datasets/jaredpalmer/kev-suites/resolve/main/v7/decision-v7/{split}.jsonl"
        if args.download_dir:
            raw = (Path(args.download_dir) / f"upstream-{split}.jsonl").read_bytes()
        else:
            request = urllib.request.Request(url, headers={"User-Agent": "rwkv-jev/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != EXPECTED[split]:
            raise ValueError(f"Upstream {split} checksum mismatch")
        records = [json.loads(line) for line in raw.splitlines() if line]
        generator.shuffle(records)
        selected = []
        counts = Counter()
        for record in records:
            source = record.get("_meta", {}).get("source")
            if source not in manifest["sources"] or counts[source] >= limit:
                continue
            group = (source, record["_meta"]["group_id"])
            state_digest = hashlib.sha256(json.dumps(record["state"], sort_keys=True).encode()).hexdigest()
            if group in seen_groups or state_digest in seen_states:
                continue
            questions = {name: value for name, value in record["questions"].items()
                         if value.get("src") == source}
            if not questions:
                continue
            selected.append({**record, "questions": questions})
            counts[source] += 1
            seen_groups.add(group)
            seen_states.add(state_digest)
        if any(counts[source] != limit for source in manifest["sources"]):
            raise ValueError(f"Insufficient records: {counts}")
        content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected)
        (destination / f"{split}.jsonl").write_text(content, encoding="utf-8")
        manifest["files"][split] = {"upstream_sha256": digest, "source_counts": dict(counts),
                                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                                    "questions": sum(len(record["questions"]) for record in selected)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
