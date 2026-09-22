import argparse
import hashlib
from pathlib import Path
import subprocess


REPOSITORY = "BlinkDL/temp-latest-training-models"
REVISION = "a7d80e331bd787f9a8ff88d993fb2af147d5b2c0"
FILENAME = "rwkv-g1k-3b-temp-5441.pth"
SIZE = 5896273469
SHA256 = "eea4fbf9fee9624958f84de3a8e3fd55a1840d76e7be53e211618d48f3ce0e58"


def verify(path):
    if path.stat().st_size != SIZE:
        raise ValueError("Checkpoint size mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch")


def main():
    parser = argparse.ArgumentParser(description="Download and verify the pinned G1k 3B preview (5.9 GB)")
    parser.add_argument("--directory", default="models")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    destination = Path(args.directory)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / FILENAME
    if final.exists():
        verify(final)
    else:
        partial = destination / (FILENAME + ".part")
        if not partial.exists() or partial.stat().st_size != SIZE:
            url = f"{args.endpoint.rstrip('/')}/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
            subprocess.run(["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "20",
                            "--max-time", "1800", "--continue-at", "-", "--output", str(partial), url], check=True)
        verify(partial)
        partial.rename(final)
    print(f"Verified {final}: {SHA256}")


if __name__ == "__main__":
    main()
