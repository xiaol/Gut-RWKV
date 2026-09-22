# Gut-RWKV

**Typed decisions from recurrent memory.** An experimental Jev-style model built
on RWKV-7: context and bounded questions in, Choice / Score / Noul probabilities
out, with no generated answer tokens.

Gut-RWKV supports **initial-state tuning plus a decision head**, or **LoRA plus
the same head**, over a frozen base. Inference encodes the input context once,
batches independent question branches, then batches candidate branches. This
uses RWKV's constant-size recurrent memory without a context-growing KV cache.

This is independent work, not TypeSafe's Jev or its private training recipe.
The G1k 3B state-tuned pilot reaches **33/48 (68.8%)** on the small held-out
development sample and **38/72 (52.8%)** on JevBench's original public subset.
It is uncalibrated and is not a general-purpose release. See
[EXPERIMENTS.md](EXPERIMENTS.md), [RESEARCH.md](RESEARCH.md) and [SOTA.md](SOTA.md)
for comparisons, prior work and the path toward a competitive benchmark result.

## Architecture

```text
learned initial WKV state (optional)
             |
context -> RWKV prefill -> shared recurrent state
                           |-- question A --|-- candidate 1 -- scalar head
                           |               |-- candidate 2 -- scalar head -> softmax
                           |-- question B --|-- candidate 1 -- scalar head
                                           |-- candidate 2 -- scalar head -> softmax
```

- **Choice:** highest-probability supplied option and the full distribution.
- **Noul:** probability of true for a proposition.
- **Score:** distribution over ordered levels and expected zero-based index.
- Candidates are scored independently, giving option-order equivariance in
  exact arithmetic. Ties use input order; floating point partitioning introduces
  small numerical differences. The head cannot jointly compare candidate content.
- Questions share only input state. They do not read each other's answers.
- `candidate_batch` bounds question and candidate batches. More questions still
  require more compute. HTTP calls are serialized; continuous batching across
  clients is not implemented.

## Install

Python 3.10+ and PyTorch 2.4+. CUDA requires a matching toolkit and `ninja` for
the included kernel; the CPU reference path needs no compiler.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

In the original workspace, `python -m venv --system-site-packages .venv` can
reuse installed CUDA PyTorch. No neighboring checkout is required: the small
RWKV runtime is vendored with its license in [THIRD_PARTY.md](THIRD_PARTY.md).

Download official `.pth` weights and `rwkv_vocab_v20230424.txt` separately.
Dimensions are inferred from weights; adapters are specific to exact base and
tokenizer hashes. The local pilot uses:

```bash
BASE=../models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
VOCAB=../models/rwkv7/rwkv_vocab_v20230424.txt
```

The requested G1k 3B checkpoint is in BlinkDL's preview repository:
[`rwkv-g1k-3b-temp-5441.pth`](https://huggingface.co/BlinkDL/temp-latest-training-models).
It is a temporary training checkpoint, distinct from the stable G1j 2.9B release.
Download the pinned, hash-verified snapshot with `scripts/download_g1k.py`.
See [RESEARCH.md](RESEARCH.md) for its exact identity.

## State tuning and training

`--state-tuning` trains per-layer initial WKV matrices and the decision head.
All base weights and time-shift vectors stay frozen; no LoRA is installed.
The learned matrices initialize the model before it reads the input context.

```bash
.venv/bin/gut-rwkv train --base "$BASE" --vocab "$VOCAB" \
  --data train.jsonl --adapter runs/state/adapter.pt \
  --state-tuning --epochs 2 --lr 1e-4
```

Replace `--state-tuning` with `--rank 8` for LoRA. Omit both to train only the head.
Combining the flags is rejected to keep the modes distinct. Trainable parameters
use fp32; the frozen base can use bf16. Base checkpoint files stay unchanged.

Training uses complete context/question/candidate sequences: the fused kernel
differentiates its initial state input but not its returned final state.
Inference shares and branches recurrent states. Oversized inputs raise errors
rather than truncating evidence.

Each JSONL line is a request plus a `label` for each question:

```json
{"state":"Refund my duplicate charge.","questions":{"team":{"type":"choice","instructions":"Route this ticket","criteria":{"billing":"Payments","technical":"Bugs"},"label":"billing"}}}
```

Choice labels are names, Noul labels booleans, Score labels integer level indices.
Split by source/template/customer and keep development, calibration and test
separate. Checkpoints record base/tokenizer/data hashes. Rejecting identical train
and evaluation files cannot detect all example overlap in differently saved files.

```bash
.venv/bin/gut-rwkv evaluate --base "$BASE" --vocab "$VOCAB" \
  --data heldout.jsonl --adapter runs/state/adapter.pt --output runs/state/evaluation.json
.venv/bin/gut-rwkv predict --base "$BASE" --vocab "$VOCAB" \
  --request examples/request.json --adapter runs/state/adapter.pt
```

Evaluation reports accuracy, NLL, multiclass Brier and 10-bin ECE overall/by type.
Score accuracy uses argmax, while its output also exposes the expectation.
Untrained heads are rejected by the API. Choice/Score `confidence` is
`(K * max(p) - 1) / (K - 1)`, or one for a single choice; this is not TypeSafe
Score formula parity or a measured correctness probability. Calibration is needed.

## Browser demo and benchmark endpoint

```bash
.venv/bin/gut-rwkv serve --base "$BASE" --vocab "$VOCAB" \
  --adapter runs/state/adapter.pt --port 8000
```

Open **http://127.0.0.1:8000** for editable context/questions and probability bars.
The server provides `POST /v1/systemone`, `GET /v1/models` and `GET /healthz`.
Its typed answers match JevBench's `typesafe` adapter. It binds only to localhost
and serializes model calls; within each request the branches are batched.
It does not claim full TypeSafe SDK compatibility or production concurrency.
`generated_tokens` is zero; no invented billing usage or hosted cost is returned.

The original workspace also contains the trained G1k pilot at
`runs/state-pilot-g1k-3b/adapter.pt`; use it with
`../models/rwkv7/rwkv-g1k-3b-temp-5441.pth`. A fresh clone needs to reproduce training;
weights and ignored `runs/` artifacts are not part of the source repository.

## Reproduce the pilot and measure batching

```bash
.venv/bin/python scripts/prepare_pilot.py
.venv/bin/gut-rwkv train --base "$BASE" --vocab "$VOCAB" \
  --data runs/pilot-data/train.jsonl --adapter runs/state-pilot-0.4b/adapter.pt \
  --state-tuning --epochs 2 --lr 1e-4
.venv/bin/python scripts/benchmark_branches.py --base "$BASE" --vocab "$VOCAB" \
  --adapter runs/state-pilot-0.4b/adapter.pt --output runs/state-pilot-0.4b/branches.json
```

The sampler verifies Kev decision-v7 hashes, selects 64 train / 16 development
records each from AG News, BoolQ and SST-5, and excludes duplicate groups/states
across splits. It never opens locked test data. If necessary, use
`--endpoint https://hf-mirror.com` or `--download-dir DIRECTORY` containing
`upstream-train.jsonl` and `upstream-development.jsonl`; both routes verify hashes.

The branch benchmark compares serial/batched questions sharing the same prefix,
includes prefill on each request, and reports latency, decisions/s, state bytes
and probability differences. It measures repeated identical questions within a
request, not diverse concurrent clients. See [RESEARCH.md](RESEARCH.md) for JevBench.
