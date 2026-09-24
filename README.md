# Gut-RWKV

<p align="center">
  <strong>Typed decisions from recurrent memory.</strong><br>
  An experimental Jev-style decision model built on RWKV-7 G1k 3B.
</p>

<p align="center">
  <a href="https://github.com/xiaol/Gut-RWKV-Jev/actions/workflows/tests.yml"><img src="https://github.com/xiaol/Gut-RWKV-Jev/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/base-RWKV--7%20G1k%203B-6f42c1" alt="RWKV-7 G1k 3B">
  <img src="https://img.shields.io/badge/status-research%20prototype-orange" alt="Research prototype">
</p>

Gut-RWKV reads a context and typed questions, then returns probabilities for
**Choice**, **Noul** (boolean) and **Score** answers. It does not generate answer
tokens. A single RWKV prefill is reused across question and candidate branches,
so inference keeps RWKV's constant-size recurrent state instead of a growing KV
cache.

## At a glance

| Track | Result | Scope |
| --- | ---: | --- |
| Lower-LR cross-entropy | **149/231 (64.5%)** | All currently public JevBench tasks; 231/231 valid |
| Pathwise typed reward | **143/231 (61.9%)** | Same public tasks; 231/231 valid |
| Lower-LR development baseline | **68.39% ± 0.45** | Three seeds on the reused development set |
| Official JevBench rank | **Unranked** | 303 organizer-held tasks are not public |

The lower-LR cross-entropy run is the current external baseline. The public
comparison is useful for reproducibility, but it is not a full 534-task JevBench
submission or an SOTA claim. Results and limitations are tracked in
[EXPERIMENTS.md](EXPERIMENTS.md), [AUDIT.md](AUDIT.md) and [SOTA.md](SOTA.md).

## What is distinctive

- **State tuning:** learns initial RWKV WKV state matrices and a typed decision
  head while keeping the base model frozen; LoRA and head-only controls are included.
- **Typed outputs:** handles Choice, Noul and ordered Score questions with one
  interface and explicit validity checks.
- **Recurrent branching:** encodes context once, then branches questions and
  candidates without a context-growing attention cache.
- **Open measurement:** keeps public-task comparisons, seed screens, loss curves
  and negative results alongside the implementation.

This is independent research, not TypeSafe's Jev implementation or its private
training recipe. The RL-style objectives are exploratory screens, not Laya recipe
parity; the current evidence favors the simpler lower-LR cross-entropy baseline.

## Research status

The project is a reproducible research prototype rather than a general-purpose
release. Development screens include proper scoring, pathwise rewards and
Gaussian REINFORCE, with learning curves and three-seed state-LR replication. See
[STATE_SCREEN.md](STATE_SCREEN.md), [POLICY_SCREEN.md](POLICY_SCREEN.md) and
[LEARNING_CURVES.md](LEARNING_CURVES.md) for protocols and raw summaries.

## Roadmap

1. Warm-start pathwise training from the cross-entropy adapter with a KL penalty.
2. Ablate reward terms and policy-sample counts under a fixed evaluation split.
3. Add a reserved calibration/source-family split before tuning benchmark settings.
4. Test a permutation-equivariant or pairwise candidate head.
5. Submit the locked adapter to the complete JevBench evaluation.

## Documentation map

| Start here | What it covers |
| --- | --- |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Reproduction commands and public JevBench comparison |
| [STATE_SCREEN.md](STATE_SCREEN.md) | State-LR, seed and objective screens |
| [POLICY_SCREEN.md](POLICY_SCREEN.md) | REINFORCE and pathwise policy experiments |
| [LEARNING_CURVES.md](LEARNING_CURVES.md) | Extracted loss curves and checkpoint summaries |
| [RESEARCH.md](RESEARCH.md) | Prior work, model identity and benchmark context |
| [SOTA.md](SOTA.md) | Ranking claims, limits and next experiments |

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

For controlled comparisons, the head is initialized before installing LoRA, so
the same seed and architecture give the same initial head in all three modes.
Older LoRA pilots used a different initialization order. Existing adapters still
load their saved weights normally.

Add `--checkpoint-every 1000` to save adapter snapshots every 1,000 updates and
at epoch ends in `adapter-checkpoints/` beside the output adapter. These snapshots
are for evaluation; they do not include optimizer state for resuming training.

State tuning also supports `--state-lr` independently of the head's `--lr`, and
`--state-max-norm` to project the learned initial WKV onto a global norm bound
after each update. `--sampling noul-balanced` balances boolean labels within
each source while preserving source/type frequencies; training records need
`src` or `_meta.source`. The default remains natural shuffling. See
[STATE_SCREEN.md](STATE_SCREEN.md) for the four-arm experimental protocol.

`--objective proper-score` optimizes log, spherical and ranked probability scores;
`--spherical-weight` and `--rps-weight` default to 0.5. The experimental
`--objective rlcd --exploration-std 0.05` adds Gaussian logit noise and uses
pathwise gradients. Its detached EMA baseline (`--baseline-decay 0.95`) centers
logged loss only and does not affect updates. These legacy objectives also use the supplied
ordering of unordered Choice candidates, a limitation of this exploratory screen.
The default objective remains `cross-entropy`.

`--objective reinforce` uses a detached reward and Gaussian policy log probability;
`--objective pathwise` differentiates the same sampled reward directly. Both use
`--policy-samples 32` paired perturbations, stable log probabilities and RPS only
for Score questions. They are independent implementations, not Laya recipe parity.
See [POLICY_SCREEN.md](POLICY_SCREEN.md) for the estimator and matched protocol.

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

For a comparison on identical questions, `scripts/audit_decisions.py` reports
metrics by source/type, the pilot subset, exact train/development overlap, and
cached/full-sequence probability differences. It verifies adapter training,
base and vocabulary hashes, and writes per-question outcomes beside the report:

```bash
.venv/bin/python scripts/audit_decisions.py --base "$BASE" --vocab "$VOCAB" \
  --adapter runs/state/adapter.pt --train train.jsonl --data heldout.jsonl \
  --subset runs/pilot-data/development.jsonl --output runs/audit/state.json
```

Use each adapter's actual training file and the same development file for all
models. Exact overlap checks do not detect paraphrases or pretraining exposure.

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
