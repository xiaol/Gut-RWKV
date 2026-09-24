# Matched G1k quality audit

The state-tuned pilot scores **901/1,468 (61.4%)** on the complete decision-v7
development set. The full-trained state adapter scores **650/1,468 (44.3%)** on
the same questions. This establishes a regression between these two adapters;
the earlier comparison of 68.8% on 48 questions against 44.3% on 1,468 did not.
Neither number is a JevBench result or an official rank.

## Controlled adaptation comparison

All models use the pinned G1k 3B preview, bf16, a 128-wide decision head, seed 42,
AdamW at `1e-4`, and gradient clipping at 1.0. The first three rows use the same
192 training questions, shuffled in the same order for 384 updates. Initial
heads match; their first training loss is 0.6910744905. Parameter budgets differ.
The existing state pilot is reused; head-only and rank-16 LoRA are new runs.
The final checkpoint is fixed at two epochs for each arm, rather than selecting
the best intermediate development accuracy.

| Adaptation | Trainable parameters | Training questions / updates | Full dev accuracy | NLL | Brier | ECE | Original 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Head only | 333,056 | 192 / 384 | 704/1,468 (48.0%) | 1.3817 | 0.7129 | 0.2095 | 30/48 |
| Initial state + head | 5,575,936 | 192 / 384 | **901/1,468 (61.4%)** | **1.1551** | **0.5557** | **0.1567** | **33/48** |
| Rank-16 LoRA + head | 23,926,016 | 192 / 384 | 800/1,468 (54.5%) | 1.2861 | 0.6805 | 0.2123 | 27/48 |
| Full-data initial state + head | 5,575,936 | 15,576 / 31,152 | 650/1,468 (44.3%) | 1.5599 | 0.7387 | 0.1876 | 20/48 |

State tuning wins this single-seed pilot comparison. This does not establish a
general advantage over LoRA: ranks, learning rates and parameter budgets were
not tuned equally, and the earlier 0.4B pilot favored LoRA. The full-data row
changes both data and update count; it cannot isolate which change caused the
regression. There are no intermediate checkpoints from that historical run.

Machine-readable results, identities, all 24 development sources, paired error
counts and ablations: [matched-g1k-audit.json](reports/matched-g1k-audit.json).
Raw per-question outcomes remain locally under `runs/matched-audit/` with hashes
in the report. No test or organizer-held data was opened.

## What fails

The full-trained model loses 529 questions that the pilot answers correctly and
gains 278, a net loss of 251 correct decisions. Several source-level regressions
are substantial:

| Source | Questions | State pilot | Full-trained state |
| --- | ---: | ---: | ---: |
| AG News | 116 | 80.2% | 51.7% |
| BoolQ | 80 | 70.0% | 38.8% |
| DBpedia14 | 116 | 79.3% | 44.0% |
| TREC | 116 | 69.0% | 24.1% |
| Amazon ordinal | 80 | 45.0% | 17.5% |
| SST-5 | 80 | 47.5% | 30.0% |

On Noul questions the full model predicts false on **471/472** inputs. Its
290/472 (61.4%) accuracy barely exceeds the always-false baseline of 289/472
(61.2%). It predicts false on all 184 AG News yes/no questions, all 80 IMDB
questions, all 80 Yelp yes/no questions, and 79/80 BoolQ questions. BoolQ's
training labels actually favor true (639/1,000), so per-source label frequency
alone does not explain this behavior. The pilot's Noul accuracy is 342/472
(72.5%). This is a collapse toward a fixed answer, not evidence of reasoning.

## State and head ablations

These exploratory interventions use only the original 48 development questions.
Swapping components of jointly trained adapters is a diagnostic, not a trained
replacement or an independent generalization result.

| Initial state | Decision head | Correct / 48 |
| --- | --- | ---: |
| Pilot | Pilot | **33** |
| Zero | Pilot | 25 |
| Full-trained | Full-trained | 20 |
| Zero | Full-trained | 22 |
| Pilot | Full-trained | 29 |
| Full-trained | Pilot | 11 |

The pilot's learned state is useful, and substituting it into the full model
recovers nine decisions on this subset. Substituting only the pilot head does
not recover quality. These observations implicate the learned state and its
interaction with the head; they do not prove a particular optimization cause.

New head-only snapshots at steps 96/192/288/384 score 30/29/29/30 out of 48.
LoRA snapshots score 17/29/30/27, with NLL 1.2085/1.7156/2.9197/1.1726. Accuracy
alone would conceal large confidence errors. No snapshot is selected using
JevBench, and no historical full-run learning curve can be reconstructed.

## Correctness and numerical limits

- No exact train/development state overlap, source/group overlap, semantic
  question overlap, or conflicting labels was found for either training file.
  These checks do not detect paraphrases or pretraining contamination.
- Labels pass the typed parser. Context, question and candidate token segments
  are shared by training and cached inference; the reference tests cover state
  isolation, candidate order and padding behavior.
- The CUDA recurrence passes a reference comparison for outputs, final state,
  initial-state gradients and all six recurrent inputs, with nonzero initial
  state and inert padding. This is a synthetic numerical test, not exhaustive
  validation of all checkpoint activations.
- On two questions per source (48 probes), cached versus complete-sequence
  inference changes the predicted class 0/48 times for head-only, pilot state
  and LoRA, and 1/48 for full-trained state. Maximum absolute probability
  differences are respectively 0.06693, 0.05973, 0.01045 and 0.03906. These are
  larger than the original small batching probes; numerical equivalence must
  not be claimed. All main scores use the same cached inference path.
- Fifteen tests pass, including CUDA/reference gradients, initialization parity,
  snapshot scheduling and exact subset-instance accounting. The subset matcher
  counts original instances, so duplicated or reordered variants do not inflate
  the original 48-question sample.

## Next experiment

The follow-up [state-control screen](STATE_SCREEN.md) is complete. Lower state
LR `1e-5` reaches 68.39% mean development accuracy across three seeds, with
0.45 percentage-point sample SD. Norm capping and boolean-label balancing fail
to recover quality in their seed-42 runs.

A subsequent matched objective screen scores 68.66% for proper scoring and
67.17% for noisy proper scoring, versus 67.92% for lower-LR cross-entropy at
seed 42. These are fixed final 1,536-update checkpoints. Proper scoring gains
11 net answers but worsens NLL; noisy scoring does not help overall. The CLI
calls the latter `rlcd`, but it uses pathwise gradients, not REINFORCE or GRPO.
See [the report](reports/rlcd-screen-seed42.json) for paired comparisons,
per-source/type metrics, prediction balance and cached/full-sequence probes.

The subsequent actual [policy-gradient comparison](POLICY_SCREEN.md) scores
67.03% for Gaussian REINFORCE and 68.53% for its matched typed-reward pathwise
control. The control also improves NLL, Brier and ECE over seed-42 cross-entropy,
but gains only nine answers. This supports replication of the pathwise candidate;
it does not support promoting the tested REINFORCE configuration.

Keep the replicated lower-LR model as the reference, test revised objectives
across seeds, reserve a separate calibration split, and use the full JevBench
protocol for the selected model. Calibration, official ranking and production
concurrency remain unestablished.

## Reproduction

Use the G1k base and vocabulary paths from the README. The new pilot controls
use the existing hashed `runs/pilot-data/train.jsonl`:

```bash
.venv/bin/gut-rwkv train --base "$BASE" --vocab "$VOCAB" \
  --data runs/pilot-data/train.jsonl --adapter runs/control/adapter.pt \
  --epochs 2 --lr 1e-4 --seed 42 --checkpoint-every 96
```

Use a fresh adapter path and add `--rank 16` for LoRA or `--state-tuning` for
state tuning. Run `scripts/audit_decisions.py` with each actual training file,
`--data runs/pilot-data/upstream-development.jsonl`, and
`--subset runs/pilot-data/development.jsonl`. See the README for the full command.

`scripts/evaluate_checkpoints.py --help` describes loading several compatible
snapshots with one base-model load, `--ablate-state` for a zero-state comparison,
and `--head-from` for a component swap. Snapshots contain adapter parameters,
not optimizer state. All experiment adapters remain outside the source repo.
