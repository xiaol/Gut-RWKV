# State-update and label-balance screening protocol

This follow-up to [AUDIT.md](AUDIT.md) tests interventions against the observed
full-training regression. The completed seed-42 screen reaches **997/1,468
(67.9%)** with a tenfold lower state learning rate. Repeating lower LR and control
with seeds 43 and 44 gives lower LR **68.39% mean accuracy, 0.45 percentage-point
sample SD**, versus control **46.46%, 19.94 points SD**. Control wins seed 44,
so lower LR improves consistency in these runs, not every seed. The prior
state-tuned pilot scores 61.4% on the same development questions. No JevBench
test data is used.

## Three-seed replication

The subsequent objective comparison is recorded in the
[proper-score and noisy-objective screen](#proper-score-and-noisy-objective-screen).
It does not replace the three-seed lower-LR evidence below.

Seeds 43 and 44 repeat the lower-LR arm and its matched control without changing
the data pool, head LR, code, 1,536-update budget or evaluation protocol. Both
arms use natural shuffling; seed changes affect initialization and example order.
The two failed seed-42 interventions were not repeated.

| Seed | Control correct / 1,468 | Control accuracy | Lower-LR correct / 1,468 | Lower-LR accuracy |
| --- | ---: | ---: | ---: | ---: |
| 42 | 519 | 35.35% | 997 | 67.92% |
| 43 | 507 | 34.54% | 1,005 | 68.46% |
| 44 | **1,020** | **69.48%** | 1,010 | 68.80% |
| Mean ± sample SD | | 46.46% ± 19.94 points | | **68.39% ± 0.45 points** |

The lower LR gains 478 and 498 correct decisions in seeds 42 and 43, but loses
10 in seed 44. Report the whole comparison rather than selecting either the
collapsed controls or the best individual run. Three initializations of the
same 1,468 examples do not create 4,404 independent test questions or establish
a population confidence interval. The best observed individual checkpoint is
the seed-44 control; lower LR is the more consistent configuration in this sample.

Control's final state norms are 9.18, 9.81 and 10.44, while lower LR's are 1.48,
1.61 and 1.72. The largest-norm control is the one that succeeds, further showing
that final norm alone does not diagnose collapse. State/head interaction and
training trajectory remain relevant.

Across the three seeds, lower-LR mean NLL is 1.3903, Brier 0.5069 and ECE 0.2103.
Control means are 1.4369, 0.6364 and 0.1314. Confidence reliability still needs
work; lower LR's mean NLL and ECE remain worse than the older pilot's 1.1551 and
0.1567. All three lower-LR models have 0/48 cached/full-sequence argmax flips in
the fixed probes, with maximum probability differences 0.0806, 0.0962 and 0.0854.
This is a limited probe, not exact numerical equivalence.

Report: [state-screen-replication.json](reports/state-screen-replication.json).
The seed-43/44 artifacts are in `runs/state-screen-seedSEED-ARM/`. All four
replication runs completed training and evaluation. No new training is queued.

The next quality gate is a predeclared longer-budget comparison with checkpoints
and calibration examples excluded from training and model selection. Do not fit
calibration and claim independent quality using this repeatedly inspected
development set. Keep the JevBench test protocol for the selected model; these
numbers remain development measurements, not an official rank.

## Completed seed-42 results

All rows use the fixed 1,536-update budget and identical development questions.
The head learning rate stays at `1e-4`; only the initial-state rate changes in
the winning arm. Values are raw and uncalibrated.

| Arm | Correct / 1,468 | Accuracy | NLL | Brier | ECE | Final state norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 519 | 35.4% | 1.5326 | 0.7181 | 0.1058 | 9.1830 |
| Lower state LR | **997** | **67.9%** | **1.4523** | **0.5089** | 0.2075 | 1.4842 |
| State norm cap | 501 | 34.1% | 1.4990 | 0.6982 | **0.0506** | 6.0000 |
| Balanced Noul | 433 | 29.5% | 1.4967 | 0.6954 | 0.0628 | 7.9187 |

The lower-rate arm gains 621 answers and loses 143 against the matched control,
a net gain of 478 (32.6 percentage points). Against the historical pilot it gains
233 and loses 137, a net gain of 96 (6.5 points). The pilot comparison changes
data and training budget, so only the control comparison isolates state LR.

Boolean accuracy rises to **388/472 (82.2%)**, with 291 false and 181 true
predictions. Control predicts false 450/472 times and scores 285/472 (60.4%).
Balanced training produces 228 false and 244 true predictions but only 237/472
(50.2%) correct: removing prediction imbalance does not itself solve the task.
The norm cap was active and still failed to recover quality; a small final norm
alone is not a sufficient explanation for the lower-rate arm's success.

Choice accuracy for the lower-rate arm is 485/756 (64.2%); Score accuracy is
124/240 (51.7%). Its state norm remains much smaller than control's. This supports
testing the update scale, rather than simply adding more updates with the old
schedule. The seed-42 lower-LR adapter is at
`runs/state-screen-v1-lower_state_lr/lower_state_lr/adapter.pt`.

**Confidence quality remains unresolved.** Compared with the pilot, NLL worsens
from 1.1551 to 1.4523 and ECE from 0.1567 to 0.2075, although Brier improves from
0.5557 to 0.5089. Better accuracy is not a universal probability-quality gain.
Low ECE on the failed arms is also not evidence of useful predictions.

Cached versus complete-sequence argmax disagreement on 48 probes is 8 for
control, 0 for lower LR, 6 for norm cap and 17 for balanced Noul. Maximum absolute
probability differences are 0.1250, 0.0806, 0.1919 and 0.1354 respectively. These
probes do not establish an exhaustive numerical bound; all primary scores use
the same cached path. Failed arms need particular caution around near ties.

Report: [state-screen-seed42.json](reports/state-screen-seed42.json), including
adapter and input hashes, source metrics, exact sampled label counts and paired
comparison counts. Raw outcomes, snapshots and logs remain under each
`runs/state-screen-v1-ARM/` directory. This is one seed on reused development
data, not a ranking or a test of general state-tuning superiority.

## Proper-score and noisy-objective screen

Two additional seed-42 runs keep the frozen G1k 3B preview, natural shuffle,
15,576-question training pool, head LR `1e-4`, state LR `1e-5`, and fixed final
1,536-update budget. They compare against the existing lower-LR cross-entropy
arm on the same 1,468 development questions. Intermediate snapshots are retained
every 384 updates; the results below use the final checkpoint, without choosing
an intermediate peak. This remains development screening, not JevBench.

| Objective | Correct / 1,468 | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-entropy, lower state LR | 997 | 67.92% | **1.4523** | 0.5089 | 0.2075 |
| Proper score | **1,008** | **68.66%** | 1.5008 | **0.4994** | **0.2058** |
| Noisy proper score (`rlcd`) | 986 | 67.17% | 1.6720 | 0.5356 | 0.2234 |
| Historical pilot, unmatched training | 901 | 61.38% | 1.1551 | 0.5557 | 0.1567 |

The deterministic proper score combines log probability, a spherical score with
weight 0.5, and negative normalized ranked probability score (RPS) with weight
0.5. The noisy arm adds independent Gaussian noise with standard deviation
0.05 to logits during training and differentiates through the perturbed logits.
Inference uses unperturbed logits for both arms.

**The CLI name `rlcd` denotes an exploratory reparameterized objective, not
REINFORCE, GRPO, or a reproduction of Laya.** Subtracting a detached EMA reward
baseline changes the logged loss but does not change the parameter gradients
or reduce their variance. Thus these results test proper scoring and logit
noise, not a policy-gradient RL recipe. Final centered training losses are not
comparable measures of model quality.

This screen also applies RPS to unordered Choice candidates in their supplied
order. That gives the training objective an arbitrary ordering dependence,
although inference still scores candidates independently. A subsequent objective
revision should restrict ordinal RPS to Score questions and separately compare
an actual policy-gradient estimator against the differentiable reward. The
current results retain the actual trained definition rather than relabeling it.

The subsequent [policy-gradient screen](POLICY_SCREEN.md) implements that
Score-only RPS change in separate `pathwise` and `reinforce` objectives and
compares the two gradient estimators under matched sampling.

Proper scoring gains 72 answers and loses 61 against cross-entropy: just 11 net
correct answers, or 0.75 percentage points, in one seed. Noisy scoring gains 70
and loses 81. This does not establish a reliable gain for proper scoring, and
the tested noise setting does not improve overall accuracy or confidence metrics.
Proper scoring improves Brier slightly but worsens NLL; calibration remains
unresolved. Keep the replicated lower-LR configuration as the reference.

| Objective | Choice / 756 | Noul / 472 | Score / 240 | Noul false / true predictions |
| --- | ---: | ---: | ---: | ---: |
| Cross-entropy | 64.15% | 82.20% | 51.67% | 291 / 181 |
| Proper score | 64.55% | 83.47% | 52.50% | 281 / 191 |
| Noisy proper score | 61.90% | 83.69% | 51.25% | 294 / 178 |

Source changes are mixed: proper scoring raises AG News from 76.72% to 80.17%
and BoolQ from 71.25% to 73.75%, while MNLI falls from 41.38% to 35.34%.
All 24 source breakdowns, input/adapter hashes, checkpoint metadata and paired
counts are in [rlcd-screen-seed42.json](reports/rlcd-screen-seed42.json).
Exact train/development state, group and question overlap checks remain zero.

Cached/full-sequence argmax disagreements on the fixed 48 probes are 0 for
cross-entropy, 0 for proper score and 4 for noisy scoring. Their maximum absolute
probability differences are 0.0806, 0.1242 and 0.0960. These are limited probes;
all reported accuracy uses cached inference. The noisy arm needs particular
caution near ties. Final state norms are 1.4842, 1.5634 and 1.5897 respectively.

Both new runs and evaluations are complete. Artifacts are under
`runs/rlcd-screen-proper-v2/` and `runs/rlcd-screen-rlcd-v2/`; the earlier
`runs/rlcd-screen-proper/` directory is an incomplete failed launch and is not
part of this comparison. No new training is queued. Code was not hashed before
these launches, so the report preserves input/adapter identities and metadata
without claiming a predeclared code manifest.

Reproduce the proper-score training with a fresh output directory:

```bash
.venv/bin/gut-rwkv train \
  --base ../models/rwkv7/rwkv-g1k-3b-temp-5441.pth \
  --vocab ../models/rwkv7/rwkv_vocab_v20230424.txt \
  --data runs/pilot-data/upstream-train.jsonl \
  --adapter runs/objective-proper-reproduction/adapter.pt \
  --state-tuning --epochs 1 --max-steps 1536 --checkpoint-every 384 \
  --lr 1e-4 --state-lr 1e-5 --seed 42 --objective proper-score \
  --spherical-weight 0.5 --rps-weight 0.5 --baseline-decay 0.95
```

Use `--objective rlcd --exploration-std 0.05` and a different adapter directory
for the noisy arm. Evaluate with `scripts/audit_decisions.py`, the complete
`upstream-development.jsonl`, and the original `development.jsonl` subset from
`runs/pilot-data/`. Reserve a separate calibration split before fitting
temperature, replicate any revised objective, and select a model before a full
JevBench evaluation. Neither new adapter has been evaluated on JevBench.

## Fixed state-control comparison

Every arm starts from the same frozen G1k 3B preview and the same randomly
initialized head, seed 42, bf16 base, fp32 trainable parameters, AdamW and
gradient clipping at 1.0. All use the same complete 15,576-question training
pool, head learning rate `1e-4`, and **1,536 optimizer updates**. This is a
screening budget, not a reproduction of the historical 31,152-update run.
All four final checkpoints are evaluated on the same 1,468-question development
file. The earlier 192-question pilot is a reference, not a training-matched arm.

| Arm | State learning rate | Global state norm cap | Sampling |
| --- | ---: | ---: | --- |
| Control | 1e-4 | None | Natural shuffle |
| Lower state LR | 1e-5 | None | Natural shuffle |
| State norm cap | 1e-4 | 6.0 | Natural shuffle |
| Balanced Noul | 1e-4 | None | Balance true/false within each Noul source |

Each intervention changes one factor relative to control. The cap of 6.0 is
chosen near the previously useful pilot state's norm (about 5.88), not from new
development outcomes. It projects the entire initial-WKV tensor onto a ball
around zero after each update; it does not cap each layer independently, reset
optimizer moments, or constrain the head. These are fresh initializations,
not continuation from the pilot.

The balanced sampler first generates the same shuffled source/type positions
as control. At Noul positions it alternates labels independently within each
source, choosing examples with replacement from that source/label pool. Thus
each prefix has true/false counts differing by at most one per Noul source;
source frequencies and all non-Noul example positions are preserved. Both
labels must exist in each affected training source. Duplicate draws and omitted
majority examples are expected; the manifest records unique-question counts and
sampled label/source counts. Logical questions represented as Choice are not
rebalanced by this intervention.

## Measurement

Snapshots are saved at updates 384, 768, 1,152 and 1,536. The primary comparison
uses the fixed final update, with accuracy, NLL, Brier, ECE, source/type metrics,
boolean prediction counts and cached/full-sequence numerical probes. State norm
is logged during training and recorded in the training summary. Snapshots allow
later learning-curve diagnostics but are not optimizer-resume checkpoints.

Interpretation rules:

- Compare each intervention with the new matched control, not just the old
  full-data result that used many more updates.
- A reduction in false-answer collapse must also improve decision quality;
  balanced predictions alone are not success.
- Compare against the 61.4% reference before promoting a checkpoint. This is
  one seed on development data; repeat promising configurations with additional
  seeds before making general claims.
- A null result at 1,536 updates does not exclude a later effect. If the norm cap
  is never reached, that arm has not tested active constraint of state growth.
- Calibration and any full JevBench evaluation follow model selection; this
  screen does not establish a rank or SOTA.

## Execution and status

```bash
.venv/bin/python scripts/state_screen.py \
  --base ../models/rwkv7/rwkv-g1k-3b-temp-5441.pth \
  --vocab ../models/rwkv7/rwkv_vocab_v20230424.txt \
  --train runs/pilot-data/upstream-train.jsonl \
  --development runs/pilot-data/upstream-development.jsonl \
  --subset runs/pilot-data/development.jsonl \
  --output runs/state-screen-v1 --wait-for-gpu
```

Use `--dry-run` to inspect the exact commands and data/sample identities without
starting work. An actual run requires a fresh output directory, preventing
accidental duplicate runs or overwriting adapters. The runner executes arms
sequentially. Before each train/evaluate process it waits for a GPU with at
least 30,000 MiB free and no reported compute processes; it does not stop other
GPU jobs or reserve hardware against unrelated future jobs.

To run arms concurrently, give separate invocations `--arm NAME`, distinct
output directories, and `--gpu-uuid UUID` for different GPUs. Each invocation
still verifies its own inputs and records a complete manifest.

`plan.json` records commands, sampled distributions and data/code SHA-256 hashes.
Inputs are rechecked before launching each process; changing experiment code or
data while queued aborts the run rather than silently changing the protocol.
`status.json` records the actual phase and process ID. Per-arm logs and adapters
are stored in subdirectories; `results.json` is updated after each evaluated arm.
Create a `STOP` file in the output directory to cancel a wait or stop before the
next subprocess. It does not interrupt a subprocess already running.

A waiting status means that the next subprocess has not started yet; it is not
a training result. Use each run's `status.json` for its current phase.
Generated checkpoints, data, logs and run manifests remain in ignored `runs/`.
