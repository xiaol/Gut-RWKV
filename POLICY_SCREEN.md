# Gaussian policy-gradient screen

This experiment compares an actual REINFORCE estimator with a differentiable
control for the same expected proper-score reward. It trains only the initial
RWKV state and decision head over the frozen G1k 3B preview. It is an independent
experiment, not a reproduction of Laya or TypeSafe's private training recipe.

The seed-42 screen found **67.03%** for REINFORCE and **68.53%** for its matched
pathwise control. A three-seed pathwise replication now averages **68.89% ± 0.31
points**, versus **68.39% ± 0.45 points** for lower-LR cross-entropy. Pathwise
wins all three matched seeds, but the gain is only 22 total answers across the
reused development questions. The public JevBench cohort still favors
cross-entropy by one answer (56 versus 55).

## Fixed protocol

Both arms use seed 42, the same natural data shuffle, the 15,576-question
decision-v7 training pool, and a final budget of 1,536 optimizer updates.
Head LR is `1e-4`; state LR is `1e-5`. Initialization, AdamW, gradient clipping
at 1.0, bf16 base, and fp32 trainable parameters match the earlier lower-LR
screen. Checkpoints are saved every 384 updates; only the final checkpoint is
the primary comparison. Evaluation uses all 1,468 development questions,
without opening JevBench test data or fitting calibration on development.

The matched arms differ only in gradient estimator:

- `pathwise`: differentiate through sampled logits and their proper-score reward.
- `reinforce`: detach sampled logits and rewards, then optimize the reward-weighted
  log probability of the sampled Gaussian action.

Both draw 32 logit vectors per question, using 16 Gaussian perturbations paired
with their negatives. Each sample has standard deviation 0.05. These samples
reuse one model forward pass; they do not require 32 RWKV forwards. Inference
uses the unperturbed model logits, with no exploration noise.

The action is a continuous logit vector whose softmax defines the decision
distribution. This differs from sampling one categorical answer for a binary
correct/incorrect reward. The reward is log probability of the target plus
0.5 times the spherical score, with an additional negative normalized RPS term
of weight 0.5 **only for ordered Score questions**. Log probabilities use
`log_softmax` so very confident mistakes retain a useful gradient in the
pathwise arm. Choice and Noul rewards do not use arbitrary candidate positions.

For logits `z`, fixed scale `sigma`, and detached action `a`, REINFORCE minimizes
`-mean((reward(a) - baseline) * log Normal(a; z, sigma))`. The baseline is the
detached proper-score reward at the unperturbed logits. It is independent of
the sampled action. With exact antithetic pairs, its gradient contribution
cancels; the paired perturbations provide the variance reduction. The Gaussian
normalization constant is omitted because its derivative is zero at fixed scale.
The EMA shown in training logs is diagnostic and is not this policy baseline.

No PPO clipping, GRPO normalization, KL reference model, or warm-start from a
selected adapter is used. This is a fresh-initialization estimator comparison;
it does not test every possible RL recipe or RL after supervised training.
Because the reward is differentiable and known, REINFORCE is not expected to
provide an automatic advantage over its direct gradient. The experiment tests
whether its optimization behavior helps at the same model-update budget.

## Execution

The runner freezes commands, source code hashes, input hashes and sampled label
counts in `plan.json` before launching. It rechecks them before training and
evaluation. Separate invocations use separate idle GPUs and output directories.

```bash
.venv/bin/python scripts/state_screen.py --suite policy --arm reinforce \
  --base ../models/rwkv7/rwkv-g1k-3b-temp-5441.pth \
  --vocab ../models/rwkv7/rwkv_vocab_v20230424.txt \
  --train runs/pilot-data/upstream-train.jsonl \
  --development runs/pilot-data/upstream-development.jsonl \
  --subset runs/pilot-data/development.jsonl \
  --output runs/policy-reinforce-reproduction \
  --steps 1536 --checkpoint-every 384 --seed 42 --wait-for-gpu
```

Use `--arm pathwise` and a fresh output directory for the matched control.
Omitting `--arm` runs both sequentially. The CLI also exposes `--policy-samples`
and `--exploration-std` for future experiments; the screen fixes them to 32 and
0.05. Legacy `proper-score` and `rlcd` retain their original reward definition
so previously reported experiments remain reproducible.

## Results

### Seed 42 estimator comparison

| Objective | Correct / 1,468 | Accuracy | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical lower-LR cross-entropy | 997 | 67.92% | 1.4523 | 0.5089 | 0.2075 |
| Historical proper score, legacy reward | 1,008 | 68.66% | 1.5008 | 0.4994 | 0.2058 |
| Typed reward, pathwise control | 1,006 | 68.53% | **1.3079** | **0.4835** | **0.1862** |
| Typed reward, REINFORCE | 984 | 67.03% | 1.5964 | 0.5140 | 0.2093 |

REINFORCE gains 83 answers and loses 105 against its matched
control, a net loss of 22 answers (1.50 percentage points). It gains 76 and loses
89 against cross-entropy. These findings are specific to this seed, initialization,
sample count, exploration scale and update budget; they do not rule out other RL
methods or supervised-then-RL training.

The pathwise control gains 73 and loses 64 against cross-entropy, only nine net
answers. It also lowers NLL, Brier and ECE relative to that seed-42 reference,
making it a candidate for replication rather than a proven replacement for the
three-seed baseline. Against legacy proper scoring, it loses just two net answers
and improves all three probability metrics, but reward definition and noise
sampling also change. This does not isolate the effect of restricting RPS.
Its NLL and ECE remain worse than the older pilot's 1.1551 and 0.1567.

| New objective | Choice / 756 | Noul / 472 | Score / 240 | Noul false / true predictions |
| --- | ---: | ---: | ---: | ---: |
| Pathwise | 65.08% | 83.26% | 50.42% | 266 / 206 |
| REINFORCE | 61.64% | 82.42% | 53.75% | 284 / 188 |

REINFORCE improves ordinal argmax accuracy but loses more Choice decisions.
For example, MNLI falls from 40.52% for pathwise to 32.76% for REINFORCE, and
AG News falls from 79.31% to 74.14%. Ordinal NLL is worse for REINFORCE despite
the higher ordinal accuracy. Source-level effects and confidence quality are
not captured by accuracy alone.

On the fixed 48 cached/full-sequence probes, pathwise has one argmax disagreement
and maximum probability difference 0.0980; REINFORCE has zero disagreements and
maximum difference 0.0501. Primary metrics use cached inference. Final state
norms are 1.6532 and 1.7613. Exact train/development state, group and question
overlap checks are zero for both arms; these do not detect paraphrases or
pretraining contamination.

Complete metrics for all 24 sources, paired comparisons, commands, code/input
hashes and adapter identities are in
[policy-screen-seed42.json](reports/policy-screen-seed42.json). Raw artifacts are
under `runs/policy-screen-seed42-pathwise/` and
`runs/policy-screen-seed42-reinforce/`. The saved manifests match in code, inputs,
sampled questions and labels; both report completed training and evaluation.
No further training is queued.

Validation: 33 CPU tests pass, and the CUDA recurrence/initial-state-gradient
test passes separately. Tests check the REINFORCE estimator against a large-sample
pathwise gradient, nonzero state updates, unordered reward permutation invariance,
stable gradients for confident errors, and matched runner settings.

Keep the replicated lower-LR cross-entropy configuration as the established
The pathwise replication is promising enough to carry forward, but it is not a
replacement for an external benchmark result. Mean pathwise NLL is 1.3661 versus
1.3903 for cross-entropy; mean Brier is 0.4956 versus 0.5069; mean ECE is 0.1992
versus 0.2103. Seed-level NLL varies substantially, and calibration remains
unresolved. Full per-source metrics and hashes are in
[policy-screen-replication.json](reports/policy-screen-replication.json).

Keep cross-entropy as the public JevBench baseline, replicate pathwise on a
reserved development/calibration protocol, and freeze model selection before the
full JevBench evaluation. Development results around 68% do not establish a
68% JevBench score.
