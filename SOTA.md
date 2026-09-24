# SOTA status and plan

## Current ranking

Gut-RWKV has **no official JevBench rank**. Our best measured local result is:

- JevBench original public subset: **56/72 = 77.8% accuracy** with lower-LR cross-entropy.
- Strict schema validity: **72/72**.
- Brier: **0.3858**; ECE: **0.2143**.
- Local A100 HTTP p50/p95: **333/377 ms**.
- Cost: **unknown**, because this is self-hosted; it must not be reported as zero.

The pathwise candidate scores **55/72 = 76.4%**, with Brier 0.4047 and ECE
0.1746. The earlier 38/72 pilot remains documented as a historical baseline.

Across all 231 currently public tasks (`easy` + `original` + public `hard`),
lower-LR cross-entropy scores **149/231 = 64.5%** and pathwise scores
**143/231 = 61.9%**, both with 231/231 strict validity. This broader public run
reverses the small development-set pathwise advantage. It is still not an
official JevBench rank because 303 organizer-held tasks and the composite cost
protocol are unavailable. See [public-231 comparison](reports/jevbench-public231-comparison.json).

The matched public-cohort rerun gives the lower-LR cross-entropy adapter **56/72
(77.8%)**, versus **55/72 (76.4%)** for the pathwise typed-reward candidate.
This confirms that lower-LR cross-entropy is the current external baseline. It
is still only the original-public cohort, not the 534-decision score or an
official rank. See [the comparison report](reports/jevbench-public-comparison.json).

That result is not comparable to JevBench's official composite until the complete
534-decision protocol and deployment rules are run. It is also not currently
competitive with the published top rows: Jev 1.13.0 is listed at 75.4, SemIf at
74.7, djev at 74.3, and openJev Verdict at 72.5 in JevBench v1.2. The exact
leaderboard can change; re-check it before publication.

The 68.8% result is from a 48-question development sample selected from three
training sources. It is useful for debugging, not a SOTA claim.

The first complete decision-v7 state-tuning run is lower: **44.3% accuracy on
1,468 held-out development questions** (NLL 1.5599, Brier 0.7387, ECE 0.1876).
The subsequent matched audit evaluates the pilot on the same complete split:
**61.4%**, compared with **54.5%** for a matched rank-16 LoRA pilot and **48.0%**
for head-only. The full-trained state model predicts false on 471/472 boolean
questions. See [AUDIT.md](AUDIT.md) for the actual regression, component
ablations and numerical differences between inference paths. One seed on
development data does not establish a benchmark rank or a general adaptation
advantage.

The follow-up [state-update screen](STATE_SCREEN.md) improves development
accuracy to **67.9%** by lowering only the state LR to `1e-5` while keeping head
LR at `1e-4`; its matched 1,536-update control scores 35.4%. This is not a JevBench
score. The new adapter's NLL and ECE are worse than the pilot's, so accuracy alone
does not justify a claim of better calibrated decisions.

The completed three-seed replication puts lower LR at **68.39% ± 0.45 percentage
points** (mean ± sample SD), versus control **46.46% ± 19.94 points**. Control
seed 44 achieves the highest individual score, **69.48%**. This supports greater
observed stability for lower LR, not uniform superiority or SOTA. All runs reuse
the same development questions; no official benchmark rank has been obtained.

The completed seed-42 objective screen reaches **68.66%** with deterministic
proper scoring and **67.17%** with noisy proper scoring, versus **67.92%** for
the matched lower-LR cross-entropy arm. These are development scores. Proper
scoring gains only 11 answers and worsens NLL; noise does not improve the
overall result. The CLI's `rlcd` label refers to a reparameterized noisy objective,
not a reproduced policy-gradient RL algorithm. Neither adapter has been evaluated
on JevBench. Details: [objective screen](STATE_SCREEN.md#proper-score-and-noisy-objective-screen).

An actual Gaussian REINFORCE follow-up scores **67.03%**, below its matched
typed-reward pathwise control at **68.53%**. The latter has NLL 1.3079, Brier
0.4835 and ECE 0.1862, all lower than the seed-42 cross-entropy reference.
The accuracy gain over cross-entropy is only nine answers, so replication is
needed before promotion. See [POLICY_SCREEN.md](POLICY_SCREEN.md). Neither
adapter has been evaluated on JevBench; this result does not establish SOTA.

The pathwise candidate was replicated at seeds 43 and 44. Across seeds 42/43/44
it averages **68.89% ± 0.31 points**, versus **68.39% ± 0.45 points** for the
matched lower-LR cross-entropy arms. It wins each matched seed, with mean NLL
1.3661 versus 1.3903, Brier 0.4956 versus 0.5069, and ECE 0.1992 versus 0.2103.
This is a small gain on reused development questions; the public JevBench cohort
still favors cross-entropy 56/72 to 55/72. See
[policy replication](reports/policy-screen-replication.json). No official rank
or full 534-decision score is established.

## What SOTA requires

1. **Lock the protocol.** Use JevBench's current commit, full public/held-out
   policy, exact adapter mapping, fresh run directory, and all required disclosures.
   Never train on the benchmark or organizer-held items.
2. **Build a strong training mixture.** Use the complete decision-v7 training
   suite plus policy/compositional examples, then hold out source families and
   templates. The current 192-question pilot is far too small.
3. **Compare adaptation modes.** Run head-only, state-tuned, LoRA and state-plus-
   LoRA controls with matched seeds, data, optimizer schedules and parameter
   budgets. Keep a final untouched test selection.
4. **Improve the head.** The current independently scored candidates are robust
   to option order but cannot model candidate interactions. Test a
   permutation-equivariant set encoder or pairwise/listwise reranker while
   preserving exact closed-set probabilities.
5. **Train calibration separately.** Fit temperature/vector calibration on a
   calibration split. Report raw and calibrated Brier/ECE, confidence coverage,
   ordinal MAE and confident-error rates.
6. **Add transfer and abstention.** Include unknown/none options, paraphrase
   pairs, unseen sources, negative examples and a confidence-gated fallback.
7. **Optimize serving after quality.** Keep measured branch batching, then add
   diverse-question batching, request queues, fused kernels and p50/p95
   measurements under controlled concurrency. Do not trade accuracy for vanity
   latency.

## The next quality gate

The four-arm screen and three-seed control/lower-LR replication are complete.
Lower LR is more consistent at this budget; the tested norm cap and boolean-label
balancing do not recover quality. Before extending training, predeclare budgets
and hold out calibration examples from both training and model selection.
Preserve the pilot and all screened adapters, and track per-source collapse and
confidence errors. A long full-data run with the original schedule regressed;
repeating it without these controls is not justified.

The objective screens also leave confidence quality unresolved. Keep the
three-seed lower-LR result as the reference and replicate the typed pathwise
candidate before promoting it. The new pathwise/REINFORCE pair already restricts
ordinal RPS to Score questions; the legacy objectives retain their old definition.
The tested REINFORCE configuration does not improve overall quality. Freeze model
selection before evaluating the full JevBench protocol; development accuracy
around 68% does not meet a 68% JevBench target by itself.

## Full-data comparison after the quality gate

Use the full decision-v7 training file with G1k 3B and three arms:

```text
head-only       rank=0, state_tuning=false
state-plus-head rank=0, state_tuning=true
lora-plus-head  rank=16, state_tuning=false
```

Use three seeds, a fixed development split for selection, a separate calibration
split, and the locked test only once for the selected checkpoint. Report parameter
count, adapter bytes, training wall time/VRAM, accuracy, NLL, Brier, ECE, ordinal
MAE, permutation sensitivity, paraphrase consistency, and JevBench speed/cost.

The current state-tuned G1k pilot is a starting point, not a final model: it
reaches 33/48 on the small development sample and 38/72 on JevBench's public
subset. SOTA is an experiment outcome, not a feature guaranteed by changing the
adapter or increasing model size alone.
