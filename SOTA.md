# SOTA status and plan

## Current ranking

Gut-RWKV has **no official JevBench rank**. The current local result is:

- JevBench original public subset: **38/72 = 52.8% accuracy**.
- Strict schema validity: **72/72**.
- Brier: **0.5455**; ECE: **0.1165**.
- Local A100 HTTP p50/p95: **362.5/366.3 ms**.
- Cost: **unknown**, because this is self-hosted; it must not be reported as zero.

That result is not comparable to JevBench's official composite until the complete
534-decision protocol and deployment rules are run. It is also not currently
competitive with the published top rows: Jev 1.13.0 is listed at 75.4, SemIf at
74.7, djev at 74.3, and openJev Verdict at 72.5 in JevBench v1.2. The exact
leaderboard can change; re-check it before publication.

The 68.8% result is from a 48-question development sample selected from three
training sources. It is useful for debugging, not a SOTA claim.

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

## The most valuable next run

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
