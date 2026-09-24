# Gut-RWKV experiments, 2026-09-21

For the subsequent matched G1k comparison and regression diagnosis, see
[AUDIT.md](AUDIT.md): state pilot 61.4%, rank-16 LoRA 54.5%, head-only 48.0%, and
full-trained state 44.3% on the same 1,468-question development set.

The subsequent [state-update screen](STATE_SCREEN.md) reaches 67.9% with a
tenfold lower state LR, against 35.4% for the matched 1,536-update control.
The norm-cap and boolean-balanced arms score 34.1% and 29.5%. This is a new
single-seed development result; confidence calibration remains unresolved.
The completed seeds 43/44 replication gives lower LR 68.39% mean accuracy
(sample SD 0.45 points), versus control 46.46% (SD 19.94 points). Control seed
44 wins individually at 69.48%; see STATE_SCREEN.md for the complete comparison.

## Same-data adaptation pilots

All runs use the same 192 training questions and 48 development questions from
Kev decision-v7, with 64/16 records each from AG News, BoolQ and SST-5. Two epochs,
384 optimizer updates, seed 42, learning rate `1e-4`, AdamW, gradient clipping 1.0,
bf16 frozen base and fp32 trainable parameters. No calibration or model selection
was performed on JevBench. Data identity: [manifest](reports/pilot-data-manifest.json).

| Base | Adaptation | Trainable parameters | Adapter file | Development accuracy | NLL | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| G1d 0.4B | Rank-8 LoRA + head | 3,672,320 | 14,782,184 bytes | 26/48 (54.2%) | 1.1610 | 0.6500 |
| G1d 0.4B | Initial WKV state + head | 1,706,240 | 6,827,784 bytes | 21/48 (43.8%) | 1.1656 | 0.6714 |
| G1k 3B preview | Initial WKV state + head | 5,575,936 | 22,306,568 bytes | 33/48 (68.8%) | 0.7919 | 0.4566 |

The matched 0.4B comparison favors LoRA on accuracy. State tuning uses fewer
parameters in this configuration but does not improve quality. The G1k result
also changes model size and base training; its improvement cannot be attributed
to state tuning. Head-only and G1k LoRA controls plus multiple seeds are needed.
The historical runs did not record initial head fingerprints. LoRA was installed
before the head; this changes CPU random-number consumption, whereas CUDA uses a
separate generator. The current code initializes the head first to enforce a
consistent initialization order across modes and devices.

G1k accuracy by type: Choice 14/16, Noul 9/16, Score 10/16. All are small samples.
Reports: [LoRA 0.4B](reports/lora-0.4b-development.json),
[state 0.4B](reports/state-0.4b-development.json),
[state G1k 3B](reports/state-g1k-3b-development.json).

## Full decision-v7 development run

The same G1k state-tuning configuration was then trained for two epochs on the
complete 15,576-question training file (31,152 optimizer steps), with the
untouched 1,468-question development file evaluated once training finished.
Accuracy was **650/1,468 = 44.3%**, with NLL **1.5599**, Brier **0.7387** and
10-bin ECE **0.1876**. By type, accuracy was Choice **39.9%** (756), Noul
**61.4%** (472), and Score **24.2%** (240). These numbers alone cannot be compared
to the pilot's 68.8% on 48 questions. The subsequent matched evaluation finds
61.4% for the pilot on this same complete development set, confirming a
regression between the two adapters. The full-run report is
[full-g1k-state-development.json](reports/full-g1k-state-development.json).

This result remains a development measurement, not a JevBench rank: no official
534-decision evaluation or matched full-data head-only/LoRA control was run for
this checkpoint. The new head-only/LoRA controls in AUDIT.md match the pilot.

State tuning updates only initial WKV matrices and the decision head, not base
weights or time-shift vectors. G1k's trainable WKV tensor is `[32,1,40,64,64]`;
its norm after training is 5.8772, confirming a nonzero learned initialization.
The base checkpoint SHA-256 is
`eea4fbf9fee9624958f84de3a8e3fd55a1840d76e7be53e211618d48f3ce0e58`.

## Within-request batching

NVIDIA A100-PCIE-40GB, bf16, batch cap 16, three-option questions, three timed
repeats per point after warmup. Both paths encode the context once per request;
neither reuses a cross-request cache. Questions are identical repetitions for
this scaling check. These results do not measure diverse clients or production load.

| Base | Questions | Serial shared-prefix median | Batched median | Speedup | Batched decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.4B state | 1 | 246.0 ms | 251.9 ms | 0.98x | 3.97 |
| 0.4B state | 8 | 1421.4 ms | 339.8 ms | 4.18x | 23.54 |
| 0.4B state | 32 | 5517.8 ms | 765.3 ms | 7.21x | 41.81 |
| G1k 3B state | 1 | 339.9 ms | 344.1 ms | 0.99x | 2.91 |
| G1k 3B state | 8 | 1932.8 ms | 457.2 ms | 4.23x | 17.50 |
| G1k 3B state | 32 | 7329.6 ms | 1018.0 ms | 7.20x | 31.44 |

Bf16 batching is not bit-exact: maximum observed probability differences are
0.012436 for 0.4B and 0.001819 for G1k across these probes. Questions are isolated
by state construction, but numerical batching differences can affect near ties.
The 0.4B recurrent state consumes 6,389,760 bytes per branch regardless of input
context length; total memory also includes model weights, branches and activations.

Reports: [0.4B](reports/state-0.4b-branches.json),
[G1k 3B](reports/state-g1k-3b-branches.json). `scripts/benchmark_branches.py` reproduces them.

## JevBench public evaluation

G1k 3B state-plus-head ran the unmodified JevBench `typesafe` HTTP adapter on
`datasets/public/original.jsonl`, 72 decisions. Harness revision:
`bb83e5da8bbb040aa772bcfe4c6275408dbeb831`. All 72 requests completed without
errors or distribution renormalization. This is not the full 534-decision suite,
an official ranking, or a comparison with the published composite scores.

- Accuracy: **38/72 = 52.8%**; schema validity: **72/72**.
- Brier: **0.5455**; ECE: **0.1165**; ordinal MAE: **0.8374**.
- Local HTTP latency: **p50 362.5 ms, p95 366.3 ms**; not public-network latency.
- Paraphrase pairs: both correct **15/36**, agreement **23/36**.
- Per family: adequacy 4/12, extraction 9/12, intent 4/12, ordinal 8/12,
  policy 6/12, routing 7/12.
- Hosted cost: **unknown**, not zero. Local execution incurred no API charge.

Report: [public summary](reports/jevbench-g1k-3b-public.json). Raw local evidence
is in `runs/jevbench-g1k-3b/`. These questions were not used to train the adapter.
No organizer-held data was requested and no leaderboard submission was made.

### Public JevBench comparison after objective screening

The exact same `original-public` cohort and JevBench v1 harness were rerun for
the replicated lower-state-LR cross-entropy adapter and the typed-reward pathwise
candidate. Each run used 72 serial localhost requests, the frozen G1k 3B base,
strict schema scoring, and a fresh result directory. The full 534-decision suite
was not available locally, so these rows are public-cohort evidence only.

| Adapter | Correct / 72 | Accuracy | Brier | ECE | p50 / p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lower-LR cross-entropy | **56** | **77.8%** | **0.3858** | 0.2143 | 333 / 377 ms |
| Pathwise typed reward | 55 | 76.4% | 0.4047 | **0.1746** | 335 / 361 ms |
| Earlier G1k pilot | 38 | 52.8% | 0.5455 | 0.1165 | 363 / 366 ms |

The lower-LR adapter gains five decisions and loses four against pathwise, with
63 identical decisions. Pathwise improves the earlier pilot by 17 answers but
does not beat the matched lower-LR baseline. Pathwise has better ECE but lower
accuracy and worse Brier on this cohort. Both runs have 72/72 strict schema
validity. Family-level results, hashes, raw-result paths and paired counts are in
[jevbench-public-comparison.json](reports/jevbench-public-comparison.json).

This is the appropriate external baseline for model selection, while the
1,468-question decision-v7 development set remains useful for fast ablations.
Do not tune repeatedly on these 72 public questions. Select using a reserved
development/calibration protocol, then run the complete JevBench set once.

### All currently public JevBench tasks

We also ran the available `easy`, `original`, and public `hard` files together:
231 tasks. The 303 organizer-held tasks are unavailable locally, so this remains
an unofficial public-only comparison. Serving copies set `max_tokens=8192`; the
preliminary 2,048-token run had 36 hard-task context failures and is excluded.

| Adapter | Correct / 231 | Accuracy | Brier | ECE | Strict validity | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lower-LR cross-entropy | **149** | **64.5%** | **0.5500** | **0.2097** | 231/231 | 347 / 507 ms |
| Pathwise typed reward | 143 | 61.9% | 0.5744 | 0.2358 | 231/231 | 331 / 476 ms |

The lower-LR adapter wins by six decisions. Pathwise's development-set advantage
does not transfer to this broader public set, so cross-entropy remains the safer
external baseline. These figures are not the official JevBench Score: the full
534-task protocol, calibration hard tier, deployment adjustment and cost axis are
not available from this local public-only run. Machine-readable details and
artifact hashes are in [jevbench-public231-comparison.json](reports/jevbench-public231-comparison.json).

The pathwise objective was then repeated with seeds 43 and 44 under the same
1,536-update protocol. Across seeds 42/43/44, pathwise averages **68.89% ± 0.31
points**, while lower-LR cross-entropy averages **68.39% ± 0.45 points**. The
paired pathwise gains are 9, 9 and 4 answers. Mean NLL/Brier/ECE also favor
pathwise, but seed variation and reused questions limit the claim. This does not
override the public JevBench comparison: cross-entropy remains 56/72 versus
pathwise 55/72. Full artifacts are in
[policy-screen-replication.json](reports/policy-screen-replication.json).

## Validation and remaining work

Nine tests pass, including state gradients, frozen base weights, save/load,
cached/full-sequence equivalence, reordered question batches, candidate order,
typed contracts and live HTTP schema/error handling. A wheel build includes
CUDA sources, the upstream license and the browser demo. G1k weight integrity
was verified against the upstream LFS SHA-256 before loading.

The measured gains justify investigating branch batching further. Broader
training, calibration, stronger baselines, multiple seeds and diverse concurrency
workloads are needed before claiming competitive quality or a novel algorithm.
