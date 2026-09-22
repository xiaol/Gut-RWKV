# Gut-RWKV experiments, 2026-09-21

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
The same seed does not give identical initial head weights across modes because
LoRA initialization consumes random draws first; this pilot is not a fully
controlled initialization-matched ablation.

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
**61.4%** (472), and Score **24.2%** (240). This is lower than the 68.8% pilot
on its deliberately small 48-question sample, so the pilot must not be treated
as representative. The full-run report is
[full-g1k-state-development.json](reports/full-g1k-state-development.json).

This result remains a development measurement, not a JevBench rank: no official
534-decision evaluation or matched head-only/LoRA control was run for this
checkpoint.

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

## Validation and remaining work

Nine tests pass, including state gradients, frozen base weights, save/load,
cached/full-sequence equivalence, reordered question batches, candidate order,
typed contracts and live HTTP schema/error handling. A wheel build includes
CUDA sources, the upstream license and the browser demo. G1k weight integrity
was verified against the upstream LFS SHA-256 before loading.

The measured gains justify investigating branch batching further. Broader
training, calibration, stronger baselines, multiple seeds and diverse concurrency
workloads are needed before claiming competitive quality or a novel algorithm.
