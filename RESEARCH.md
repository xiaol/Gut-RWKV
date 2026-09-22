# Research positioning and benchmarks

## What is and is not novel

Typed non-generative decisions already exist in Jev, NanoJev, Kev, SemIf and
others. RWKV initial-state tuning, constant-memory recurrence and cloned state
branches are established techniques. An inference-only RWKV Jev-like wrapper
also predates this project. Claiming Gut-RWKV invents any of these is unsupported.

Our project-specific direction is **state-tuned recurrent decision models with
dynamic candidate heads and measured branch batching**:

1. Can learned initial WKV states replace LoRA at useful accuracy/calibration,
   keeping base weights unchanged?
2. How do adapter bytes, training memory, throughput and quality compare for
   head-only, state-plus-head and LoRA-plus-head on identical data?
3. Does bounded recurrent-state branching improve throughput as independent
   question count and context length grow?
4. What accuracy tradeoff comes from independent candidate scoring, which gives
   option-order equivariance but excludes joint candidate interactions?

These are testable research questions and engineering contributions, not yet
established scientific novelty or superiority. Small-state memory is an RWKV
property. Quality and speed advantages need evidence against strong baselines.

## JevBench

**[JevBench](https://github.com/fstandhartinger/jevbench)** and its
**[Benchmark Heaven leaderboard](https://benchmarkheaven.com/jev-models)** are
directly relevant. It is independent, not endorsed by TypeSafe. As checked on
2026-09-21, v1.2 evaluates 534 decisions with public and organizer-held items.
The composite is a geometric mean of intelligence, calibration, speed and cost
with equal weights. No separate official TypeSafe prize competition was verified.

Read its current rules before comparing. It includes estimated cost bases and
deployment-dependent latency adjustments; our local A100 timing is not equivalent
to published endpoint latency. Unknown hosted cost must stay unknown, not zero.
A public-subset run is not an official ranked submission.

Start Gut-RWKV's server, then use JevBench's native adapter:

```bash
git clone https://github.com/fstandhartinger/jevbench
cd jevbench
python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env '' \
  --model gut-rwkv --cost-basis self_hosted_local_gpu --reserve-usd 0 \
  --results RUN/results.jsonl --raw-dir RUN/raw --ledger RUN/ledger.jsonl
python -m jevbench.cli summarize --tasks datasets/public/original.jsonl \
  --results RUN/results.jsonl --public-export RUN/summary.json
```

Use a fresh run directory. Keep benchmark data out of training. Organizer
submission needs reproducible code, weight identity, deployment/pricing
disclosures and their acceptance. No submission or contact has been made.

Other useful comparisons:

- [Kev](https://github.com/jaredpalmer/kev): decision-v7 and transfer-v4/v9;
  calibration, source transfer and none-of-the-above behavior.
- [NanoJev](https://github.com/TianyuCodings/NanoJev): game policies, requiring
  matched observations and controllers for a meaningful comparison.
- [SemIf](https://github.com/TheoLeeCJ/SemIf): authored semantic decisions.
- [RWKV wrapper](https://github.com/1cyberlangke1/rwkv-jev-like): untuned token
  scoring baseline, to test on the same tasks and hardware.
- [TypeSafe](https://docs.typesafe.ai/introduction): interface reference, not
  a full public architecture or training specification.

## Controlled experiment

Freeze base/tokenizer/data hashes. Compare head-only, state-plus-head and rank-8
LoRA-plus-head with the same head, examples, seed schedule and stopping rule.
Report trainable parameters/bytes; state tuning is not automatically smaller
than every LoRA rank. Use multiple seeds before interpreting small differences.

Report held-out accuracy by type/source, NLL, Brier, ECE, ordinal MAE, coverage
at a fixed error budget, permutation sensitivity and question isolation. Fit
temperature on separate calibration data. Use the locked test only after
selecting a final configuration.

Measure 1/8/32/128 questions over several context lengths with both identical
and diverse rubrics. Report warm/cold latency, p50/p95, decisions/s, peak VRAM
and recurrent-state bytes. Separate branch batching within requests from
concurrent clients. The current server serializes requests; continuous batching
and faster inference kernels remain future work.

## Requested G1k 3B

Found in [BlinkDL/temp-latest-training-models](https://huggingface.co/BlinkDL/temp-latest-training-models),
not the stable `rwkv7-g1` repository. Identity observed 2026-09-21:

- File: `rwkv-g1k-3b-temp-5441.pth`.
- Repository revision: `a7d80e331bd787f9a8ff88d993fb2af147d5b2c0`.
- Size: 5,896,273,469 bytes.
- SHA-256: `eea4fbf9fee9624958f84de3a8e3fd55a1840d76e7be53e211618d48f3ce0e58`.
- Model card license: Apache-2.0; status: cutting-edge preview/training checkpoint.

The downloader pins this revision so a moving `main` cannot silently change the
experiment. This was the 3B G1k file in the accessible listing, not a promise that
future temporary releases will keep this filename. An adapter trained on the 0.4B
base cannot be loaded onto it; a separate state/head training run is required.
