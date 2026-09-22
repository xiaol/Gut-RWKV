# RWKV-Jev pilot, 2026-09-21

Historical report for the original LoRA-only prototype, now named Gut-RWKV.
For current state-tuned models, batched inference and JevBench results, see
[EXPERIMENTS.md](EXPERIMENTS.md). The runtime figures below predate question batching.

The first local adapter is available at `runs/pilot-0.4b/adapter.pt` (14,782,184
bytes). It runs against the existing RWKV-7 G1d 0.4B checkpoint. It is a small
feasibility adapter, with substantial weaknesses in boolean and ordinal tasks.

## Training

- Base: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`, frozen bf16 weights.
- Rank-8 fp32 adapters on TimeMix and ChannelMix linear projections, plus a
  128-wide decision head. No vocabulary head or answer-token decoding.
- Data: 192 questions, 64 each from AG News, BoolQ and SST-5 in Kev decision-v7.
- Two epochs, 384 optimizer updates, learning rate `1e-4`, seed 42, one question
  per optimizer update. This was one pilot configuration, without tuning on dev.
- Upstream and selected dataset hashes: `runs/pilot-data/manifest.json`.
- Base/tokenizer/training-file hashes are stored inside the adapter metadata.
- Log: `runs/pilot-training.log`; summary: `runs/pilot-0.4b/train-summary.json`.

## Held-out development results

These are 48 selected development questions, not the upstream complete benchmark
or a comparison against Jev. No locked test examples were opened.

| Type / source | Correct | Accuracy | NLL | Brier |
| --- | --- | --- | --- | --- |
| Choice / AG News | 14/16 | 87.5% | 0.4063 | 0.2397 |
| Noul / BoolQ | 8/16 | 50.0% | 1.3028 | 0.8513 |
| Score / SST-5 | 4/16 | 25.0% | 1.7740 | 0.8589 |
| Overall | 26/48 | 54.2% | 1.1610 | 0.6500 |

Score accuracy here uses the most probable discrete level; the public answer
also returns the expected level. Overall 10-bin ECE is 0.2535, on a small sample.
Probabilities are uncalibrated. BoolQ and SST-5 results do not establish useful
decision quality; more training data and an independent calibration set are
needed before choosing confidence thresholds.

Machine-readable report: `runs/pilot-0.4b/evaluation.json`.

## Correctness and runtime checks

Five CPU tests pass: cached/full-sequence parity, question isolation, candidate
permutations including shared-prefix names, training gradients with frozen base
weights, adapter round-trip, input validation and typed answers.

The real 0.4B adapter was additionally checked on NVIDIA A100-PCIE-40GB, bf16:

- Asking the sample Choice alone or with other questions gives identical output.
- Reversing its candidate order changes mapped probabilities by zero.
- Cached versus full-sequence execution differs by at most 0.007997 probability
  on that sample. Different bf16 sequence partitioning is not numerically exact;
  this single check is not a general error bound.
- Five warm-model runs of the three-question example have median latency
  **818.7 ms**, with a range of 813.4–820.1 ms. The complete state is processed
  on every request; no results or state are cached across requests. This is
  prototype latency, not a match for Jev's speed claims.

Evidence: `runs/pilot-0.4b/integration.json` and
`runs/pilot-0.4b/example-response.json`. The example routes a duplicate-charge
refund request to billing, but one demonstration does not establish domain quality.

## Run the adapter

From `rwkv-jev/`:

```bash
.venv/bin/rwkv-jev predict \
  --base ../models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --vocab ../models/rwkv7/rwkv_vocab_v20230424.txt \
  --adapter runs/pilot-0.4b/adapter.pt \
  --request examples/request.json
```

The next useful experiment is broader mixed-task training, followed by
independent calibration and evaluation on unseen sources. Kernel fusion and
batching question branches are separate performance work; increasing training
data will not automatically fix latency.
