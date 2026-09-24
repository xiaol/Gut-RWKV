# Learning curves

Training logs record the loss at step 1 and every tenth optimizer update. The
four checkpoint steps (`384`, `768`, `1152`, `1536`) are also retained as adapter
snapshots. The machine-readable extraction is
[reports/learning-curves.json](reports/learning-curves.json); recreate it with
`scripts/summarize_learning_curves.py`.

## Logged curves

| Arm | First loss | Minimum logged loss (step) | Last logged loss | Final summary loss | Final state norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-entropy seed 42 | 0.699 | 0.000 (1430) | 0.000 | 0.706 | 1.484 |
| Cross-entropy seed 43 | 0.699 | 0.000 (990) | 0.265 | 0.019 | 1.613 |
| Cross-entropy seed 44 | 4.360 | 0.000 (1510) | 0.304 | 0.000 | 1.724 |
| Pathwise seed 42 | 0.348 | -0.500 (870) | -0.500 | 0.501 | 1.653 |
| Pathwise seed 43 | 0.348 | -0.500 (1120) | -0.412 | -0.476 | 1.549 |
| Pathwise seed 44 | 4.306 | -0.500 (1090) | -0.352 | -0.500 | 1.515 |
| REINFORCE seed 42 | -0.002 | -0.063 (210) | 0.000 | -0.001 | 1.761 |
| Proper score seed 42 | 0.474 | -2.024 (960) | -1.303 | not recorded beside log | not recorded beside log |
| Noisy proper score seed 42 | 0.396 | -2.090 (700) | -1.975 | not recorded beside log | not recorded beside log |

The objective scales differ: cross-entropy is positive, pathwise loss is the
negative sampled reward, and REINFORCE is a reward-weighted Gaussian log-density
surrogate. Their raw losses must not be ranked against each other. The logged
point at step 1530 and the final summary at step 1536 can differ because the
training loop prints every tenth step while the summary records the unprinted
last update.

Loss curves alone do not establish generalization. The useful checkpoints are
the fixed final adapters evaluated on held-out development and public JevBench
tasks. The broader 231-task public run selected lower-LR cross-entropy over
pathwise despite pathwise's better mean development curve, which is why future
selection must use a reserved external gate.

## Further experiments

1. **Warm-start pathwise from cross-entropy.** Keep a frozen cross-entropy
   reference, train only the initial state for a short pathwise phase, and add a
   KL penalty so RL cannot erase calibrated decisions.
2. **Typed reward ablation.** Compare correctness-only, proper score, ordinal
   distance, and confidence penalties separately. Keep RPS restricted to Score.
3. **Checkpoint selection gate.** Reserve source families for development,
   calibration, and external selection. Never choose the best step on the same
   public JevBench items used for reporting.
4. **Long-context capability.** Train/evaluate with the 8,192-token serving
   configuration and report short versus long-task accuracy separately; the
   original 2,048-token cap caused 36 public hard-task failures.
5. **Head interaction.** Add a permutation-equivariant candidate set encoder or
   pairwise scorer, then measure candidate-order sensitivity and Score MAE.
6. **Complete benchmark submission.** Obtain the organizer-held evaluation route
   or submit the selected reproducible endpoint; only then compute the official
   four-axis JevBench Score and rank.
