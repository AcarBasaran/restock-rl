# DQN tuning round 1 (night 3)

**Result in one line: reward centering closed the gap. On the tuning years the three centered variants match or slightly beat the baseline, while the uncentered V0 stays about 7% below. The selected variant (V2), evaluated once on the 50 test years, earns 97,128 ± 80 per year against the baseline's 96,614 (+514, +0.5%). It is better in 49 of 50 years.**

## What was run (decided in advance: D16, D18, D20)

| Variant | Change (each builds on the previous one) | Config |
|---|---|---|
| V0 | Current configuration (SB3 defaults, 2×256, γ 0.99, reward / 100) | `configs/tuning_round_1/v0.yaml` |
| V1 | + reward centering: training reward = (true reward − 264.4512) / 100, where 264.4512 is the baseline's mean daily reward on the tuning years | `v1.yaml` |
| V2 | + learning rate 5e-5, target_update_interval 20,000, batch size 128 | `v2.yaml` |
| V3 | + exploration_fraction 0.3 | `v3.yaml` |

- **Seeds:** 3 per variant, training seeds 200000–200002 (the same for every variant; Q22). Each run is 2M steps.
- **Checkpoints:** saved every 100k steps (20 per run), and every one is scored greedily on the 20 **tuning** years (100000–100019).
- **Selection:**
  - Best checkpoint per seed: highest mean tuning-year reward.
  - Best variant: highest mean of those best-checkpoint rewards across the 3 seeds.
  - Test years: only the selected variant's 3 checkpoints were evaluated, once. `src.tuning test` derives the selection itself and refuses a second run.
- **Units:** all numbers are true reward (uncentered, unscaled). The centering constant is recomputed by a test.
- **Code:**
  - Runs used commit `bc48e5c`. V0 and V2 seed 0 recorded `61f0ce4`, a README/notes-only commit made seconds after they were submitted; `src/` and `configs/` are identical in both.
  - V2 seeds 1–2 and V3 were re-packed to use fewer job slots (see "Jobs"). They ran 1 torch thread per process instead of 2; that doesn't change the algorithm.

## Tuning years: best checkpoint per seed, mean ± std over 3 seeds

| Variant | 365-day reward | Fill A | Fill B | Fill overall | Stock depot | Stock A | Stock B | Best checkpoint per seed |
|---|---|---|---|---|---|---|---|---|
| V0 current setup | 89,730 ± 978 | 97.0% ± 1.2% | 94.5% ± 4.9% | 96.1% ± 2.4% | 3.2 ± 2.6 | 10.2 ± 0.9 | 9.5 ± 4.0 | 400k, 1600k, 600k |
| V1 + reward centering | 96,345 ± 173 | 96.2% ± 0.3% | 94.0% ± 1.4% | 95.5% ± 0.6% | 1.1 ± 0.3 | 5.1 ± 0.2 | 5.6 ± 0.7 | 2000k, 300k, 400k |
| **V2 + lr 5e-5, target 20k, batch 128** | **97,027 ± 31** | 96.3% ± 0.5% | 93.1% ± 0.3% | 95.2% ± 0.4% | 1.3 ± 0.2 | 4.7 ± 0.4 | 4.5 ± 0.1 | 600k, 1000k, 900k |
| V3 + exploration 0.3 | 97,016 ± 14 | 97.0% ± 0.2% | 92.7% ± 0.7% | 95.6% ± 0.3% | 1.4 ± 0.2 | 5.0 ± 0.1 | 4.7 ± 0.4 | 1100k, 1000k, 1000k |
| Baseline (60 / 24 / 32) | 96,525 | 95.7% | 92.7% | 94.7% | 4.2 | 4.0 | 3.7 | — |

**Selected: V2** (97,027). V3 is only 11 lower, which is a tie in practice. The rule picks V2.

**Per seed** (tuning years). The "final" column is the 2M-step model, for comparison with last night's final-model evaluation:

| Variant | Seed | Best checkpoint | Best reward | Fill overall | Final (2M) reward |
|---|---|---|---|---|---|
| V0 | 200000 | 400k | 90,765 | 97.0% | 36,333 |
| V0 | 200001 | 1,600k | 88,821 | 97.9% | 60,380 |
| V0 | 200002 | 600k | 89,604 | 93.5% | 85,627 |
| V1 | 200000 | 2,000k | 96,185 | 94.8% | 96,185 |
| V1 | 200001 | 300k | 96,530 | 95.8% | 93,460 |
| V1 | 200002 | 400k | 96,322 | 95.8% | 94,952 |
| V2 | 200000 | 600k | 97,002 | 95.0% | 96,762 |
| V2 | 200001 | 1,000k | 97,061 | 95.7% | 96,629 |
| V2 | 200002 | 900k | 97,017 | 94.9% | 96,284 |
| V3 | 200000 | 1,100k | 97,010 | 95.6% | 96,896 |
| V3 | 200001 | 1,000k | 97,006 | 95.2% | 96,607 |
| V3 | 200002 | 1,000k | 97,031 | 95.8% | 96,876 |

## Plots

Not committed: `results/tuning1/20261115-114120/`.

- **`learning_curves.png`:** training reward per simulated year, mean of 3 seeds per variant (50-year rolling mean, ε-greedy, true units), with the baseline's tuning-year reward as a dashed line.
  - V0 repeats last night's pattern: a peak of about 83k at 0.2M steps, then a slide to about 60–65k with sharp drops.
  - V1 and V2 climb to about 93–95k by 0.2M steps and **stay there** for the remaining 1.8M steps.
  - V3 rises more slowly, because exploration lasts longer, and joins them at about 0.6M.
  - The training curves stay below the baseline line because of the 5% random actions; the greedy checkpoints don't.
- **`checkpoint_scores.png`:** every checkpoint's greedy tuning-year score, one line per seed.
  - V0 swings between about 22k and 90k from one checkpoint to the next.
  - V1 stays at 91–96k throughout.
  - V2 and V3 sit at 95–97k from about 0.2–0.3M steps on.
  - Their 100k checkpoints are far below zero (as low as −1,070k for V3, still mostly exploring at that point). The panel marks them at the bottom edge.

## Test years (used once, for V2 only)

V2's three selected checkpoints (600k, 1,000k, 900k), greedy, on the 50 test years (900000–900049). Same demands and starting stock as the baseline.

| Metric (50 test years) | V2, mean ± std over 3 seeds | Baseline (60 / 24 / 32) | First DQN run (night 2, final models, 5 seeds) |
|---|---|---|---|
| 365-day reward | **97,128 ± 80** | 96,614 | 67,764 ± 21,110 |
| Fill rate A | 96.2% ± 0.5% | 95.6% | 69.3% ± 28.9% |
| Fill rate B | 93.0% ± 0.4% | 92.7% | 81.1% ± 29.0% |
| Fill rate overall | 95.1% ± 0.4% | 94.6% | 73.3% ± 27.2% |
| Avg stock depot | 1.3 ± 0.3 | 4.2 | 3.8 ± 4.7 |
| Avg stock A | 4.7 ± 0.4 | 4.0 | 3.9 ± 3.8 |
| Avg stock B | 4.6 ± 0.1 | 3.8 | 11.5 ± 11.8 |

- **Per seed:** 97,036 / 97,181 / 97,166.
- **Paired** (same test year, V2 averaged over seeds, minus baseline): +514 per year, std 317 across years. That is a standard error of about 45, so the gap is about 11 standard errors. V2 is better in 49 of 50 years. Per seed it is better in 44, 49 and 47 of 50.
- **Where the gain comes from:**
  - Revenue +521 per year, from about 52 fewer lost units per year (Store A −42, Store B −10).
  - Holding cost +7.5 per year, essentially the same.
  - DQN keeps much less at the depot (1.3 vs 4.2 units, at 1 per unit per day) and a little more on the store shelves (4.7 / 4.6 vs 4.0 / 3.8, at 2 per unit per day). That buys more sales for the same holding bill.
- The test gain (+514) matches the tuning-year gain (+502). So checkpoint selection on the tuning years doesn't look like it inflated the result.

Raw output (not committed) is in `results/tuning1/20261115-114254_test/`: `test.md`, `test.json`, per-year CSVs and the three checkpoints. There is also a marker file at `/scratch/$USER/restock-rl/runs/tuning1/TEST_EVALUATED.json`.

## Honest assessment

**Tuning closed the gap, and the decisive change was reward centering.**

- **Centering (V1)** alone moves the mean best-checkpoint reward from 89.7k to 96.3k. It also removes the collapse and oscillation seen in V0 and in last night's run, in both the training curves and the checkpoint scores.
- **V2's extra changes** (smaller learning rate, slower target updates, larger batches) add about 700 more and make the seeds very consistent (std 31).
- **Slower exploration (V3)** adds nothing measurable.

This fits last night's main hypothesis. Without centering, every Q-value carries a large common offset (about 264 / (1 − γ)), and the small differences between actions get lost in it. That would make the greedy action unstable.

**V2's lead is real but small**: +0.5% profit, in 49 of 50 test years.

**Limits:**
- Only 3 seeds, not the 5 in D18 (Q26).
- Selection used 20 checkpoints × 4 variants on the tuning years. The test result agrees, though.
- The baseline is the simple order-up-to rule as decided (D8–D12); for example, the depot ignores store stock. A stronger heuristic might close or reverse a 0.5% gap (Q28).

So the answer to the research question is currently "DQN can slightly beat a properly tuned order-up-to policy in this setup, by about 0.5%, mainly by holding less stock at the depot and more on store shelves". It is not "DQN is much better".

**On D20:** the best tuned DQN isn't below the baseline on the tuning years, so the provisional trigger for a scope reduction hasn't fired. Whether to stop tuning here is a team decision (Q24, Q26).

## Jobs and runtimes

| Job | What | Node(s) | State | Elapsed |
|---|---|---|---|---|
| 2024398 | timing: V3 settings, 200k steps, scoring included | ai06 | COMPLETED | 0:03:15 (1,141 steps/s) |
| 2024443_0–2 | V0, seeds 200000–200002 | ai02 | COMPLETED | 0:57:29 / 0:55:22 / 1:05:27 |
| 2024444_0–2 | V1, seeds 200000–200002 | ai01 | COMPLETED | 0:35:18 / 0:40:14 / 0:38:02 |
| 2024445_0 | V2, seed 200000 | ai01 | COMPLETED | 0:35:50 |
| 2024445_1, _2 | V2, seeds 200001–200002 | — | CANCELLED (re-packed; _1 had run 24 s, partial output deleted) | — |
| 2024446_0–2 | V3, seeds 200000–200002 | — | CANCELLED while pending (re-packed) | — |
| 2024734 | V2, seeds 200001–200002, packed (2 processes, 1 core each) | ai01 | COMPLETED | 0:56:59 |
| 2024733 | V3, seeds 200000–200002, packed (3 processes, 1 core each) | ai01 | COMPLETED | 0:55:25 |

No job failed. Scoring 20 checkpoints took about 13 s per run. Wall clock was 09:53 (timing job) to 11:40 (last packed job). The summary (5 s) and the test evaluation (23 s) ran on the login node.
