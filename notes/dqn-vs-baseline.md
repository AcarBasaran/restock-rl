# DQN vs tuned baseline — first run (night 2)

**Result in one line: the first DQN run does not beat the baseline, and isn't close. Across 5 seeds it earns 67,764 ± 21,110 per year against the baseline's 96,614, and every seed is worse than the baseline on every one of the 50 test years.**

## Setup (as decided; nothing tuned)

- **DQN:** SB3 2.9.2 DQN with SB3 defaults, a 2×256 ReLU MLP, γ = 0.99, and flat 405 actions without masking (D13–D16). Observations are divided by `[60, 30, 30, 40, 30, 30]` (D16, Q13). Rewards are divided by 100 during training only (D17).
- **Training:** 2M steps per seed (D15), training seeds 200000–200004 (Q12). Each seed ran as one SLURM array task on VALAR (job 2022023, account/partition `ai`, 2 CPUs, no GPU), at code commit `e7a1664`.
- **Evaluation:** each seed's **final** model with greedy actions (argmax Q) on the 50 test years, seeds 900000–900049 (D18). These are the same demands and starting stocks the baseline sees. No warm-up exclusion. Undiscounted 365-day reward in true units (D17, D19).
- **Baseline:** order-up-to with targets depot 60 / Store A 24 / Store B 32, from the D12 grid search on the tuning years.
- **Command:** `python -m src.compare --runs-dir /scratch/$USER/restock-rl/runs/dqn` (about 30 s on the login node).
- **Raw output:** `results/dqn/20261115-070241/`, not committed. It contains `comparison.md`, `comparison.json`, per-year CSVs, `learning_curves.png`, `test_reward_by_seed.png`, and copies of the models and monitor logs.

## Comparison (50 test years)

| Metric (50 test years) | DQN, mean ± std over 5 seeds | Baseline (60 / 24 / 32) |
|---|---|---|
| 365-day reward | 67,764 ± 21,110 | 96,614 (std over years 1,156) |
| Fill rate Store A | 69.3% ± 28.9% | 95.6% |
| Fill rate Store B | 81.1% ± 29.0% | 92.7% |
| Fill rate overall | 73.3% ± 27.2% | 94.6% |
| Average stock depot | 3.8 ± 4.7 | 4.2 |
| Average stock Store A | 3.9 ± 3.8 | 4.0 |
| Average stock Store B | 11.5 ± 11.8 | 3.8 |
| Lost sales Store A per year | 2,244 ± 2,115 | 323 |
| Lost sales Store B per year | 691 ± 1,062 | 267 |

The "±" is the standard deviation of the five per-seed means (ddof = 1). Fill rates are pooled over years, and stock is measured at the end of the day after shipments leave; Q2 and Q3 are still open.

**Per seed:**

| Training seed | Reward | Fill A / B / overall | Avg stock depot / A / B | Years better than baseline |
|---|---|---|---|---|
| 200000 | 36,362 | 35% / 30% / 34% | 1.0 / 0.1 / 0.0 | 0 of 50 |
| 200001 | 56,037 | 41% / 90% / 58% | 1.0 / 0.7 / 8.8 | 0 of 50 |
| 200002 | 85,433 | 81% / 86% / 83% | 2.8 / 3.0 / 3.0 | 0 of 50 |
| 200003 | 77,131 | 94% / 100% / 96% | 2.0 / 7.8 / 29.6 | 0 of 50 |
| 200004 | 83,859 | 95% / 99% / 96% | 12.2 / 8.0 / 15.9 | 0 of 50 |

**Paired difference** (same test year, DQN averaged over seeds, minus baseline): −28,850 per year, std 689 across years. DQN is better in 0 of 50 years. Per-seed mean differences are −60,252 / −40,578 / −11,181 / −19,484 / −12,755. Even the best seed (200002) loses between 8,990 and 13,393 in every year.

## What the plots show

- `learning_curves.png` shows training reward per simulated year: a 50-year rolling mean in true units under ε-greedy, on training years. All five seeds rise quickly to about 83k at about 0.2M steps. That is the point where ε reaches its floor of 0.05 (exploration_fraction 0.1 × 2M). After that, all five **decline** to about 60–68k and stay noisy, with sharp drops of 20–40k that recover within a few dozen years. The curves never reach the baseline line.
- `test_reward_by_seed.png` shows each seed's greedy test reward against the baseline line. The spread between seeds (36k to 85k) is far larger than the year-to-year spread (std 1,156 for the baseline, 55–1,934 per DQN seed).

## Diagnostic: saved checkpoints (tuning years, not the result)

To tell a single bad final snapshot apart from a general problem, I evaluated the checkpoints saved during training (500k, 1M, 1.5M, final) greedily on the **20 tuning years** (100000–100019). I used tuning years, not test years, so the test set stays unused for any kind of selection. This diagnostic selects nothing and changes nothing. The script was a throwaway in the scratch area and isn't committed.

| Seed index | 500k | 1M | 1.5M | 2M (final) |
|---|---|---|---|---|
| 0 | 86,974 (91%) | 78,622 (78%) | 78,793 (89%) | 36,333 (34%) |
| 1 | 82,695 (86%) | 35,549 (34%) | 64,694 (81%) | 56,116 (58%) |
| 2 | 73,045 (92%) | 38,397 (40%) | 75,619 (92%) | 85,564 (83%) |
| 3 | 93,727 (94%) | 73,330 (85%) | 228 (1%) | 77,059 (96%) |
| 4 | 86,517 (92%) | 68,936 (75%) | 78,039 (86%) | 83,598 (96%) |

Each cell shows mean reward and overall fill rate. The baseline scores 96,525 on these years.

The greedy policy **jumps around a lot between snapshots**. Seed 3 at 1.5M almost stops supplying the stores, scoring 228 per year. The best snapshot anywhere is seed 3 at 500k, at 93.7k, which is still below the baseline. So picking a better checkpoint would narrow the gap but not close it, and the problem is not one unlucky final snapshot.

## Honest assessment

With the provisional setup (SB3 defaults, 2M steps, flat 405 actions), DQN clearly fails to match a properly tuned order-up-to policy. On average it loses about 30% of the baseline's profit. It is also unreliable: two seeds learn policies that starve Store A (fill rate 35–41%). Two other seeds reach high fill rates by holding far more stock at Store B (16–30 units against the baseline's 3.8), so holding costs eat their sales. Only seed 200002 holds a baseline-like amount of stock, and it under-supplies both stores. The learning curves and checkpoints show unstable learning rather than slow learning: performance peaks early and then degrades and oscillates. More steps with the same settings would probably not help by themselves. Nothing was tuned tonight (as instructed), so this is a statement about this first configuration, not about DQN on this problem in general.

## Likely causes (open questions; nothing was changed)

These are hypotheses, not findings. Each is logged in `notes/open-questions.md` (Q19–Q21). Choosing among them, and deciding whether to trigger a scope reduction, is **D20 (still open)**.

1. **Instability after exploration ends (D16).** The decline starts exactly when ε hits 0.05. Plausible contributors at SB3 defaults:
   - a hard target-network copy every 10,000 steps (`tau = 1`)
   - learning rate 1e-4 with a 1M buffer (half of all training data)
   - Q-value overestimation spread across 405 actions
   - long-horizon bootstrapping with γ = 0.99, where values are about 100× the per-day reward
2. **Flat 405-action space (D14, D20).** Most actions are near-duplicates. The value of each one is learned from few samples, and a single wrong argmax (for example shipping 0 to Store A) is costly. The proposal's fallback ("fewer actions" or "distribution only") targets exactly this.
3. **Final-model evaluation (Q18).** Because the policy oscillates, the final snapshot is partly luck. Periodic greedy evaluation on tuning years with best-checkpoint selection would give a fairer picture. That is a protocol decision.
4. **Training budget (D15).** Probably not the main issue, given that the peak comes early and then degrades. Still, 2M steps is only about 5,500 simulated years.
5. **Reward/observation scaling (D16, D17).** A reward scale of 100 still gives per-step rewards around 2.5 and Q-values around 250. That may be large for SB3's Huber loss with gradient clipping at 10. This is unverified.
