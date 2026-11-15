# Overnight report: environment, evaluation, baseline, grid search

Branch `feature/env-baseline` (from `main` at `127308d`). Committed locally only: **nothing pushed, nothing merged, no GitHub issues touched.** No DQN code, no torch or Stable-Baselines3.

## Headline

- All 8 tasks were done. **53 tests pass.**
- The tuned baseline targets are **depot 60, Store A 24, Store B 32**: mean 365-day reward **96,525** on 20 tuning years, fill rate 94.7%, none on a range edge.
- **11 open questions** are logged in `notes/open-questions.md`. Two need attention first:
  - **Q1:** how a Gymnasium step maps onto the day. I implemented a provisional version.
  - **Q11:** the pinned NumPy needs Python ≥ 3.12, but D21 says ≥ 3.10.

## 1. What changed (files)

| File | What |
|---|---|
| `CLAUDE.md` | D21 and D25 marked DECIDED (Section 3 only). |
| `README.md` | New "Setup" block in the decisions section (D25, D21); new "How to run" section. |
| `requirements.txt` | New: gymnasium 1.4.0, numpy 2.5.4, pytest 9.2.0, PyYAML 6.0.3 (only the four allowed packages). |
| `.gitignore` | Added `results/`. |
| `pytest.ini` | `pythonpath = .` so tests import `src.*`. |
| `configs/default.yaml` | Every number: demand, lead times, costs, action grids, γ, start-stock means, seed ranges, grid ranges. |
| `src/env.py` | `RestockEnv` (Gymnasium) per Section 2 and D1–D7. |
| `src/evaluate.py` | Same-demand evaluation of any obs → action policy; random and do-nothing policies; sanity-check CLI. |
| `src/baseline.py` | Order-up-to policy (D8–D11), grid search (D12), runtime estimate, optional coarse→fine mode. |
| `tests/test_env.py`, `tests/test_evaluate.py`, `tests/test_baseline.py` | Tests (below). |
| `notes/sanity-check.md`, `notes/grid-search.md`, `notes/open-questions.md`, this report | Notes. `notes/` is a new top-level folder, created because the task asked for these files. |

**Python (D21):** the system `python3` on the VALAR login node is 3.6.8, which is too old. I used the cluster's `anaconda3/2025.06` module, which has Python 3.13.9, to build `.venv`. This is recorded in D21 and the README.

## Commits (on `feature/env-baseline`, oldest first)

```
e1138ba Record D21 (Python venv) and D25 (repo layout) as decided; add requirements.txt
fe3b4e6 Add Gymnasium environment (Section 2, D1–D7) with config and tests
8e2898f Add same-demand evaluation and sanity check with random and do-nothing policies
4aaeffb Add order-up-to baseline (D8–D11) and grid search (D12) with tests
b1b3de5 Add grid-search summary, open questions, and README How to run section
3949e53 Self-review: take random policy's action count and seed offset from config
```

(This report is committed on top of these.)

## 2. Tests run and results

`.venv/bin/python -m pytest` → **53 passed** (19 env, 6 evaluate, 28 baseline). No test was weakened. All passed the first time they were run, except for my own unclear comment in one baseline test, which I rewrote.

- **Environment:**
  - one hand-traced day, plus a second day with a lost sale
  - pipeline arrival day for the depot (day 2), Store A (day 1) and Store B (day 3)
  - Store B and depot pipeline slots shifting one per day
  - the 13 → 8 / 4 / 1 split, both as a function and inside the environment
  - holding cost after shipments leave (34, against 58 if it were charged before shipping)
  - sales and lost sales in the reward, and no other costs
  - `truncated=True` exactly at day 365 and `terminated` never true
  - the same seed gives the same starting stock and demand whatever the policy does
  - observation order
  - Gymnasium's `check_env`, action encoding, and start-stock and demand means
- **Do the tests catch mistakes?** I ran the environment tests against three deliberately broken copies, and each one was caught:
  - holding cost charged before shipping: 2 tests failed
  - split rounding up: 3 failed
  - depot lead time 3 instead of 2: 8 failed
- **Evaluation:** a hand-computed do-nothing year (−10,720), warm-up exclusion, reproducibility, discounting, seed ranges not overlapping, pooled fill rate.
- **Baseline:**
  - rounding: 6 → 8, 25 → 30 (depot), 25 → 24 (store), and other ties go up
  - caps at 32 and 40, and never below 0
  - the D9 example (need 25 → ship 24)
  - Store B counting shipments on the way
  - the depot counting stock after today's shipments, and ignoring store stock
  - the short-depot split
  - grid-search helpers

## Sanity check (`notes/sanity-check.md`)

Sanity seeds 0–4: do-nothing **−10,702** per year, random **−2,380** (std 34,721). Both match hand estimates. Do-nothing loses almost all demand while the depot's ~30 units sit at holding cost 1/day. Random starves Store A and lets Store B's stock wander, so holding cost varies widely between years.

## Grid search results (`notes/grid-search.md`)

- **Setup:** full D12 grid of 4,608 candidates, 20 years each, on tuning seeds 100000–100019. Test seeds start at 900000, with the count left open (D18).
- **Runtime:** the timing test estimated about 5 min on one core, and the real run took 2.8 min with 4 low-priority workers. That is far under 4 hours, so I ran no coarse/fine pass.
- **Best targets:** depot **60**, Store A **24**, Store B **32**.
  - Reward: 96,525 ± 921
  - Discounted reward: 25,607
  - Fill rate: A 95.7%, B 92.7%, overall 94.7%
  - Average stock (depot / A / B): 4.2 / 4.0 / 3.7
  - Lost sales per year: A 310, B 269
- **Edges:** no best value is on the edge of its range, so the widening rule didn't apply. Store A's best value is one step above its lower bound, but that bound itself scores clearly worse.
- **Discounted vs. undiscounted:** both rankings pick the same top three, so Q6 doesn't change the choice.
- **How close the top is:** the top three are within about 240 of each other, roughly two standard errors.

## Self-review (task 7)

I re-read Section 2 and D1–D12 line by line against the code.

| Item | Status |
|---|---|
| Network, lead times 2/1/3, Poisson 20/10, lost sales, lost orders hidden from the agent | ✓ config + env; lost sales only in `info` |
| Daily order arrivals → demand → decision; 6-value observation in spec order; no Store A pipeline slot | ✓ tested |
| 405 actions, grids {0..40 step 10}, {0..32 step 4} ×2 | ✓ tested |
| Proportional split, round down, leftover stays (D4) | ✓ tested |
| +10 per unit sold; holding 2/store, 1/depot after shipments (D3); in transit free; no other costs | ✓ tested |
| γ 0.99; continuing task; truncation, never termination, at 365 | ✓ tested |
| D1 arrival at t+L; D2 immediate deduction; D5 no caps; D6 Poisson start stock, empty pipelines; D7 unlimited supplier | ✓ tested |
| D8 depot stock after today's shipments + supplier orders on the way, no store stock | ✓ tested |
| D9 shelf + shipments on the way; D10 nearest step, ties up, caps; D11 same split | ✓ tested |
| D12 ranges, 20 years, separate tuning/test seeds, edge check, chosen targets reported | ✓ |
| Same demands and starting stock for every policy (Section 2 evaluation) | ✓ demand pre-drawn per seed, tested with different policies |
| True reward only; seeded RNGs; no demand files | ✓ |
| Section 6: all numbers in config | Fixed one gap: the random policy's action count and seed offset are now from the config (commit `3949e53`) |

No other mismatches found. The step/day mapping is the one spec gap I had to fill provisionally (Q1).

## 3. Choices made under Section 4

- **Code structure:**
  - `RestockEnv` keeps its state as plain Python ints and pre-draws the whole year's demand at `reset`. This is faster, and it guarantees the same demand for any policy.
  - Flat action index = `(i_depot × 9 + i_A) × 9 + i_B`.
  - Lead times are general in code; the default config gives exactly the 6-value observation.
  - `reset(options=...)` can fix the starting stock and demand, for tests.
- **Observation:** float32, raw counts, upper bound infinite. Gymnasium's checker warns about the infinite bound; that warning is intended under D5 and filtered in that one test.
- **Policy interface:** a policy is a callable obs → action index, with an optional `reset(seed)`.
- **Evaluation output:**
  - Each year also reports discounted reward, revenue and holding cost.
  - The summary gives the mean and standard deviation of reward over years.
- **Grid search operation:** 4 workers at niceness 10, results written to CSV and JSON in `results/grid_search/`.
- **Tests:**
  - The additional tests listed above.
  - The mutation check, run in the scratchpad and not committed.
- **Helper:** `pytest.ini` so tests import `src.*`.

## 4. Open questions logged (`notes/open-questions.md`)

The code needed some behavior to run for Q1–Q3, Q6 and Q7. For those it uses the suggested default, marked provisional. Nothing was decided.

| ID | Related | Question | Suggested default |
|---|---|---|---|
| Q1 | new | How one Gymnasium step maps onto the day, and whether the first decision comes before day 1's demand | Implemented: `reset` = day-0 decision point; step reward = sales of day t+1 − holding of day t |
| Q2 | D19 | When "average stock" is measured | Implemented: stock that pays holding (after shipments leave) |
| Q3 | D19 | Fill rate over several years: pooled or mean of yearly rates | Implemented: pooled |
| Q4 | D18 | Test seed count, DQN training seed range | Test 900000–900049; training from 200000 |
| Q5 | D18 | Warm-up exclusion vs. "profit over 365 days" | Evaluate keeps `warmup_days=0` |
| Q6 | new (D12) | Grid search maximizes undiscounted or discounted reward | Implemented: undiscounted (both give the same top 3) |
| Q7 | new (D12) | "20 years" = 20 separate years or one 20-year run | Implemented: 20 separate years |
| Q8 | D16 sub-item | Observation scaling/clipping for the network | Nothing done |
| Q9 | D22 | Where heavy compute runs (grid search ran on the login node, 3 min) | DQN training via SLURM |
| Q10 | new (docs) | CLAUDE.md Sections 0/5/6 still call Sections 5–6 "proposals" after D25 | Reword to point to D25 |
| Q11 | new (D21) | Pinned NumPy needs Python ≥ 3.12; D21 says ≥ 3.10; demand may differ across NumPy versions | Require Python ≥ 3.12 |

**Still OPEN in CLAUDE.md and not touched:** D13 (SB3 vs. PyTorch), D14–D20, D22–D24.

**Next decisions needed:**
- **Q1:** it fixes the meaning of the yearly profit for both methods.
- **D18 / Q4:** the test seed count; final evaluation of the baseline can't run until it's set.
- **Q11:** the Python minimum.
- **D13–D17:** before any DQN work.
