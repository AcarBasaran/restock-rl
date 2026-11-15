# Open questions

Logged during the overnight runs. Items still open are not decided; where code needed some behavior, it uses the suggested default, marked provisional.
Still open after night 2: Q2, Q3, Q6, Q7, Q10, and everything from Q12 on.

Each item names the related decision ID, or "new" if it is not in CLAUDE.md yet
(I propose adding those to Section 3).

## Q1 · new · How does one environment step map onto the day, and what happens on the first day?

**Status: DECIDED** as D26 (provisional, by Acar): keep the current step boundaries.

The spec gives the daily order (arrivals → demand → decision) and says the
state is observed at the decision point, but not where a Gymnasium step
starts and ends, or whether the first decision comes before or after the
first day's demand.

- **Provisional (in `src/env.py`):** `reset()` returns the decision point
  of "day 0": the random starting stock, nothing in transit, no demand yet.
  Each step applies the decision of day t, charges day t's holding cost
  (after shipments leave), then simulates day t+1's arrivals and demand.
  Step reward = sales of day t+1 − holding of day t. A year of 365 steps
  therefore counts holding on days 0–364 and sales on days 1–365.
- **Alternative:** `reset()` also simulates day 1's demand before the first
  decision. Then day 1's sales belong to no step, and the year counts
  sales on days 1–365 only if they are added separately.
- **Why it matters:** it shifts about one day of holding and sales per year
  (small, and the same for both methods), but it fixes what "profit over
  365 days" means exactly. Every reward in the reward function is a proper
  function of (state, action, next day's demand) only under the
  provisional version, which is what DQN needs.
- **Suggested default:** keep the provisional version.

## Q2 · D19 · At what point in the day is "average stock" measured?

- **Provisional (in `src/evaluate.py`):** the stock that is charged for
  holding: end of the decision day, after shipments left. In-transit units
  are not counted.
- **Alternatives:** stock right after sales (before shipping), or
  including units on the road.
- **Suggested default:** provisional version (it matches the cost the
  methods pay).

## Q3 · D19 · How are fill rates combined over several years?

- **Provisional:** the summary pools all years (total sales ÷ total demand);
  per-year fill rates are also kept in each episode's results.
- **Alternative:** mean of the per-year fill rates (almost identical here,
  because demand per year barely varies).
- **Suggested default:** pooled.

## Q4 · D18 · Test and training seed ranges

**Status: partly decided** by D18: 50 test years from the test range (seeds 900000–900049), 5 DQN training seeds. The DQN training seed *values* are still not decided; see Q12.

- `configs/default.yaml` reserves test seeds starting at 900000 but leaves
  the count empty, so code refuses to run on the test range until D18 is
  decided. No DQN training seed range is defined.
- Ranges in use: sanity 0–4, tuning 100000–100019 (grid search).
- **Suggested default:** test 900000–900049 (50 years); DQN training
  demand seeds from 200000 upward; network/exploration seeds 1–5.

## Q5 · D18 · Warm-up exclusion

**Status: DECIDED** by D18: no warm-up exclusion.

- `evaluate.py` takes `warmup_days` (default 0). The D18 default (drop the
  first 30 days) conflicts with the settled main metric "profit over 365
  days" in Section 2; this was flagged before and is still open.
- **Suggested default:** report the full 365 days as the main metric;
  optionally also report the metric without the first 30 days.

## Q6 · new (D12 detail) · Which number does the grid search maximize?

- **Provisional (in `src/baseline.py`):** the mean undiscounted 365-day
  reward over the tuning years (the main evaluation metric).
- **Alternative:** the mean discounted reward with γ = 0.99 (DQN's training
  objective). Both are saved for every candidate in the results CSV, so
  the choice can be changed without re-running. See notes/grid-search.md
  for whether they pick the same targets.
- **Suggested default:** undiscounted (matches the main metric).

## Q7 · new (D12 detail) · What are "20 simulated years per candidate"?

- **Provisional:** 20 separate 365-day years, each starting from fresh random
  stock, on tuning seeds 100000–100019, the same 20 for every candidate.
- **Alternative:** one continuous 20-year run per candidate.
- **Suggested default:** provisional version (it matches how both methods
  are evaluated).

## Q8 · D16 sub-item · Observation scaling and clipping

**Status: DECIDED** by D16: divide each feature by a fixed constant from the config, chosen from baseline runs. The exact rule for picking the constants is Q13.

- The environment returns raw, unbounded unit counts (D5), so Gymnasium's
  checker warns that the observation upper bound is infinite. This is
  intended; how the network sees the values is part of the open D16
  sub-item. Nothing was done here.

## Q9 · D22 · Where does heavy compute run?

**Status: DECIDED** by D22: training only as SLURM jobs.

- The grid search ran on the VALAR login node (`login01`, no SLURM job), with
  4 worker processes at lowered priority, because the whole grid takes only
  a few minutes. DQN training (1–2M steps × several seeds) is much heavier
  and should probably go through SLURM, which is D22.

## Q10 · new (documentation) · CLAUDE.md still calls Sections 5–6 "proposals"

- D25 is now decided as a starting point, but Section 0 ("Sections 5 and 6
  contain proposals"), the Section 5 heading ("proposed — see D25") and
  Section 6 ("Proposed conventions … ask before relying on them") still
  say otherwise. I left that wording alone because only Section 3 was in
  scope.
- **Suggested change:** update those three phrases to point to D25.

## Q11 · new (D21 follow-up) · Same demand on every teammate's machine

**Status: DECIDED** in D21: minimum Python 3.12.

- Demand is reproducible from the seed, but NumPy does not promise the
  same random stream across versions. `requirements.txt` pins
  `numpy==2.5.4`, which needs **Python 3.12 or newer**, while D21 allows
  3.10 or newer. A teammate on Python 3.10/3.11 would have to install a
  different NumPy and could get different demands for the same seed.
- **Options:** (a) require Python ≥ 3.12 (Colab and the VALAR module both
  qualify); (b) pin an older NumPy that installs on 3.10 and re-pin
  everyone to it.
- **Suggested default:** (a), and record the Python minimum in D21.
