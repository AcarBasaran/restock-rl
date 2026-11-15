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

---

## Logged during night 2

## Q12 · D18 detail · Which numbers are the 5 DQN training seeds?

- D18 fixes 5 seeds but not their values. **Provisional (in the config):**
  200000–200004. SB3 uses the seed for the network, exploration and the
  first `reset()`; later training years continue that environment's random
  stream, so training years never coincide with tuning (100000–100019) or
  test (900000–900049) years, which always start from a fresh seed.
- **Suggested default:** keep.

## Q13 · D16 detail · Exact rule for the observation scaling constants

- D16 says "chosen from the observed ranges in the baseline runs" but not
  how. **Provisional rule:** the maximum of each feature while the tuned
  baseline runs on the 20 tuning years (53, 28, 21, 40, 28, 28), rounded up
  to the next multiple of 10 → `[60, 30, 30, 40, 30, 30]`. The test years were
  not used. There is no clipping, so states DQN reaches that the baseline
  never does give scaled values above 1 (early in training, Store B stock of
  ~300 gives 10).
- **Alternatives:** 99th percentile; or use the action caps (40, 32, 32) for
  the pipeline features.
- **Suggested default:** keep the provisional rule.

## Q14 · D22 detail · Which SLURM account and partition?

- The CPU partitions `short`/`mid`/`long` only accept account `users`, which
  this user does not have. The default course account may only use
  `t4_ai` (GPU nodes, 8 h limit), where `sbatch --test-only` estimated a
  start ~8.5 h later for any job size. Account `ai` on partition `ai` (the
  account the user's other jobs were using tonight) starts at once.
- **Used:** account/partition `ai`, QOS `ai`, 2 CPUs, 4 GB, no GPU, 3 h limit.
  Total use tonight: about 5 × 1 h × 2 CPUs.
- **Question:** is the `ai` account fine for this course project, or should
  the team ask the admins for `users` access (CPU partitions) or run under
  the default course account on `t4_ai` and accept the wait? Switching is one line in
  `scripts/train_dqn.slurm`.

## Q15 · new (infrastructure) · `/home` is over its quota

- `beegfs-quota-fs` reports **1.01 TiB used of a 750 GiB hard limit** on
  `/home` (not caused by this repo, which is ~2 MB). Writes there may start
  failing. VALAR's login message also asks to run active jobs from
  `/scratch/$USER`.
- **Done tonight:** the venv now lives at `/scratch/$USER/restock-rl/venv`
  and `.venv` is a symlink to it (same path for D21); job outputs go to
  `/scratch/$USER/restock-rl/`; only small final artifacts (models of
  ~1 MB, CSVs, plots) are copied into `results/`. `/scratch` is described as
  temporary, so anything worth keeping should not live only there.
- **Suggested action:** free space in `/home`.

## Q16 · new (documentation) · Section 2 still says the seed count is open

- Section 2 (Evaluation) still reads "DQN is run with several random seeds
  … The number is open; see D18". D18 is now decided (5 seeds). Section 2
  needs approval to change, so it was left alone.
- **Suggested change:** "The number is decided; see D18."

## Q17 · D23 · Logging (still open)

- Nothing beyond what tonight's tasks needed: SB3's Monitor CSV per run
  (per-year training reward in true units, used for the learning curves),
  SB3's console log in the SLURM output files, and a `run.json` per run
  (seed, git commit, versions, config, runtime). TensorBoard is installed
  and pinned (allowed) but not enabled.
- **Suggested default:** TensorBoard + CSV, as in CLAUDE.md.

## Q18 · new (evaluation detail) · What the learning curves show, and model selection

- The curves are the training reward per simulated year under ε-greedy
  exploration (ε = 0.05 after the first 200k steps) on training years, so
  they sit below what the greedy policy earns and are not directly
  comparable to the test results.
- Each seed is evaluated with its **final** model (as the task asked). Picking
  the best checkpoint instead would need a periodic greedy evaluation on
  separate (e.g. tuning) years, which is a protocol decision.
- **Suggested default:** keep final-model evaluation for the first run.

## Q19 · D20 · The first DQN run clearly fails: what next?

**Status (night 3): decided (provisional) in D20.** One focused tuning round first; scope reduction if the best tuned DQN is still clearly below the baseline on the tuning years. Which reduction is still open.

- Result (see `notes/dqn-vs-baseline.md`): DQN scores 67,764 ± 21,110 per year
  over 5 seeds, against the baseline's 96,614 on the 50 test years. It is
  worse in every year for every seed.
- D20's suggested trigger is "DQN cannot match the baseline after 2 tuning
  rounds". No tuning round has happened yet (nothing was tuned tonight, as
  instructed), so the trigger hasn't fired.
- **Question:** run tuning rounds first (Q20), or go straight to a scope
  reduction (fewer actions, or distribution between stores only)? Also,
  what counts as one "tuning round"?
- **Suggested default:** one tuning round on the full 405-action problem
  first, using tuning years for selection, then decide D20.

## Q20 · D16 · Which hyperparameters to revisit first (likely causes)

**Status (night 3): decided (provisional) in D16.** Tuning round 1 = fixed variants V0–V3, 3 seeds each.

- Observations, not tested fixes:
  - Training reward peaks (about 83k) exactly when ε reaches 0.05 at 0.2M
    steps, then declines to about 60–68k with sharp drops.
  - Greedy checkpoints jump around a lot. One seed went from 93.7k to
    73k, then 0.2k, then 77k on the tuning years.
- Candidates:
  - softer or less frequent target updates (`tau`, `target_update_interval`)
  - learning rate
  - buffer size
  - longer exploration or a higher final ε
  - Double DQN (not in SB3's DQN; would need a different library or a
    custom implementation, which touches D13)
  - a reward scale that keeps Q-values near 1 (D17)
  - action handling (D14)
- **Question:** which of these does the team want in round 1, and how many
  configurations × seeds per round?
- **Suggested default:** none chosen; this needs a team decision.

## Q21 · Q18 follow-up · Checkpoint diagnostic on the tuning years

**Status (night 3): decided (provisional) in D18.** DQN may select on the tuning years; test years only once for the selected configuration.

- To diagnose the failure I evaluated the 500k / 1M / 1.5M / 2M checkpoints
  greedily on the 20 tuning years. I didn't use the test years, and nothing
  was selected or reported as the result. The tuning years are also the
  baseline's grid-search years.
- **Question:** is using the tuning years for DQN checkpoint or
  hyperparameter selection acceptable? The alternative is a separate DQN
  validation seed range.
- **Suggested default:** a separate validation range (e.g. seeds
  300000–300019), so that neither method's selection years overlap with the
  other's. This needs a team decision.

## Logged during night 3

## Q22 · D16/D18 detail · Which 3 seeds per variant in tuning round 1?

- I used the first three training seeds (200000, 200001, 200002) for every
  variant, so the variants are compared on the same seeds. V0 on these seeds
  is the same setup as night 2's seed indices 0–2, which doubles as a
  reproducibility check (only the checkpoint frequency differs, and it does
  not affect training).
- **Question:** is that fine, or should tuning use seeds separate from the
  5 "final" training seeds?
- **Suggested default:** keep it. Seeds 200003–200004 were not used tonight.

## Q23 · D22 detail · The `ai` QOS is shared with your other jobs

- QOS `ai` allows 8 running jobs per user (50 submitted). Your other
  projects run
  on the same QOS, so tonight's 12 runs competed with them for those 8
  slots. I did not touch the other jobs.
- **Question:** is that acceptable, or should restock-rl runs use fewer
  concurrent slots (e.g. `--array=0-2%2`)?

## Q24 · D20 detail · What counts as "clearly below the baseline"?

- D20 says to move to the scope reduction if the best tuned DQN is "still
  clearly below" the baseline on the tuning years. No threshold is defined.
- **Question:** what is the threshold? For example: below the baseline in
  most tuning years, a gap larger than the seed-to-seed std, or a fixed
  percentage.
- **Suggested default:** none. The numbers are reported in
  `notes/tuning-round-1.md` and the call is the team's.

## Q25 · documentation · Section 2's fallback line is partly stale

- Section 2 (Fallback) says "Which reduction and when is open; see D20".
  The "when" part is now decided (provisional) in D20. Section 2 needs
  approval to change, so it was left alone (same situation as Q16).
- **Suggested change:** "Which reduction is open; when is decided; see D20."

## Q26 · D18/D20 · What comes after tuning round 1?

- V2 beat the baseline on the test years by +514 per year (+0.5%; better
  in 49 of 50 years) with 3 seeds (see `notes/tuning-round-1.md`). D18
  says 5 DQN training seeds; seeds 200003 and 200004 of V2 were not
  trained.
- The test years have now been used once for V2. Training two more V2
  seeds and testing them would be a second use for the same configuration.
- **Question:** is the 3-seed result the final DQN result, or should V2
  get 5 seeds? If 5, how should the second use of the test years be
  handled? Is tuning finished (D20's trigger did not fire)?
- **Suggested default:** none; this needs a team decision.

## Q27 · D19 · How to report a small difference

- The gap is small in relative terms (0.5%) but very consistent: paired
  per-year std 317, so the gap is about 11 standard errors.
- **Question:** which uncertainty goes in the report? Options: the spread
  across seeds (± 80), paired per-year differences, or both. Is a formal
  test wanted?
- **Suggested default:** report both: mean ± std across seeds, and the
  paired per-year difference with the number of years DQN is better.

## Q28 · Section 2 / D8 · Is the order-up-to baseline strong enough for a 0.5% claim?

- DQN's gain comes from holding less stock at the depot and more on the
  store shelves. Under D8 the baseline's depot rule ignores store stock,
  and the targets move in grid steps (depot 10, stores 4), so the baseline
  may be leaving some of that on the table.
- This is a question, not a proposal to change the settled baseline
  (Section 2, D8–D12). Any change needs a team decision.
- **Question:** is the comparison against the settled baseline enough for
  the report? Or should the report discuss this limitation, or add a
  stronger heuristic as an extra reference?

**Status of Q24 (night 3):** the best tuned DQN is above the baseline on
the tuning years (97,027 vs 96,525), so the "clearly below" threshold did
not matter this round. It is still undefined.
