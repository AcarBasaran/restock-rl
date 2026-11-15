# Overnight report 2 — DQN training and comparison (2026-11-15)

Branch `feature/env-baseline`, all commits local. Nothing was pushed or merged, and no GitHub issues were touched.

## Headline

**The first DQN run clearly fails to beat the tuned baseline.** On the 50 test years, DQN scores **67,764 ± 21,110** per year across 5 seeds; the baseline scores **96,614**. Every seed is worse than the baseline in every one of the 50 years. Nothing was tuned (as instructed). The likely causes are logged as open questions Q19–Q21, and D20 is still open. Details are in `notes/dqn-vs-baseline.md`.

| Metric (50 test years) | DQN, mean ± std over 5 seeds | Baseline (60 / 24 / 32) |
|---|---|---|
| 365-day reward | 67,764 ± 21,110 | 96,614 (std over years 1,156) |
| Fill rate Store A | 69.3% ± 28.9% | 95.6% |
| Fill rate Store B | 81.1% ± 29.0% | 92.7% |
| Fill rate overall | 73.3% ± 27.2% | 94.6% |
| Average stock depot | 3.8 ± 4.7 | 4.2 |
| Average stock Store A | 3.9 ± 3.8 | 4.0 |
| Average stock Store B | 11.5 ± 11.8 | 3.8 |

Per-seed rewards: 36,362 / 56,037 / 85,433 / 77,131 / 83,859 (seeds 200000–200004). The paired difference (seed-averaged DQN minus baseline, same year) is −28,850 ± 689; DQN is better in 0 of 50 years.

## Section 7 summary

### 1. What I changed (files)

- **Step 0 — decisions:**
  - `CLAUDE.md`: D26 is new (step boundary, was Q1). D13–D19 and D22 are marked `DECIDED (provisional, decided by Acar; team can object)`, and D21 has minimum Python 3.12 (was Q11). Section 2 is untouched (see Q16).
  - `README.md`: new block "DQN and evaluation", plus the Python 3.12 setup note.
- **Task 1:** `src/baseline.py` gained `evaluate` and `obs-ranges` subcommands and `tuned_policy`. `configs/default.yaml` has a `baseline:` section (60/24/32) and the 50-year test range. `notes/baseline-test.md` holds the baseline result on the test years: **96,614**, fill rate 94.6%.
- **Task 2:**
  - `src/train_dqn.py`:
    - wrappers `ScaleObservation` and `ScaleReward`
    - `make_training_env`, with Monitor placed inside the scaling so it logs true reward
    - `build_model`, with every hyperparameter from config
    - `GreedyDQNPolicy`
    - `train` and a CLI
  - `configs/default.yaml` `dqn:` section.
  - `tests/test_train_dqn.py` (11 tests).
  - `requirements.txt` pins torch 2.15.0+cpu, stable-baselines3 2.9.2, matplotlib 3.11.2 and tensorboard 2.21.0.
- **Task 3:** `scripts/train_dqn.slurm`, a 5-task array on account/partition/qos `ai`, 2 CPUs, 4G, 3 h.
- **Task 4:** `src/compare.py` evaluates the baseline and each final model on the test years, computes paired differences, and draws two plots.
- **Task 5:** `notes/dqn-vs-baseline.md`.
- **Task 6:** self-review fixes. Activation and optimizer moved into config, and the `evaluate.py` docstring was updated.
- **Task 7:** README "How to run" now covers setup (torch CPU index, the scratch venv), tests, the smoke run, SLURM timing and full runs, and the comparison.
- `notes/open-questions.md`: status lines for Q1, Q4, Q5, Q8, Q9 and Q11; new Q12–Q21.

### 2. Tests

- `pytest -q`: **64 passed** in 21.5 s (19 env, 6 evaluate, 28 baseline, 11 train_dqn). This was the last run, after the final code change.
- Key new test: `test_sb3_stores_day_365_as_timeout_not_terminal`. In SB3's replay buffer, step 365 is stored with `done = 1` and `timeout = 1`, and the done flag used in the TD target is 0. The stored next observation is the real end-of-year state, not the next reset. Other tests check:
  - the wrappers don't change dynamics or info
  - Monitor logs true reward
  - config values equal SB3 defaults (except net_arch)
  - the built network is 6→256→256→405 with ReLU and Adam
  - the greedy policy equals `predict(deterministic=True)`
- Smoke run on the login node: 40k steps in 0.7 min (956 steps/s). The pipeline works end to end.

### 3. Choices I made under Section 4

- `GreedyDQNPolicy` calls the Q-network directly instead of `model.predict`. It gives identical actions (tested) and made evaluation about 9× faster. `torch.set_num_threads(1)` is used for tests and evaluation on the shared login node.
- Checkpoints every 500k steps, and SB3 console logging every 100 episodes. These are logging settings, not hyperparameters. They live in code defaults and are not in the config (D23 is still open).
- Output layout:
  - runs under `/scratch/$USER/restock-rl/runs/<run_name>/seed_index_<i>/`
  - comparison results under `results/dqn/<timestamp>/`
  - the comparison copies each model (about 1 MB), monitor CSV and `run.json` into `results/`, so that folder is self-contained (gitignored)
- Plot styling:
  - seeds drawn as neutral grey replicate lines, the seed mean in blue, and the baseline as a dashed reference with a direct label
  - labels placed in empty space with a surface box (fixed after inspecting the first render, where two labels collided with lines)
- Test seeds: the test range was set to 900000–900049 (50 years) under D18. In `test_evaluate.py`, the range is checked for size and non-overlap.

### 4. Open decisions that block the next step

- **D20 (OPEN)** — what to do now that DQN fails (Q19).
- **D16 tuning round** — which knobs to try (Q20).
- **D23 (OPEN)** — logging (Q17).
- **D24 (OPEN)** — module ownership.
- Full list below.

## Commits tonight (oldest first)

| Commit | Message |
|---|---|
| 522fce2 | Record provisional DQN, evaluation and setup decisions (D13–D19, D21, D22, D26) |
| fea00f2 | Evaluate tuned baseline on the 50 test years (D18) |
| 529599f | Add SB3 DQN training with observation and reward scaling wrappers (D13–D17) |
| e7a1664 | Add SLURM job array script for DQN training; record git commit per run |
| 4f466bb | Add DQN-vs-baseline evaluation with learning-curve plots; fast greedy policy |
| b43cffd | README: training, SLURM and evaluation instructions; log night-2 open questions |
| f20ce43 | Self-review: activation and optimizer in config (D16); update evaluate docstring |
| 42a19f7 | Add DQN vs baseline comparison (first run); fix learning-curve label placement; log Q19–Q21 |
| (next) | Add overnight report 2 |

The training runs used code at `e7a1664`, which is recorded in each `run.json`. Later commits did not change training behaviour. f20ce43 only moved the already-default ReLU/Adam into the config explicitly, so those two keys are missing from the runs' config snapshot.

## SLURM jobs and runtimes

Cluster setup:

- CPU partitions need account `users`, which I don't have.
- The default account `comx29` can only use `t4_ai`, where a test submission estimated a start about 8.5 h later.
- I used account/partition/qos **`ai`** with 2 CPUs, 4G, no GPU, and a 3 h limit (Q14).
- The venv lives at `/scratch/$USER/restock-rl/venv`, because `/home` is over quota (Q15).

| Job | What | Node | State | Elapsed | Speed |
|---|---|---|---|---|---|
| 2021957 | timing run, 50k steps | ai02 | COMPLETED | 00:01:54 | 566 steps/s |
| 2022023_0 | seed 200000, 2M steps | ai02 | COMPLETED | 01:05:35 | 511 steps/s |
| 2022023_1 | seed 200001, 2M steps | ai04 | COMPLETED | 00:37:28 | 906 steps/s |
| 2022023_2 | seed 200002, 2M steps | ai04 | COMPLETED | 00:38:48 | 874 steps/s |
| 2022023_3 | seed 200003, 2M steps | ai04 | COMPLETED | 00:39:03 | 869 steps/s |
| 2022023_4 | seed 200004, 2M steps | ai04 | COMPLETED | 00:40:05 | 846 steps/s |

All tasks started at 05:55:36 and the array ended at 07:01:11. No job failed, so nothing was resubmitted. SLURM logs are in `/scratch/$USER/restock-rl/slurm/`.

Login-node use (D22 allows light work):

- the test suite, about 21 s (one earlier run took 111 s before I limited torch threads)
- the 0.7 min smoke run
- baseline evaluations
- one comparison test run that took about 4 min before the speed-up; now about 30 s
- the final comparison, 30 s
- a 31 s checkpoint diagnostic

No training ran on the login node.

## Self-review (Task 6) against Section 2 and the DECIDED items

- **Section 2 network, actions, reward and costs:** the environment code is unchanged since night 1, apart from a docstring, and its 19 tests pass. No costs were added. The wrappers change only what the network sees (tested).
- **Truncation, not termination:** the env returns `truncated=True` at step 365, and SB3 stores it as a timeout and bootstraps (tested).
- **γ = 0.99:** passed to SB3 from the config (tested).
- **Same demands and initial stocks:**
  - Both methods go through `evaluate()` with the same 50 test seeds.
  - The demand and the starting stock come from the seed at reset.
- **True reward units (D17):**
  - Monitor sits inside the reward scaling.
  - Evaluation uses the unwrapped env.
  - Every reported number is in true units.
- **D13–D17:** SB3 DQN, flat 405 actions, 2M steps, SB3 defaults plus 2×256, observations divided by config constants, reward divided by 100 in training only. All hyperparameters are in the config.
- **D18:** 50 test years, 5 seeds, no warm-up. **D19:** undiscounted reward, fill rate per store and overall, and average stock per location.
- **D21:** Python 3.13.9 (≥ 3.12). **D22:** training ran only through SLURM.
- **D26:** step boundary as implemented and documented.
- **Firm rules:**
  - no commits to main, no push
  - results/ and models not committed (checked with `git check-ignore` and `git ls-files`)
  - only allowed packages, pinned
  - seeded RNGs, no demand files
- **Found and fixed:** activation and optimizer were implicit SB3 defaults, not in the config, so I moved them into it (D16 says every hyperparameter goes in the config). The `evaluate.py` docstring still said D18/D19 were open.
- **Not changed (needs approval):** Section 2 still says the seed count is open (Q16).

## Open questions (all in `notes/open-questions.md`)

Still **OPEN** in CLAUDE.md:

- **D20:** scope reduction and trigger (now pressing, see Q19)
- **D23:** logging (Q17)
- **D24:** module ownership

Open questions:

- **Q2:** when in the day average stock is measured.
- **Q3:** how fill rates are combined over years (currently pooled).
- **Q6:** which number the grid search maximizes.
- **Q7:** what "20 simulated years per candidate" means exactly.
- **Q10:** CLAUDE.md still calls Sections 5–6 "proposals".
- **Q12:** DQN training seed values (200000–200004, provisional).
- **Q13:** exact rule for the observation scaling constants.
- **Q14:** whether the `ai` SLURM account/partition is acceptable.
- **Q15:** `/home` is over quota; the venv was moved to `/scratch`.
- **Q16:** Section 2's stale "number is open" seed-count text.
- **Q17:** logging is minimal for now (Monitor CSV, stdout, `run.json`; TensorBoard not enabled).
- **Q18:** learning curves show ε-greedy training reward; final model rather than best checkpoint.
- **Q19 (new):** the first DQN run fails. Tune first, or reduce scope (D20)? And what is a "tuning round"?
- **Q20 (new):** which hyperparameters to try first. Training peaks when ε reaches 0.05 at 0.2M steps and then degrades, and greedy checkpoints jump around a lot.
- **Q21 (new):** whether the tuning years may be used for DQN checkpoint or hyperparameter selection, or a separate validation range is needed. Tonight's checkpoint diagnostic used the tuning years and selected nothing.
