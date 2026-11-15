# Overnight report 3 — DQN tuning round 1 (2026-11-15)

Branch `feature/env-baseline`. Pushed to GitHub and open as PR #6 into `main`, not merged. `main` and history are untouched.

## Headline

**Reward centering closed the gap.** The selected variant (V2: centering + learning rate 5e-5, target update every 20k steps, batch 128) earns **97,128 ± 80** per year on the 50 test years (3 seeds, best checkpoints selected on the tuning years). The tuned baseline earns **96,614**. That is +514 per year (+0.5%), and V2 is better in 49 of 50 years. Details are in `notes/tuning-round-1.md`.

**Tuning years** (selection), best checkpoint per seed, mean ± std over 3 seeds:

| Variant | 365-day reward | Fill A | Fill B | Fill overall | Stock depot | Stock A | Stock B |
|---|---|---|---|---|---|---|---|
| V0 current setup | 89,730 ± 978 | 97.0% ± 1.2% | 94.5% ± 4.9% | 96.1% ± 2.4% | 3.2 ± 2.6 | 10.2 ± 0.9 | 9.5 ± 4.0 |
| V1 + reward centering | 96,345 ± 173 | 96.2% ± 0.3% | 94.0% ± 1.4% | 95.5% ± 0.6% | 1.1 ± 0.3 | 5.1 ± 0.2 | 5.6 ± 0.7 |
| **V2 + lr 5e-5, target 20k, batch 128** | **97,027 ± 31** | 96.3% ± 0.5% | 93.1% ± 0.3% | 95.2% ± 0.4% | 1.3 ± 0.2 | 4.7 ± 0.4 | 4.5 ± 0.1 |
| V3 + exploration 0.3 | 97,016 ± 14 | 97.0% ± 0.2% | 92.7% ± 0.7% | 95.6% ± 0.3% | 1.4 ± 0.2 | 5.0 ± 0.1 | 4.7 ± 0.4 |
| Baseline (60 / 24 / 32) | 96,525 | 95.7% | 92.7% | 94.7% | 4.2 | 4.0 | 3.7 |

**Test years** (used once, V2 only):

| Metric (50 test years) | V2, mean ± std over 3 seeds | Baseline (60 / 24 / 32) |
|---|---|---|
| 365-day reward | **97,128 ± 80** | 96,614 |
| Fill rate A / B / overall | 96.2% / 93.0% / 95.1% | 95.6% / 92.7% / 94.6% |
| Avg stock depot / A / B | 1.3 / 4.7 / 4.6 | 4.2 / 4.0 / 3.8 |

**Paired difference** (V2 minus baseline, same year): +514 ± 317. V2 is better in 49 of 50 years; per seed, 44, 49 and 47 of 50.

The gain is about 52 fewer lost sales per year at the same holding cost (+7.5). DQN holds less at the depot and more on the store shelves.

## Section 7 summary

### 1. What I changed (files)

- **Step 0 — decisions** (commit `8e1c27f`):
  - `CLAUDE.md`:
    - D20: provisional trigger (one tuning round first); which reduction is still OPEN.
    - D16: tuning-round-1 variants V0–V3 and the protocol.
    - D18: DQN may select on the tuning years; the test years are used once.
  - `README.md` "DQN and evaluation" block: the same three decisions.
  - `notes/open-questions.md`: Q19–Q21 marked decided.
- **Task 1 — variants and centering** (commit `364e9d7`):
  - `configs/default.yaml`:
    - `baseline.tuning_mean_daily_reward: 264.4512` (= 96,524.7 / 365)
    - `dqn.reward_centering: false`
    - `dqn.checkpoint_every: 100000`
  - `configs/tuning_round_1/v0–v3.yaml`: each lists only its change on top of the previous file, via `base:`.
  - `src/train_dqn.py`:
    - `load_run_config` (`base:` inheritance)
    - the `CenterReward` wrapper, with the explanation of why it leaves the optimal policy unchanged
    - checkpointing from the config
    - `score_checkpoints` and `select_best`, run on the tuning years at the end of each job
    - `--config`, `--score-only` and `--no-score`
  - `tests/test_train_dqn.py`: 9 new tests:
    - centering wrapper
    - centered and scaled reward with Monitor logging true reward
    - centering only in V1–V3
    - the centering constant recomputed from the baseline
    - centering shifts every policy's discounted return by the same constant
    - variant files differ only as decided
    - V3 model hyperparameters
    - checkpoint selection
    - scoring uses the tuning years and true reward
  - `scripts/train_dqn.slurm`: `CONFIG` variable, git module, 4 h limit.
- **Round tooling** (commit `bc48e5c`, plot fix later): `src/tuning.py` and `tests/test_tuning.py`.
  - `summarize`: tables, plots, and variant selection from the tuning years only.
  - `test`: re-derives the selection itself, has no option to name a variant, and refuses a second run.
- **Slot optimization** (commit `bda88d5`): `scripts/train_dqn_packed.slurm` runs several seeds per job, one core and one thread each.
- **README** (commit `61f0ce4`): tuning-round instructions.
- **Public-repo cleanup** (commit `da7f1e7`, option C): removed the cluster username from paths, the course account name, and other job names from the current files. History and `main` were not rewritten. This repo's `user.email` is now the GitHub noreply address.
- **Task 3/5 notes:**
  - `notes/tuning-round-1.md`
  - `notes/open-questions.md`: Q22–Q28
  - this report

### 2. Tests

- `pytest -q`: **78 passed** in about 23 s. Breakdown: 19 env, 6 evaluate, 28 baseline, 20 train_dqn, 5 tuning.
- End-to-end smoke run on the login node: V3 config, 20k steps, checkpoints every 10k, scoring included, 42 s.
- Dry runs of `src.tuning summarize` and `test` on a fake round built from smoke models. For the `test` dry run, the "test" seeds were patched to the sanity seeds, so the real test years were not touched.

### 3. Choices I made under Section 4

- **Variant files:** they inherit (`base:`) instead of being full copies, so the files themselves show "V2 = V1 + …". A test checks that each differs from the previous one only in the decided keys.
- **Scoring inside the job:** checkpoint scoring runs at the end of each SLURM job, on the compute node, and writes `checkpoint_scores.csv` and `best_checkpoint.json`.
- **Ties:**
  - Between checkpoints: the earliest wins.
  - Between variants: the earlier variant wins. This never happened; V2 beat V3 by 11.
- **Plots:**
  - Learning curves: one plot, mean of 3 seeds per variant, categorical colors in fixed order (palette slots 1–4), a legend plus direct end labels, and the baseline as a dashed line.
  - Extra small-multiples plot of every checkpoint's tuning score, with selected checkpoints circled. Its y-axis is clipped at 0, and the very early, very negative checkpoints are marked at the bottom edge.
  - I couldn't run the palette validator (no `node` on the cluster).
- **Repacking:** when you asked to free slots, the running jobs were past halfway, so I left them alone. I counted V2 seed 1, which had started only 3 s earlier, as not started: I cancelled it and deleted its partial output. Repacked jobs run from a worktree pinned at `bc48e5c` (`/scratch/$USER/restock-rl/code-bc48e5c`), so the code commit is identical.
- **Thread count:** to get 1 thread per process without editing `src/train_dqn.py` or the config (which says 2), the packed script uses a small launcher that fixes torch at 1 thread. Training is otherwise identical.

### 4. Open decisions that need an answer next

- **Q26 (D18, D20):** is the 3-seed V2 result final, or should V2 get 5 seeds, and how would a second use of the test years work? Is tuning finished?
- **Q28:** is the settled order-up-to baseline strong enough for a 0.5% claim?
- Still OPEN in CLAUDE.md:
  - D20: which reduction (not triggered this round).
  - D23: logging.
  - D24: module ownership.

## Commits tonight (oldest first)

| Commit | Message |
|---|---|
| 8e1c27f | Record provisional decisions for tuning round 1 (D20 trigger, D16 variants, D18 tuning-year selection) |
| 364e9d7 | Tuning round 1 setup: variant configs V0-V3, reward-centering wrapper, checkpoint scoring on tuning years |
| bc48e5c | Add tuning-round summary, variant selection on tuning years, and one-time test evaluation |
| 61f0ce4 | README: tuning round 1 instructions; log Q22-Q25 |
| bda88d5 | Add packed SLURM script: several seeds per job, one core and one thread each |
| da7f1e7 | Remove cluster username, course account name and unrelated job names from notes and script comments |
| (next) | Tuning round 1 results; checkpoint-plot axis fix; overnight report 3 |

**Push and PR:** `feature/env-baseline` pushed; PR #6 opened (https://github.com/AcarBasaran/restock-rl/pull/6), not merged. Before pushing, I scanned all branch commits.
- **Not found:** emails, tokens, keys, large files, results, models, logs, the venv or checkpoints.
- **Found:** your cluster username and hostname in the commit author email of all branch commits and 4 on `main`, plus a few strings in notes and a script comment. You chose option C: the current files are clean, and history is unchanged.

## Jobs and runtimes

| Job | What | Node | State | Elapsed |
|---|---|---|---|---|
| 2024398 | timing: V3 settings, 200k steps, scoring | ai06 | COMPLETED | 0:03:15 (1,141 steps/s) |
| 2024443_0–2 | V0 × 3 seeds | ai02 | COMPLETED | 0:57:29 / 0:55:22 / 1:05:27 |
| 2024444_0–2 | V1 × 3 seeds | ai01 | COMPLETED | 0:35:18 / 0:40:14 / 0:38:02 |
| 2024445_0 | V2, seed 200000 | ai01 | COMPLETED | 0:35:50 |
| 2024445_1–2 | V2, seeds 200001–2 | — | CANCELLED for repacking (_1 after 24 s; partial output deleted) | — |
| 2024446_0–2 | V3 × 3 seeds | — | CANCELLED while pending, for repacking | — |
| 2024734 | V2, seeds 200001–2, packed | ai01 | COMPLETED | 0:56:59 |
| 2024733 | V3 × 3 seeds, packed | ai01 | COMPLETED | 0:55:25 |

- No failures and no resubmissions for errors.
- Your other jobs share the `ai` QOS (max 8 running per user). After repacking, tuning held at most 5 slots, and 2 once V0 finished.
- **Login-node use:**
  - tests, about 23 s
  - one 42 s smoke run
  - fake-round dry runs
  - baseline tuning-year evaluation
  - summary, 5 s
  - test evaluation, 23 s
- No training ran on the login node.

**Reproducibility check:** V0 seed 200000 trained on `ai02`, the same node type as night 2's seed-index-0 run. Its final model scores exactly 36,333 on the tuning years, the same as night 2. Seeds 200001–2 ran on `ai04` in night 2 and on `ai02` tonight, and their final models differ (e.g. 60,380 vs 56,116). So training is bit-reproducible on the same hardware, but not across node types.

## Open questions (all in `notes/open-questions.md`)

- **Still OPEN in CLAUDE.md:**
  - D20: which scope reduction (trigger not fired).
  - D23: logging.
  - D24: module ownership.
- **Older:**
  - **Q2, Q3:** how average stock and fill rate are measured.
  - **Q6, Q7:** details of the grid search.
  - **Q10:** CLAUDE.md wording of Sections 5–6.
  - **Q12:** training seed values.
  - **Q13:** obs-scale rule.
  - **Q14:** the `ai` account.
  - **Q15:** `/home` quota.
  - **Q16:** stale Section 2 text.
  - **Q17:** logging.
  - **Q18:** learning curves / model selection. Partly answered by D18 tonight.
- **New tonight:**
  - **Q22:** seeds 200000–200002 for every variant.
  - **Q23:** the shared `ai` QOS slots.
  - **Q24:** what "clearly below" means. It didn't matter this round.
  - **Q25:** Section 2 fallback text partly stale.
  - **Q26:** 3 vs 5 seeds, and whether tuning is finished.
  - **Q27:** how to report a small but consistent difference.
  - **Q28:** whether the baseline is strong enough for a 0.5% claim.
