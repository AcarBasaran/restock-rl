# CLAUDE.md — restock-rl

Project title: **Deep Reinforcement Learning for Inventory Management in a Two-Store Chain**
Team of three students. This repo compares a DQN agent against a tuned order-up-to policy on a simulated depot + two-store inventory system.

---

## 0. The most important rule

This is a team project. Many design choices are **not settled** and must be agreed by the whole team. Making them on your own creates conflicts later.

- Section 2 is the **settled specification**, taken from the team's project proposal. Implement it exactly. Do not "improve" it, simplify it, or add costs, constraints, or features to it.
- Section 3 lists **open decisions**. For any item marked `OPEN`:
  - **Stop and ask the user before writing any code that depends on it.** Name the decision by its ID (e.g. "D4 is still open").
  - Do **not** silently implement the suggested default. Defaults are only there to make the discussion faster.
  - Do **not** hide the choice behind a guess. Making something a config option is fine only after the user confirms that approach.
  - When a decision is made, the user updates this file to `DECIDED: <answer>`. Treat only what is written in this file as decided, not something said earlier in a session.
- Items tagged **[team]** need a team discussion; the user will come back with an answer. Items tagged **[Acar]** the user can answer directly.
- Sections 5 and 6 contain **proposals**, not team decisions (see D25). Follow them only as a starting point and ask before relying on them in a way that is hard to change later.
- If you notice a new ambiguity that is not listed here and that changes results, behavior, or the comparison, treat it as `OPEN`: stop, describe it, and propose adding it to Section 3.
- If something in the settled spec looks wrong to you, say so and explain why, but do not change it without the user's approval.

---

## 1. Project goal

Research question: in a two-store system where the stores differ in demand level and delivery time, can DQN achieve higher profit than a simple, properly tuned order-up-to policy?

No external dataset is used. The environment simulator generates all experience; customer demand is drawn from a Poisson distribution.

Tools named in the proposal: Python; PyTorch or Stable-Baselines3 (see D13); the team's own computers or Google Colab (see D22).

---

## 2. Settled specification (from the proposal — implement exactly)

**Network**
- One supplier → one depot → two stores (A and B).
- Supplier → depot delivery time: 2 days.
- Depot → Store A delivery time: 1 day. Store A demand: Poisson, mean 20 per day.
- Depot → Store B delivery time: 3 days. Store B demand: Poisson, mean 10 per day.
- Lost sales: unmet demand is lost and not carried over to the next day.
- The number of lost customer orders is recorded for analysis but is **not** shown to the agent.

**Daily sequence**
1. Previously placed orders and deliveries arrive.
2. Customer demand is met.
3. At the end of the day, the agent decides the new depot order and the shipments to each store.

**Observation / state (6 values)**
1. Depot stock
2. Store A stock
3. Store B stock
4. Depot order arriving the next day
5. Store B shipment arriving in 1 day
6. Store B shipment arriving in 2 days

There is no pending-shipment variable for Store A, since its delivery time is one day. Daily demand does not depend on the past, so this state is Markov.

**Actions (405 discrete)**
- Depot order from the supplier: {0, 10, 20, 30, 40}.
- Shipment to Store A: {0, 4, 8, …, 32}.
- Shipment to Store B: {0, 4, 8, …, 32}.
- If depot stock is not enough for both requested shipments, the available stock is divided **proportionally** between the requested amounts. The rounding rule is decided; see D4.

**Reward**
- +10 per unit sold.
- At the end of the day: holding cost 2 per unit remaining at each store, 1 per unit remaining at the depot. The exact timing relative to shipping is decided; see D3.
- **No other costs.** No purchase cost, no fixed ordering or shipping cost, no lost-sales penalty. Do not add any.

**Objective and task type**
- Maximize long-term total reward with discount factor γ = 0.99.
- Continuing task. Training and evaluation use 365-day periods, but the end of the year is **not** a terminal state. Whatever framework is used, the cutoff must be treated as a time-limit truncation (e.g. `truncated=True` in Gymnasium, never `terminated=True`), so value targets still bootstrap at the cutoff.
- Initial inventories are low and random; the exact distribution is decided, see D6. Pending deliveries start at zero.

**Baseline**
- A simple order-up-to policy: whenever stock plus pending deliveries falls below a target level, order the difference. Target levels are set by grid search for a fair comparison. Details are decided; see D8–D12.

**Evaluation (settled parts)**
- Main metric: cumulative reward (profit) over 365 days.
- Also measured: fill rate (percentage of customer demand met) and average stock levels.
- Both methods are tested with the same customer demands and initial stocks.
- DQN is run with several random seeds and results are averaged. The number is open; see D18.

**Fallback (settled in principle)**
- If DQN training is unstable or too slow, the scope can be reduced, either by reducing the number of actions or by focusing only on distribution between stores. Which reduction and when is open; see D20.
- Demand and cost values may be updated after first trials. Any such change is a team decision.

---

## 3. Open decisions

Format: ID · tag · status · question · suggested default (NOT approved).

### Environment rules

- **D1** · [team] · `DECIDED` · Delivery-time convention.
  DECIDED: an order placed at the end of day t arrives at the start of day t+L: depot t+2, Store A t+1, Store B t+3. Example: a Store A shipment placed Monday at closing arrives Tuesday morning, before customers.
- **D2** · [team] · `DECIDED` · When shipped units leave depot stock.
  DECIDED: shipped units are deducted from depot stock immediately when the decision is made.
- **D3** · [team] · `DECIDED` · When holding cost is charged.
  DECIDED: holding cost is charged after shipments leave the depot. Units in transit are not charged.
- **D4** · [team] · `DECIDED` · Integer rounding in the proportional split.
  DECIDED: when the depot is short, each store's proportional share is rounded down. Leftover units stay in the depot for the next day.
- **D5** · [team] · `DECIDED` · Stock caps.
  DECIDED: no stock caps in the environment dynamics. How observations are scaled or clipped for the network is deferred to the DQN setup.
- **D6** · [team] · `DECIDED` · Initial inventory distribution ("low and random").
  DECIDED: initial inventories are drawn from Poisson distributions with means equal to one day's demand: depot 30, Store A 20, Store B 10. Pending deliveries start at zero.
- **D7** · [team] · `DECIDED` · Supplier reliability.
  DECIDED: the supplier is unlimited and always delivers full orders on time.
- **D26** · [Acar] · `DECIDED (provisional)` · How one environment step maps onto the day (was open question Q1).
  DECIDED (provisional, decided by Acar; team can object): `reset()` returns the decision point of day 0 (random starting stock, nothing in transit, no demand yet). Each step applies the decision of day t, charges day t's holding cost after shipments leave, then simulates day t+1's arrivals and demand. Step reward = 10 × units sold on day t+1 − holding cost of day t. A 365-step year counts holding on days 0–364 and sales on days 1–365.

### Baseline design

Note: the baseline involves no learning. Only its three target numbers (depot, Store A, Store B) are tuned, by the grid search in D12.

- **D8** · [team] · `DECIDED` · What the depot's "stock plus pending deliveries" counts.
  DECIDED: the depot orders up to its target counting only depot stock plus supplier orders on the way. Depot stock is counted after today's shipments to the stores have left. Store stock is not counted, to keep the baseline simple.
- **D9** · [team] · `DECIDED` · Store shipment rule.
  DECIDED: each store is topped up to its own target, counting its shelf stock plus shipments on the way. Example: target 40, 15 on the shelf, nothing on the way → ship 25 (before the D10 rounding, which makes it 24).
- **D10** · [team] · `DECIDED` · Whether baseline quantities are rounded to the DQN action grid or exact.
  DECIDED: baseline quantities are rounded to the nearest allowed step, the same as DQN: depot orders in multiples of 10 (max 40), store shipments in multiples of 4 (max 32). Ties round up (e.g. 6 → 8, 25 → 30), since shipping slightly more is the safer side. Reason for the steps: fixed truck sizes, big trucks from supplier to depot and small trucks from depot to stores.
- **D11** · [team] · `DECIDED` · Whether the baseline uses the same proportional split when the depot is short.
  DECIDED: yes, the baseline uses the same round-down proportional split (D4). The environment applies it automatically.
- **D12** · [team] · `DECIDED` · Grid search ranges for the targets and the number of simulated years per candidate.
  DECIDED: grid search over Store A and Store B targets from 20 to 80 in steps of 4, and the depot target from 30 to 200 in steps of 10. About 20 simulated years per combination, using tuning seeds that are separate from the test seeds. If the best value is on the edge of a range, widen that range. If the full grid is too slow, a coarse pass followed by a finer pass around the best values is allowed. Report the chosen targets.

### DQN setup

- **D13** · [team] · `DECIDED (provisional)` · Stable-Baselines3 DQN or a custom PyTorch implementation, and whether the environment follows the Gymnasium API.
  DECIDED (earlier, by the team): the environment follows the Gymnasium API.
  DECIDED (provisional, decided by Acar; team can object): Stable-Baselines3 DQN with the Gymnasium environment.
- **D14** · [team] · `DECIDED (provisional)` · Action handling.
  DECIDED (provisional, decided by Acar; team can object): flat 405 discrete actions, no masking.
- **D15** · [team] · `DECIDED (provisional)` · Training budget per seed.
  DECIDED (provisional, decided by Acar; team can object): 2M training steps per seed for the first run.
- **D16** · [team] · `DECIDED (provisional)` · Hyperparameters (network size, learning rate, ε schedule, buffer size) and observation scaling.
  DECIDED (provisional, decided by Acar; team can object): SB3 DQN defaults with a 2×256 MLP and γ = 0.99. Observation scaling: divide each feature by a fixed constant from the config, chosen from the observed ranges in the baseline runs. Never change the environment dynamics. Every hyperparameter goes in the config.
  DECIDED (provisional, decided by Acar; team can object; was open question Q20): tuning round 1 is this fixed set of variants, decided in advance, with no other changes. Each variant gets 3 seeds × 2M steps, runs on SLURM only, uses the same environment, and is evaluated on the tuning years.
  - V0: the current configuration, rerun with 3 seeds as a reference.
  - V1: V0 + reward centering. A fixed constant is subtracted from every training reward: the baseline's mean daily reward on the tuning years, taken from the config. The /100 scaling is still applied after centering.
  - V2: V1 + learning rate 5e-5, target_update_interval 20000, batch size 128.
  - V3: V2 + exploration_fraction 0.3.
  For every run, a checkpoint is saved every 100k steps and scored greedily on the tuning years, and the best checkpoint per seed (by mean tuning-year reward) is selected. The best variant is the one with the highest mean best-checkpoint tuning-year reward. Only that variant's 3 selected checkpoints are evaluated once on the 50 test years.
- **D17** · [team] · `DECIDED (provisional)` · Reward scaling during training.
  DECIDED (provisional, decided by Acar; team can object): rewards are divided by 100 during training only. All reported results use true reward.

### Evaluation protocol

- **D18** · [team] · `DECIDED (provisional)` · Number of test years, number of DQN seeds, warm-up exclusion, and seed separation.
  DECIDED (provisional, decided by Acar; team can object): 50 test years (seeds from the existing test range in the config), 5 DQN training seeds, no warm-up exclusion.
  DECIDED (provisional, decided by Acar; team can object; was open question Q21): DQN may use the tuning years (the same seeds the baseline grid search used) to select checkpoints and settings. The test years are used only once, for the final evaluation of the selected configuration, and never for any selection.
  DECIDED (provisional, decided by Acar; team can object; was open question Q26): the selected tuning-round-1 variant (V2) gets 5 seeds, as this item requires. The 2 extra seeds (200003, 200004) are trained with exactly the same commit, config, 2M steps and checkpoint schedule. Each new seed's best checkpoint is picked on the tuning years by the same rule. Nothing else is selected, so this is not a second selection on the test years. Those 2 checkpoints are then evaluated once on the same 50 test years, and the comparison is reported as mean ± std over all 5 seeds.
- **D19** · [team] · `DECIDED (provisional)` · Metric details.
  DECIDED (provisional, decided by Acar; team can object): undiscounted 365-day reward; fill rate per store and overall; average stock per location.
- **D20** · [team] · `DECIDED (provisional)` for the trigger · `OPEN` for which reduction · Which scope reduction to use if needed, and when to trigger it.
  Options from the proposal: fewer actions, or distribution between stores only.
  DECIDED (provisional, decided by Acar; team can object): run one focused tuning round first (the fixed variants in D16). If the best tuned DQN is still clearly below the baseline on the tuning years, the next step is the scope reduction.
  Still OPEN: which reduction. It will be decided after tuning round 1, not now.

### Setup

- **D21** · [Acar] · `DECIDED` · Python version and environment tool.
  DECIDED: a Python virtual environment in `.venv` with `requirements.txt`, using the Python available on this machine. On the VALAR login node the system `python3` is 3.6.8 (too old), so the venv is built from the `anaconda3/2025.06` module's Python 3.13.9 (`/opt/ohpc/pub/compiler/anaconda3/2025.06/bin/python`).
  DECIDED (provisional, decided by Acar; team can object): minimum Python 3.12 (was 3.10; open question Q11), so that every teammate can install the pinned NumPy and gets identical random draws.
- **D22** · [Acar] · `DECIDED (provisional)` · Where training runs.
  DECIDED (provisional, decided by Acar; team can object): all training runs as SLURM jobs on VALAR, never on the login node. Light work (tests, short smoke runs under ~2 minutes, evaluation) may run on the login node.
- **D23** · [Acar] · `OPEN` · Logging.
  Default: TensorBoard + CSV.

### Team workflow

- **D24** · [team] · `OPEN` · Module ownership: who works on env, baseline, DQN.
  Until decided, ask before editing any file outside the task you were given.
- **D25** · [team] · `DECIDED` (starting point) · Repository layout and engineering conventions in Sections 5–6 (folder structure, config file format, required tests).
  DECIDED as a starting point: use the layout and conventions in Sections 5–6 (`configs/`, `src/`, `tests/`, one config file for all numbers, environment tests first, a feature branch per task). Teammates can object.

---

## 4. What you may decide on your own

These do not change results or the comparison. Decide them, and mention what you chose in your summary:
- Code structure inside a module, function and variable names, docstrings, type hints.
- Test design and additional tests.
- Plot styling and small helper utilities.
- Refactors that keep behavior identical (confirm with tests).

When unsure whether something belongs here or in Section 3, assume Section 3 and ask.

---

## 5. Repository layout (proposed — see D25)

```
restock-rl/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/                 # all numbers from Section 2
├── src/
│   ├── env.py               # environment
│   ├── baseline.py          # order-up-to policy + grid search
│   ├── train_dqn.py
│   └── evaluate.py          # same-demand evaluation, metrics
├── tests/
├── results/                 # gitignored
└── notebooks/
```

Do not create new top-level folders without asking.

---

## 6. Engineering rules

**Firm rules (always follow):**
- Never change the settled spec in Section 2 without the user's approval.
- **Git.**
  - Never commit to `main`, push, force-push, or rewrite history without explicit permission.
  - Keep commits small with clear messages.
- **Dependencies.** Ask before adding any package not already in `requirements.txt`.
- **Large files.** Never commit trained models, logs, or `results/`.
- **Reporting.** Results are always reported in true reward units, never scaled values.
- **Reproducibility.** Demand must be reproducible, so both methods can be tested on the same demands (Section 2). Use seeded random number generators. Do not store demand data files unless asked.
- Code and comments in English.

**Proposed conventions (see D25 — ask before relying on them):**
- Keep demand means, delivery times, costs, action grids, γ, and seeds in a config file instead of hardcoding them.
- Build and test the environment before any training code. Tests should cover:
  - one hand-traced day with fixed demand
  - pipeline shifting for depot, A, and B
  - the proportional split
  - reward computation
  - truncation at day 365
- Work on a feature branch per task.

---

## 7. How to end a task

Finish every task with a short summary:
1. What you changed (files).
2. Which tests you ran and their results.
3. Any choices you made under Section 4.
4. Any open decision (by ID) that blocked you or that you need answered next.
