# CLAUDE.md — restock-rl

Project title: **Deep Reinforcement Learning for Inventory Management in a Two-Store Chain**
Team of three students. This repo compares a DQN agent against a tuned order-up-to policy on a simulated depot + two-store inventory system.

---

## 0. The most important rule

This is a team project. Some design choices are **not settled** and must be agreed by the whole team. Making them on your own creates conflicts later.

- Section 2 is the **settled specification**. Implement it exactly. Do not "improve" it, simplify it, or add costs, constraints, or features to it.
- Section 3 lists **open decisions**. For any item marked `OPEN`:
  - **Stop and ask the user before writing any code that depends on it.** Name the decision by its ID (e.g. "D4 is still open").
  - Do **not** silently implement the suggested default. The default is only there to make the discussion faster.
  - Do **not** write code that hides the choice behind a guess. A config option is fine only after the user confirms it should be a config option.
  - When the user answers, they will update this file to `DECIDED: <answer>`. Treat only what is written here as decided, not something said earlier in a session.
- Items tagged **[team]** need a team discussion; the user will come back with an answer. Items tagged **[Acar]** the user can answer directly.
- If you notice a new ambiguity that is not listed here and that changes results, behavior, or the comparison, treat it as `OPEN`: stop, describe it, and propose adding it to Section 3.
- If something in the settled spec looks wrong to you, say so and explain why, but do not change it without the user's approval.

---

## 1. Project goal

Research question: in a two-store system where the stores differ in demand level and delivery time, can DQN achieve higher profit than a simple, properly tuned order-up-to policy?

No external dataset is used. The environment simulator generates all experience; Poisson demand is drawn from seeded random number generators.

---

## 2. Settled specification (implement exactly)

**Network**
- One supplier → one depot → two stores (A and B).
- Supplier → depot lead time: 2 days.
- Depot → Store A lead time: 1 day. Store A demand: Poisson, mean 20 per day.
- Depot → Store B lead time: 3 days. Store B demand: Poisson, mean 10 per day.
- Lost sales: unmet demand is lost and not carried over. Lost demand is recorded for analysis but is **not** part of the observation.

**Daily sequence**
1. Previously placed orders and shipments that are due arrive.
2. Customer demand at each store is realized and met from store stock.
3. At the end of the day, the agent decides the depot order and the shipments to each store.

**Observation / state (6 values)**
1. Depot stock
2. Store A stock
3. Store B stock
4. Depot order arriving tomorrow
5. Store B shipment arriving in 1 day
6. Store B shipment arriving in 2 days

There is no pending-shipment variable for Store A. Demand is i.i.d., so this state is Markov.

**Actions (405 discrete)**
- Depot order from the supplier: {0, 10, 20, 30, 40}.
- Shipment to Store A: {0, 4, 8, …, 32}.
- Shipment to Store B: {0, 4, 8, …, 32}.
- If depot stock is insufficient for both requested shipments, available stock is split **proportionally** between the requested amounts. The rounding rule is open; see D4.

**Reward**
- +10 per unit sold.
- Holding cost: 2 per unit left in stock at each store, 1 per unit left at the depot, charged at the end of the day. The exact timing is open; see D3.
- **No other costs.** No purchase cost, no fixed ordering or shipping cost, no lost-sales penalty. Do not add any.

**Objective and task type**
- Maximize discounted return with γ = 0.99.
- Continuing task. Training and evaluation run in 365-day periods, but day 365 is **not** terminal. In Gymnasium, end-of-period must be `truncated=True`, never `terminated=True`, so DQN bootstraps at the cutoff.
- Initial inventories are low and random. The range is open; see D6. Pending deliveries start at zero.

**Baseline**
- An order-up-to policy: when inventory position (stock + pending deliveries) is below the target, order the difference. Targets are tuned by grid search. Details are open; see D8–D12.

**Evaluation (settled parts)**
- Both methods face identical demand sequences and initial stocks (common random numbers via fixed seeds).
- Train, tune, and test seeds are disjoint.
- Metrics: total profit over 365 days, fill rate, and average stock levels.
- DQN is run with several random seeds and results are averaged.

---

## 3. Open decisions

Format: ID · tag · status · question · suggested default (NOT approved).

### Environment dynamics

- **D1** · [team] · `OPEN` · Lead-time convention.
  Default: an order placed at the end of day t arrives at the start of day t+L (depot t+2, A t+1, B t+3), consistent with the state definition.
- **D2** · [team] · `OPEN` · When shipped units leave depot stock.
  Default: they are deducted immediately at decision time.
- **D3** · [team] · `OPEN` · When holding cost is charged: after sales but before shipping, or after shipments leave the depot? In-transit units are not charged in either case.
  Default: after shipments leave.
- **D4** · [team] · `OPEN` · Integer rounding in the proportional split.
  Default: floor each store's share; leftover units stay at the depot.
- **D5** · [team] · `OPEN` · Stock caps: none, observation clipping only, or hard caps on inventory (e.g. 100 depot / 50 per store)?
  Default: no caps in the dynamics; clip the observation for the network only.
- **D6** · [team] · `OPEN` · Initial inventory distribution.
  Default: depot uniform on 0–20, each store uniform on 0–10.
- **D7** · [team] · `OPEN` · Supplier reliability.
  Default: unlimited supply, full orders always delivered on time.

### Baseline

- **D8** · [team] · `OPEN` · What the depot's inventory position counts: depot stock + depot pipeline only, or echelon (also store stock and depot→store pipeline)?
  Default: depot stock + depot pipeline.
- **D9** · [team] · `OPEN` · Store shipment rule.
  Default: ship each store up to its own target, counting store stock + shipments already on the way.
- **D10** · [team] · `OPEN` · Whether baseline quantities are rounded to the DQN action grid or exact.
  Default: rounded to the same grid.
- **D11** · [team] · `OPEN` · Whether the baseline uses the same proportional split when the depot is short.
  Default: yes.
- **D12** · [team] · `OPEN` · Grid search ranges for the three targets and the number of simulated years per candidate.
  Default: ~20 years per candidate on tuning seeds.

### DQN

- **D13** · [team] · `OPEN` · Stable-Baselines3 DQN or a custom PyTorch implementation.
  Default: SB3 first.
- **D14** · [team] · `OPEN` · Action handling: flat 405 actions, action masking, or a factored head.
  Default: flat, no masking.
- **D15** · [team] · `OPEN` · Training budget per seed.
  Default: 1–2M steps for the first run.
- **D16** · [team] · `OPEN` · Hyperparameters (network size, learning rate, ε schedule, buffer size).
  Default: SB3 defaults with a 2×256 MLP; every change is recorded in the config.
- **D17** · [team] · `OPEN` · Reward scaling during training.
  Default: divide by 100 for training only; always report true profit.

### Evaluation

- **D18** · [team] · `OPEN` · Number of test years, number of DQN seeds, warm-up exclusion.
  Default: 50 test years, 5 seeds, first 30 days excluded from metrics.
- **D19** · [team] · `OPEN` · Metric details.
  Default: undiscounted 365-day profit; fill rate per store and overall; average stock per location.
- **D20** · [team] · `OPEN` · When to fall back to the reduced scope (depot with unlimited stock, 81 actions).
  Default: if DQN cannot match the baseline after 2 tuning rounds.

### Setup and workflow

- **D21** · [Acar] · `OPEN` · Python version and environment tool.
  Default: Python 3.11, venv, `requirements.txt`.
- **D22** · [Acar] · `OPEN` · Where training runs: laptops, Colab, or the VALAR cluster. If VALAR, SLURM scripts are needed.
  Default: develop locally, run full experiments on VALAR.
- **D23** · [Acar] · `OPEN` · Logging.
  Default: TensorBoard + CSV.
- **D24** · [team] · `OPEN` · Module ownership: who owns env, baseline, DQN.
  Until decided, ask before editing any file outside the task you were given.

---

## 4. What you may decide on your own

These do not change results or the comparison. Decide them, and mention what you chose in your summary:
- Code structure inside a module, function and variable names, docstrings, type hints.
- Test design and additional tests.
- Plot styling and helper utilities.
- Refactors that keep behavior identical (confirm with tests).

When unsure whether something belongs here or in Section 3, assume Section 3 and ask.

---

## 5. Repository layout (target)

```
restock-rl/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/default.yaml     # every number from Section 2 lives here
├── src/
│   ├── env.py               # Gymnasium environment
│   ├── baseline.py          # order-up-to policy + grid search
│   ├── train_dqn.py
│   └── evaluate.py          # common-random-number evaluation, metrics
├── tests/test_env.py
├── results/                 # gitignored
└── notebooks/
```

Do not create new top-level folders without asking.

---

## 6. Engineering rules

- **No hardcoded numbers.** Demand means, lead times, costs, action grids, γ, and seeds all come from `configs/default.yaml`.
- **Environment first, with tests.** Before any training code, `tests/test_env.py` must cover:
  - one hand-traced day with fixed demand
  - the pipeline shifting correctly for depot, A, and B
  - the proportional split
  - reward computation
  - `truncated` vs `terminated` at day 365
- **Seeding.** The environment takes a seed. Evaluation demand must be reproducible from seeds alone. Do not store demand files.
- **Never report scaled rewards as results.** Results are always in true profit units.
- **Git.**
  - Work on a feature branch.
  - Never commit to `main`, never push, never force-push, and never rewrite history without explicit permission.
  - Keep commits small with clear messages.
- **Dependencies.** Ask before adding any package not already in `requirements.txt`.
- **Large files.** Never commit trained models, logs, or `results/`.
- Code and comments in English.

---

## 7. How to end a task

Finish every task with a short summary:
1. What you changed (files).
2. Which tests you ran and their results.
3. Any choices you made under Section 4.
4. Any open decision (by ID) that blocked you or that you need answered next.
