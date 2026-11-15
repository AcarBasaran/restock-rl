# restock-rl
DQN vs. a tuned order-up-to policy for inventory control in a depot + two-store chain with lost sales and different lead times.

## How to run

**Setup** (once). Needs Python 3.12 or newer (D21), so that everyone installs the same pinned NumPy and gets identical random demand. On the VALAR login node the system `python3` is 3.6, so use the Anaconda module's Python (3.13):

```bash
module load anaconda3/2025.06        # VALAR only; elsewhere use Python >= 3.12
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Tests:**

```bash
.venv/bin/python -m pytest
```

**Sanity check** (random and do-nothing policies on the sanity seeds):

```bash
.venv/bin/python -m src.evaluate
```

**Baseline grid search** (D12, on the tuning seeds; results go to `results/grid_search/`, which is not committed):

```bash
.venv/bin/python -m src.baseline time                      # runtime estimate
.venv/bin/python -m src.baseline grid-search --workers 4   # full grid
.venv/bin/python -m src.baseline grid-search --coarse 2    # coarse pass, then fine pass
```

All numbers (demand, delivery times, costs, action grids, seed ranges, grid ranges) are in `configs/default.yaml`.

## Design decisions beyond the proposal

These decisions were made during planning and are not in the proposal. If you disagree, comment on the related GitHub issue.

### Environment

- An order placed at the end of a day arrives at the start of the day one delivery time later: 2 days for the depot, 1 day for Store A, 3 days for Store B. Example: a shipment to Store A placed Monday at closing arrives Tuesday morning, before the first customers. [D1]
- When the depot can't cover both requested shipments, its stock is split in proportion to the requests, each store's share is rounded down, and leftover units stay in the depot for the next day. Example: the depot has 13 units and the agent asks for 16 for Store A and 8 for Store B, so Store A gets 8 (from 8.67), Store B gets 4 (from 4.33), and 1 unit stays in the depot. [D4]
- The end-of-day holding cost is charged after shipments have left the depot, and units on the road are not charged. [D3]
- Each year starts with random stock drawn from a Poisson distribution whose mean is one day's demand (depot 30, Store A 20, Store B 10), with nothing in transit. [D6]
- Units sent to a store are removed from depot stock as soon as the shipping decision is made, not when they arrive. [D2]
- The supplier has unlimited stock and always delivers full orders on time. [D7]
- Stock can grow without a cap, and how stock values are scaled or clipped for the neural network will be decided with the DQN setup. [D5]
- The simulator follows the standard Gymnasium environment API. [D13]

### Baseline

- The depot orders up to its target counting only its own stock (after today's shipments to the stores have left) plus supplier orders on the way, not store stock, to keep the baseline simple. [D8]
- Each store is topped up to its own target, counting its shelf stock plus shipments on the way. Example: target 40, 15 on the shelf and nothing on the way means a shipment of 25, which the truck-size rounding below turns into 24. [D9]
- Like DQN's actions, baseline quantities are rounded to the nearest truck size: depot orders in multiples of 10 (max 40, big trucks from the supplier) and store shipments in multiples of 4 (max 32, small trucks from the depot), with ties rounded up (6 → 8, 25 → 30) because shipping slightly more is the safer side. [D10]
- The three targets are tuned by grid search (store targets 20–80 in steps of 4, depot target 30–200 in steps of 10, about 20 simulated years per combination on tuning seeds separate from the test seeds), widening any range whose best value lands on its edge, allowing a coarse pass then a finer pass around the best values if the full grid is too slow, and reporting the chosen targets. [D12]
- When the depot is short, the baseline gets the same round-down proportional split as DQN, applied automatically by the environment. [D11]

### DQN and evaluation

These were decided by Acar as provisional choices; the team can object.

- The learning method is Stable-Baselines3's DQN, trained directly on the Gymnasium environment. [D13]
- Both methods are tested on the same 50 test years (seeds from the test range in the config), DQN with 5 training seeds, and every day of the year counts (no warm-up exclusion). [D18]
- One environment step starts at the end-of-day decision and ends at the next decision, so a step's reward is the next day's sales minus today's holding cost, and the year starts at the day-0 decision before any demand. [D26]
- Results are reported as undiscounted 365-day profit, fill rate per store and overall, and average stock per location. [D19]
- DQN picks one of the 405 order/shipment combinations directly, with no actions blocked. [D14]
- Each training seed gets 2 million steps (about 5,500 simulated years) in the first run. [D15]
- DQN uses Stable-Baselines3's default settings with a two-layer network of 256 units each and a discount factor of 0.99, and each observation value is divided by a fixed constant from the config (chosen from the stock levels seen in baseline runs) without changing the simulation itself. [D16]
- Rewards are divided by 100 during training only, and every reported number is in true reward units. [D17]
- All training runs as batch jobs on the VALAR cluster, never on the login node, which is only used for tests, short smoke runs and evaluation. [D22]

### Setup

- The repository follows the layout and conventions proposed in CLAUDE.md Sections 5–6 (`configs/`, `src/`, `tests/`, one config file for all numbers, environment tests first, one feature branch per task) as a starting point that teammates can still object to. [D25]
- Code runs in a Python virtual environment in `.venv` with packages from `requirements.txt`, built from Python 3.12 or newer so every teammate can install the pinned NumPy and gets identical random draws (on the VALAR login node: the `anaconda3/2025.06` module's Python 3.13.9, since the system `python3` is 3.6). [D21]
