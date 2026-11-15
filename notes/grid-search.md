# Baseline grid search (D12)

- **Command:** `python -m src.baseline grid-search --workers 4 --nice 10`
- **Grid:** depot target 30–200 step 10 (18 values) × Store A target 20–80 step 4 (16) × Store B target 20–80 step 4 (16) = **4,608 candidates**, full grid.
- **Years per candidate:** 20 separate 365-day years on **tuning seeds 100000–100019**, the same 20 for every candidate. Test seeds are reserved from 900000 (count still open, D18), so the two never overlap. Both ranges are in `configs/default.yaml`.
- **Runtime:** a timing test estimated 0.067 s per candidate on one core (≈ 5 min for the full grid). The actual run took **2.8 min** with 4 workers, so no coarse/fine pass was needed (D12 allows one only if the full grid is too slow).
- **Selection rule (provisional, Q6):** highest mean undiscounted 365-day reward on the tuning years.
- **Raw results:** `results/grid_search/20261115-052007_full.csv` (every candidate, all metrics) and `20261115-052007_best.json`; not committed.

## Chosen targets

| Depot target | Store A target | Store B target |
|---|---|---|
| **60** | **24** | **32** |

| Metric (20 tuning years) | Value |
|---|---|
| Mean 365-day reward | **96,525** (std 921 across years) |
| Mean discounted reward (γ = 0.99) | 25,607 |
| Fill rate: Store A / Store B / overall | 95.7% / 92.7% / 94.7% |
| Average stock: depot / A / B | 4.2 / 4.0 / 3.7 units |
| Lost sales per year: A / B | 310 / 269 units |

Average stock is measured at the end of the day after shipments left (the stock that pays holding cost; Q2), which is why it is low: stores are nearly empty at closing time and are refilled overnight.

## Is any best value on the edge of its range?

**No.** All three are inside their ranges, so D12's "widen the range" rule did not apply. Store A's best value (24) is one step above the lower bound (20), but 20 is clearly worse (93.8k vs 96.5k), so the peak is inside the range.

One-at-a-time sweep around the best (other two targets held at the best values), mean reward in thousands:

- Depot: 30: 53.2 · 40: 70.3 · 50: 88.3 · **60: 96.5** · 70: 94.1 · 80: 90.5 · … · 200: 47.8
- Store A: 20: 93.8 · **24: 96.5** · 28: 96.3 · 32: 94.9 · 36: 93.0 · … · 80: 64.6
- Store B: 20: 87.8 · 24: 92.3 · 28: 95.3 · **32: 96.5** · 36: 96.0 · 40: 94.2 · … · 80: 65.9

## Top candidates

| Depot | A | B | Mean reward | Std | Discounted | Fill all |
|---|---|---|---|---|---|---|
| 60 | 24 | 32 | 96,525 | 921 | 25,607 | 94.7% |
| 60 | 28 | 36 | 96,305 | 1,172 | 25,568 | 96.7% |
| 60 | 28 | 32 | 96,283 | 931 | 25,559 | 95.8% |
| 60 | 24 | 36 | 96,003 | 1,036 | 25,453 | 95.6% |
| 60 | 32 | 36 | 95,502 | 1,220 | 25,348 | 97.2% |

- The undiscounted and discounted rankings pick the **same top three in the same order**, so Q6 does not change the result.
- The top three are within about 240 of each other (≈ 2 standard errors of a 20-year mean). All candidates share the same 20 years, which makes the comparison sharper than that suggests, but if the team wants more certainty, re-running only the top few on more tuning years would settle it.
- The targets look sensible: Store A (lead time 1 day, demand 20) is refilled to about one day's demand plus a safety margin; Store B (lead 3 days, demand 10) to about three days' demand; the depot (lead 2 days, ~30 per day outflow) to about two days of outflow.
