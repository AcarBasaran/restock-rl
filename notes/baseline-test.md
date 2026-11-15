# Baseline final test

- **Policy:** order-up-to with the tuned targets **depot 60, Store A 24, Store B 32** (from the D12 grid search on the tuning years; stored under `baseline:` in `configs/default.yaml`).
- **Test years:** 50 years, seeds **900000–900049** (D18), never used for tuning. No warm-up exclusion (D18). True reward units.
- **Command:** `python -m src.baseline evaluate --seeds test`
- **Raw results:** `results/baseline/20261115-054209_test_years.csv` (one row per year) and `20261115-054209_test_summary.json`; not committed.

| Metric (50 test years) | Value |
|---|---|
| Mean 365-day reward | **96,614** (std 1,156 across years; standard error of the mean ≈ 164) |
| Mean discounted reward (γ = 0.99) | 25,639 |
| Fill rate: Store A / Store B / overall | 95.6% / 92.7% / 94.6% |
| Average stock: depot / Store A / Store B | 4.2 / 4.0 / 3.8 units |
| Lost sales per year: Store A / Store B | 323 / 267 units |

On the 20 tuning years the same targets scored 96,525 with a 94.7% fill rate, so the test result matches the tuning result closely and shows no sign that the targets were fitted to the tuning years. Fill rates are pooled over all years and average stock is measured at the end of the day after shipments leave; both are still open details (Q2, Q3).
