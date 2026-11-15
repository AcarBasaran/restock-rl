# Sanity check: random and do-nothing policies

Run: `python -m src.evaluate` (sanity seeds 0–4, one 365-day year each, no warm-up).
All values are in true reward units.

| Policy | Years | Mean reward | Std | Fill A | Fill B | Fill all | Avg depot | Avg A | Avg B | Lost A / yr | Lost B / yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 5 | -2,380 | 34,721 | 47.2% | 96.2% | 63.6% | 8.5 | 1.4 | 93.5 | 3,870 | 141 |
| do_nothing | 5 | -10,702 | 2,682 | 0.3% | 0.3% | 0.3% | 30.0 | 0.1 | 0.0 | 7,309 | 3,670 |

Random policy, per year:

| Seed | Reward | Revenue | Holding cost | Avg B stock | Fill A | Fill B |
|---|---|---|---|---|---|---|
| 0 | 8,028 | 70,780 | 62,752 | 80.4 | 46.9% | 97.4% |
| 1 | -40,676 | 70,220 | 110,896 | 147.0 | 48.0% | 97.8% |
| 2 | 15,980 | 67,180 | 51,200 | 64.2 | 44.9% | 94.2% |
| 3 | 40,232 | 69,570 | 29,338 | 35.2 | 47.0% | 93.9% |
| 4 | -35,463 | 72,290 | 107,753 | 141.0 | 49.3% | 97.6% |

## Do the numbers look reasonable?

Yes. Both match what a hand calculation predicts. **Do-nothing** never orders or ships, so after the first day or two the stores are empty, almost all demand is lost (fill rate 0.3%), and the depot's starting stock (~30 units) sits there all year at a holding cost of 1 per unit per day. Expected result: about −30 × 365 ≈ −10,950 plus a few hundred from first-day store sales. The measured −10,702 fits. **Random** orders 20 per day on average but requests 16 + 16 = 32 of shipments, so the depot is nearly always empty (average 8.5) and the proportional split sends each store roughly 10 per day. Store A (demand 20) is therefore starved (fill 47%), while Store B (demand 10) gets about its demand, so its stock wanders like a random walk. That is why B's average stock is large and varies a lot between years (35–147). Revenue is stable (~70k), but B's holding cost swings from 29k to 111k, which explains the near-zero mean and the large standard deviation. The upper bound on revenue is 30 units/day × 10 × 365 = 109,500 per year, so a good policy should end up well above both of these.
