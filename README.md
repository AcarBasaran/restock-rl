# restock-rl
DQN vs. a tuned order-up-to policy for inventory control in a depot + two-store chain with lost sales and different lead times.

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
- The simulator follows the standard Gymnasium environment API, while the choice between Stable-Baselines3 and our own PyTorch DQN is still open. [D13]

### Baseline

- The depot orders up to its target counting only its own stock (after today's shipments to the stores have left) plus supplier orders on the way, not store stock, to keep the baseline simple. [D8]
- Each store is topped up to its own target, counting its shelf stock plus shipments on the way. Example: target 40, 15 on the shelf and nothing on the way means a shipment of 25, which the truck-size rounding below turns into 24. [D9]
- Like DQN's actions, baseline quantities are rounded to the nearest truck size: depot orders in multiples of 10 (max 40, big trucks from the supplier) and store shipments in multiples of 4 (max 32, small trucks from the depot), with ties rounded up (6 → 8, 25 → 30) because shipping slightly more is the safer side. [D10]
- The three targets are tuned by grid search (store targets 20–80 in steps of 4, depot target 30–200 in steps of 10, about 20 simulated years per combination on tuning seeds separate from the test seeds), widening any range whose best value lands on its edge, allowing a coarse pass then a finer pass around the best values if the full grid is too slow, and reporting the chosen targets. [D12]
- When the depot is short, the baseline gets the same round-down proportional split as DQN, applied automatically by the environment. [D11]
