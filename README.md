# restock-rl
DQN vs. a tuned order-up-to policy for inventory control in a depot + two-store chain with lost sales and different lead times.

## Design decisions beyond the proposal

The project proposal left some details of the simulator open. The team has agreed on the following:

- **Delivery times.** An order placed at the end of a day arrives at the start of the day that is one delivery time later: 2 days for the depot, 1 day for Store A, 3 days for Store B. Example: a shipment to Store A placed Monday at closing arrives Tuesday morning, before the first customers.
- **Shipped units leave the depot right away.** Units sent to a store are removed from depot stock as soon as the shipping decision is made, not when they arrive.
- **Holding cost timing.** The end-of-day holding cost is charged after shipments have left the depot. Units on the road are not charged.
- **Rounding when the depot is short.** If the depot can't cover both requested shipments, its stock is split in proportion to the requests and each store's share is rounded down. Leftover units stay in the depot for the next day. Example: the depot has 13 units and the agent asks for 16 for Store A and 8 for Store B. Store A gets 8 (from 8.67), Store B gets 4 (from 4.33), and 1 unit stays in the depot.
- **No stock limits.** Stock at the depot and stores can grow without a cap. How stock values are scaled or clipped before they go into the neural network will be decided with the rest of the DQN setup.
- **Starting stock.** Each year starts with random stock drawn from a Poisson distribution whose mean is one day's demand: depot 30, Store A 20, Store B 10. Nothing is in transit at the start.
- **Reliable supplier.** The supplier has unlimited stock and always delivers full orders on time.
- **Gymnasium interface.** The simulator follows the standard Gymnasium environment API. Whether the DQN uses Stable-Baselines3 or our own PyTorch code is still open.
