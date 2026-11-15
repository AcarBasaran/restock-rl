"""Depot + two-store inventory environment (Gymnasium API).

Implements CLAUDE.md Section 2 and decisions D1–D7. All numbers come from
``configs/default.yaml``.

Timing of one environment step
------------------------------
The observation is the state at the end-of-day decision point of day ``t``
(after that day's customer demand has been met). ``step(action)`` then:

1. Applies the decision of day ``t``: the requested shipments are split
   proportionally (rounded down) if the depot is short (D4), deducted from
   depot stock immediately (D2), and put on the road together with the new
   depot order.
2. Charges the end-of-day holding cost of day ``t`` on what is left at the
   depot and the stores after shipments have left (D3). Units on the road
   are free.
3. Moves to day ``t+1``: arrivals (an order placed at the end of day ``t``
   arrives at the start of day ``t+L``, D1), then customer demand with lost
   sales.

Step reward = 10 x units sold on day ``t+1`` - holding cost of day ``t``.
``reset()`` returns the decision point of "day 0": the random starting
stock (D6) with nothing in transit, before any demand. One simulated year
is ``episode_length`` (365) steps, after which ``truncated=True``; the
episode is never ``terminated`` (continuing task).

This step/day boundary is decision D26 (provisional; was open question Q1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (defaults to ``configs/default.yaml``)."""
    with open(path or DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def split_shipments(depot_stock: int, request_a: int, request_b: int) -> tuple[int, int]:
    """Return the units actually shipped to Store A and Store B.

    If the depot has enough stock, both requests are shipped in full.
    Otherwise depot stock is divided in proportion to the requests and each
    share is rounded down; leftover units stay at the depot (D4).
    Example: depot 13, requests 16 and 8 -> 8 and 4, 1 unit stays.
    """
    total = request_a + request_b
    if total <= depot_stock:
        return request_a, request_b
    return depot_stock * request_a // total, depot_stock * request_b // total


class RestockEnv(gym.Env):
    """Supplier -> depot -> Store A / Store B with lost sales.

    Observation (with the default lead times, exactly the 6 values of
    Section 2, in this order):
        0 depot stock
        1 Store A stock
        2 Store B stock
        3 depot order arriving the next day
        4 Store B shipment arriving in 1 day
        5 Store B shipment arriving in 2 days
    In general: the three stocks, then the depot, Store A and Store B
    pipelines (lead time - 1 values each, soonest arrival first). Values are
    raw unit counts with no clipping or scaling (D5; network-side scaling is
    part of D16).

    Action: one integer in ``[0, 405)``; see ``decode_action``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        cfg = config if config is not None else load_config()
        self.config = cfg

        self.episode_length = int(cfg["episode_length"])
        self.price = int(cfg["price"])
        self.mean_a = float(cfg["demand_mean"]["store_a"])
        self.mean_b = float(cfg["demand_mean"]["store_b"])
        self.init_mean = (
            float(cfg["initial_stock_mean"]["depot"]),
            float(cfg["initial_stock_mean"]["store_a"]),
            float(cfg["initial_stock_mean"]["store_b"]),
        )
        self.h_depot = int(cfg["holding_cost"]["depot"])
        self.h_a = int(cfg["holding_cost"]["store_a"])
        self.h_b = int(cfg["holding_cost"]["store_b"])
        self.lead_depot = int(cfg["lead_time"]["depot"])
        self.lead_a = int(cfg["lead_time"]["store_a"])
        self.lead_b = int(cfg["lead_time"]["store_b"])
        for name, lead in (("depot", self.lead_depot), ("store_a", self.lead_a), ("store_b", self.lead_b)):
            if lead < 1:
                raise ValueError(f"lead_time.{name} must be at least 1 day, got {lead}")

        self.depot_grid = [int(x) for x in cfg["actions"]["depot_order"]]
        self.ship_a_grid = [int(x) for x in cfg["actions"]["ship_store_a"]]
        self.ship_b_grid = [int(x) for x in cfg["actions"]["ship_store_b"]]
        self.n_actions = len(self.depot_grid) * len(self.ship_a_grid) * len(self.ship_b_grid)

        self.obs_size = 3 + (self.lead_depot - 1) + (self.lead_a - 1) + (self.lead_b - 1)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

        # State (plain Python ints for speed). Pipelines hold lead-1 entries at
        # the decision point; index 0 arrives at the start of the next day.
        self._depot = 0
        self._a = 0
        self._b = 0
        self._pipe_depot: list[int] = []
        self._pipe_a: list[int] = []
        self._pipe_b: list[int] = []
        self._demand: list[list[int]] = []
        self._t = 0
        self._needs_reset = True

    # ------------------------------------------------------------------ actions
    def decode_action(self, action: int) -> tuple[int, int, int]:
        """Flat index -> (depot order, requested shipment A, requested shipment B).

        index = (i_depot * n_a + i_a) * n_b + i_b
        """
        action = int(action)
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside [0, {self.n_actions})")
        n_a, n_b = len(self.ship_a_grid), len(self.ship_b_grid)
        i_depot, rest = divmod(action, n_a * n_b)
        i_a, i_b = divmod(rest, n_b)
        return self.depot_grid[i_depot], self.ship_a_grid[i_a], self.ship_b_grid[i_b]

    def encode_action(self, depot_order: int, ship_a: int, ship_b: int) -> int:
        """(depot order, shipment A, shipment B) -> flat index. Values must be on the grid."""
        i_depot = self.depot_grid.index(int(depot_order))
        i_a = self.ship_a_grid.index(int(ship_a))
        i_b = self.ship_b_grid.index(int(ship_b))
        return (i_depot * len(self.ship_a_grid) + i_a) * len(self.ship_b_grid) + i_b

    # ------------------------------------------------------------- observation
    def _obs(self) -> np.ndarray:
        return np.array(
            [self._depot, self._a, self._b, *self._pipe_depot, *self._pipe_a, *self._pipe_b],
            dtype=np.float32,
        )

    def parse_observation(self, obs) -> dict[str, Any]:
        """Split an observation into named integer parts."""
        values = [int(round(float(x))) for x in obs]
        i = 3
        pipe_depot = values[i : i + self.lead_depot - 1]
        i += self.lead_depot - 1
        pipe_a = values[i : i + self.lead_a - 1]
        i += self.lead_a - 1
        pipe_b = values[i : i + self.lead_b - 1]
        return {
            "depot": values[0],
            "store_a": values[1],
            "store_b": values[2],
            "pipe_depot": pipe_depot,
            "pipe_a": pipe_a,
            "pipe_b": pipe_b,
        }

    # ------------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Start a new year.

        Random draws, in a fixed order that does not depend on the policy:
        starting stock (depot, A, B; Poisson, D6), then the whole year's
        demand (Poisson). The same seed therefore always gives the same
        starting stock and the same demand, whatever the policy does.

        ``options`` (mainly for tests) may override:
            "initial_stock": (depot, store_a, store_b)
            "demand": array-like of shape (episode_length, 2), columns A, B
        """
        super().reset(seed=seed)
        rng = self.np_random
        init = rng.poisson(self.init_mean)
        demand = rng.poisson((self.mean_a, self.mean_b), size=(self.episode_length, 2))

        options = options or {}
        if "initial_stock" in options:
            init = options["initial_stock"]
        if "demand" in options:
            demand = np.asarray(options["demand"], dtype=np.int64)
            if demand.shape != (self.episode_length, 2):
                raise ValueError(f"demand must have shape ({self.episode_length}, 2), got {demand.shape}")

        self._depot, self._a, self._b = (int(x) for x in init)
        self._pipe_depot = [0] * (self.lead_depot - 1)
        self._pipe_a = [0] * (self.lead_a - 1)
        self._pipe_b = [0] * (self.lead_b - 1)
        self._demand = np.asarray(demand).astype(int).tolist()
        self._t = 0
        self._needs_reset = False
        return self._obs(), {"day": 0}

    # -------------------------------------------------------------------- step
    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before step(), and again after truncation")

        order, request_a, request_b = self.decode_action(action)

        # 1. Decision of day t: proportional split if short (D4), deduct now (D2).
        ship_a, ship_b = split_shipments(self._depot, request_a, request_b)
        self._depot -= ship_a + ship_b
        self._pipe_depot.append(order)
        self._pipe_a.append(ship_a)
        self._pipe_b.append(ship_b)

        # 2. End-of-day holding cost of day t, after shipments left (D3).
        stock_depot, stock_a, stock_b = self._depot, self._a, self._b
        holding_cost = self.h_depot * stock_depot + self.h_a * stock_a + self.h_b * stock_b

        # 3. Day t+1: arrivals (D1), then demand with lost sales.
        self._t += 1
        self._depot += self._pipe_depot.pop(0)
        self._a += self._pipe_a.pop(0)
        self._b += self._pipe_b.pop(0)

        demand_a, demand_b = self._demand[self._t - 1]
        sales_a = min(self._a, demand_a)
        sales_b = min(self._b, demand_b)
        self._a -= sales_a
        self._b -= sales_b

        revenue = self.price * (sales_a + sales_b)
        reward = revenue - holding_cost

        truncated = self._t >= self.episode_length
        if truncated:
            self._needs_reset = True

        info = {
            "day": self._t,
            "depot_order": order,
            "requested_a": request_a,
            "requested_b": request_b,
            "shipped_a": ship_a,
            "shipped_b": ship_b,
            # stock charged for holding on the decision day (after shipments left)
            "held_depot": stock_depot,
            "held_a": stock_a,
            "held_b": stock_b,
            "holding_cost": holding_cost,
            "demand_a": demand_a,
            "demand_b": demand_b,
            "sales_a": sales_a,
            "sales_b": sales_b,
            # recorded for analysis only; never part of the observation
            "lost_a": demand_a - sales_a,
            "lost_b": demand_b - sales_b,
            "revenue": revenue,
        }
        return self._obs(), float(reward), False, truncated, info
