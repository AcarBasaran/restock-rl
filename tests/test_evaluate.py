"""Tests for src/evaluate.py."""

import math

import numpy as np
import pytest

from src.env import RestockEnv, load_config
from src.evaluate import RandomPolicy, do_nothing_policy, evaluate, run_episode, seed_range


class FixedYearEnv(RestockEnv):
    """Environment whose reset ignores randomness and uses fixed options."""

    def __init__(self, initial_stock, demand):
        super().__init__(load_config())
        self._fixed = {"initial_stock": initial_stock, "demand": demand}

    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=self._fixed)


def test_do_nothing_hand_computed():
    # Start: depot 30, A 25, B 5. Demand every day: A 20, B 10. Never act.
    # Day 1: sell A 20 (A -> 5), sell B 5 (lose 5). Day 2: sell A 5, lose 15; lose 10 at B.
    # Afterwards nothing is sold and the depot keeps 30 units forever.
    # Holding: day 0: 30 + 2*25 + 2*5 = 90; day 1: 30 + 2*5 = 40; days 2..364: 30 each.
    # Total = 10*(20 + 5 + 5) - (90 + 40 + 363*30) = 300 - 11020 = -10720
    demand = np.tile([20, 10], (365, 1))
    env = FixedYearEnv((30, 25, 5), demand)
    r = run_episode(env, do_nothing_policy, seed=0)
    assert r["total_reward"] == -10720
    assert (r["sales_a"], r["sales_b"]) == (25, 5)
    assert (r["lost_a"], r["lost_b"]) == (365 * 20 - 25, 365 * 10 - 5)
    assert r["fill_rate_a"] == pytest.approx(25 / 7300)
    assert r["fill_rate"] == pytest.approx(30 / 10950)
    assert r["avg_stock_depot"] == 30
    # Store A held: 25 (day 0), 5 (day 1), 0 after -> 30 / 365
    assert r["avg_stock_a"] == pytest.approx(30 / 365)


def test_warmup_excludes_first_days():
    demand = np.tile([20, 10], (365, 1))
    env = FixedYearEnv((30, 25, 5), demand)
    r = run_episode(env, do_nothing_policy, seed=0, warmup_days=2)
    # Only days 3..365 count: no sales, holding 30 per day for 363 steps.
    assert r["days_counted"] == 363
    assert r["total_reward"] == -30 * 363
    assert r["sales_a"] == 0 and r["fill_rate_a"] == 0


def test_same_seeds_same_results():
    config = load_config()
    seeds = [1, 2, 3]
    r1 = evaluate(RandomPolicy(config), seeds, config)
    r2 = evaluate(RandomPolicy(config), seeds, config)
    assert [e["total_reward"] for e in r1["episodes"]] == [e["total_reward"] for e in r2["episodes"]]


def test_discounted_reward_uses_gamma():
    demand = np.zeros((365, 2), dtype=int)
    env = FixedYearEnv((10, 0, 0), demand)
    r = run_episode(env, do_nothing_policy, seed=0)
    # Holding 10 per day: discounted sum = -10 * (1 - 0.99**365) / (1 - 0.99)
    assert r["discounted_reward"] == pytest.approx(-10 * (1 - 0.99**365) / 0.01)
    assert r["total_reward"] == -3650


def test_seed_ranges_do_not_overlap():
    config = load_config()
    sanity = set(seed_range(config, "sanity"))
    tuning = set(seed_range(config, "tuning"))
    assert not sanity & tuning
    test = set(seed_range(config, "test"))
    assert len(test) == 50  # D18
    assert not (sanity | tuning) & test


def test_summary_fill_rates_pooled():
    config = load_config()
    result = evaluate(do_nothing_policy, [5, 6], config)
    eps = result["episodes"]
    s = result["summary"]
    expected = (eps[0]["sales_a"] + eps[1]["sales_a"]) / (eps[0]["demand_a"] + eps[1]["demand_a"])
    assert math.isclose(s["fill_rate_a"], expected)
