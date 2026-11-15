"""Tests for src/baseline.py against D8–D11 (expected values traced by hand)."""

import numpy as np
import pytest

from src.baseline import OrderUpToPolicy, candidate_grid, edge_report, fine_candidates, round_to_grid
from src.env import RestockEnv, load_config

DEPOT_GRID = [0, 10, 20, 30, 40]
STORE_GRID = [0, 4, 8, 12, 16, 20, 24, 28, 32]


def obs(depot, a, b, depot_next=0, b_in_1=0, b_in_2=0):
    return np.array([depot, a, b, depot_next, b_in_1, b_in_2], dtype=np.float32)


# -------------------------------------------------------------------- rounding
@pytest.mark.parametrize(
    "x, grid, expected",
    [
        (6, STORE_GRID, 8),     # tie between 4 and 8 -> up
        (25, DEPOT_GRID, 30),   # tie between 20 and 30 -> up
        (25, STORE_GRID, 24),   # nearest is 24 (not a tie)
        (27, STORE_GRID, 28),
        (2, STORE_GRID, 4),     # tie between 0 and 4 -> up
        (1, STORE_GRID, 0),
        (5, DEPOT_GRID, 10),    # tie between 0 and 10 -> up
        (24, DEPOT_GRID, 20),
        (16, STORE_GRID, 16),   # exact values stay
    ],
)
def test_round_to_grid(x, grid, expected):
    assert round_to_grid(x, grid) == expected


@pytest.mark.parametrize(
    "x, grid, expected",
    [
        (50, STORE_GRID, 32),   # store cap
        (34, STORE_GRID, 32),
        (31, STORE_GRID, 32),
        (100, DEPOT_GRID, 40),  # depot cap
        (45, DEPOT_GRID, 40),
        (-7, STORE_GRID, 0),    # never negative
        (-30, DEPOT_GRID, 0),
        (0, DEPOT_GRID, 0),
    ],
)
def test_caps_and_floor(x, grid, expected):
    assert round_to_grid(x, grid) == expected


# ---------------------------------------------------------- order-up-to logic
@pytest.fixture
def config():
    return load_config()


def test_store_top_up_example_from_d9(config):
    # Target A 40, 15 on the shelf, nothing on the way -> need 25 -> ship 24.
    p = OrderUpToPolicy(depot_target=0, store_a_target=40, store_b_target=0, config=config)
    order, ship_a, ship_b = p.decide(obs(depot=100, a=15, b=50))
    assert ship_a == 24
    assert ship_b == 0  # B is above its target


def test_store_b_counts_shipments_on_the_way(config):
    # Target B 40, shelf 10, on the way 8 + 4 -> need 18 -> tie 16/20 -> 20.
    p = OrderUpToPolicy(depot_target=0, store_a_target=0, store_b_target=40, config=config)
    _, _, ship_b = p.decide(obs(depot=100, a=50, b=10, b_in_1=8, b_in_2=4))
    assert ship_b == 20


def test_depot_counts_stock_after_todays_shipments(config):
    # Depot 50, order on the way 10. A: target 40, shelf 16 -> need 24 -> 24.
    # B: target 40, shelf 20, nothing on the way -> need 20 -> 20.
    # Shipments 24 + 20 = 44 <= 50, so 6 stay at the depot.
    # Depot target 40: need = 40 - (6 + 10) = 24 -> nearest 20.
    # (Counting stock before shipping would give 40 - (50 + 10) < 0 -> 0.)
    p = OrderUpToPolicy(depot_target=40, store_a_target=40, store_b_target=40, config=config)
    assert p.decide(obs(depot=50, a=16, b=20, depot_next=10)) == (20, 24, 20)


def test_depot_ignores_store_stock(config):
    # Stores are full, so nothing ships. Depot 15, nothing on the way,
    # target 50 -> need 35 -> tie 30/40 -> 40, whatever the stores hold.
    p = OrderUpToPolicy(depot_target=50, store_a_target=20, store_b_target=20, config=config)
    assert p.decide(obs(depot=15, a=100, b=100)) == (40, 0, 0)
    assert p.decide(obs(depot=15, a=30, b=30)) == (40, 0, 0)


def test_depot_short_uses_same_split(config):
    # Depot 13; A target 40, shelf 24 -> need 16 -> 16; B target 40, shelf 32 -> need 8 -> 8.
    # Split as in the environment: A 8, B 4, so 1 stays at the depot.
    # Depot target 30, nothing on the way: need = 30 - 1 = 29 -> 30.
    p = OrderUpToPolicy(depot_target=30, store_a_target=40, store_b_target=40, config=config)
    order, req_a, req_b = p.decide(obs(depot=13, a=24, b=32))
    assert (req_a, req_b) == (16, 8)  # requests; the environment applies the split
    assert order == 30

    env = RestockEnv(config)
    env.reset(seed=0, options={"initial_stock": (13, 24, 32), "demand": np.zeros((365, 2), int)})
    o, _, _, _, info = env.step(p(obs(13, 24, 32)))
    assert (info["shipped_a"], info["shipped_b"]) == (8, 4)
    assert o[0] == 1 and o[3] == 30


def test_policy_returns_valid_action_index(config):
    env = RestockEnv(config)
    p = OrderUpToPolicy(100, 44, 48, config)
    o, _ = env.reset(seed=3)
    truncated = False
    while not truncated:
        a = p(o)
        assert 0 <= a < 405
        o, _, _, truncated, info = env.step(a)
        assert info["requested_a"] <= 32 and info["requested_b"] <= 32 and info["depot_order"] <= 40


def test_reasonable_targets_beat_simple_policies(config):
    # Not a tuning result, just a plausibility check: sensible targets should
    # sell most of the demand.
    from src.evaluate import evaluate

    result = evaluate(OrderUpToPolicy(100, 44, 48, config), [0, 1, 2], config)
    assert result["summary"]["fill_rate"] > 0.8
    assert result["summary"]["mean_total_reward"] > 50_000


# ----------------------------------------------------------- grid search helpers
def test_candidate_grid_matches_d12(config):
    g = candidate_grid(config)
    assert g["store_a_target"] == list(range(20, 81, 4))
    assert g["store_b_target"] == list(range(20, 81, 4))
    assert g["depot_target"] == list(range(30, 201, 10))


def test_coarse_grid_keeps_both_ends(config):
    g = candidate_grid(config, coarse_factor=3)
    assert g["store_a_target"][0] == 20 and g["store_a_target"][-1] == 80
    assert g["depot_target"][0] == 30 and g["depot_target"][-1] == 200


def test_fine_candidates_around_best(config):
    best = {"depot_target": 100, "store_a_target": 40, "store_b_target": 60}
    fine = fine_candidates(best, config, coarse_factor=2)
    depots = sorted({c[0] for c in fine})
    stores_a = sorted({c[1] for c in fine})
    assert depots == [80, 90, 100, 110, 120]
    assert stores_a == [32, 36, 40, 44, 48]


def test_edge_report(config):
    g = candidate_grid(config)
    best = {"depot_target": 200, "store_a_target": 20, "store_b_target": 48}
    assert edge_report(best, g) == {"depot_target": "upper edge", "store_a_target": "lower edge",
                                    "store_b_target": None}
