"""Tests for src/env.py against CLAUDE.md Section 2 and D1–D7.

Expected numbers are traced by hand in the comments, not computed by the
code under test.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.env import RestockEnv, load_config, split_shipments

T = 365


@pytest.fixture
def env():
    return RestockEnv(load_config())


def demand_with(first_days, total=T):
    """Demand array: the given (A, B) pairs for the first days, zero afterwards."""
    d = np.zeros((total, 2), dtype=int)
    for i, pair in enumerate(first_days):
        d[i] = pair
    return d


def act(env, order, ship_a, ship_b):
    return env.encode_action(order, ship_a, ship_b)


# --------------------------------------------------------------------- basics
def test_spaces_match_spec(env):
    assert env.action_space.n == 405  # 5 x 9 x 9
    assert env.observation_space.shape == (6,)


# The checker warns that the observation upper bound is infinite. That is
# intended: D5 says no stock caps; network-side clipping is part of D16.
@pytest.mark.filterwarnings("ignore:.*maximum value is infinity")
def test_gymnasium_env_checker(env):
    check_env(env, skip_render_check=True)


def test_action_encoding_roundtrip(env):
    seen = set()
    for a in range(env.action_space.n):
        triple = env.decode_action(a)
        assert env.encode_action(*triple) == a
        seen.add(triple)
    assert len(seen) == 405
    assert env.decode_action(0) == (0, 0, 0)
    assert env.decode_action(404) == (40, 32, 32)


# ------------------------------------------------------------ observation order
def test_observation_order(env):
    # Start: depot 31, A 17, B 9, nothing in transit -> [31, 17, 9, 0, 0, 0].
    obs, _ = env.reset(seed=0, options={"initial_stock": (31, 17, 9), "demand": demand_with([])})
    assert obs.tolist() == [31, 17, 9, 0, 0, 0]

    # Day 0 decision: order 40, ship A 4, ship B 8 (depot has enough).
    # Depot 31 - 12 = 19. Day 1: A gets its shipment (17 + 4 = 21); the depot
    # order is due tomorrow (slot 3); B's 8 units are 2 days away (slot 5).
    obs, *_ = env.step(act(env, 40, 4, 8))
    assert obs.tolist() == [19, 21, 9, 40, 0, 8]

    # Day 1 decision: nothing. Day 2: depot order arrives (19 + 40 = 59);
    # B's 8 units are now 1 day away (slot 4).
    obs, *_ = env.step(act(env, 0, 0, 0))
    assert obs.tolist() == [59, 21, 9, 0, 8, 0]

    # Day 3: B's shipment arrives (9 + 8 = 17).
    obs, *_ = env.step(act(env, 0, 0, 0))
    assert obs.tolist() == [59, 21, 17, 0, 0, 0]


# ------------------------------------------------------------ hand-traced day
def test_hand_traced_days(env):
    # Start: depot 50, A 5, B 3. Demand day 1: A 7, B 2. Day 2: A 3, B 5.
    obs, _ = env.reset(
        seed=0, options={"initial_stock": (50, 5, 3), "demand": demand_with([(7, 2), (3, 5)])}
    )

    # Day 0 decision: order 20, ship A 8, ship B 12 (20 <= 50, no split).
    #   depot after shipping: 50 - 20 = 30
    #   holding (after shipments): 1*30 + 2*5 + 2*3 = 46
    # Day 1: arrivals: depot 0 (order due day 2), A +8 -> 13, B 0 (due day 3).
    #   demand A 7 -> sell 7, A = 6; demand B 2 -> sell 2, B = 1
    #   reward = 10*(7+2) - 46 = 44
    obs, reward, terminated, truncated, info = env.step(act(env, 20, 8, 12))
    assert obs.tolist() == [30, 6, 1, 20, 0, 12]
    assert reward == 44
    assert (terminated, truncated) == (False, False)
    assert (info["sales_a"], info["sales_b"], info["lost_a"], info["lost_b"]) == (7, 2, 0, 0)
    assert info["holding_cost"] == 46

    # Day 1 decision: nothing.
    #   holding: 1*30 + 2*6 + 2*1 = 44
    # Day 2: depot +20 -> 50; A +0; B +0 (12 due day 3).
    #   demand A 3 -> sell 3, A = 3; demand B 5 -> sell 1, lose 4, B = 0
    #   reward = 10*(3+1) - 44 = -4
    obs, reward, _, _, info = env.step(act(env, 0, 0, 0))
    assert obs.tolist() == [50, 3, 0, 0, 12, 0]
    assert reward == -4
    assert (info["sales_b"], info["lost_b"]) == (1, 4)


# ---------------------------------------------------------- pipeline shifting
@pytest.mark.parametrize(
    "location, action, lead, stock_index",
    [
        ("depot", (10, 0, 0), 2, 0),
        ("store_a", (0, 4, 0), 1, 1),
        ("store_b", (0, 0, 4), 3, 2),
    ],
)
def test_pipeline_arrival_day(env, location, action, lead, stock_index):
    # A single order placed at the end of day 0 must arrive at the start of
    # day 0 + L (D1). Zero demand, so stock only changes through arrivals.
    # Depot starts at 100 and stores at 0; for store shipments the depot
    # stock drops at once and the store stock rises on the arrival day.
    obs, _ = env.reset(seed=0, options={"initial_stock": (100, 0, 0), "demand": demand_with([])})
    qty = sum(action)
    stock_before = 100 if location == "depot" else 0

    obs, *_ = env.step(act(env, *action))  # obs is now the end of day 1
    for day in range(1, 6):
        expected = stock_before + (qty if day >= lead else 0)
        assert obs[stock_index] == expected, f"{location}: day {day} expected {expected}, got {obs[stock_index]}"
        obs, *_ = env.step(act(env, 0, 0, 0))


def test_store_b_pipeline_shifts_one_slot_per_day(env):
    # Two B shipments on consecutive days: 4 (day 0) and 8 (day 1).
    obs, _ = env.reset(seed=0, options={"initial_stock": (100, 0, 0), "demand": demand_with([])})
    obs, *_ = env.step(act(env, 0, 0, 4))
    assert obs[4:6].tolist() == [0, 4]  # 4 arrives on day 3, i.e. in 2 days
    obs, *_ = env.step(act(env, 0, 0, 8))
    assert obs[4:6].tolist() == [4, 8]  # 4 in 1 day, 8 in 2 days
    obs, *_ = env.step(act(env, 0, 0, 0))
    assert obs[2] == 4 and obs[4:6].tolist() == [8, 0]
    obs, *_ = env.step(act(env, 0, 0, 0))
    assert obs[2] == 12 and obs[4:6].tolist() == [0, 0]


def test_depot_pipeline_slot(env):
    obs, _ = env.reset(seed=0, options={"initial_stock": (0, 0, 0), "demand": demand_with([])})
    obs, *_ = env.step(act(env, 30, 0, 0))
    assert obs[0] == 0 and obs[3] == 30  # arrives tomorrow
    obs, *_ = env.step(act(env, 10, 0, 0))
    assert obs[0] == 30 and obs[3] == 10
    obs, *_ = env.step(act(env, 0, 0, 0))
    assert obs[0] == 40 and obs[3] == 0


# ---------------------------------------------------------- proportional split
def test_split_function_example():
    assert split_shipments(13, 16, 8) == (8, 4)  # 8.67 -> 8, 4.33 -> 4; 1 stays


def test_split_function_cases():
    assert split_shipments(30, 16, 8) == (16, 8)  # enough stock: full requests
    assert split_shipments(24, 16, 8) == (16, 8)  # exactly enough
    assert split_shipments(0, 16, 8) == (0, 0)
    assert split_shipments(1, 4, 4) == (0, 0)  # 0.5 / 0.5 -> 0 / 0, 1 stays
    assert split_shipments(10, 0, 32) == (0, 10)


def test_split_13_8_4_1_in_env(env):
    obs, _ = env.reset(seed=0, options={"initial_stock": (13, 0, 0), "demand": demand_with([])})
    obs, reward, _, _, info = env.step(act(env, 0, 16, 8))
    assert (info["shipped_a"], info["shipped_b"]) == (8, 4)
    assert obs[0] == 1  # leftover stays in the depot
    assert obs[1] == 8  # A receives 8 the next day
    assert obs[5] == 4  # B's 4 units arrive in 2 days


# --------------------------------------------------------------------- reward
def test_holding_cost_after_shipments_leave(env):
    # Depot 40, A 6, B 3, zero demand. Ship 16 to A and 8 to B, no order.
    # After shipments leave: depot 16, A 6, B 3 (shipped units are on the road).
    #   holding = 1*16 + 2*6 + 2*3 = 34  -> reward = -34
    # (Charging before shipping would give 1*40 + 12 + 6 = 58.)
    env.reset(seed=0, options={"initial_stock": (40, 6, 3), "demand": demand_with([])})
    _, reward, _, _, info = env.step(act(env, 0, 16, 8))
    assert info["holding_cost"] == 34
    assert reward == -34


def test_reward_sales_and_holding(env):
    # Depot 0, A 25, B 12; demand day 1: A 20, B 15. No action.
    #   holding day 0: 0 + 2*25 + 2*12 = 74
    #   sales day 1: A 20, B 12 (3 lost) -> revenue 320
    #   reward = 320 - 74 = 246
    env.reset(seed=0, options={"initial_stock": (0, 25, 12), "demand": demand_with([(20, 15)])})
    _, reward, _, _, info = env.step(act(env, 0, 0, 0))
    assert reward == 246
    assert (info["revenue"], info["lost_b"]) == (320, 3)


def test_no_other_costs(env):
    # Ordering and shipping cost nothing: with zero stock, any action's reward
    # is 0 on the first day (no holding, no sales).
    env.reset(seed=0, options={"initial_stock": (0, 0, 0), "demand": demand_with([])})
    _, reward, *_ = env.step(act(env, 40, 32, 32))
    assert reward == 0


# ------------------------------------------------------------------ truncation
def test_truncation_not_termination_at_365(env):
    env.reset(seed=1)
    for day in range(1, T + 1):
        _, _, terminated, truncated, info = env.step(act(env, 0, 0, 0))
        assert terminated is False
        assert truncated is (day == T)
        assert info["day"] == day
    with pytest.raises(RuntimeError):
        env.step(act(env, 0, 0, 0))


# --------------------------------------------------------------- reproducibility
def run_year(env, seed, policy):
    obs, _ = env.reset(seed=seed)
    start = obs.tolist()
    demand = []
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step(policy(obs))
        demand.append((info["demand_a"], info["demand_b"]))
    return start, demand


def test_same_seed_same_demand_and_initial_stock(env):
    rng = np.random.default_rng(123)
    do_nothing = lambda obs: 0
    random_policy = lambda obs: int(rng.integers(405))

    start1, demand1 = run_year(env, 42, do_nothing)
    start2, demand2 = run_year(env, 42, random_policy)  # different actions, same seed
    assert start1 == start2
    assert demand1 == demand2

    start3, demand3 = run_year(env, 43, do_nothing)
    assert demand3 != demand1

    # A fresh environment object with the same seed gives the same year too.
    start4, demand4 = run_year(RestockEnv(load_config()), 42, do_nothing)
    assert (start4, demand4) == (start1, demand1)


def test_initial_stock_and_demand_statistics(env):
    # Starting stock ~ Poisson(30, 20, 10) (D6); demand ~ Poisson(20, 10).
    starts = []
    demands = []
    for seed in range(300):
        obs, _ = env.reset(seed=seed)
        starts.append(obs[:3])
        assert obs[3:].tolist() == [0, 0, 0]  # nothing in transit
        demands.append(np.array(env._demand))
    starts = np.array(starts)
    demands = np.concatenate(demands)
    assert np.allclose(starts.mean(axis=0), [30, 20, 10], atol=1.0)
    assert np.allclose(demands.mean(axis=0), [20, 10], atol=0.1)
