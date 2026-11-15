"""Tests for the DQN wrappers and SB3's handling of the day-365 cutoff."""

import inspect

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import DQN  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # tests run on the shared login node; keep them light

from src.env import RestockEnv, load_config  # noqa: E402
from src.train_dqn import (  # noqa: E402
    SB3_KWARGS,
    GreedyDQNPolicy,
    ScaleObservation,
    ScaleReward,
    build_model,
    make_training_env,
)

T = 365


@pytest.fixture
def config():
    return load_config()


def zero_demand():
    return np.zeros((T, 2), dtype=int)


# ------------------------------------------------------------------- wrappers
def test_scale_observation_divides_each_feature(config):
    scale = [60, 30, 30, 40, 30, 30]
    env = ScaleObservation(RestockEnv(config), scale)
    obs, _ = env.reset(seed=0, options={"initial_stock": (30, 15, 6), "demand": zero_demand()})
    assert obs.dtype == np.float32
    assert np.allclose(obs, [30 / 60, 15 / 30, 6 / 30, 0, 0, 0])
    obs, *_ = env.step(env.unwrapped.encode_action(40, 4, 8))
    # raw: depot 30 - 12 = 18, A 15 + 4 = 19, B 6, depot order 40 tomorrow, B 8 in 2 days
    assert np.allclose(obs, [18 / 60, 19 / 30, 6 / 30, 40 / 40, 0, 8 / 30])


def test_scale_observation_does_not_clip(config):
    env = ScaleObservation(RestockEnv(config), [60, 30, 30, 40, 30, 30])
    obs, _ = env.reset(seed=0, options={"initial_stock": (600, 300, 90), "demand": zero_demand()})
    assert np.allclose(obs[:3], [10, 10, 3])


def test_scale_observation_rejects_bad_scale(config):
    with pytest.raises(ValueError):
        ScaleObservation(RestockEnv(config), [1, 2, 3])
    with pytest.raises(ValueError):
        ScaleObservation(RestockEnv(config), [60, 30, 0, 40, 30, 30])


def test_scale_reward_divides_by_constant(config):
    raw = RestockEnv(config)
    wrapped = ScaleReward(RestockEnv(config), 100)
    raw.reset(seed=5)
    wrapped.reset(seed=5)
    for a in [0, 17, 404, 200, 81]:
        _, r_raw, *_ = raw.step(a)
        _, r_wrapped, *_ = wrapped.step(a)
        assert r_wrapped == pytest.approx(r_raw / 100)


def test_wrappers_do_not_change_dynamics(config):
    # Same seed and actions: wrapped observations x scale == raw observations,
    # wrapped rewards x 100 == raw rewards, and info (sales, demand) identical.
    raw = RestockEnv(config)
    wrapped = make_training_env(config)
    scale = np.array(config["dqn"]["obs_scale"], dtype=np.float32)
    o_raw, _ = raw.reset(seed=11)
    o_wr, _ = wrapped.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(T):
        a = int(rng.integers(405))
        assert np.allclose(o_wr * scale, o_raw)
        o_raw, r_raw, term_raw, trunc_raw, info_raw = raw.step(a)
        o_wr, r_wr, term_wr, trunc_wr, info_wr = wrapped.step(a)
        assert r_wr * config["dqn"]["reward_scale"] == pytest.approx(r_raw)
        assert (term_wr, trunc_wr) == (term_raw, trunc_raw)
        assert info_wr["sales_a"] == info_raw["sales_a"] and info_wr["demand_b"] == info_raw["demand_b"]
    assert trunc_raw and not term_raw


def test_monitor_logs_true_reward(config):
    env = make_training_env(config)
    env.reset(seed=3)
    total_scaled = 0.0
    truncated = False
    while not truncated:
        _, r, _, truncated, info = env.step(0)
        total_scaled += r
    # Monitor sits inside the reward scaling, so its episode return is in true units.
    assert info["episode"]["l"] == T
    assert info["episode"]["r"] == pytest.approx(total_scaled * config["dqn"]["reward_scale"], abs=1e-3)


# ------------------------------------------------------- config vs SB3 defaults
def test_config_hyperparameters_are_sb3_defaults(config):
    # D16: SB3 defaults except the 2x256 network; gamma 0.99 is also SB3's default.
    defaults = {k: v.default for k, v in inspect.signature(DQN.__init__).parameters.items()}
    for key in SB3_KWARGS:
        assert config["dqn"][key] == defaults[key], key
    assert config["gamma"] == defaults["gamma"] == 0.99
    assert config["dqn"]["net_arch"] == [256, 256]

    from stable_baselines3.dqn.policies import DQNPolicy

    policy_defaults = {k: v.default for k, v in inspect.signature(DQNPolicy.__init__).parameters.items()}
    assert config["dqn"]["activation_fn"] == policy_defaults["activation_fn"].__name__ == "ReLU"
    assert config["dqn"]["optimizer"] == policy_defaults["optimizer_class"].__name__ == "Adam"
    assert policy_defaults["optimizer_kwargs"] is None  # PyTorch's own Adam settings


def test_built_model_uses_config(config):
    env = make_training_env(config)
    model = build_model(config, env, seed=0)
    assert model.gamma == 0.99
    assert model.batch_size == config["dqn"]["batch_size"]
    assert model.target_update_interval == config["dqn"]["target_update_interval"]
    layers = [m for m in model.q_net.q_net if m.__class__.__name__ == "Linear"]
    assert [(l.in_features, l.out_features) for l in layers] == [(6, 256), (256, 256), (256, 405)]
    assert all(m.__class__.__name__ == "ReLU" for m in model.q_net.q_net if m.__class__.__name__ != "Linear")
    assert model.policy.optimizer.__class__.__name__ == "Adam"
    assert model.policy.optimizer.defaults["eps"] == 1e-8


# ----------------------------------------------------- truncation, not termination
def test_sb3_stores_day_365_as_timeout_not_terminal(config):
    """SB3 must treat the end of the year as truncation so targets bootstrap.

    Run 365 + 10 steps without any gradient update and inspect the replay
    buffer: the transition into day 365 is marked done (episode boundary)
    AND timeout, and the done flag SB3 uses in its targets is 0 there.
    """
    small = dict(config)
    small["dqn"] = dict(config["dqn"], buffer_size=1000, learning_starts=10_000)
    env = make_training_env(small)
    model = build_model(small, env, seed=0)
    assert model.replay_buffer.handle_timeout_termination is True

    model.learn(total_timesteps=T + 10)
    rb = model.replay_buffer
    dones = rb.dones[: T + 10, 0]
    timeouts = rb.timeouts[: T + 10, 0]
    assert dones[T - 1] == 1 and timeouts[T - 1] == 1  # step 365 ends the year by truncation
    assert dones.sum() == 1 and timeouts.sum() == 1  # nothing else ends an episode

    batch = rb._get_samples(np.array([T - 1, 0, T]))
    assert batch.dones.flatten().tolist() == [0.0, 0.0, 0.0]  # no terminal in the TD target

    # The stored next observation at the cutoff is the real final state of the
    # year (from terminal_observation), not the first state of the next year.
    assert not np.allclose(rb.next_observations[T - 1, 0], rb.observations[T, 0])


def test_greedy_policy_returns_valid_actions(config):
    env = make_training_env(config)
    model = build_model(config, env, seed=0)
    policy = GreedyDQNPolicy(model, config)
    raw = RestockEnv(config)
    obs, _ = raw.reset(seed=1)
    for _ in range(5):
        a = policy(obs)
        assert 0 <= a < 405
        obs, *_ = raw.step(a)


def test_greedy_policy_matches_sb3_predict(config):
    # Train briefly so Q-values are not all near zero, then compare the fast
    # greedy path with SB3's predict(deterministic=True) on real observations.
    small = dict(config)
    small["dqn"] = dict(config["dqn"], buffer_size=5000, learning_starts=200)
    env = make_training_env(small)
    model = build_model(small, env, seed=1)
    model.learn(total_timesteps=2000)
    policy = GreedyDQNPolicy(model, small)
    scale = np.array(small["dqn"]["obs_scale"], dtype=np.float32)
    raw = RestockEnv(small)
    obs, _ = raw.reset(seed=2)
    rng = np.random.default_rng(0)
    for _ in range(300):
        expected, _ = model.predict(obs / scale, deterministic=True)
        assert policy(obs) == int(expected)
        obs, *_ = raw.step(int(rng.integers(405)))
