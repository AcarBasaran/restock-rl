"""Tests for the DQN wrappers and SB3's handling of the day-365 cutoff."""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import DQN  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)  # tests run on the shared login node; keep them light

from src.env import RestockEnv, load_config  # noqa: E402
from src.train_dqn import (  # noqa: E402
    SB3_KWARGS,
    CenterReward,
    GreedyDQNPolicy,
    ScaleObservation,
    ScaleReward,
    build_model,
    load_run_config,
    make_training_env,
    score_checkpoints,
    select_best,
    train,
)

ROUND_1 = Path(__file__).resolve().parent.parent / "configs" / "tuning_round_1"

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


# ------------------------------------------------- reward centering (round 1)
def test_center_reward_subtracts_constant(config):
    raw = RestockEnv(config)
    centered = CenterReward(RestockEnv(config), 264.4512)
    raw.reset(seed=7)
    centered.reset(seed=7)
    for a in [0, 17, 404, 200, 81]:
        _, r_raw, *_ = raw.step(a)
        _, r_c, *_ = centered.step(a)
        assert r_c == pytest.approx(r_raw - 264.4512)


def test_centered_training_env_scales_after_centering_and_monitor_logs_true_reward():
    cfg = load_run_config(ROUND_1 / "v1.yaml")
    c = cfg["baseline"]["tuning_mean_daily_reward"]
    raw = RestockEnv(cfg)
    env = make_training_env(cfg)
    raw.reset(seed=4)
    env.reset(seed=4)
    rng = np.random.default_rng(1)
    true_total = 0.0
    truncated = False
    while not truncated:
        a = int(rng.integers(405))
        _, r_raw, *_ = raw.step(a)
        _, r, _, truncated, info = env.step(a)
        assert r == pytest.approx((r_raw - c) / cfg["dqn"]["reward_scale"])
        true_total += r_raw
    # Learning curves come from Monitor: uncentered, unscaled true reward.
    assert info["episode"]["r"] == pytest.approx(true_total, abs=1e-3)


def _wrapper_names(env):
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_centering_wrapper_only_in_v1_to_v3():
    for name, centered in [("v0", False), ("v1", True), ("v2", True), ("v3", True)]:
        names = _wrapper_names(make_training_env(load_run_config(ROUND_1 / f"{name}.yaml")))
        assert ("CenterReward" in names) == centered, name
        # Monitor (true reward) sits inside every reward wrapper.
        assert names.index("Monitor") > names.index("ScaleReward")


def test_centering_constant_is_baseline_mean_daily_reward_on_tuning_years(config):
    from src.baseline import tuned_policy
    from src.evaluate import evaluate, seed_range

    summary = evaluate(tuned_policy(config), seed_range(config, "tuning"), config)["summary"]
    expected = summary["mean_total_reward"] / config["episode_length"]
    assert config["baseline"]["tuning_mean_daily_reward"] == pytest.approx(expected, abs=1e-3)


def test_centering_shifts_every_policy_by_the_same_amount(config):
    # The algebra behind CenterReward's docstring, on real trajectories: over
    # the same horizon the discounted centered return differs from the true one
    # by the same constant whatever the actions, so action rankings are unchanged.
    gamma, c = config["gamma"], 264.4512
    shifts = []
    for policy_seed in (0, 1, 2):
        env = RestockEnv(config)
        env.reset(seed=9)
        rng = np.random.default_rng(policy_seed)
        g_true = g_centered = 0.0
        for k in range(T):
            _, r, *_ = env.step(int(rng.integers(405)))
            g_true += gamma**k * r
            g_centered += gamma**k * (r - c)
        shifts.append(g_centered - g_true)
    expected = -c * (1 - gamma**T) / (1 - gamma)
    assert shifts == pytest.approx([expected] * 3)


# --------------------------------------------------------- variant configs
def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_variant_configs_differ_only_as_decided(config):
    v = {name: _flatten(load_run_config(ROUND_1 / f"{name.lower()}.yaml")) for name in ["V0", "V1", "V2", "V3"]}
    base = _flatten(config)

    def diff(a, b):
        return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

    assert diff(base, v["V0"]) == {"variant"}
    assert diff(v["V0"], v["V1"]) == {"variant", "dqn.reward_centering"} and v["V1"]["dqn.reward_centering"] is True
    assert diff(v["V1"], v["V2"]) == {"variant", "dqn.learning_rate", "dqn.target_update_interval", "dqn.batch_size"}
    assert (v["V2"]["dqn.learning_rate"], v["V2"]["dqn.target_update_interval"], v["V2"]["dqn.batch_size"]) == (
        5e-5, 20000, 128)
    assert diff(v["V2"], v["V3"]) == {"variant", "dqn.exploration_fraction"} and v["V3"]["dqn.exploration_fraction"] == 0.3
    for name in v:
        assert v[name]["variant"] == name
        assert v[name]["dqn.checkpoint_every"] == 100_000
        assert v[name]["dqn.total_timesteps"] == 2_000_000


def test_v2_model_uses_variant_hyperparameters():
    cfg = load_run_config(ROUND_1 / "v3.yaml")
    model = build_model(cfg, make_training_env(cfg), seed=0)
    assert model.batch_size == 128 and model.target_update_interval == 20000
    assert model.exploration_fraction == 0.3
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


# ---------------------------------------------------- checkpoint selection
def test_select_best_picks_highest_reward_earliest_on_ties():
    rows = [{"steps": 300, "mean_total_reward": 5.0}, {"steps": 100, "mean_total_reward": 7.0},
            {"steps": 200, "mean_total_reward": 7.0}]
    assert select_best(rows)["steps"] == 100


def test_score_checkpoints_uses_tuning_years_and_true_reward(tmp_path, config):
    from src.evaluate import evaluate

    cfg = load_run_config(ROUND_1 / "v1.yaml")
    cfg["dqn"] = dict(cfg["dqn"], buffer_size=5000, learning_starts=200, torch_threads=1)
    cfg["seeds"] = dict(cfg["seeds"], tuning={"start": 100000, "count": 2})  # 2 years keep the test fast
    train(cfg, seed=0, out_dir=tmp_path, timesteps=1000, checkpoint_every=500)
    result = score_checkpoints(tmp_path, cfg)
    with open(tmp_path / "checkpoint_scores.csv") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["steps"]) for r in rows] == [500, 1000]
    assert result["selected_on"] == "tuning" and result["tuning_seeds"] == [100000, 100001]
    assert json.loads((tmp_path / "best_checkpoint.json").read_text())["best"] == result["best"]

    # Scores are the true (uncentered, unscaled) reward of the greedy policy on those years.
    model = DQN.load(tmp_path / "checkpoints" / "dqn_1000_steps.zip", device="cpu")
    direct = evaluate(GreedyDQNPolicy(model, cfg), [100000, 100001], cfg)["summary"]["mean_total_reward"]
    assert float(rows[1]["mean_total_reward"]) == pytest.approx(direct)
