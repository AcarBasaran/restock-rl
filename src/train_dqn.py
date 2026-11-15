"""Train Stable-Baselines3 DQN on the inventory environment (D13–D17).

Wrapper order (outermost last):
    RestockEnv -> Monitor (logs TRUE reward per year) -> ScaleObservation (D16)
               -> CenterReward (only if dqn.reward_centering; tuning round 1 V1-V3)
               -> ScaleReward (D17, training only)
The environment dynamics are never changed; the wrappers only rescale what
the network sees and the reward it learns from. Everything reported (Monitor
learning curves, checkpoint scores, evaluations) is TRUE reward: uncentered
and unscaled.

Each run saves a checkpoint every ``dqn.checkpoint_every`` steps and, after
training, scores every checkpoint greedily on the TUNING years (never the
test years) and records the best one (D16, D18).

The end of a simulated year is a time-limit truncation (``truncated=True``),
which SB3's replay buffer stores as a timeout, so value targets still
bootstrap there (tested in tests/test_train_dqn.py).

Usage:
    python -m src.train_dqn --seed-index 0 --out-dir /scratch/$USER/restock-rl/runs/seed0
    python -m src.train_dqn --config configs/tuning_round_1/v1.yaml --seed-index 0 --out-dir ...
    python -m src.train_dqn --seed 200000 --timesteps 20000 --out-dir /tmp/smoke   # smoke run
    python -m src.train_dqn --config ... --score-only --out-dir <run dir>          # (re)score checkpoints
All hyperparameters come from the ``dqn:`` section of the config
(configs/default.yaml, or a variant file that overrides it via ``base:``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import yaml

from src.env import RestockEnv, load_config

# SB3 DQN keyword arguments that are read straight from the config.
SB3_KWARGS = [
    "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "train_freq",
    "gradient_steps", "target_update_interval", "exploration_fraction",
    "exploration_initial_eps", "exploration_final_eps", "max_grad_norm",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_run_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a config; a file with ``base: <relative path>`` overrides that file.

    ``None`` gives configs/default.yaml. Variant files (e.g.
    configs/tuning_round_1/v2.yaml) list only what differs from their base,
    so "V2 = V1 + ..." is literally what the files say.
    """
    if path is None:
        return load_config()
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    base_rel = raw.pop("base", None)
    if base_rel is None:
        return raw
    return _deep_merge(load_run_config(path.parent / base_rel), raw)


class ScaleObservation(gym.ObservationWrapper):
    """Divide each observation feature by a fixed constant (D16). No clipping."""

    def __init__(self, env: gym.Env, scale):
        super().__init__(env)
        self.scale = np.asarray(scale, dtype=np.float32)
        if self.scale.shape != env.observation_space.shape:
            raise ValueError(f"obs_scale has shape {self.scale.shape}, observation has {env.observation_space.shape}")
        if np.any(self.scale <= 0):
            raise ValueError("obs_scale values must be positive")
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return (np.asarray(observation, dtype=np.float32) / self.scale).astype(np.float32)


class ScaleReward(gym.RewardWrapper):
    """Divide the reward by a constant during training only (D17)."""

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        if scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.scale = float(scale)

    def reward(self, reward):
        return reward / self.scale


class CenterReward(gym.RewardWrapper):
    """Subtract a fixed constant from every training reward (tuning round 1, V1-V3).

    Why this does not change the optimal policy: this is a continuing task
    with discount gamma < 1 (Section 2), and the end of a simulated year is a
    truncation that SB3 bootstraps through, never a terminal state (tested in
    tests/test_train_dqn.py). So the values DQN learns are infinite-horizon
    discounted values. Subtracting a constant c from every reward subtracts
    the same amount from every value:
        Q_c(s, a) = E[sum_k gamma^k (r_k - c)] = Q(s, a) - c / (1 - gamma)
    for every state, action and policy. The shift is identical for all
    actions, so the argmax, the greedy policy and the optimal policy are
    unchanged. (With real termination this would NOT hold: episode length
    could then depend on the policy, and a negative shift would reward ending
    early. Here there is no termination.)

    What it does change is the scale the network must fit. Uncentered, every
    Q-value carries a large common offset (about 264 / (1 - 0.99) = 26,445
    true units, 264 after /100), and the differences between actions that
    decide the policy are small next to it. Centering on the baseline's daily
    reward removes most of that offset. The constant is
    ``baseline.tuning_mean_daily_reward`` from the config.
    """

    def __init__(self, env: gym.Env, center: float):
        super().__init__(env)
        self.center = float(center)

    def reward(self, reward):
        return reward - self.center


def make_training_env(config: dict[str, Any], monitor_file: str | Path | None = None) -> gym.Env:
    """RestockEnv wrapped for training: true-reward Monitor, then (centering and) scaling wrappers."""
    from stable_baselines3.common.monitor import Monitor

    dqn = config["dqn"]
    env: gym.Env = RestockEnv(config)
    env = Monitor(env, filename=str(monitor_file) if monitor_file else None)
    env = ScaleObservation(env, dqn["obs_scale"])
    if dqn.get("reward_centering", False):
        env = CenterReward(env, config["baseline"]["tuning_mean_daily_reward"])
    env = ScaleReward(env, dqn["reward_scale"])
    return env


def build_model(config: dict[str, Any], env: gym.Env, seed: int, verbose: int = 0):
    """SB3 DQN with every hyperparameter from the config (D16)."""
    import torch
    from stable_baselines3 import DQN

    dqn = config["dqn"]
    kwargs = {k: dqn[k] for k in SB3_KWARGS}
    policy_kwargs = {
        "net_arch": list(dqn["net_arch"]),
        "activation_fn": getattr(torch.nn, dqn["activation_fn"]),
        "optimizer_class": getattr(torch.optim, dqn["optimizer"]),
    }
    return DQN(
        "MlpPolicy",
        env,
        gamma=float(config["gamma"]),
        policy_kwargs=policy_kwargs,
        seed=int(seed),
        verbose=verbose,
        device="cpu",
        **kwargs,
    )


class GreedyDQNPolicy:
    """Observation -> action for evaluation: scale like in training, greedy action.

    Same result as ``model.predict(scaled_obs, deterministic=True)`` (argmax of
    the Q-network; tested), but calls the Q-network directly, which is much
    faster for step-by-step evaluation.
    """

    def __init__(self, model, config: dict[str, Any]):
        import torch

        self._torch = torch
        self.model = model
        self.model.policy.set_training_mode(False)
        self.scale = np.asarray(config["dqn"]["obs_scale"], dtype=np.float32)

    def __call__(self, obs) -> int:
        x = self._torch.as_tensor(np.asarray(obs, dtype=np.float32) / self.scale).unsqueeze(0)
        with self._torch.no_grad():
            q_values = self.model.q_net(x)
        return int(q_values.argmax(dim=1).item())


def _git_commit() -> str | None:
    import subprocess

    try:
        repo = Path(__file__).resolve().parent.parent
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


SCORE_KEYS = ["mean_total_reward", "std_total_reward", "mean_discounted_reward", "fill_rate_a", "fill_rate_b",
              "fill_rate", "avg_stock_depot", "avg_stock_a", "avg_stock_b", "mean_lost_a", "mean_lost_b"]


def checkpoint_steps(path: Path) -> int:
    """Training step of an SB3 checkpoint file named ``dqn_<steps>_steps.zip``."""
    return int(path.stem.split("_")[1])


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Checkpoint with the highest mean tuning-year reward; ties go to the earliest."""
    if not rows:
        raise ValueError("no checkpoint scores")
    return max(sorted(rows, key=lambda r: r["steps"]), key=lambda r: r["mean_total_reward"])


def score_checkpoints(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Score every checkpoint of a run greedily on the TUNING years (D16, D18).

    Writes checkpoint_scores.csv (one row per checkpoint, true reward units)
    and best_checkpoint.json. The test years are never touched here.
    """
    from stable_baselines3 import DQN

    from src.evaluate import evaluate, seed_range

    seeds = seed_range(config, "tuning")
    paths = sorted((run_dir / "checkpoints").glob("dqn_*_steps.zip"), key=checkpoint_steps)
    if not paths:
        raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
    rows = []
    for path in paths:
        summary = evaluate(GreedyDQNPolicy(DQN.load(path, device="cpu"), config), seeds, config)["summary"]
        rows.append({"steps": checkpoint_steps(path), "checkpoint": path.name,
                     **{k: float(summary[k]) for k in SCORE_KEYS}})
        print(f"  {path.name}: {summary['mean_total_reward']:,.0f} on tuning years", flush=True)
    with open(run_dir / "checkpoint_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = select_best(rows)
    result = {"selected_on": "tuning", "tuning_seeds": [seeds[0], seeds[-1]], "best": best}
    with open(run_dir / "best_checkpoint.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"best checkpoint: {best['checkpoint']} ({best['mean_total_reward']:,.0f})", flush=True)
    return result


def train(config: dict[str, Any], seed: int, out_dir: Path, timesteps: int | None = None,
          checkpoint_every: int | None = None, config_file: str | None = None) -> dict[str, Any]:
    import stable_baselines3
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback

    out_dir.mkdir(parents=True, exist_ok=True)
    dqn = config["dqn"]
    torch.set_num_threads(int(dqn.get("torch_threads", 1)))
    total = int(timesteps if timesteps is not None else dqn["total_timesteps"])
    if checkpoint_every is None:
        checkpoint_every = int(dqn.get("checkpoint_every", 100_000))

    env = make_training_env(config, monitor_file=out_dir / "monitor")
    model = build_model(config, env, seed, verbose=1)
    callback = CheckpointCallback(save_freq=checkpoint_every, save_path=str(out_dir / "checkpoints"),
                                  name_prefix="dqn")

    meta = {
        "variant": config.get("variant"),
        "config_file": config_file,
        "seed": seed,
        "total_timesteps": total,
        "git_commit": _git_commit(),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__,
                     "gymnasium": gym.__version__},
        "config": config,
    }
    start = time.time()
    model.learn(total_timesteps=total, callback=callback, log_interval=100, progress_bar=False)
    meta["train_seconds"] = time.time() - start
    meta["steps_per_second"] = total / meta["train_seconds"]
    model.save(out_dir / "model.zip")
    with open(out_dir / "run.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done: {total} steps in {meta['train_seconds'] / 60:.1f} min "
          f"({meta['steps_per_second']:.0f} steps/s), saved {out_dir / 'model.zip'}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--seed", type=int, help="training seed value")
    group.add_argument("--seed-index", type=int, help="index into dqn.training_seeds (e.g. SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=None, help="override dqn.total_timesteps (smoke/timing runs)")
    parser.add_argument("--config", type=Path, default=None, help="config or variant file (default configs/default.yaml)")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="override dqn.checkpoint_every (smoke runs)")
    parser.add_argument("--no-score", action="store_true", help="skip scoring checkpoints on the tuning years")
    parser.add_argument("--score-only", action="store_true", help="only (re)score the checkpoints in --out-dir")
    args = parser.parse_args()

    config = load_run_config(args.config)
    if args.score_only:
        import torch

        torch.set_num_threads(int(config["dqn"].get("torch_threads", 1)))
        score_checkpoints(args.out_dir, config)
        return
    if args.seed is None and args.seed_index is None:
        parser.error("--seed or --seed-index is required")
    seed = args.seed if args.seed is not None else int(config["dqn"]["training_seeds"][args.seed_index])
    train(config, seed, args.out_dir, args.timesteps, args.checkpoint_every,
          str(args.config) if args.config else None)
    if not args.no_score:
        start = time.time()
        score_checkpoints(args.out_dir, config)
        print(f"scored checkpoints in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
