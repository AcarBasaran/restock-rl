"""Train Stable-Baselines3 DQN on the inventory environment (D13–D17).

Wrapper order (outermost last):
    RestockEnv -> Monitor (logs TRUE reward per year) -> ScaleObservation (D16)
               -> ScaleReward (D17, training only)
The environment dynamics are never changed; the wrappers only rescale what
the network sees and the reward it learns from.

The end of a simulated year is a time-limit truncation (``truncated=True``),
which SB3's replay buffer stores as a timeout, so value targets still
bootstrap there (tested in tests/test_train_dqn.py).

Usage:
    python -m src.train_dqn --seed-index 0 --out-dir /scratch/$USER/restock-rl/runs/seed0
    python -m src.train_dqn --seed 200000 --timesteps 20000 --out-dir /tmp/smoke   # smoke run
All hyperparameters come from the ``dqn:`` section of configs/default.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.env import RestockEnv, load_config

# SB3 DQN keyword arguments that are read straight from the config.
SB3_KWARGS = [
    "learning_rate", "buffer_size", "learning_starts", "batch_size", "tau", "train_freq",
    "gradient_steps", "target_update_interval", "exploration_fraction",
    "exploration_initial_eps", "exploration_final_eps", "max_grad_norm",
]


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


def make_training_env(config: dict[str, Any], monitor_file: str | Path | None = None) -> gym.Env:
    """RestockEnv wrapped for training: true-reward Monitor, then scaling wrappers."""
    from stable_baselines3.common.monitor import Monitor

    dqn = config["dqn"]
    env: gym.Env = RestockEnv(config)
    env = Monitor(env, filename=str(monitor_file) if monitor_file else None)
    env = ScaleObservation(env, dqn["obs_scale"])
    env = ScaleReward(env, dqn["reward_scale"])
    return env


def build_model(config: dict[str, Any], env: gym.Env, seed: int, verbose: int = 0):
    """SB3 DQN with every hyperparameter from the config (D16)."""
    from stable_baselines3 import DQN

    dqn = config["dqn"]
    kwargs = {k: dqn[k] for k in SB3_KWARGS}
    return DQN(
        "MlpPolicy",
        env,
        gamma=float(config["gamma"]),
        policy_kwargs={"net_arch": list(dqn["net_arch"])},
        seed=int(seed),
        verbose=verbose,
        device="cpu",
        **kwargs,
    )


class GreedyDQNPolicy:
    """Observation -> action for evaluation: scale like in training, greedy action."""

    def __init__(self, model, config: dict[str, Any]):
        self.model = model
        self.scale = np.asarray(config["dqn"]["obs_scale"], dtype=np.float32)

    def __call__(self, obs) -> int:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32) / self.scale, deterministic=True)
        return int(action)


def train(config: dict[str, Any], seed: int, out_dir: Path, timesteps: int | None = None,
          checkpoint_every: int = 500_000) -> dict[str, Any]:
    import stable_baselines3
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback

    out_dir.mkdir(parents=True, exist_ok=True)
    dqn = config["dqn"]
    torch.set_num_threads(int(dqn.get("torch_threads", 1)))
    total = int(timesteps if timesteps is not None else dqn["total_timesteps"])

    env = make_training_env(config, monitor_file=out_dir / "monitor")
    model = build_model(config, env, seed, verbose=1)
    callback = CheckpointCallback(save_freq=checkpoint_every, save_path=str(out_dir / "checkpoints"),
                                  name_prefix="dqn")

    meta = {
        "seed": seed,
        "total_timesteps": total,
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, help="training seed value")
    group.add_argument("--seed-index", type=int, help="index into dqn.training_seeds (e.g. SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=None, help="override dqn.total_timesteps (smoke/timing runs)")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else int(config["dqn"]["training_seeds"][args.seed_index])
    train(config, seed, args.out_dir, args.timesteps)


if __name__ == "__main__":
    main()
