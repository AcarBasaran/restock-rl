"""Same-demand evaluation of any policy (observation -> action).

Every policy is run on the same list of demand seeds, so all methods see the
same customer demand and starting stock (Section 2). All values are in true
reward units.

Metrics follow D19 (undiscounted 365-day reward, fill rate per store and
overall, average stock per location) and D18 (no warm-up exclusion, so
``warmup_days`` defaults to 0). Two details are still open and use
provisional choices: when average stock is measured (Q2) and how fill rates
are pooled over years (Q3); see notes/open-questions.md.

Usage (sanity check with simple policies):
    python -m src.evaluate --policy random --policy do_nothing
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from src.env import RestockEnv, load_config

Policy = Callable[[np.ndarray], int]


def run_episode(env: RestockEnv, policy: Policy, seed: int, warmup_days: int = 0) -> dict[str, Any]:
    """Run one simulated year and return its metrics.

    Steps 1..warmup_days are simulated but left out of every metric. Average
    stock is the stock charged for holding (end of the decision day, after
    shipments left), averaged over the counted steps.
    """
    if not 0 <= warmup_days < env.episode_length:
        raise ValueError(f"warmup_days must be in [0, {env.episode_length}), got {warmup_days}")
    if hasattr(policy, "reset"):
        policy.reset(seed)

    gamma = float(env.config["gamma"])
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    discounted_reward = 0.0
    discount = 1.0
    sums = dict.fromkeys(
        ["revenue", "holding_cost", "demand_a", "demand_b", "sales_a", "sales_b",
         "lost_a", "lost_b", "held_depot", "held_a", "held_b"],
        0,
    )
    counted = 0
    truncated = False
    step = 0
    while not truncated:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        if terminated:
            raise RuntimeError("the environment should never terminate (continuing task)")
        step += 1
        if step <= warmup_days:
            continue
        counted += 1
        total_reward += reward
        discounted_reward += discount * reward
        discount *= gamma
        for key in sums:
            sums[key] += info[key]

    def ratio(num: float, den: float) -> float:
        return num / den if den > 0 else float("nan")

    return {
        "seed": seed,
        "days_counted": counted,
        "total_reward": total_reward,
        "discounted_reward": discounted_reward,
        "revenue": sums["revenue"],
        "holding_cost": sums["holding_cost"],
        "demand_a": sums["demand_a"],
        "demand_b": sums["demand_b"],
        "sales_a": sums["sales_a"],
        "sales_b": sums["sales_b"],
        "lost_a": sums["lost_a"],
        "lost_b": sums["lost_b"],
        "fill_rate_a": ratio(sums["sales_a"], sums["demand_a"]),
        "fill_rate_b": ratio(sums["sales_b"], sums["demand_b"]),
        "fill_rate": ratio(sums["sales_a"] + sums["sales_b"], sums["demand_a"] + sums["demand_b"]),
        "avg_stock_depot": sums["held_depot"] / counted,
        "avg_stock_a": sums["held_a"] / counted,
        "avg_stock_b": sums["held_b"] / counted,
    }


def summarize(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-year results.

    Rewards and average stocks: mean and standard deviation over years.
    Fill rates: pooled (total sales / total demand over all years).
    Lost sales: mean per year.
    """
    def col(key):
        return np.array([e[key] for e in episodes], dtype=float)

    def pooled(sales_keys, demand_keys):
        d = sum(col(k).sum() for k in demand_keys)
        return sum(col(k).sum() for k in sales_keys) / d if d > 0 else float("nan")

    rewards = col("total_reward")
    return {
        "n_years": len(episodes),
        "mean_total_reward": rewards.mean(),
        "std_total_reward": rewards.std(ddof=1) if len(episodes) > 1 else 0.0,
        "mean_discounted_reward": col("discounted_reward").mean(),
        "fill_rate_a": pooled(["sales_a"], ["demand_a"]),
        "fill_rate_b": pooled(["sales_b"], ["demand_b"]),
        "fill_rate": pooled(["sales_a", "sales_b"], ["demand_a", "demand_b"]),
        "avg_stock_depot": col("avg_stock_depot").mean(),
        "avg_stock_a": col("avg_stock_a").mean(),
        "avg_stock_b": col("avg_stock_b").mean(),
        "mean_lost_a": col("lost_a").mean(),
        "mean_lost_b": col("lost_b").mean(),
    }


def evaluate(
    policy: Policy,
    seeds: Sequence[int],
    config: dict[str, Any] | None = None,
    warmup_days: int = 0,
) -> dict[str, Any]:
    """Run ``policy`` for one year per seed. Returns per-year results and a summary."""
    env = RestockEnv(config if config is not None else load_config())
    episodes = [run_episode(env, policy, int(s), warmup_days) for s in seeds]
    return {"episodes": episodes, "summary": summarize(episodes)}


def seed_range(config: dict[str, Any], name: str) -> list[int]:
    """Seeds of a named range in the config (e.g. "tuning")."""
    spec = config["seeds"][name]
    if spec.get("count") is None:
        raise ValueError(f"seeds.{name}.count is not set (see open decisions)")
    return list(range(int(spec["start"]), int(spec["start"]) + int(spec["count"])))


# ----------------------------------------------------------- simple policies
class RandomPolicy:
    """Uniformly random action; its own RNG is re-seeded per year for reproducibility.

    The policy RNG uses seed ``offset + demand seed`` (offset from
    ``seeds.random_policy_offset`` in the config), so its randomness is
    reproducible but separate from the demand stream.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config if config is not None else load_config()
        self.n_actions = RestockEnv(cfg).n_actions
        self.offset = int(cfg["seeds"]["random_policy_offset"])
        self.rng = np.random.default_rng(self.offset)

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(self.offset + seed)

    def __call__(self, obs) -> int:
        return int(self.rng.integers(self.n_actions))


def do_nothing_policy(obs) -> int:
    """Never order, never ship (action index 0 = (0, 0, 0))."""
    return 0


def format_summary(name: str, s: dict[str, Any]) -> str:
    return (
        f"| {name} | {s['n_years']} | {s['mean_total_reward']:,.0f} | {s['std_total_reward']:,.0f} "
        f"| {s['fill_rate_a']:.1%} | {s['fill_rate_b']:.1%} | {s['fill_rate']:.1%} "
        f"| {s['avg_stock_depot']:.1f} | {s['avg_stock_a']:.1f} | {s['avg_stock_b']:.1f} "
        f"| {s['mean_lost_a']:,.0f} | {s['mean_lost_b']:,.0f} |"
    )


TABLE_HEADER = (
    "| Policy | Years | Mean reward | Std | Fill A | Fill B | Fill all | Avg depot | Avg A | Avg B "
    "| Lost A / yr | Lost B / yr |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple policies on the sanity seeds.")
    parser.add_argument("--policy", action="append", choices=["random", "do_nothing"])
    parser.add_argument("--seeds", default="sanity", help="seed range name in the config")
    parser.add_argument("--warmup-days", type=int, default=0)
    args = parser.parse_args()

    config = load_config()
    seeds = seed_range(config, args.seeds)
    policies = {"random": RandomPolicy(config), "do_nothing": do_nothing_policy}
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} years), warm-up {args.warmup_days} days\n")
    print(TABLE_HEADER)
    for name in args.policy or ["random", "do_nothing"]:
        result = evaluate(policies[name], seeds, config, args.warmup_days)
        print(format_summary(name, result["summary"]))


if __name__ == "__main__":
    main()
