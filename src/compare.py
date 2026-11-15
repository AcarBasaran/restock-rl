"""Evaluate trained DQN models against the tuned baseline on the test years.

For every training seed's final model: greedy actions on the 50 test years
(D18), the same years and starting stock as the baseline. All numbers are in
true reward units (D17). Also draws the learning curves from the Monitor logs
(training reward per simulated year, true units).

Usage:
    python -m src.compare --runs-dir /scratch/$USER/restock-rl/runs/dqn
Outputs go to results/dqn/ (not committed).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.baseline import save_rows, tuned_policy
from src.env import load_config
from src.evaluate import evaluate, seed_range

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "dqn"

METRICS = [
    ("mean_total_reward", "365-day reward", "{:,.0f}"),
    ("fill_rate_a", "Fill rate Store A", "{:.1%}"),
    ("fill_rate_b", "Fill rate Store B", "{:.1%}"),
    ("fill_rate", "Fill rate overall", "{:.1%}"),
    ("avg_stock_depot", "Average stock depot", "{:.1f}"),
    ("avg_stock_a", "Average stock Store A", "{:.1f}"),
    ("avg_stock_b", "Average stock Store B", "{:.1f}"),
]

# Chart styling (light surface, reference data-viz palette).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
REPLICATE = "#b9b8b2"


def find_runs(runs_dir: Path) -> list[Path]:
    runs = sorted(p for p in runs_dir.glob("seed_index_*") if (p / "model.zip").exists())
    if not runs:
        raise FileNotFoundError(f"no finished runs (model.zip) under {runs_dir}")
    return runs


def load_monitor(run: Path) -> np.ndarray:
    """Per-year training rewards (true units) from the SB3 Monitor CSV."""
    with open(run / "monitor.monitor.csv") as f:
        next(f)  # JSON header line
        rows = list(csv.DictReader(f))
    return np.array([float(r["r"]) for r in rows])


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return np.array([])
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def mean_std(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def comparison_table(dqn: list[dict[str, Any]], baseline: dict[str, Any], baseline_std_reward: float) -> str:
    lines = [
        f"| Metric (50 test years) | DQN, mean ± std over {len(dqn)} seeds | Baseline (60 / 24 / 32) |",
        "|---|---|---|",
    ]
    for key, label, fmt in METRICS:
        m, s = mean_std([d[key] for d in dqn])
        b = fmt.format(baseline[key])
        if key == "mean_total_reward":
            b += f" (std over years {baseline_std_reward:,.0f})"
        lines.append(f"| {label} | {fmt.format(m)} ± {fmt.format(s)} | {b} |")
    return "\n".join(lines)


def plot_learning_curves(runs: list[Path], baseline_reward: float, out: Path, window: int = 50,
                         steps_per_year: int = 365) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = [rolling_mean(load_monitor(r), window) for r in runs]
    n = min(len(c) for c in curves)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for c in curves:
        x = (np.arange(len(c)) + window) * steps_per_year / 1e6
        ax.plot(x, c / 1000, color=REPLICATE, linewidth=1)
    mean_curve = np.mean([c[:n] for c in curves], axis=0)
    x = (np.arange(n) + window) * steps_per_year / 1e6
    ax.plot(x, mean_curve / 1000, color=SERIES_1, linewidth=2)
    ax.axhline(baseline_reward / 1000, color=TEXT_SECONDARY, linewidth=1.5, linestyle=(0, (5, 4)))

    ax.annotate("DQN, mean of seeds", xy=(x[-1], mean_curve[-1] / 1000), xytext=(-4, 8),
                textcoords="offset points", ha="right", color=TEXT_PRIMARY, fontsize=9)
    ax.annotate(f"Baseline on test years ({baseline_reward / 1000:.1f}k)", xy=(x[0], baseline_reward / 1000),
                xytext=(2, 6), textcoords="offset points", ha="left", color=TEXT_SECONDARY, fontsize=9)
    ax.annotate("individual seeds", xy=(x[n // 3], curves[0][n // 3] / 1000), xytext=(0, -14),
                textcoords="offset points", ha="center", color=TEXT_SECONDARY, fontsize=8)

    ax.set_title(f"DQN training reward per simulated year ({window}-year rolling mean, ε-greedy)",
                 color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel("Training steps (millions)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Reward per year (thousands, true units)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_test_rewards(dqn: list[dict[str, Any]], seeds: list[int], baseline_reward: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [d["mean_total_reward"] / 1000 for d in dqn]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.arange(len(values))
    ax.scatter(xs, values, s=48, color=SERIES_1, edgecolors=SURFACE, linewidths=2, zorder=3)
    for x, v in zip(xs, values):
        ax.annotate(f"{v:.1f}k", xy=(x, v), xytext=(0, 8), textcoords="offset points", ha="center",
                    color=TEXT_PRIMARY, fontsize=8)
    ax.axhline(baseline_reward / 1000, color=TEXT_SECONDARY, linewidth=1.5, linestyle=(0, (5, 4)))
    ax.annotate(f"Baseline {baseline_reward / 1000:.1f}k", xy=(len(values) - 0.5, baseline_reward / 1000),
                xytext=(0, 6), textcoords="offset points", ha="right", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(xs, [f"seed {s}" for s in seeds])
    ax.set_xlim(-0.6, len(values) - 0.4)
    ax.set_title("Mean 365-day reward on the 50 test years (greedy DQN)", color=TEXT_PRIMARY,
                 fontsize=11, loc="left")
    ax.set_ylabel("Reward per year (thousands)", color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    from stable_baselines3 import DQN

    from src.train_dqn import GreedyDQNPolicy

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50, help="rolling window (years) for learning curves")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)  # tiny network; avoids oversubscribing the shared login node
    config = load_config()
    test_seeds = seed_range(config, "test")
    runs = find_runs(args.runs_dir)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    base = evaluate(tuned_policy(config), test_seeds, config)
    save_rows(base["episodes"], out_dir / "baseline_test_years.csv")

    dqn_summaries, train_seeds, per_year = [], [], []
    for run in runs:
        meta = json.loads((run / "run.json").read_text())
        model = DQN.load(run / "model.zip", device="cpu")
        result = evaluate(GreedyDQNPolicy(model, config), test_seeds, config)
        save_rows(result["episodes"], out_dir / f"dqn_{run.name}_test_years.csv")
        dqn_summaries.append(result["summary"])
        train_seeds.append(meta["seed"])
        per_year.append([e["total_reward"] for e in result["episodes"]])
        keep = out_dir / "runs" / run.name  # small: model ~1 MB, monitor CSV, metadata
        keep.mkdir(parents=True, exist_ok=True)
        for name in ("model.zip", "monitor.monitor.csv", "run.json"):
            shutil.copy2(run / name, keep / name)
        print(f"{run.name} (seed {meta['seed']}): {result['summary']['mean_total_reward']:,.0f}", flush=True)

    base_years = np.array([e["total_reward"] for e in base["episodes"]])
    dqn_years = np.array(per_year)  # seeds x years
    diff = dqn_years.mean(axis=0) - base_years  # per test year, seed-averaged DQN minus baseline
    paired = {
        "mean_difference_per_year": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)),
        "years_dqn_better": int((diff > 0).sum()),
        "per_seed_mean_difference": [float((row - base_years).mean()) for row in dqn_years],
    }

    table = comparison_table(dqn_summaries, base["summary"], base["summary"]["std_total_reward"])
    plot_learning_curves(runs, base["summary"]["mean_total_reward"], out_dir / "learning_curves.png", args.window)
    plot_test_rewards(dqn_summaries, train_seeds, base["summary"]["mean_total_reward"], out_dir / "test_reward_by_seed.png")

    out = {"test_seeds": [test_seeds[0], test_seeds[-1]], "training_seeds": train_seeds,
           "runs": [str(r) for r in runs], "baseline": base["summary"],
           "dqn_per_seed": dict(zip(map(str, train_seeds), dqn_summaries)), "paired": paired}
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    (out_dir / "comparison.md").write_text(table + "\n")
    print(table)
    print(json.dumps(paired, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
