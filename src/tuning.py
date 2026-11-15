"""Tuning round 1 (D16, D18, D20): summarize variants, pick the best, test it once.

Two steps, in this order:

    python -m src.tuning summarize --round-dir /scratch/$USER/restock-rl/runs/tuning1
        Reads every run's checkpoint scores (greedy, TUNING years, written by
        src.train_dqn at the end of each job). Per variant: mean ± std across
        seeds of the best-checkpoint tuning-year metrics, the baseline's
        tuning-year numbers, learning curves, and the selected variant (highest
        mean best-checkpoint tuning-year reward). No test years are used.

    python -m src.tuning test --round-dir /scratch/$USER/restock-rl/runs/tuning1
        Re-derives the selection from the tuning-year scores alone (there is no
        option to name a variant), then evaluates that variant's selected
        checkpoints ONCE on the 50 test years, next to the baseline. Refuses to
        run a second time (marker file in the round directory).

All numbers are true reward units (uncentered, unscaled). Outputs go to
results/tuning1/<timestamp>/ (not committed).
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
from src.compare import GRID, REPLICATE, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, load_monitor, mean_std, rolling_mean
from src.evaluate import evaluate, seed_range
from src.train_dqn import load_run_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tuning1"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "tuning_round_1"
VARIANTS = ["v0", "v1", "v2", "v3"]
SEEDS_PER_VARIANT = 3
TEST_MARKER = "TEST_EVALUATED.json"

# Categorical slots 1-4 of the reference palette, fixed order V0..V3.
VARIANT_COLORS = {"v0": "#2a78d6", "v1": "#eb6834", "v2": "#1baf7a", "v3": "#eda100"}
VARIANT_LABELS = {
    "v0": "V0 current setup",
    "v1": "V1 + reward centering",
    "v2": "V2 + lr 5e-5, target 20k, batch 128",
    "v3": "V3 + exploration 0.3",
}

METRICS = [
    ("mean_total_reward", "365-day reward", "{:,.0f}"),
    ("fill_rate_a", "Fill rate A", "{:.1%}"),
    ("fill_rate_b", "Fill rate B", "{:.1%}"),
    ("fill_rate", "Fill rate overall", "{:.1%}"),
    ("avg_stock_depot", "Avg stock depot", "{:.1f}"),
    ("avg_stock_a", "Avg stock A", "{:.1f}"),
    ("avg_stock_b", "Avg stock B", "{:.1f}"),
]


# ------------------------------------------------------------------ loading
def load_scores(run: Path) -> list[dict[str, float]]:
    with open(run / "checkpoint_scores.csv") as f:
        return [{k: (v if k == "checkpoint" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def load_round(round_dir: Path, require_complete: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Per variant: list of runs with their scores, best checkpoint and metadata."""
    out: dict[str, list[dict[str, Any]]] = {}
    for variant in VARIANTS:
        runs = []
        for run in sorted((round_dir / variant).glob("seed_index_*")):
            if not (run / "best_checkpoint.json").exists():
                continue
            meta = json.loads((run / "run.json").read_text())
            best = json.loads((run / "best_checkpoint.json").read_text())
            if best["selected_on"] != "tuning":
                raise ValueError(f"{run}: selection not made on the tuning years")
            runs.append({"dir": run, "seed": meta["seed"], "meta": meta, "best": best["best"],
                         "scores": load_scores(run)})
        if require_complete and len(runs) != SEEDS_PER_VARIANT:
            raise RuntimeError(f"{variant}: {len(runs)} finished runs, expected {SEEDS_PER_VARIANT}")
        if runs:
            out[variant] = runs
    return out


def select_variant(round_data: dict[str, list[dict[str, Any]]]) -> str:
    """Variant with the highest mean (over seeds) best-checkpoint TUNING-year reward."""
    means = {v: float(np.mean([r["best"]["mean_total_reward"] for r in runs])) for v, runs in round_data.items()}
    return max(VARIANTS, key=lambda v: means.get(v, -np.inf))


# ------------------------------------------------------------------- tables
def variant_table(round_data: dict[str, list[dict[str, Any]]], baseline: dict[str, Any]) -> str:
    head = "| Variant | " + " | ".join(label for _, label, _ in METRICS) + " | Best checkpoint (steps) per seed |"
    lines = [head, "|" + "---|" * (len(METRICS) + 2)]
    for variant, runs in round_data.items():
        cells = []
        for key, _, fmt in METRICS:
            m, s = mean_std([r["best"][key] for r in runs])
            cells.append(f"{fmt.format(m)} ± {fmt.format(s)}")
        steps = ", ".join(f"{int(r['best']['steps']) // 1000}k" for r in runs)
        lines.append(f"| {VARIANT_LABELS[variant]} | " + " | ".join(cells) + f" | {steps} |")
    lines.append("| **Baseline (60 / 24 / 32)** | " + " | ".join(fmt.format(baseline[k]) for k, _, fmt in METRICS)
                 + " | — |")
    return "\n".join(lines)


def per_seed_table(round_data: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["| Variant | Training seed | Best checkpoint | Best tuning reward | Fill overall | Final (2M) tuning reward |",
             "|---|---|---|---|---|---|"]
    for variant, runs in round_data.items():
        for r in runs:
            final = max(r["scores"], key=lambda row: row["steps"])
            lines.append(f"| {variant.upper()} | {r['seed']} | {int(r['best']['steps']):,} | "
                         f"{r['best']['mean_total_reward']:,.0f} | {r['best']['fill_rate']:.1%} | "
                         f"{final['mean_total_reward']:,.0f} |")
    return "\n".join(lines)


# -------------------------------------------------------------------- plots
def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def _end_labels(ax, ends: list[tuple[float, float, str, str]], min_gap: float) -> None:
    """Direct labels at line ends (text ink, small colored key), nudged apart vertically."""
    ends = sorted(ends, key=lambda e: e[1])
    placed: list[float] = []
    for x, y, text, color in ends:
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(f"■ ", xy=(x, y_label), xytext=(6, 0), textcoords="offset points", va="center",
                    color=color, fontsize=8, annotation_clip=False)
        ax.annotate(text, xy=(x, y_label), xytext=(16, 0), textcoords="offset points", va="center",
                    color=TEXT_PRIMARY, fontsize=8, annotation_clip=False)


def plot_learning_curves(round_data, baseline_reward: float, out: Path, window: int = 50) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ends = []
    for variant, runs in round_data.items():
        curves = [rolling_mean(load_monitor(r["dir"]), window) for r in runs]
        n = min(len(c) for c in curves)
        mean_curve = np.mean([c[:n] for c in curves], axis=0) / 1000
        x = (np.arange(n) + window) * 365 / 1e6
        ax.plot(x, mean_curve, color=VARIANT_COLORS[variant], linewidth=2, label=VARIANT_LABELS[variant])
        ends.append((x[-1], float(mean_curve[-1]), variant.upper(), VARIANT_COLORS[variant]))
    ax.axhline(baseline_reward / 1000, color=TEXT_SECONDARY, linewidth=1.5, linestyle=(0, (5, 4)))
    ax.annotate(f"Baseline on tuning years ({baseline_reward / 1000:.1f}k)",
                xy=(ax.get_xlim()[0], baseline_reward / 1000),
                xytext=(2, 6), textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)
    _end_labels(ax, ends, min_gap=(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.035)
    _style(ax, f"Training reward per simulated year, mean of {SEEDS_PER_VARIANT} seeds "
               f"({window}-year rolling mean, ε-greedy, true units)",
           "Training steps (millions)", "Reward per year (thousands)")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 0.96, 1))  # room for the end labels
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_checkpoint_scores(round_data, baseline_reward: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = list(round_data)
    fig, axes = plt.subplots(1, len(variants), figsize=(3.2 * len(variants), 3.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, variant in zip(axes, variants):
        for r in round_data[variant]:
            steps = np.array([row["steps"] for row in r["scores"]]) / 1e6
            vals = np.array([row["mean_total_reward"] for row in r["scores"]]) / 1000
            ax.plot(steps, vals, color=VARIANT_COLORS[variant], linewidth=1.2, marker="o", markersize=2.5)
            b = r["best"]
            ax.scatter([b["steps"] / 1e6], [b["mean_total_reward"] / 1000], s=40, facecolor="none",
                       edgecolor=TEXT_PRIMARY, linewidth=1, zorder=4)
        ax.axhline(baseline_reward / 1000, color=TEXT_SECONDARY, linewidth=1.2, linestyle=(0, (5, 4)))
        _style(ax, VARIANT_LABELS[variant], "Training steps (millions)", "")
        ax.title.set_fontsize(9)
    axes[0].set_ylabel("Greedy reward on tuning years (thousands)", color=TEXT_SECONDARY, fontsize=9)
    axes[0].annotate(f"baseline {baseline_reward / 1000:.1f}k", xy=(axes[0].get_xlim()[0], baseline_reward / 1000),
                     xytext=(2, 4), textcoords="offset points", color=TEXT_SECONDARY, fontsize=8)
    fig.suptitle("Every checkpoint scored greedily on the 20 tuning years (one line per seed; ○ = selected)",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- commands
def _copy_run_files(round_data, out_dir: Path) -> None:
    for variant, runs in round_data.items():
        for r in runs:
            keep = out_dir / "runs" / variant / r["dir"].name
            keep.mkdir(parents=True, exist_ok=True)
            for name in ("checkpoint_scores.csv", "best_checkpoint.json", "run.json", "monitor.monitor.csv"):
                shutil.copy2(r["dir"] / name, keep / name)


def cmd_summarize(args) -> None:
    config = load_run_config(CONFIG_DIR / "v0.yaml")
    round_data = load_round(args.round_dir, require_complete=not args.allow_incomplete)
    base = evaluate(tuned_policy(config), seed_range(config, "tuning"), config)["summary"]
    selected = select_variant(round_data)

    out_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = variant_table(round_data, base)
    seeds = per_seed_table(round_data)
    plot_learning_curves(round_data, base["mean_total_reward"], out_dir / "learning_curves.png", args.window)
    plot_checkpoint_scores(round_data, base["mean_total_reward"], out_dir / "checkpoint_scores.png")
    _copy_run_files(round_data, out_dir)

    summary = {
        "selected_on": "tuning", "tuning_seeds": [seed_range(config, "tuning")[0], seed_range(config, "tuning")[-1]],
        "selected_variant": selected, "baseline_tuning": base,
        "variants": {v: {"seeds": [r["seed"] for r in runs],
                         "best": [r["best"] for r in runs],
                         "mean_best_reward": float(np.mean([r["best"]["mean_total_reward"] for r in runs])),
                         "train_seconds": [r["meta"]["train_seconds"] for r in runs]}
                     for v, runs in round_data.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "summary.md").write_text(f"{table}\n\n{seeds}\n\nSelected variant: {selected.upper()}\n")
    print(table, seeds, f"Selected variant (tuning years only): {selected.upper()}", f"saved to {out_dir}",
          sep="\n\n")


def cmd_test(args) -> None:
    import torch
    from stable_baselines3 import DQN

    from src.train_dqn import GreedyDQNPolicy

    torch.set_num_threads(1)
    marker = args.round_dir / TEST_MARKER
    if marker.exists():
        raise SystemExit(f"test years were already used for this round ({marker}); refusing to run again")
    round_data = load_round(args.round_dir, require_complete=True)
    selected = select_variant(round_data)  # from tuning-year scores only
    config = load_run_config(CONFIG_DIR / f"{selected}.yaml")
    test_seeds = seed_range(config, "test")

    out_dir = RESULTS_DIR / (datetime.now().strftime("%Y%m%d-%H%M%S") + "_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = evaluate(tuned_policy(config), test_seeds, config)
    save_rows(base["episodes"], out_dir / "baseline_test_years.csv")

    dqn, per_year, used = [], [], []
    for r in round_data[selected]:
        ckpt = r["dir"] / "checkpoints" / r["best"]["checkpoint"]
        result = evaluate(GreedyDQNPolicy(DQN.load(ckpt, device="cpu"), config), test_seeds, config)
        save_rows(result["episodes"], out_dir / f"dqn_{selected}_{r['dir'].name}_test_years.csv")
        keep = out_dir / "models" / r["dir"].name
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, keep / ckpt.name)
        dqn.append(result["summary"])
        per_year.append([e["total_reward"] for e in result["episodes"]])
        used.append({"seed": r["seed"], "checkpoint": str(ckpt),
                     "tuning_reward": r["best"]["mean_total_reward"],
                     "test_reward": result["summary"]["mean_total_reward"]})
        print(f"{r['seed']} {ckpt.name}: test {result['summary']['mean_total_reward']:,.0f}", flush=True)

    base_years = np.array([e["total_reward"] for e in base["episodes"]])
    diff = np.array(per_year).mean(axis=0) - base_years
    paired = {"mean_difference_per_year": float(diff.mean()), "std_difference": float(diff.std(ddof=1)),
              "years_dqn_better": int((diff > 0).sum()),
              "per_seed_years_better": [int((np.array(row) > base_years).sum()) for row in per_year]}
    lines = [f"| Metric (50 test years) | {selected.upper()} best checkpoints, mean ± std over {len(dqn)} seeds "
             f"| Baseline (60 / 24 / 32) |", "|---|---|---|"]
    for key, label, fmt in METRICS:
        m, s = mean_std([d[key] for d in dqn])
        lines.append(f"| {label} | {fmt.format(m)} ± {fmt.format(s)} | {fmt.format(base['summary'][key])} |")
    table = "\n".join(lines)
    record = {"evaluated_at": datetime.now().isoformat(timespec="seconds"), "selected_variant": selected,
              "selected_on": "tuning", "test_seeds": [test_seeds[0], test_seeds[-1]], "checkpoints": used,
              "baseline_test": base["summary"], "dqn_test": dqn, "paired": paired, "results_dir": str(out_dir)}
    (out_dir / "test.json").write_text(json.dumps(record, indent=2, default=float))
    (out_dir / "test.md").write_text(table + "\n")
    marker.write_text(json.dumps(record, indent=2, default=float))
    print(table, json.dumps(paired, indent=2), f"saved to {out_dir}", sep="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarize", help="tuning-year summary, plots and variant selection")
    s.add_argument("--round-dir", type=Path, required=True)
    s.add_argument("--window", type=int, default=50)
    s.add_argument("--allow-incomplete", action="store_true", help="preview with unfinished runs")
    t = sub.add_parser("test", help="evaluate the selected variant once on the test years")
    t.add_argument("--round-dir", type=Path, required=True)
    args = parser.parse_args()
    cmd_summarize(args) if args.command == "summarize" else cmd_test(args)


if __name__ == "__main__":
    main()
