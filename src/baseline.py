"""Order-up-to baseline (D8–D11) and its grid search (D12).

The baseline involves no learning: only three target numbers (depot,
Store A, Store B) are tuned.

Decision at the end of each day, from the observation alone:
1. Store top-up (D9): need = target - (shelf stock + shipments on the way).
2. Round each need to the nearest allowed shipment, ties up, within
   0..32 (D10): 6 -> 8, 25 -> 24.
3. Work out what will actually leave the depot, using the same
   proportional round-down split as the environment (D11, D4).
4. Depot order (D8): need = target - (depot stock after today's shipments
   + supplier orders on the way); store stock is not counted.
5. Round to the nearest allowed order, ties up, within 0..40 (D10): 25 -> 30.

Usage:
    python -m src.baseline time                  # estimate grid-search runtime
    python -m src.baseline grid-search           # full grid (D12)
    python -m src.baseline grid-search --coarse  # coarse pass, then fine pass
    python -m src.baseline evaluate --seeds test # tuned targets on the 50 test years
    python -m src.baseline obs-ranges            # observation ranges (for D16 scaling)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from bisect import bisect_left
from collections.abc import Sequence
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from src.env import RestockEnv, load_config, split_shipments
from src.evaluate import evaluate, seed_range

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "grid_search"


def round_to_grid(x: float, grid: Sequence[int]) -> int:
    """Nearest value in the sorted ``grid``; ties go to the larger value.

    Values below the grid give its minimum, values above give its maximum (cap).
    """
    if x <= grid[0]:
        return grid[0]
    if x >= grid[-1]:
        return grid[-1]
    i = bisect_left(grid, x)  # grid[i-1] < x <= grid[i]
    lo, hi = grid[i - 1], grid[i]
    return hi if (x - lo) >= (hi - x) else lo


class OrderUpToPolicy:
    """Order-up-to policy with targets for the depot, Store A and Store B."""

    def __init__(self, depot_target: int, store_a_target: int, store_b_target: int,
                 config: dict[str, Any] | None = None):
        self.depot_target = int(depot_target)
        self.store_a_target = int(store_a_target)
        self.store_b_target = int(store_b_target)
        # Used only for the action grids, observation layout and action encoding.
        self._spec = RestockEnv(config if config is not None else load_config())

    def decide(self, obs) -> tuple[int, int, int]:
        """Return (depot order, requested shipment A, requested shipment B)."""
        s = self._spec.parse_observation(obs)

        need_a = self.store_a_target - (s["store_a"] + sum(s["pipe_a"]))
        need_b = self.store_b_target - (s["store_b"] + sum(s["pipe_b"]))
        request_a = round_to_grid(need_a, self._spec.ship_a_grid)
        request_b = round_to_grid(need_b, self._spec.ship_b_grid)

        ship_a, ship_b = split_shipments(s["depot"], request_a, request_b)
        depot_after_shipping = s["depot"] - ship_a - ship_b

        need_depot = self.depot_target - (depot_after_shipping + sum(s["pipe_depot"]))
        order = round_to_grid(need_depot, self._spec.depot_grid)
        return order, request_a, request_b

    def __call__(self, obs) -> int:
        return self._spec.encode_action(*self.decide(obs))


# ---------------------------------------------------------------- grid search
def inclusive_range(spec: dict[str, int]) -> list[int]:
    return list(range(int(spec["start"]), int(spec["stop"]) + 1, int(spec["step"])))


def candidate_grid(config: dict[str, Any], coarse_factor: int = 1) -> dict[str, list[int]]:
    """Target values per location from the config; every ``coarse_factor``-th
    value (always keeping both ends) for a coarse pass."""
    grids = {}
    for key in ("depot_target", "store_a_target", "store_b_target"):
        values = inclusive_range(config["grid_search"][key])
        if coarse_factor > 1:
            coarse = values[::coarse_factor]
            if coarse[-1] != values[-1]:
                coarse.append(values[-1])
            values = coarse
        grids[key] = values
    return grids


_worker: dict[str, Any] = {}


def _init_worker(config: dict[str, Any], seeds: list[int], niceness: int) -> None:
    if niceness:
        try:
            os.nice(niceness)
        except OSError:
            pass
    _worker["config"] = config
    _worker["seeds"] = seeds


def evaluate_targets(targets: tuple[int, int, int]) -> dict[str, Any]:
    """Evaluate one (depot, A, B) target triple on the tuning seeds."""
    config, seeds = _worker["config"], _worker["seeds"]
    policy = OrderUpToPolicy(*targets, config=config)
    summary = evaluate(policy, seeds, config)["summary"]
    return {"depot_target": targets[0], "store_a_target": targets[1],
            "store_b_target": targets[2], **summary}


def run_grid(candidates: list[tuple[int, int, int]], config: dict[str, Any], seeds: list[int],
             workers: int, niceness: int, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = time.time()
    with Pool(workers, initializer=_init_worker, initargs=(config, seeds, niceness)) as pool:
        for i, row in enumerate(pool.imap_unordered(evaluate_targets, candidates, chunksize=8), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(candidates):
                elapsed = time.time() - start
                print(f"[{label}] {i}/{len(candidates)} candidates, {elapsed / 60:.1f} min", flush=True)
    rows.sort(key=lambda r: (r["depot_target"], r["store_a_target"], r["store_b_target"]))
    return rows


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Highest mean undiscounted 365-day reward on the tuning seeds."""
    return max(rows, key=lambda r: r["mean_total_reward"])


def edge_report(best: dict[str, Any], grids: dict[str, list[int]]) -> dict[str, str | None]:
    report = {}
    for key, values in grids.items():
        if best[key] == values[0]:
            report[key] = "lower edge"
        elif best[key] == values[-1]:
            report[key] = "upper edge"
        else:
            report[key] = None
    return report


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def time_estimate(config: dict[str, Any], seeds: list[int], n_candidates: int = 12) -> float:
    """Seconds per candidate on one core (median-free simple average)."""
    _init_worker(config, seeds, 0)
    grids = candidate_grid(config)
    sample = list(itertools.product(grids["depot_target"][::6], grids["store_a_target"][::6],
                                    grids["store_b_target"][::6]))[:n_candidates]
    start = time.time()
    for t in sample:
        evaluate_targets(t)
    return (time.time() - start) / len(sample)


def fine_candidates(best: dict[str, Any], config: dict[str, Any], coarse_factor: int) -> list[tuple[int, int, int]]:
    """All full-resolution values within one coarse step of the coarse best."""
    full = candidate_grid(config)
    near = {}
    for key, values in full.items():
        step = values[1] - values[0]
        radius = step * coarse_factor
        near[key] = [v for v in values if abs(v - best[key]) <= radius]
    return list(itertools.product(near["depot_target"], near["store_a_target"], near["store_b_target"]))


def tuned_policy(config: dict[str, Any]) -> OrderUpToPolicy:
    """Order-up-to policy with the tuned targets from ``config["baseline"]``."""
    b = config["baseline"]
    return OrderUpToPolicy(b["depot_target"], b["store_a_target"], b["store_b_target"], config)


def observation_ranges(policy, seeds: Sequence[int], config: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Min, max and 99th percentile of every observation feature while ``policy`` runs."""
    import numpy as np

    env = RestockEnv(config)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        rows.append(obs)
        truncated = False
        while not truncated:
            obs, _, _, truncated, _ = env.step(policy(obs))
            rows.append(obs)
    data = np.array(rows)
    names = OBS_NAMES[: data.shape[1]]
    return {
        name: {"min": float(data[:, i].min()), "p99": float(np.percentile(data[:, i], 99)),
               "max": float(data[:, i].max())}
        for i, name in enumerate(names)
    }


OBS_NAMES = ["depot_stock", "store_a_stock", "store_b_stock", "depot_arriving_next_day",
             "store_b_arriving_in_1_day", "store_b_arriving_in_2_days"]


def run_tuned(command: str, seeds_name: str, config: dict[str, Any]) -> None:
    seeds = seed_range(config, seeds_name)
    policy = tuned_policy(config)
    targets = (policy.depot_target, policy.store_a_target, policy.store_b_target)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR.parent / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    if command == "evaluate":
        result = evaluate(policy, seeds, config)
        save_rows(result["episodes"], out_dir / f"{stamp}_{seeds_name}_years.csv")
        out = {"targets": targets, "seeds": [seeds[0], seeds[-1]], "summary": result["summary"]}
        path = out_dir / f"{stamp}_{seeds_name}_summary.json"
    else:
        out = {"targets": targets, "seeds": [seeds[0], seeds[-1]],
               "ranges": observation_ranges(policy, seeds, config)}
        path = out_dir / f"{stamp}_{seeds_name}_obs_ranges.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print(f"saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("time", help="estimate the full grid-search runtime")
    gs = sub.add_parser("grid-search", help="run the grid search on the tuning seeds")
    gs.add_argument("--workers", type=int, default=4)
    gs.add_argument("--nice", type=int, default=10, help="lower the CPU priority of workers")
    gs.add_argument("--coarse", type=int, default=0, metavar="FACTOR",
                    help="coarse pass using every FACTOR-th value, then a fine pass around the best")
    ev = sub.add_parser("evaluate", help="evaluate the tuned targets from the config")
    ev.add_argument("--seeds", default="test", help="seed range name in the config")
    rg = sub.add_parser("obs-ranges", help="observation ranges of the tuned baseline")
    rg.add_argument("--seeds", default="tuning", help="seed range name in the config")
    args = parser.parse_args()

    config = load_config()
    if args.command in ("evaluate", "obs-ranges"):
        run_tuned(args.command, args.seeds, config)
        return
    seeds = seed_range(config, "tuning")
    grids = candidate_grid(config)
    n_full = len(grids["depot_target"]) * len(grids["store_a_target"]) * len(grids["store_b_target"])

    if args.command == "time":
        per = time_estimate(config, seeds)
        print(f"Tuning seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} years per candidate)")
        print(f"Full grid: {n_full} candidates")
        print(f"{per:.3f} s per candidate on one core -> {per * n_full / 3600:.2f} h on one core")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Tuning seeds {seeds[0]}..{seeds[-1]}, {args.workers} workers", flush=True)
    if args.coarse > 1:
        coarse_grids = candidate_grid(config, args.coarse)
        coarse = list(itertools.product(*coarse_grids.values()))
        rows = run_grid(coarse, config, seeds, args.workers, args.nice, "coarse")
        save_rows(rows, RESULTS_DIR / f"{stamp}_coarse.csv")
        fine = fine_candidates(best_row(rows), config, args.coarse)
        rows = run_grid(fine, config, seeds, args.workers, args.nice, "fine")
        save_rows(rows, RESULTS_DIR / f"{stamp}_fine.csv")
    else:
        rows = run_grid(list(itertools.product(*grids.values())), config, seeds,
                        args.workers, args.nice, "full")
        save_rows(rows, RESULTS_DIR / f"{stamp}_full.csv")

    best = best_row(rows)
    edges = edge_report(best, grids)
    out = {"best": best, "edges": edges, "tuning_seeds": [seeds[0], seeds[-1]],
           "grid": config["grid_search"], "mode": "coarse+fine" if args.coarse > 1 else "full"}
    with open(RESULTS_DIR / f"{stamp}_best.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
