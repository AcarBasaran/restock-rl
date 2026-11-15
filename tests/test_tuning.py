"""Tests for tuning-round selection logic (no training)."""

import json
from argparse import Namespace

import pytest

pytest.importorskip("stable_baselines3")

from src import tuning  # noqa: E402


def _runs(*rewards):
    return [{"best": {"mean_total_reward": r}} for r in rewards]


def test_select_variant_uses_mean_best_checkpoint_reward():
    data = {"v0": _runs(80, 81, 82), "v1": _runs(95, 60, 60), "v2": _runs(85, 85, 86), "v3": _runs(70, 70, 70)}
    # v1 has the single best seed but v2 the best mean.
    assert tuning.select_variant(data) == "v2"


def test_select_variant_tie_goes_to_earlier_variant():
    assert tuning.select_variant({"v0": _runs(1, 2, 3), "v1": _runs(3, 2, 1)}) == "v0"


def _fake_run(path, seed, selected_on="tuning"):
    path.mkdir(parents=True)
    best = {"steps": 100000, "checkpoint": "dqn_100000_steps.zip", "mean_total_reward": 1.0}
    (path / "best_checkpoint.json").write_text(json.dumps({"selected_on": selected_on, "best": best}))
    (path / "run.json").write_text(json.dumps({"seed": seed, "train_seconds": 1.0}))
    (path / "checkpoint_scores.csv").write_text("steps,checkpoint,mean_total_reward\n100000,dqn_100000_steps.zip,1.0\n")


def test_load_round_requires_three_seeds_per_variant(tmp_path):
    for v in tuning.VARIANTS:
        for i in range(3 if v != "v2" else 2):
            _fake_run(tmp_path / v / f"seed_index_{i}", 200000 + i)
    with pytest.raises(RuntimeError, match="v2"):
        tuning.load_round(tmp_path)
    assert len(tuning.load_round(tmp_path, require_complete=False)["v2"]) == 2


def test_load_round_rejects_selection_not_made_on_tuning_years(tmp_path):
    _fake_run(tmp_path / "v0" / "seed_index_0", 200000, selected_on="test")
    with pytest.raises(ValueError, match="tuning"):
        tuning.load_round(tmp_path, require_complete=False)


def test_test_command_refuses_second_use_of_test_years(tmp_path):
    (tmp_path / tuning.TEST_MARKER).write_text("{}")
    with pytest.raises(SystemExit, match="already used"):
        tuning.cmd_test(Namespace(round_dir=tmp_path))
