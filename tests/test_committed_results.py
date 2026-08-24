from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_committed_baselines_are_validation_only() -> None:
    result = json.loads(
        (ROOT / "evaluation/results/grouped_v1_baselines.json").read_text(encoding="utf-8")
    )
    assert result["evaluated_splits"] == ["train", "validation"]
    assert result["frozen_test_loaded"] is False
    assert result["bootstrap_resamples"] >= 1000
    assert all(run["evaluated_split"] == "validation" for run in result["runs"])
    assert all("test_metrics" not in run for run in result["runs"])
    assert all(run["validation_group_bootstrap"] for run in result["runs"])
    assert all(run["slice_metrics"] for run in result["runs"])


def test_no_baseline_only_selection_receipt_is_published() -> None:
    assert not (
        ROOT / "evaluation/selection/grouped_v1_baseline_selection_draft.json"
    ).exists()
