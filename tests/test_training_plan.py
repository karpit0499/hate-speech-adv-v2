import json
from pathlib import Path

import pytest

from hate_speech_adv.integrity import payload_sha256
from hate_speech_adv.training_plan import ALLOWED_ADAPTATIONS, ALLOWED_LOSSES, build_run_plan, load_matrix, validate_matrix
from hate_speech_adv.transformer_pipeline import (
    prune_matrix_checkpoint_storage,
    require_production_seed_last,
)

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/train/experiment_matrix.json"
MANIFEST = ROOT / "evaluation/manifests/grouped_v1.json"


def test_matrix_covers_declared_models_adaptations_losses_and_seeds():
    matrix = load_matrix(MATRIX)
    validate_matrix(matrix)
    assert matrix["seeds"] == [13, 42, 73]
    assert {candidate["adaptation"] for candidate in matrix["candidates"]} == set(ALLOWED_ADAPTATIONS)
    assert {candidate["loss"] for candidate in matrix["candidates"]} == set(ALLOWED_LOSSES)
    assert {candidate["encoder"] for candidate in matrix["candidates"]} == {"current_roberta", "twitter_roberta"}
    assert matrix["selection"]["production_seed"] == 42
    assert matrix["selection"]["required_bootstrap_resamples"] >= 1000
    assert "learning_rate" not in matrix["defaults"]
    assert all(matrix["adaptation"][name]["learning_rate"] > 0 for name in ALLOWED_ADAPTATIONS)


def test_smoke_plan_has_27_validation_only_runs_and_all_losses():
    plan = build_run_plan(MATRIX, MANIFEST)
    assert len(plan["runs"]) == 27
    assert plan["frozen_test_loaded"] is False
    assert all(run["allowed_splits"] == ["train", "validation"] for run in plan["runs"])
    assert set(plan["loss_smoke"]) == set(ALLOWED_LOSSES)
    assert all(value > 0 for value in plan["loss_smoke"].values())
    assert plan["offline_training_smoke"]["steps"] == 1
    assert plan["offline_training_smoke"]["network_accessed"] is False
    assert plan["offline_training_smoke"]["loss_after"] < plan["offline_training_smoke"]["loss_before"]


def _write_result(path: Path, candidate_id: str, seed: int, macro_f1: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    document = {
        "candidate_id": candidate_id,
        "seed": seed,
        "metrics": {
            "macro_f1": macro_f1,
            "hate_speech_recall": macro_f1,
            "hate_speech_precision": 0.5,
        },
    }
    document["integrity"] = {"payload_sha256": payload_sha256(document)}
    (path / "run_result.json").write_text(json.dumps(document), encoding="utf-8")
    (path / "selected_checkpoint").mkdir()
    (path / "selected_checkpoint" / "weights.bin").write_bytes(b"weights")


def test_matrix_runner_keeps_only_complete_leader_seed_42_checkpoint(tmp_path):
    first = "current_lora_weighted_ce"
    with pytest.raises(ValueError, match="must run last"):
        require_production_seed_last(MATRIX, first, 42, tmp_path / first / "42")
    for seed in (13, 42, 73):
        _write_result(tmp_path / first / str(seed), first, seed, 0.6)
    assert require_production_seed_last(MATRIX, first, 42, tmp_path / first / "42") == tmp_path
    retention = prune_matrix_checkpoint_storage(MATRIX, tmp_path)
    assert retention["retained_checkpoint"] == f"{first}/42/selected_checkpoint"
    assert len(list(tmp_path.glob("*/*/selected_checkpoint"))) == 1

    second = "current_lora_focal"
    for seed in (13, 42, 73):
        _write_result(tmp_path / second / str(seed), second, seed, 0.7)
    retention = prune_matrix_checkpoint_storage(MATRIX, tmp_path)
    assert retention["retained_checkpoint"] == f"{second}/42/selected_checkpoint"
    assert len(list(tmp_path.glob("*/*/selected_checkpoint"))) == 1
