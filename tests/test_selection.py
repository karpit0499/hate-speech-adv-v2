import json
from pathlib import Path
import subprocess

import pytest

from hate_speech_adv.integrity import canonical_json_sha256, file_sha256, payload_sha256, semantic_file_sha256
from hate_speech_adv.manifest import load_manifest, verify_protocol_commit_lock
from hate_speech_adv.selection import (
    build_protocol_commit_lock,
    create_selection_draft,
    verify_commit_is_ancestor,
)
from hate_speech_adv.training_plan import SECURE_TRAINING_STACK, load_matrix

ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/train/experiment_matrix.json"
MANIFEST_PATH = ROOT / "evaluation/manifests/grouped_v1.json"
SLICES_PATH = ROOT / "configs/evaluation/slices_v1.json"


def _complete_metrics(value: float, supports: dict[str, int]) -> dict[str, object]:
    labels = ("hate_speech", "offensive", "neither")
    n = sum(supports[label] for label in labels)
    return {
        "n": n,
        "accuracy": value,
        "macro_precision": value,
        "macro_recall": value,
        "macro_f1": value,
        "weighted_f1": value,
        "hate_speech_precision": 0.40,
        "hate_speech_recall": 0.60,
        "hate_speech_f1": 0.48,
        "per_class": {
            label: {
                "precision": value,
                "recall": value,
                "f1": value,
                "support": supports[label],
            }
            for label in labels
        },
        "confusion_matrix_rows_true_columns_predicted": [
            [supports["hate_speech"], 0, 0],
            [0, supports["offensive"], 0],
            [0, 0, supports["neither"]],
        ],
    }


def _bootstrap(*, estimate: float, resamples: int, seed: int) -> dict[str, object]:
    return {
        "estimate": estimate,
        "confidence": 0.95,
        "lower": max(0.0, estimate - 0.01),
        "upper": min(1.0, estimate + 0.01),
        "bootstrap_mean": estimate,
        "bootstrap_standard_error": 0.005,
        "resamples": resamples,
        "seed": seed,
        "groups": 10,
    }


def _complete_result_set(tmp_path: Path) -> list[Path]:
    matrix = load_matrix(MATRIX_PATH)
    manifest = load_manifest(MANIFEST_PATH)
    slices = json.loads(SLICES_PATH.read_text(encoding="utf-8"))
    validation_support = manifest["summary"]["split_support"]["validation"]
    software = {
        "python": manifest["protocol"]["generator"]["python"],
        "numpy": manifest["protocol"]["generator"]["numpy"],
        "torch": SECURE_TRAINING_STACK["torch"],
        "transformers": SECURE_TRAINING_STACK["transformers"],
        "peft": SECURE_TRAINING_STACK["peft"],
        "datasets": manifest["protocol"]["generator"]["datasets"],
    }
    paths: list[Path] = []
    for candidate_index, candidate in enumerate(matrix["candidates"]):
        for seed in matrix["seeds"]:
            candidate_id = candidate["id"]
            value = 0.60 + candidate_index / 100 + seed / 1_000_000
            checkpoint_files = [
                {"path": "config.json", "size_bytes": 2, "sha256": "a" * 64},
                {"path": "model.safetensors", "size_bytes": 3, "sha256": "b" * 64},
                {"path": "tokenizer_config.json", "size_bytes": 4, "sha256": "c" * 64},
            ]
            slice_metrics = {
                category: {
                    name: {
                        "support": 1,
                        "metrics": _complete_metrics(
                            value,
                            {"hate_speech": 1, "offensive": 0, "neither": 0},
                        ),
                    }
                    for name in slices[category]
                }
                for category in ("ambiguity_slices", "hard_negative_slices")
            }
            document = {
                "schema_version": 2,
                "protocol_id": "grouped_v1",
                "run_id": f"{candidate_id}__seed_{seed}",
                "candidate_id": candidate_id,
                "seed": seed,
                "evaluated_split": "validation",
                "frozen_test_loaded": False,
                "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
                "matrix_sha256": file_sha256(MATRIX_PATH),
                "matrix_semantic_sha256": semantic_file_sha256(MATRIX_PATH),
                "slices_sha256": file_sha256(SLICES_PATH),
                "slices_semantic_sha256": semantic_file_sha256(SLICES_PATH),
                "candidate": candidate,
                "encoder": matrix["encoders"][candidate["encoder"]],
                "software": software,
                "training_hyperparameters": {
                    **matrix["defaults"],
                    "learning_rate": matrix["adaptation"][candidate["adaptation"]]["learning_rate"],
                },
                "metrics": _complete_metrics(value, validation_support),
                "grouped_bootstrap": {
                    "macro_f1": _bootstrap(
                        estimate=value,
                        resamples=matrix["selection"]["required_bootstrap_resamples"],
                        seed=seed,
                    ),
                    "hate_speech_recall": _bootstrap(
                        estimate=0.60,
                        resamples=matrix["selection"]["required_bootstrap_resamples"],
                        seed=seed,
                    ),
                },
                "slice_metrics": slice_metrics,
                "checkpoint": "selected_checkpoint",
                "checkpoint_format": (
                    "merged_full_model" if candidate["adaptation"] == "lora" else "full_model"
                ),
                "checkpoint_artifacts": {
                    "files": checkpoint_files,
                    "tree_sha256": canonical_json_sha256(checkpoint_files),
                    "size_bytes": 9,
                },
            }
            document["integrity"] = {"payload_sha256": payload_sha256(document)}
            path = tmp_path / f"{candidate_id}__seed_{seed}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            paths.append(path)
    return paths


def test_selection_requires_and_uses_all_27_frozen_runs(tmp_path):
    paths = _complete_result_set(tmp_path)
    receipt = create_selection_draft(
        paths,
        matrix_path=MATRIX_PATH,
        manifest_path=MANIFEST_PATH,
        slices_path=SLICES_PATH,
        repository_root=tmp_path,
    )
    assert receipt["observed_unique_runs"] == receipt["required_runs"] == 27
    assert receipt["selection_locked"] is False
    assert receipt["frozen_test_evaluated"] is False
    assert receipt["selected"]["production_seed"] == 42
    assert receipt["selected"]["production_run_id"].endswith("__seed_42")
    assert len(receipt["candidate_summaries"]) == 9


def test_selection_rejects_partial_or_duplicate_matrix(tmp_path):
    paths = _complete_result_set(tmp_path)
    with pytest.raises(ValueError, match="exactly 27"):
        create_selection_draft(
            paths[:-1],
            matrix_path=MATRIX_PATH,
            manifest_path=MANIFEST_PATH,
            slices_path=SLICES_PATH,
            repository_root=tmp_path,
        )
    with pytest.raises(ValueError, match="duplicate candidate/seed"):
        create_selection_draft(
            paths + [paths[0]],
            matrix_path=MATRIX_PATH,
            manifest_path=MANIFEST_PATH,
            slices_path=SLICES_PATH,
            repository_root=tmp_path,
        )


def test_selection_rejects_tampered_result_integrity(tmp_path):
    paths = _complete_result_set(tmp_path)
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    document["metrics"]["macro_f1"] = 0.99
    paths[0].write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash"):
        create_selection_draft(
            paths,
            matrix_path=MATRIX_PATH,
            manifest_path=MANIFEST_PATH,
            slices_path=SLICES_PATH,
            repository_root=tmp_path,
        )


def test_selection_rejects_fabricated_incomplete_evidence(tmp_path):
    paths = _complete_result_set(tmp_path)
    original = json.loads(paths[0].read_text(encoding="utf-8"))

    def missing_encoder(document):
        document.pop("encoder")

    def altered_software(document):
        document["software"]["torch"] = "0.0.0"

    def incomplete_checkpoint(document):
        files = [
            item
            for item in document["checkpoint_artifacts"]["files"]
            if item["path"] != "tokenizer_config.json"
        ]
        document["checkpoint_artifacts"] = {
            "files": files,
            "tree_sha256": canonical_json_sha256(files),
            "size_bytes": sum(item["size_bytes"] for item in files),
        }

    def missing_bootstrap(document):
        document["grouped_bootstrap"].pop("hate_speech_recall")

    def incomplete_slices(document):
        document["slice_metrics"]["ambiguity_slices"].pop("disputed")

    cases = (
        (missing_encoder, "encoder/revision"),
        (altered_software, "software stack"),
        (incomplete_checkpoint, "standalone model tree"),
        (missing_bootstrap, "exact grouped-bootstrap metrics"),
        (incomplete_slices, "slice topology"),
    )
    for mutate, message in cases:
        document = json.loads(json.dumps(original))
        mutate(document)
        document["integrity"] = {"payload_sha256": payload_sha256(document)}
        paths[0].write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            create_selection_draft(
                paths,
                matrix_path=MATRIX_PATH,
                manifest_path=MANIFEST_PATH,
                slices_path=SLICES_PATH,
                repository_root=tmp_path,
            )


def _git(tmp_path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_protocol_lock_rejects_current_byte_bypass_and_omitted_gate(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "protocol@example.invalid")
    _git(tmp_path, "config", "user.name", "Protocol Test")
    gate = tmp_path / "open_gate.py"
    gate.write_text("ALLOW = False\n", encoding="utf-8")
    _git(tmp_path, "add", "open_gate.py")
    _git(tmp_path, "commit", "-m", "protocol")
    commit = _git(tmp_path, "rev-parse", "HEAD")
    lock = build_protocol_commit_lock(tmp_path, commit, paths=("open_gate.py",))
    receipt = {"protocol_lock": lock}
    verify_protocol_commit_lock(receipt, tmp_path, required_paths=("open_gate.py",))
    with pytest.raises(PermissionError, match="complete ordered"):
        verify_protocol_commit_lock(
            receipt,
            tmp_path,
            required_paths=("open_gate.py", "evaluate_frozen.py"),
        )
    gate.write_text("ALLOW = True\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="current protocol bytes"):
        verify_protocol_commit_lock(receipt, tmp_path, required_paths=("open_gate.py",))


def test_commit_ancestry_rejects_unreachable_orphan_objects(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "protocol@example.invalid")
    _git(tmp_path, "config", "user.name", "Protocol Test")
    path = tmp_path / "protocol.txt"
    path.write_text("protocol\n", encoding="utf-8")
    _git(tmp_path, "add", "protocol.txt")
    _git(tmp_path, "commit", "-m", "protocol")
    protocol_commit = _git(tmp_path, "rev-parse", "HEAD")
    path.write_text("protocol plus results\n", encoding="utf-8")
    _git(tmp_path, "commit", "-am", "results")
    results_commit = _git(tmp_path, "rev-parse", "HEAD")
    tree = _git(tmp_path, "rev-parse", f"{results_commit}^{{tree}}")
    orphan = _git(tmp_path, "commit-tree", tree, "-m", "unreachable orphan")
    verify_commit_is_ancestor(tmp_path, protocol_commit, results_commit)
    with pytest.raises(PermissionError, match="ancestry gate failed"):
        verify_commit_is_ancestor(tmp_path, results_commit, orphan)
