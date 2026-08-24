from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .constants import PROTOCOL_LOCK_PATHS
from .integrity import (
    canonical_json_sha256,
    file_sha256,
    payload_sha256,
    semantic_file_sha256,
    verify_payload_integrity,
)
from .manifest import load_manifest
from .training_plan import SECURE_TRAINING_STACK, load_matrix, validate_matrix


_COMPLETE_METRICS = {
    "n",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "hate_speech_precision",
    "hate_speech_recall",
    "hate_speech_f1",
    "per_class",
    "confusion_matrix_rows_true_columns_predicted",
}


def _finite_probability(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return number


def validate_complete_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_n: int | None = None,
    expected_support: Mapping[str, int] | None = None,
) -> None:
    if set(metrics) != _COMPLETE_METRICS:
        raise ValueError("metric report does not contain the exact frozen metric schema")
    n = int(metrics["n"])
    if n < 1 or (expected_n is not None and n != expected_n):
        raise ValueError("metric report support differs from the frozen split")
    for field in _COMPLETE_METRICS - {
        "n",
        "per_class",
        "confusion_matrix_rows_true_columns_predicted",
    }:
        _finite_probability(metrics[field], field)
    labels = ("hate_speech", "offensive", "neither")
    per_class = metrics["per_class"]
    if tuple(per_class) != labels:
        raise ValueError("per-class metrics must use the exact three-class order")
    supports: dict[str, int] = {}
    for label in labels:
        report = per_class[label]
        if set(report) != {"precision", "recall", "f1", "support"}:
            raise ValueError("per-class metric schema is incomplete")
        for field in ("precision", "recall", "f1"):
            _finite_probability(report[field], f"{label}.{field}")
        supports[label] = int(report["support"])
        if supports[label] < 0:
            raise ValueError("class support cannot be negative")
    if sum(supports.values()) != n:
        raise ValueError("per-class supports do not sum to metric support")
    if expected_support is not None and supports != {
        label: int(expected_support[label]) for label in labels
    }:
        raise ValueError("per-class support differs from the frozen manifest")
    matrix = metrics["confusion_matrix_rows_true_columns_predicted"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(row, list) or len(row) != 3 for row in matrix)
    ):
        raise ValueError("confusion matrix must be exactly 3x3")
    if any(int(value) < 0 for row in matrix for value in row):
        raise ValueError("confusion matrix contains a negative count")
    if [sum(int(value) for value in row) for row in matrix] != [
        supports[label] for label in labels
    ]:
        raise ValueError("confusion-matrix support differs from per-class support")


def validate_grouped_bootstrap(
    report: Mapping[str, Any],
    *,
    estimate: float,
    resamples: int,
    seed: int,
) -> None:
    required = {
        "estimate",
        "confidence",
        "lower",
        "upper",
        "bootstrap_mean",
        "bootstrap_standard_error",
        "resamples",
        "seed",
        "groups",
    }
    if set(report) != required:
        raise ValueError("grouped-bootstrap schema is incomplete")
    if int(report["resamples"]) != resamples or int(report["seed"]) != seed:
        raise ValueError("grouped bootstrap differs from the frozen resample plan")
    if int(report["groups"]) < 1:
        raise ValueError("grouped bootstrap must contain at least one group")
    observed_estimate = _finite_probability(report["estimate"], "bootstrap estimate")
    if not math.isclose(observed_estimate, float(estimate), abs_tol=1e-12):
        raise ValueError("grouped-bootstrap estimate differs from the point metric")
    confidence = _finite_probability(report["confidence"], "bootstrap confidence")
    if not math.isclose(confidence, 0.95, abs_tol=1e-12):
        raise ValueError("grouped-bootstrap confidence must be 0.95")
    lower = _finite_probability(report["lower"], "bootstrap lower")
    upper = _finite_probability(report["upper"], "bootstrap upper")
    _finite_probability(report["bootstrap_mean"], "bootstrap mean")
    standard_error = float(report["bootstrap_standard_error"])
    if not math.isfinite(standard_error) or standard_error < 0 or lower > upper:
        raise ValueError("grouped-bootstrap interval is invalid")


def validate_slice_report(
    report: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> None:
    categories = ("ambiguity_slices", "hard_negative_slices")
    if tuple(report) != categories:
        raise ValueError("slice report categories differ from the frozen configuration")
    for category in categories:
        expected_names = tuple(configuration[category])
        if tuple(report[category]) != expected_names:
            raise ValueError(f"slice topology differs for {category}")
        for name in expected_names:
            item = report[category][name]
            if set(item) != {"support", "metrics"}:
                raise ValueError(f"slice report schema is incomplete: {category}/{name}")
            support = int(item["support"])
            if support < 0:
                raise ValueError("slice support cannot be negative")
            if support == 0:
                if item["metrics"] is not None:
                    raise ValueError("empty slices must not contain fabricated metrics")
            else:
                validate_complete_metrics(item["metrics"], expected_n=support)

def _git_blob(root: Path, commit: str, relative_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def _current_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_commit_is_ancestor(
    repository_root: str | Path,
    ancestor: str,
    descendant: str,
) -> None:
    """Reject valid-but-unreachable objects that are outside the intended chain."""
    if not re.fullmatch(r"[0-9a-f]{40}", ancestor) or not re.fullmatch(
        r"[0-9a-f]{40}", descendant
    ):
        raise PermissionError("ancestry gates require full 40-character commit SHAs")
    root = Path(repository_root).resolve()
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 1:
        raise PermissionError(f"commit ancestry gate failed: {ancestor} is not an ancestor of {descendant}")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "git merge-base failed"
        raise PermissionError(f"could not verify commit ancestry: {detail}")


def build_protocol_commit_lock(
    repository_root: str | Path,
    protocol_commit: str,
    paths: Sequence[str] = PROTOCOL_LOCK_PATHS,
) -> dict[str, Any]:
    """Bind the selection to exact committed protocol/configuration bytes."""
    if not re.fullmatch(r"[0-9a-f]{40}", protocol_commit):
        raise ValueError("protocol commit must be a full 40-character SHA")
    root = Path(repository_root).resolve()
    files: list[dict[str, str]] = []
    for relative_path in paths:
        working_path = root / relative_path
        committed = _git_blob(root, protocol_commit, relative_path)
        if committed != working_path.read_bytes():
            raise ValueError(f"working protocol differs from {protocol_commit}: {relative_path}")
        files.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(committed).hexdigest(),
                "semantic_sha256": semantic_file_sha256(working_path),
            }
        )
    return {
        "commit": protocol_commit,
        "files": files,
        "byte_tree_sha256": canonical_json_sha256(
            [{"path": item["path"], "sha256": item["sha256"]} for item in files]
        ),
        "semantic_tree_sha256": canonical_json_sha256(
            [
                {"path": item["path"], "semantic_sha256": item["semantic_sha256"]}
                for item in files
            ]
        ),
    }


def collect_validation_runs(
    paths: Sequence[Path],
    *,
    matrix: Mapping[str, Any],
    matrix_path: str | Path,
    slices_path: str | Path,
    manifest: Mapping[str, Any],
    repository_root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    expected_matrix_hash = file_sha256(matrix_path)
    expected_matrix_semantic_hash = semantic_file_sha256(matrix_path)
    expected_slices_hash = file_sha256(slices_path)
    expected_slices_semantic_hash = semantic_file_sha256(slices_path)
    root = Path(repository_root).resolve()
    expected = {
        (str(candidate["id"]), int(seed))
        for candidate in matrix["candidates"]
        for seed in matrix["seeds"]
    }
    candidates = {str(candidate["id"]): candidate for candidate in matrix["candidates"]}
    slice_configuration = json.loads(Path(slices_path).read_text(encoding="utf-8"))
    validation_support = manifest["summary"]["split_support"]["validation"]
    expected_software = {
        "python": manifest["protocol"]["generator"]["python"],
        "numpy": manifest["protocol"]["generator"]["numpy"],
        "torch": SECURE_TRAINING_STACK["torch"],
        "transformers": SECURE_TRAINING_STACK["transformers"],
        "peft": SECURE_TRAINING_STACK["peft"],
        "datasets": manifest["protocol"]["generator"]["datasets"],
    }
    seen: set[tuple[str, int]] = set()
    runs: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        verify_payload_integrity(document)
        if document.get("schema_version") != 2 or document.get("protocol_id") != "grouped_v1":
            raise ValueError(f"unsupported validation-run schema: {path}")
        if document.get("frozen_test_loaded") is not False or "test_metrics" in document:
            raise ValueError(f"test-derived content is forbidden in selection input: {path}")
        if document.get("evaluated_split") != "validation":
            raise ValueError(f"selection accepts validation-only runs: {path}")
        if document.get("public_manifest_payload_sha256") != manifest["integrity"][
            "payload_sha256"
        ]:
            raise ValueError(f"run uses another public manifest: {path}")
        if document.get("matrix_sha256") != expected_matrix_hash:
            raise ValueError(f"run uses another experiment-matrix byte hash: {path}")
        if document.get("matrix_semantic_sha256") != expected_matrix_semantic_hash:
            raise ValueError(f"run uses another experiment-matrix semantic hash: {path}")
        if document.get("slices_sha256") != expected_slices_hash:
            raise ValueError(f"run uses another slice-configuration byte hash: {path}")
        if document.get("slices_semantic_sha256") != expected_slices_semantic_hash:
            raise ValueError(f"run uses another slice-configuration semantic hash: {path}")
        candidate_id = str(document.get("candidate_id", ""))
        seed = int(document.get("seed", -1))
        key = (candidate_id, seed)
        if key not in expected:
            raise ValueError(f"undeclared candidate/seed run: {key}")
        if key in seen:
            raise ValueError(f"duplicate candidate/seed run: {key}")
        if document.get("run_id") != f"{candidate_id}__seed_{seed}":
            raise ValueError(f"run ID is not canonical: {path}")
        if document.get("candidate") != candidates[candidate_id]:
            raise ValueError(f"run candidate configuration differs from frozen matrix: {path}")
        expected_encoder = matrix["encoders"][candidates[candidate_id]["encoder"]]
        if document.get("encoder") != expected_encoder:
            raise ValueError(f"run encoder/revision differs from frozen matrix: {path}")
        if document.get("software") != expected_software:
            raise ValueError(f"run software stack differs from the frozen secure stack: {path}")
        expected_hyperparameters = {
            **matrix["defaults"],
            "learning_rate": float(
                matrix["adaptation"][candidates[candidate_id]["adaptation"]]["learning_rate"]
            ),
        }
        if document.get("training_hyperparameters") != expected_hyperparameters:
            raise ValueError(f"run hyperparameters differ from frozen matrix: {path}")
        checkpoint = document.get("checkpoint_artifacts", {})
        if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("tree_sha256", ""))):
            raise ValueError(f"run lacks a checkpoint tree hash: {path}")
        if not checkpoint.get("files"):
            raise ValueError(f"run checkpoint inventory is empty: {path}")
        checkpoint_files = checkpoint["files"]
        if checkpoint.get("tree_sha256") != canonical_json_sha256(checkpoint_files):
            raise ValueError(f"run checkpoint tree hash is internally inconsistent: {path}")
        if int(checkpoint.get("size_bytes", -1)) != sum(
            int(item["size_bytes"]) for item in checkpoint_files
        ):
            raise ValueError(f"run checkpoint byte count is internally inconsistent: {path}")
        checkpoint_paths = [str(item.get("path", "")) for item in checkpoint_files]
        if (
            checkpoint_paths != sorted(checkpoint_paths)
            or len(checkpoint_paths) != len(set(checkpoint_paths))
            or any(
                not item_path
                or Path(item_path).is_absolute()
                or ".." in Path(item_path).parts
                for item_path in checkpoint_paths
            )
            or any(
                int(item.get("size_bytes", -1)) < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
                for item in checkpoint_files
            )
            or "config.json" not in checkpoint_paths
            or "tokenizer_config.json" not in checkpoint_paths
            or not any(item_path.endswith(".safetensors") for item_path in checkpoint_paths)
        ):
            raise ValueError(f"run checkpoint inventory is not a standalone model tree: {path}")
        expected_format = (
            "merged_full_model"
            if candidates[candidate_id]["adaptation"] == "lora"
            else "full_model"
        )
        if document.get("checkpoint") != "selected_checkpoint" or document.get(
            "checkpoint_format"
        ) != expected_format:
            raise ValueError(f"run checkpoint format differs from the adaptation: {path}")
        validate_complete_metrics(
            document.get("metrics", {}),
            expected_n=int(validation_support["rows"]),
            expected_support=validation_support,
        )
        bootstrap = document.get("grouped_bootstrap", {})
        if set(bootstrap) != {"macro_f1", "hate_speech_recall"}:
            raise ValueError(f"run lacks the exact grouped-bootstrap metrics: {path}")
        for metric in ("macro_f1", "hate_speech_recall"):
            validate_grouped_bootstrap(
                bootstrap[metric],
                estimate=float(document["metrics"][metric]),
                resamples=int(matrix["selection"]["required_bootstrap_resamples"]),
                seed=seed,
            )
        validate_slice_report(document.get("slice_metrics", {}), slice_configuration)
        seen.add(key)
        runs.append(document)
        try:
            relative_path = str(path.resolve().relative_to(root))
        except ValueError as error:
            raise ValueError(f"selection result must be stored inside the repository: {path}") from error
        inputs.append(
            {
                "path": relative_path,
                "sha256": file_sha256(path),
                "payload_sha256": str(document["integrity"]["payload_sha256"]),
            }
        )
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra or len(runs) != len(expected):
        raise ValueError(
            f"selection requires exactly {len(expected)} unique frozen runs; "
            f"missing={missing}, extra={extra}"
        )
    return sorted(runs, key=lambda run: (str(run["candidate_id"]), int(run["seed"]))), sorted(
        inputs, key=lambda item: item["path"]
    )


def _seed_variance(candidate_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        metric: {
            "mean": float(np.mean([run["metrics"][metric] for run in candidate_runs])),
            "sample_std": float(np.std([run["metrics"][metric] for run in candidate_runs], ddof=1)),
            "minimum": float(np.min([run["metrics"][metric] for run in candidate_runs])),
            "maximum": float(np.max([run["metrics"][metric] for run in candidate_runs])),
        }
        for metric in (
            "macro_f1",
            "hate_speech_precision",
            "hate_speech_recall",
            "hate_speech_f1",
        )
    }


def create_selection_draft(
    result_paths: Sequence[Path],
    *,
    matrix_path: str | Path,
    manifest_path: str | Path,
    slices_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    validate_matrix(matrix)
    manifest = load_manifest(manifest_path)
    runs, inputs = collect_validation_runs(
        result_paths,
        matrix=matrix,
        matrix_path=matrix_path,
        slices_path=slices_path,
        manifest=manifest,
        repository_root=repository_root,
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["candidate_id"])].append(run)
    minimum_precision = float(matrix["selection"]["minimum_hate_speech_precision"])
    candidates: list[dict[str, Any]] = []
    for candidate_id, candidate_runs in sorted(grouped.items()):
        variance = _seed_variance(candidate_runs)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "seeds": [int(run["seed"]) for run in candidate_runs],
                "seed_variance": variance,
                "validation_macro_f1_mean": variance["macro_f1"]["mean"],
                "validation_macro_f1_std": variance["macro_f1"]["sample_std"],
                "validation_hate_recall_mean": variance["hate_speech_recall"]["mean"],
                "validation_hate_precision_mean": variance["hate_speech_precision"]["mean"],
                "passes_precision_constraint": variance["hate_speech_precision"]["mean"]
                >= minimum_precision,
            }
        )
    eligible = [candidate for candidate in candidates if candidate["passes_precision_constraint"]]
    if not eligible:
        raise ValueError("no candidate satisfies the frozen hate-precision constraint")
    selected_summary = max(
        eligible,
        key=lambda item: (
            item["validation_macro_f1_mean"],
            item["validation_hate_recall_mean"],
            item["candidate_id"],
        ),
    )
    production_seed = int(matrix["selection"]["production_seed"])
    production_run = next(
        run
        for run in runs
        if run["candidate_id"] == selected_summary["candidate_id"]
        and int(run["seed"]) == production_seed
    )
    draft: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "matrix_sha256": file_sha256(matrix_path),
        "matrix_semantic_sha256": semantic_file_sha256(matrix_path),
        "slices_sha256": file_sha256(slices_path),
        "slices_semantic_sha256": semantic_file_sha256(slices_path),
        "evaluated_split": "validation",
        "selection_locked": False,
        "frozen_test_evaluated": False,
        "required_runs": len(matrix["candidates"]) * len(matrix["seeds"]),
        "observed_unique_runs": len(runs),
        "required_seeds": list(matrix["seeds"]),
        "selection_rule": matrix["selection"],
        "selected": {
            **selected_summary,
            "production_seed": production_seed,
            "production_run_id": production_run["run_id"],
            "production_result_payload_sha256": production_run["integrity"]["payload_sha256"],
            "production_checkpoint": production_run["checkpoint_artifacts"],
            "production_hyperparameters": production_run["training_hyperparameters"],
        },
        "candidate_summaries": candidates,
        "result_inputs": inputs,
        "result_set_sha256": canonical_json_sha256(inputs),
        "protocol_lock": None,
        "result_lock": None,
    }
    draft["integrity"] = {"payload_sha256": payload_sha256(draft)}
    return draft


def lock_selection_receipt(
    draft: Mapping[str, Any],
    *,
    repository_root: str | Path,
    protocol_commit: str,
    results_commit: str,
) -> dict[str, Any]:
    verify_payload_integrity(draft)
    if draft.get("selection_locked") is not False:
        raise ValueError("only an unlocked selection draft can be commit-locked")
    lock = build_protocol_commit_lock(repository_root, protocol_commit)
    if not re.fullmatch(r"[0-9a-f]{40}", results_commit):
        raise ValueError("results commit must be a full 40-character SHA")
    root = Path(repository_root).resolve()
    head = _current_head(root)
    verify_commit_is_ancestor(root, protocol_commit, results_commit)
    verify_commit_is_ancestor(root, results_commit, head)
    verify_commit_is_ancestor(root, protocol_commit, head)
    result_files: list[dict[str, str]] = []
    for expected in draft["result_inputs"]:
        committed = _git_blob(root, results_commit, str(expected["path"]))
        observed_hash = hashlib.sha256(committed).hexdigest()
        if observed_hash != expected["sha256"]:
            raise ValueError(f"results commit differs from selection input: {expected['path']}")
        result_files.append({"path": str(expected["path"]), "sha256": observed_hash})
    result_lock = {
        "commit": results_commit,
        "files": result_files,
        "byte_tree_sha256": canonical_json_sha256(result_files),
    }
    receipt = {key: value for key, value in draft.items() if key != "integrity"}
    receipt["protocol_lock"] = lock
    receipt["result_lock"] = result_lock
    receipt["selection_locked"] = True
    receipt["integrity"] = {"payload_sha256": payload_sha256(receipt)}
    return receipt


def create_selection_receipt(
    result_paths: Sequence[Path],
    *,
    matrix_path: str | Path,
    manifest_path: str | Path,
    slices_path: str | Path,
    repository_root: str | Path,
    protocol_commit: str | None = None,
    results_commit: str | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper: create a draft, and lock only when explicitly requested."""
    draft = create_selection_draft(
        result_paths,
        matrix_path=matrix_path,
        manifest_path=manifest_path,
        slices_path=slices_path,
        repository_root=repository_root,
    )
    if protocol_commit is None:
        return draft
    if results_commit is None:
        raise ValueError("results_commit is required to commit-lock a selection")
    return lock_selection_receipt(
        draft,
        repository_root=repository_root,
        protocol_commit=protocol_commit,
        results_commit=results_commit,
    )
