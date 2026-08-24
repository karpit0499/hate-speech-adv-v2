from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from .integrity import (
    artifact_descriptor,
    canonical_json_sha256,
    file_sha256,
    payload_sha256,
    semantic_file_sha256,
    verify_payload_integrity,
)
from .manifest import (
    load_manifest,
    load_tuning_artifact,
    records_for_tuning,
    require_external_private_path,
)
from .metrics import (
    classification_metrics,
    grouped_bootstrap_interval,
    hate_recall_value,
    macro_f1_value,
)
from .protocol import read_split_secret
from .slices import slice_metric_report
from .selection import validate_complete_metrics, validate_slice_report


def evaluate_validation_artifact(
    runtime: str,
    artifact: str | Path,
    tokenizer_path: str | Path,
    manifest_path: str | Path,
    tuning_artifact_path: str | Path,
    split_secret_path: str | Path,
    slices_path: str | Path,
    *,
    batch_size: int = 32,
    max_length: int = 256,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    """Evaluate fp32 or q8 on a literal train/validation-only artifact."""
    if runtime not in {"pytorch", "onnx"}:
        raise ValueError("runtime must be pytorch or onnx")
    if max_length != 256:
        raise ValueError("artifact evaluation must use the frozen max length of 256")
    if bootstrap_resamples < 1000:
        raise ValueError("reported artifact intervals require at least 1000 grouped resamples")
    manifest = load_manifest(manifest_path)
    tuning_artifact_path = require_external_private_path(
        tuning_artifact_path,
        Path(__file__).resolve().parents[2],
        "private train/validation artifact",
    )
    tuning_artifact = load_tuning_artifact(
        tuning_artifact_path, manifest, read_split_secret(split_secret_path)
    )
    records = records_for_tuning(manifest, ["validation"], tuning_artifact)
    rows = records
    texts = [str(record["tweet"]) for record in records]
    labels = [int(record["class"]) for record in records]
    groups = [str(record["group_token"]) for record in records]
    predictions: list[int] = []
    started = time.perf_counter()

    if runtime == "pytorch":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            artifact, local_files_only=True
        ).eval()
        with torch.inference_mode():
            for offset in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[offset : offset + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                predictions.extend(model(**encoded).logits.argmax(dim=-1).tolist())
    else:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        input_names = {item.name for item in session.get_inputs()}
        for offset in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[offset : offset + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="np",
            )
            feeds = {key: value for key, value in encoded.items() if key in input_names}
            predictions.extend(np.argmax(session.run(None, feeds)[0], axis=-1).tolist())

    if len(predictions) != len(labels):
        raise RuntimeError("artifact produced an incomplete validation prediction set")
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "runtime": runtime,
        "evaluated_split": "validation",
        "frozen_test_loaded": False,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "artifact": {"locator": str(artifact), **artifact_descriptor(artifact)},
        "tokenizer": {"locator": str(tokenizer_path), **artifact_descriptor(tokenizer_path)},
        "batch_size": batch_size,
        "max_length": max_length,
        "bootstrap_resamples": bootstrap_resamples,
        "slices_sha256": file_sha256(slices_path),
        "slices_semantic_sha256": semantic_file_sha256(slices_path),
        "elapsed_seconds": elapsed,
        "metrics": classification_metrics(labels, predictions),
        "grouped_bootstrap": {
            "macro_f1": grouped_bootstrap_interval(
                labels,
                predictions,
                groups,
                macro_f1_value,
                seed=42,
                resamples=bootstrap_resamples,
            ),
            "hate_speech_recall": grouped_bootstrap_interval(
                labels,
                predictions,
                groups,
                hate_recall_value,
                seed=42,
                resamples=bootstrap_resamples,
            ),
        },
        "slice_metrics": slice_metric_report(
            rows,
            labels,
            predictions,
            json.loads(Path(slices_path).read_text(encoding="utf-8")),
        ),
        "prediction_columns": ["row_token", "group_token", "label", "prediction"],
        "predictions": [
            [record["row_token"], record["group_token"], label, prediction]
            for record, label, prediction in zip(records, labels, predictions)
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "runtime_package": (
                importlib.metadata.version("torch")
                if runtime == "pytorch"
                else importlib.metadata.version("onnxruntime")
            ),
        },
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    return result


def compare_validation_artifacts(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    configuration: dict[str, Any],
    export_manifest: dict[str, Any],
    slice_configuration: dict[str, Any],
    *,
    configuration_sha256: str,
    slices_sha256: str,
    slices_semantic_sha256: str,
) -> dict[str, Any]:
    for document in (reference, candidate, export_manifest):
        verify_payload_integrity(document)
    for document in (reference, candidate):
        if document.get("evaluated_split") != "validation" or document.get("frozen_test_loaded") is not False:
            raise PermissionError("q8 tuning comparison accepts validation-only artifacts")
        if document.get("runtime") != "onnx":
            raise ValueError("fp32-to-q8 gate requires two ONNX-runtime evaluations")
        if int(document.get("max_length", -1)) != int(configuration["max_length"]):
            raise ValueError("artifact evaluation used a different max length")
        if document.get("slices_sha256") != slices_sha256 or document.get(
            "slices_semantic_sha256"
        ) != slices_semantic_sha256:
            raise ValueError("artifact evaluation used another slice configuration")
        validate_complete_metrics(document.get("metrics", {}))
        validate_slice_report(document.get("slice_metrics", {}), slice_configuration)
    if reference["public_manifest_payload_sha256"] != candidate["public_manifest_payload_sha256"]:
        raise ValueError("artifacts were not evaluated on the same public manifest")
    if export_manifest.get("export_config_sha256") != configuration_sha256:
        raise ValueError("export manifest uses another export configuration")
    if int(export_manifest.get("opset", -1)) != int(configuration["opset"]):
        raise ValueError("export manifest uses another ONNX opset")
    if export_manifest.get("quantization") != configuration.get("quantization"):
        raise ValueError("export manifest uses another q8 quantization configuration")

    exported = {
        str(item["path"]): str(item["sha256"])
        for item in export_manifest["output_artifacts"]["files"]
    }
    reference_hash = str(reference["artifact"].get("sha256", ""))
    candidate_hash = str(candidate["artifact"].get("sha256", ""))
    fp32_hashes = {
        sha for path, sha in exported.items() if path in export_manifest["fp32_onnx_paths"]
    }
    q8_hashes = {
        sha for path, sha in exported.items() if path in export_manifest["q8_onnx_paths"]
    }
    if reference_hash not in fp32_hashes:
        raise ValueError("reference evaluation is not bound to the exported fp32 ONNX")
    if candidate_hash not in q8_hashes:
        raise ValueError("candidate evaluation is not bound to the exported q8 ONNX")
    quantization_provenance = export_manifest.get("quantization_provenance", {})
    if quantization_provenance != {
        "source_fp32_path": quantization_provenance.get("source_fp32_path"),
        "source_fp32_sha256": reference_hash,
        "q8_path": quantization_provenance.get("q8_path"),
        "q8_sha256": candidate_hash,
        "method": configuration["quantization"]["method"],
        "target": configuration["quantization"]["target"],
        "per_channel": configuration["quantization"]["per_channel"],
    }:
        raise ValueError("q8 evaluation is not bound to exact quantization provenance")
    if quantization_provenance["source_fp32_path"] not in export_manifest["fp32_onnx_paths"]:
        raise ValueError("quantization source is not a declared fp32 ONNX output")
    if quantization_provenance["q8_path"] not in export_manifest["q8_onnx_paths"]:
        raise ValueError("quantized artifact is not a declared q8 ONNX output")

    def exported_subtree(prefix: str) -> dict[str, Any]:
        files = [
            {
                **item,
                "path": str(item["path"])[len(prefix) :],
            }
            for item in export_manifest["output_artifacts"]["files"]
            if str(item["path"]).startswith(prefix)
        ]
        return {
            "kind": "directory",
            "files": files,
            "tree_sha256": canonical_json_sha256(files),
            "size_bytes": sum(int(item["size_bytes"]) for item in files),
        }

    if reference.get("tokenizer") != {
        "locator": reference["tokenizer"]["locator"],
        **exported_subtree("fp32/"),
    }:
        raise ValueError("reference tokenizer/config tree is not bound to the fp32 export")
    if candidate.get("tokenizer") != {
        "locator": candidate["tokenizer"]["locator"],
        **exported_subtree("q8/"),
    }:
        raise ValueError("candidate tokenizer/config tree is not bound to the q8 export")

    reference_rows = reference["predictions"]
    candidate_rows = candidate["predictions"]
    if [row[:3] for row in reference_rows] != [row[:3] for row in candidate_rows]:
        raise ValueError("prediction rows do not align")
    if not reference_rows:
        raise ValueError("cannot gate empty prediction artifacts")
    disagreements = sum(left[3] != right[3] for left, right in zip(reference_rows, candidate_rows))
    disagreement_rate = disagreements / len(reference_rows)
    macro_drop = float(reference["metrics"]["macro_f1"] - candidate["metrics"]["macro_f1"])
    hate_recall_drop = float(
        reference["metrics"]["hate_speech_recall"]
        - candidate["metrics"]["hate_speech_recall"]
    )
    gate = configuration["quality_gate"]
    checks = {
        "macro_f1_drop": macro_drop <= float(gate["maximum_macro_f1_drop"]),
        "hate_recall_drop": hate_recall_drop <= float(gate["maximum_hate_recall_drop"]),
        "prediction_disagreement_rate": disagreement_rate
        <= float(gate["maximum_prediction_disagreement_rate"]),
    }
    slice_gate = gate["slice_quality_gate"]
    minimum_support = int(slice_gate["minimum_support"])
    maximum_slice_macro_drop = float(slice_gate["maximum_macro_f1_drop"])
    maximum_slice_accuracy_drop = float(slice_gate["maximum_accuracy_drop"])
    slice_checks: dict[str, dict[str, Any]] = {}
    for category in ("ambiguity_slices", "hard_negative_slices"):
        slice_checks[category] = {}
        for name in slice_configuration[category]:
            left = reference["slice_metrics"][category][name]
            right = candidate["slice_metrics"][category][name]
            if int(left["support"]) != int(right["support"]):
                raise ValueError(f"artifact slice supports differ: {category}/{name}")
            support = int(left["support"])
            eligible = support >= minimum_support
            if support and (left["metrics"] is None or right["metrics"] is None):
                raise ValueError(f"artifact slice metrics are missing: {category}/{name}")
            macro_slice_drop = (
                float(left["metrics"]["macro_f1"] - right["metrics"]["macro_f1"])
                if support
                else 0.0
            )
            accuracy_slice_drop = (
                float(left["metrics"]["accuracy"] - right["metrics"]["accuracy"])
                if support
                else 0.0
            )
            passed = (
                not eligible
                or (
                    macro_slice_drop <= maximum_slice_macro_drop
                    and accuracy_slice_drop <= maximum_slice_accuracy_drop
                )
            )
            slice_checks[category][name] = {
                "support": support,
                "eligible": eligible,
                "minimum_support": minimum_support,
                "macro_f1_drop": macro_slice_drop,
                "maximum_macro_f1_drop": maximum_slice_macro_drop,
                "accuracy_drop": accuracy_slice_drop,
                "maximum_accuracy_drop": maximum_slice_accuracy_drop,
                "passed": passed,
            }
    checks["slice_metrics"] = all(
        item["passed"]
        for category in slice_checks.values()
        for item in category.values()
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "evaluated_split": "validation",
        "frozen_test_loaded": False,
        "public_manifest_payload_sha256": reference[
            "public_manifest_payload_sha256"
        ],
        "reference_evaluation_payload_sha256": reference["integrity"]["payload_sha256"],
        "candidate_evaluation_payload_sha256": candidate["integrity"]["payload_sha256"],
        "reference_artifact_sha256": reference_hash,
        "candidate_artifact_sha256": candidate_hash,
        "export_manifest_payload_sha256": export_manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": export_manifest[
            "selection_receipt_payload_sha256"
        ],
        "export_config_sha256": configuration_sha256,
        "slices_sha256": slices_sha256,
        "slices_semantic_sha256": slices_semantic_sha256,
        "opset": int(configuration["opset"]),
        "max_length": int(configuration["max_length"]),
        "quantization_provenance": quantization_provenance,
        "macro_f1_drop": macro_drop,
        "hate_recall_drop": hate_recall_drop,
        "prediction_disagreements": disagreements,
        "prediction_disagreement_rate": disagreement_rate,
        "thresholds": gate,
        "checks": checks,
        "slice_checks": slice_checks,
        "passed": all(checks.values()),
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    return result


def validate_quality_gate_result(
    result: dict[str, Any],
    *,
    export_manifest: dict[str, Any],
    selection_receipt: dict[str, Any],
    configuration: dict[str, Any],
    configuration_path: str | Path,
    slice_configuration: dict[str, Any] | None = None,
    slices_path: str | Path | None = None,
) -> None:
    """Validate a committed passing gate without trusting its self-attested flag alone."""
    for document in (result, export_manifest, selection_receipt):
        verify_payload_integrity(document)
    if result.get("schema_version") != 2 or result.get("protocol_id") != "grouped_v1":
        raise ValueError("unsupported q8 quality-gate result")
    if result.get("evaluated_split") != "validation" or result.get(
        "frozen_test_loaded"
    ) is not False:
        raise PermissionError("q8 gate must be validation-only")
    if result.get("selection_receipt_payload_sha256") != selection_receipt["integrity"][
        "payload_sha256"
    ]:
        raise ValueError("q8 gate refers to another selection receipt")
    if result.get("export_manifest_payload_sha256") != export_manifest["integrity"][
        "payload_sha256"
    ]:
        raise ValueError("q8 gate refers to another export manifest")
    if result.get("export_config_sha256") != file_sha256(configuration_path):
        raise ValueError("q8 gate uses another export configuration")
    provenance = export_manifest["quantization_provenance"]
    if result.get("reference_artifact_sha256") != provenance["source_fp32_sha256"]:
        raise ValueError("q8 gate reference is not the exact fp32 export")
    if result.get("candidate_artifact_sha256") != provenance["q8_sha256"]:
        raise ValueError("q8 gate candidate is not the exact q8 export")
    expected_checks = {
        "macro_f1_drop",
        "hate_recall_drop",
        "prediction_disagreement_rate",
        "slice_metrics",
    }
    if set(result.get("checks", {})) != expected_checks or not all(
        value is True for value in result["checks"].values()
    ):
        raise PermissionError("q8 quality gate did not pass every frozen check")
    if result.get("passed") is not True:
        raise PermissionError("q8 quality gate is not passing")
    gate = configuration["quality_gate"]
    numeric_checks = (
        ("macro_f1_drop", "maximum_macro_f1_drop"),
        ("hate_recall_drop", "maximum_hate_recall_drop"),
        ("prediction_disagreement_rate", "maximum_prediction_disagreement_rate"),
    )
    for observed, threshold in numeric_checks:
        value = float(result[observed])
        if not math.isfinite(value) or value > float(gate[threshold]):
            raise PermissionError(f"q8 quality gate exceeds {threshold}")
    if slice_configuration is not None:
        if slices_path is None:
            raise ValueError("slice path is required with slice configuration")
        if result.get("slices_sha256") != file_sha256(slices_path) or result.get(
            "slices_semantic_sha256"
        ) != semantic_file_sha256(slices_path):
            raise ValueError("q8 gate uses another slice configuration")
        categories = ("ambiguity_slices", "hard_negative_slices")
        if tuple(result.get("slice_checks", {})) != categories:
            raise ValueError("q8 slice-gate categories are incomplete")
        for category in categories:
            if tuple(result["slice_checks"][category]) != tuple(slice_configuration[category]):
                raise ValueError(f"q8 slice-gate topology differs: {category}")
            for item in result["slice_checks"][category].values():
                if set(item) != {
                    "support",
                    "eligible",
                    "minimum_support",
                    "macro_f1_drop",
                    "maximum_macro_f1_drop",
                    "accuracy_drop",
                    "maximum_accuracy_drop",
                    "passed",
                }:
                    raise ValueError("q8 slice-gate item schema is incomplete")
                if item["passed"] is not True:
                    raise PermissionError("q8 slice gate contains a failed slice")
