import json
from pathlib import Path

from hate_speech_adv.artifact_evaluation import compare_validation_artifacts
from hate_speech_adv.export_pipeline import validate_export_inputs
from hate_speech_adv.integrity import artifact_tree, canonical_json_sha256, file_sha256, payload_sha256, semantic_file_sha256
from hate_speech_adv.metrics import classification_metrics


ROOT = Path(__file__).resolve().parents[1]
SLICES_PATH = ROOT / "configs/evaluation/slices_v1.json"
SLICES = json.loads(SLICES_PATH.read_text(encoding="utf-8"))


def _empty_slices():
    return {
        category: {
            name: {"support": 0, "metrics": None}
            for name in SLICES[category]
        }
        for category in ("ambiguity_slices", "hard_negative_slices")
    }


def _document(predictions, macro_f1, hate_recall, artifact_hash, tokenizer_locator):
    tokenizer_files = [
        {"path": "model.onnx", "sha256": artifact_hash, "size_bytes": 1}
    ]
    document = {
        "runtime": "onnx",
        "evaluated_split": "validation",
        "frozen_test_loaded": False,
        "public_manifest_payload_sha256": "a" * 64,
        "artifact": {"kind": "file", "sha256": artifact_hash, "size_bytes": 1},
        "tokenizer": {
            "locator": tokenizer_locator,
            "kind": "directory",
            "files": tokenizer_files,
            "tree_sha256": canonical_json_sha256(tokenizer_files),
            "size_bytes": 1,
        },
        "max_length": 256,
        "slices_sha256": file_sha256(SLICES_PATH),
        "slices_semantic_sha256": semantic_file_sha256(SLICES_PATH),
        "metrics": classification_metrics(
            [int(row[2]) for row in predictions],
            [int(row[3]) for row in predictions],
        ),
        "slice_metrics": _empty_slices(),
        "predictions": predictions,
    }
    document["integrity"] = {"payload_sha256": payload_sha256(document)}
    return document


def test_q8_validation_gate_is_bound_to_exact_exported_artifacts():
    rows = [["r0", "g0", 0, 0], ["r1", "g1", 1, 1], ["r2", "g2", 2, 2]]
    reference = _document(rows, 0.8, 0.7, "1" * 64, "fp32")
    candidate = _document(rows, 0.795, 0.69, "2" * 64, "q8")
    config = {
        "opset": 17,
        "max_length": 256,
        "quantization": {
            "method": "dynamic_int8",
            "target": "avx2",
            "per_channel": False,
        },
        "quality_gate": {
            "maximum_macro_f1_drop": 0.01,
            "maximum_hate_recall_drop": 0.02,
            "maximum_prediction_disagreement_rate": 0.02,
            "slice_quality_gate": {
                "minimum_support": 25,
                "maximum_macro_f1_drop": 0.03,
                "maximum_accuracy_drop": 0.03,
            },
        },
    }
    export_manifest = {
        "export_config_sha256": "3" * 64,
        "opset": 17,
        "quantization": config["quantization"],
        "selection_receipt_payload_sha256": "4" * 64,
        "fp32_onnx_paths": ["fp32/model.onnx"],
        "q8_onnx_paths": ["q8/model.onnx"],
        "output_artifacts": {
            "files": [
                {"path": "fp32/model.onnx", "sha256": "1" * 64, "size_bytes": 1},
                {"path": "q8/model.onnx", "sha256": "2" * 64, "size_bytes": 1},
            ]
        },
        "quantization_provenance": {
            "source_fp32_path": "fp32/model.onnx",
            "source_fp32_sha256": "1" * 64,
            "q8_path": "q8/model.onnx",
            "q8_sha256": "2" * 64,
            **config["quantization"],
        },
    }
    export_manifest["integrity"] = {"payload_sha256": payload_sha256(export_manifest)}
    comparison = compare_validation_artifacts(
        reference,
        candidate,
        config,
        export_manifest,
        SLICES,
        configuration_sha256="3" * 64,
        slices_sha256=file_sha256(SLICES_PATH),
        slices_semantic_sha256=semantic_file_sha256(SLICES_PATH),
    )
    assert comparison["passed"] is True
    assert comparison["reference_artifact_sha256"] == "1" * 64
    assert comparison["candidate_artifact_sha256"] == "2" * 64


def test_export_rejects_checkpoint_not_named_by_selection(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    receipt = {
        "evaluated_split": "validation",
        "selection_locked": True,
        "frozen_test_evaluated": False,
        "selected": {
            "production_checkpoint": artifact_tree(checkpoint),
            "production_hyperparameters": {"max_length": 256},
        },
    }
    receipt["integrity"] = {"payload_sha256": payload_sha256(receipt)}
    config = {
        "format": "onnx",
        "opset": 17,
        "max_length": 256,
        "quantization": {
            "method": "dynamic_int8",
            "target": "avx2",
            "per_channel": False,
        },
        "quality_gate": {"evaluated_split": "validation"},
    }
    monkeypatch.setattr(
        "hate_speech_adv.export_pipeline.verify_locked_selection_receipt",
        lambda *_: (receipt, "f" * 40),
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    validate_export_inputs(
        checkpoint,
        receipt,
        config,
        receipt_path=receipt_path,
        repository_root=tmp_path,
    )
    (checkpoint / "config.json").write_text('{"tampered":true}', encoding="utf-8")
    import pytest

    with pytest.raises(PermissionError, match="checkpoint bytes"):
        validate_export_inputs(
            checkpoint,
            receipt,
            config,
            receipt_path=receipt_path,
            repository_root=tmp_path,
        )
