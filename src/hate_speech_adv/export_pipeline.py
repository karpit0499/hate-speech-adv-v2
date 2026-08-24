from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from pathlib import Path
from typing import Any

from .integrity import (
    artifact_tree,
    canonical_json_sha256,
    file_sha256,
    payload_sha256,
    semantic_file_sha256,
    verify_payload_integrity,
)
from .manifest import verify_locked_selection_receipt


def validate_export_inputs(
    checkpoint: Path,
    receipt: dict[str, Any],
    configuration: dict[str, Any],
    *,
    receipt_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    committed_receipt, _ = verify_locked_selection_receipt(receipt_path, repository_root)
    if committed_receipt != receipt:
        raise PermissionError("export receipt bytes differ from the verified committed receipt")
    verify_payload_integrity(receipt)
    if receipt.get("evaluated_split") != "validation" or not receipt.get("selection_locked"):
        raise PermissionError("export requires a commit-locked validation-only selection receipt")
    if receipt.get("frozen_test_evaluated") is not False:
        raise PermissionError("export input receipt must predate frozen-test evaluation")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint = artifact_tree(checkpoint)
    expected_checkpoint = receipt.get("selected", {}).get("production_checkpoint")
    if observed_checkpoint != expected_checkpoint:
        raise PermissionError("checkpoint bytes do not match the selected production-seed run")
    if configuration.get("format") != "onnx":
        raise ValueError("only the declared ONNX pipeline is supported")
    if configuration.get("quantization", {}).get("method") != "dynamic_int8":
        raise ValueError("only the declared dynamic-int8 quantization is supported")
    if configuration.get("quantization", {}).get("target") != "avx2":
        raise ValueError("grouped_v1 q8 export requires the declared AVX2 target")
    if configuration.get("quantization", {}).get("per_channel") is not False:
        raise ValueError("grouped_v1 q8 export requires the declared per-tensor weights")
    if int(configuration.get("opset", -1)) != 17:
        raise ValueError("grouped_v1 export requires the frozen ONNX opset 17")
    max_length = int(configuration.get("max_length", -1))
    selected_max_length = int(
        receipt.get("selected", {}).get("production_hyperparameters", {}).get("max_length", -2)
    )
    if max_length != 256 or selected_max_length != max_length:
        raise ValueError("export max length must exactly match the selected training run")
    if configuration.get("quality_gate", {}).get("evaluated_split") != "validation":
        raise ValueError("q8 quality gate must use validation")
    return observed_checkpoint


def _validated_inventory_files(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("export inventory is empty")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in files:
        relative = str(item.get("path", ""))
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen
        ):
            raise ValueError("export inventory contains an unsafe or duplicate path")
        digest = str(item.get("sha256", ""))
        size = int(item.get("size_bytes", -1))
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or size < 0:
            raise ValueError("export inventory contains an invalid descriptor")
        seen.add(relative)
        normalized.append({"path": relative, "size_bytes": size, "sha256": digest})
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise ValueError("export inventory must be sorted canonically")
    if inventory.get("tree_sha256") != canonical_json_sha256(normalized):
        raise ValueError("export inventory tree hash is inconsistent")
    if int(inventory.get("size_bytes", -1)) != sum(item["size_bytes"] for item in normalized):
        raise ValueError("export inventory byte count is inconsistent")
    return normalized


def validate_export_manifest_and_tree(
    export_manifest: dict[str, Any],
    receipt: dict[str, Any],
    configuration: dict[str, Any],
    export_root: str | Path,
    *,
    configuration_path: str | Path,
) -> dict[str, Any]:
    """Verify the manifest bindings and every external export byte, with no extras."""
    verify_payload_integrity(export_manifest)
    verify_payload_integrity(receipt)
    if export_manifest.get("schema_version") != 2 or export_manifest.get(
        "protocol_id"
    ) != "grouped_v1":
        raise ValueError("unsupported export manifest")
    if export_manifest.get("selection_receipt_payload_sha256") != receipt["integrity"][
        "payload_sha256"
    ]:
        raise ValueError("export manifest refers to another selection receipt")
    if export_manifest.get("production_run_id") != receipt.get("selected", {}).get(
        "production_run_id"
    ):
        raise ValueError("export manifest refers to another production run")
    if export_manifest.get("source_checkpoint") != receipt.get("selected", {}).get(
        "production_checkpoint"
    ):
        raise ValueError("export manifest refers to another production checkpoint")
    if export_manifest.get("export_config_sha256") != file_sha256(configuration_path):
        raise ValueError("export manifest uses another export configuration")
    if export_manifest.get("export_config_semantic_sha256") != semantic_file_sha256(
        configuration_path
    ):
        raise ValueError("export manifest uses another export configuration meaning")
    if int(export_manifest.get("opset", -1)) != int(configuration["opset"]):
        raise ValueError("export manifest uses another ONNX opset")
    if int(export_manifest.get("max_length", -1)) != int(configuration["max_length"]):
        raise ValueError("export manifest uses another maximum length")
    if export_manifest.get("quantization") != configuration.get("quantization"):
        raise ValueError("export manifest uses another quantization contract")

    inventory = export_manifest.get("output_artifacts", {})
    files = _validated_inventory_files(inventory)
    root = Path(export_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_paths = {item["path"] for item in files}
    actual_paths = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) != "export_manifest.json"
    }
    if actual_paths != expected_paths:
        raise PermissionError(
            f"export tree path set differs: missing={sorted(expected_paths-actual_paths)}, "
            f"extra={sorted(actual_paths-expected_paths)}"
        )
    for expected in files:
        path = (root / expected["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PermissionError("export artifact escapes the export root") from error
        if path.is_symlink() or not path.is_file():
            raise PermissionError(f"export artifact is not a regular file: {expected['path']}")
        if path.stat().st_size != expected["size_bytes"] or file_sha256(path) != expected[
            "sha256"
        ]:
            raise PermissionError(f"export artifact bytes differ: {expected['path']}")

    fp32_paths = export_manifest.get("fp32_onnx_paths")
    q8_paths = export_manifest.get("q8_onnx_paths")
    if not isinstance(fp32_paths, list) or len(fp32_paths) != 1:
        raise ValueError("exactly one fp32 ONNX artifact is required")
    if not isinstance(q8_paths, list) or len(q8_paths) != 1:
        raise ValueError("exactly one q8 ONNX artifact is required")
    if fp32_paths[0] not in expected_paths or q8_paths[0] not in expected_paths:
        raise ValueError("declared ONNX paths are outside the export inventory")
    provenance = export_manifest.get("quantization_provenance", {})
    by_path = {item["path"]: item["sha256"] for item in files}
    if provenance != {
        "source_fp32_path": fp32_paths[0],
        "source_fp32_sha256": by_path[fp32_paths[0]],
        "q8_path": q8_paths[0],
        "q8_sha256": by_path[q8_paths[0]],
        "method": configuration["quantization"]["method"],
        "target": configuration["quantization"]["target"],
        "per_channel": configuration["quantization"]["per_channel"],
    }:
        raise ValueError("quantization provenance is not exact")
    return {"root": root, "files": files, "by_path": by_path}


def _assert_onnx_opset(paths: list[Path], expected_opset: int) -> None:
    import onnx

    if not paths:
        raise RuntimeError("export produced no ONNX files")
    for path in paths:
        model = onnx.load(str(path), load_external_data=False)
        default_domains = [item.version for item in model.opset_import if item.domain in ("", "ai.onnx")]
        if default_domains != [expected_opset]:
            raise RuntimeError(
                f"{path.name} has ONNX opsets {default_domains}, expected exactly [{expected_opset}]"
            )


def execute_export(
    checkpoint: str | Path,
    selection_receipt_path: str | Path,
    configuration_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Export exactly the selected checkpoint into a new, empty artifact tree."""
    import inspect

    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = Path(checkpoint)
    receipt_path = Path(selection_receipt_path)
    configuration_path = Path(configuration_path)
    output_dir = Path(output_dir)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    repository_root = Path(repository_root or Path(__file__).resolve().parents[2])
    checkpoint_inventory = validate_export_inputs(
        checkpoint,
        receipt,
        configuration,
        receipt_path=receipt_path,
        repository_root=repository_root,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("export output directory must be empty to prevent stale artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir = output_dir / "fp32"
    q8_dir = output_dir / "q8"
    fp32_dir.mkdir()
    q8_dir.mkdir()

    opset = int(configuration["opset"])
    max_length = int(configuration["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.model_max_length = max_length
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True
    ).eval()
    encoded = tokenizer(
        "offline ONNX export shape probe",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    accepted_inputs = set(inspect.signature(model.forward).parameters)
    input_names = [name for name in encoded if name in accepted_inputs]
    if not input_names:
        raise RuntimeError("tokenizer/model produced no compatible ONNX inputs")

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped, names):
            super().__init__()
            self.wrapped = wrapped
            self.names = names

        def forward(self, *values):
            return self.wrapped(**dict(zip(self.names, values))).logits

    fp32_model_path = fp32_dir / "model.onnx"
    dynamic_axes = {
        name: {0: "batch", 1: "sequence"} for name in input_names
    }
    dynamic_axes["logits"] = {0: "batch"}
    torch.onnx.export(
        LogitsOnly(model, input_names),
        tuple(encoded[name] for name in input_names),
        str(fp32_model_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    model.config.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)

    target = configuration["quantization"]["target"]
    if target != "avx2":
        raise ValueError("unsupported quantization target")
    q8_model_path = q8_dir / "model.onnx"
    quantize_dynamic(
        model_input=str(fp32_model_path),
        model_output=str(q8_model_path),
        per_channel=bool(configuration["quantization"]["per_channel"]),
        weight_type=QuantType.QInt8,
    )
    model.config.save_pretrained(q8_dir)
    tokenizer.save_pretrained(q8_dir)

    fp32_onnx = sorted(fp32_dir.rglob("*.onnx"))
    q8_onnx = sorted(q8_dir.rglob("*.onnx"))
    _assert_onnx_opset(fp32_onnx, opset)
    _assert_onnx_opset(q8_onnx, opset)
    output_inventory = artifact_tree(output_dir)
    output_hashes = {
        str(item["path"]): str(item["sha256"])
        for item in output_inventory["files"]
    }
    fp32_model_relative = str(fp32_model_path.relative_to(output_dir))
    q8_model_relative = str(q8_model_path.relative_to(output_dir))
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": "grouped_v1",
        "evaluated_split": "validation",
        "frozen_test_loaded": False,
        "selection_receipt_sha256": file_sha256(receipt_path),
        "selection_receipt_payload_sha256": receipt["integrity"]["payload_sha256"],
        "production_run_id": receipt["selected"]["production_run_id"],
        "production_result_payload_sha256": receipt["selected"][
            "production_result_payload_sha256"
        ],
        "source_checkpoint": checkpoint_inventory,
        "export_config_sha256": file_sha256(configuration_path),
        "export_config_semantic_sha256": semantic_file_sha256(configuration_path),
        "opset": opset,
        "max_length": max_length,
        "quantization": configuration["quantization"],
        "output_artifacts": output_inventory,
        "output_inventory_scope": (
            "all fp32/q8 ONNX, tokenizer, and model-config artifacts; "
            "this self-referential manifest is intentionally excluded"
        ),
        "fp32_onnx_paths": [str(path.relative_to(output_dir)) for path in fp32_onnx],
        "q8_onnx_paths": [str(path.relative_to(output_dir)) for path in q8_onnx],
        "quantization_provenance": {
            "source_fp32_path": fp32_model_relative,
            "source_fp32_sha256": output_hashes[fp32_model_relative],
            "q8_path": q8_model_relative,
            "q8_sha256": output_hashes[q8_model_relative],
            "method": configuration["quantization"]["method"],
            "target": configuration["quantization"]["target"],
            "per_channel": configuration["quantization"]["per_channel"],
        },
        "quality_gate": configuration["quality_gate"],
        "quality_gate_status": "pending_validation_comparison",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": importlib.metadata.version("onnx"),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    (output_dir / "export_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
