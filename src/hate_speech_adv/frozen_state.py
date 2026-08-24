from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import PROTOCOL_ID
from .integrity import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    payload_sha256,
    semantic_file_sha256,
    verify_payload_integrity,
)
from .manifest import (
    FROZEN_TEST_ACKNOWLEDGEMENT,
    authorize_frozen_test,
    load_manifest,
    require_external_private_path,
    verify_file_is_committed_at_head,
    verify_locked_selection_receipt,
)
from .metrics import (
    classification_metrics,
    grouped_bootstrap_interval,
    hate_recall_value,
    macro_f1_value,
)
from .protocol import read_split_secret
from .selection import (
    validate_complete_metrics,
    validate_grouped_bootstrap,
    validate_slice_report,
)
from .slices import matches_slice


EVIDENCE_COLUMNS = (
    "source_row_id",
    "group_sha256",
    "label",
    "count",
    "hate_speech_votes",
    "offensive_votes",
    "neither_votes",
    "slice_memberships",
    "fp32_prediction",
    "q8_prediction",
)


def _required_sha256(document: Mapping[str, Any], field: str) -> str:
    value = str(document.get(field, ""))
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _durable_create_json(path: str | Path, document: Mapping[str, Any], mode: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _evidence_records(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    if tuple(evidence.get("columns", ())) != EVIDENCE_COLUMNS:
        raise ValueError("private evidence columns differ from grouped_v1")
    raw_rows = evidence.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("private frozen-test evidence has no rows")
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for raw in raw_rows:
        if not isinstance(raw, list) or len(raw) != len(EVIDENCE_COLUMNS):
            raise ValueError("private evidence row has the wrong width")
        record = dict(zip(EVIDENCE_COLUMNS, raw))
        source_id = int(record["source_row_id"])
        if source_id < 0 or source_id in seen_ids:
            raise ValueError("private evidence has duplicate or invalid source IDs")
        seen_ids.add(source_id)
        if not re.fullmatch(r"[0-9a-f]{64}", str(record["group_sha256"])):
            raise ValueError("private evidence has an invalid group hash")
        label = int(record["label"])
        fp32 = int(record["fp32_prediction"])
        q8 = int(record["q8_prediction"])
        if any(value not in (0, 1, 2) for value in (label, fp32, q8)):
            raise ValueError("private evidence has an invalid label or prediction")
        votes = [
            int(record["hate_speech_votes"]),
            int(record["offensive_votes"]),
            int(record["neither_votes"]),
        ]
        if any(value < 0 for value in votes) or sum(votes) != int(record["count"]):
            raise ValueError("private evidence has inconsistent annotation votes")
        memberships = record["slice_memberships"]
        if (
            not isinstance(memberships, list)
            or memberships != sorted(set(str(value) for value in memberships))
        ):
            raise ValueError("private evidence slice memberships are not canonical")
        records.append(record)
    if [int(record["source_row_id"]) for record in records] != sorted(seen_ids):
        raise ValueError("private evidence rows must be ordered by source row ID")
    return records


def _membership_records(records: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            int(row["source_row_id"]),
            str(row["group_sha256"]),
            int(row["label"]),
            int(row["count"]),
            int(row["hate_speech_votes"]),
            int(row["offensive_votes"]),
            int(row["neither_votes"]),
        ]
        for row in records
    ]


def _prediction_records(records: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    return [
        [
            int(row["source_row_id"]),
            int(row["fp32_prediction"]),
            int(row["q8_prediction"]),
        ]
        for row in records
    ]


def validate_private_evidence(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    verify_payload_integrity(evidence)
    if evidence.get("schema_version") != 1 or evidence.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported private frozen-test evidence")
    if evidence.get("scope") != "frozen_test" or evidence.get("contains_raw_text") is not False:
        raise ValueError("private evidence has the wrong scope or raw-text policy")
    serialized_columns = json.dumps(evidence.get("columns", [])).lower()
    if "text" in serialized_columns or "tweet" in serialized_columns:
        raise ValueError("private evidence must not serialize raw text")
    records = _evidence_records(evidence)
    if evidence.get("membership_records_sha256") != canonical_json_sha256(
        _membership_records(records)
    ):
        raise ValueError("private evidence membership hash differs")
    if evidence.get("prediction_records_sha256") != canonical_json_sha256(
        _prediction_records(records)
    ):
        raise ValueError("private evidence prediction hash differs")
    for field in (
        "public_manifest_payload_sha256",
        "selection_receipt_payload_sha256",
        "export_manifest_payload_sha256",
        "quality_gate_payload_sha256",
        "one_shot_ledger_sha256",
    ):
        _required_sha256(evidence, field)
    return records


def _slice_report_from_memberships(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[int],
    slice_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    output: dict[str, Any] = {}
    for category in ("ambiguity_slices", "hard_negative_slices"):
        output[category] = {}
        for name in slice_configuration[category]:
            key = f"{category}/{name}"
            indices = [
                index
                for index, record in enumerate(records)
                if key in set(record["slice_memberships"])
            ]
            output[category][name] = {
                "support": len(indices),
                "metrics": (
                    classification_metrics(
                        [labels[index] for index in indices],
                        [int(predictions[index]) for index in indices],
                    )
                    if indices
                    else None
                ),
            }
    return output


def build_frozen_metric_evidence(
    records: Sequence[Mapping[str, Any]],
    slice_configuration: Mapping[str, Any],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    groups = [str(record["group_sha256"]) for record in records]
    reports: dict[str, Any] = {}
    for artifact in ("fp32", "q8"):
        predictions = [int(record[f"{artifact}_prediction"]) for record in records]
        metrics = classification_metrics(labels, predictions)
        reports[artifact] = {
            "metrics": metrics,
            "grouped_bootstrap": {
                "macro_f1": grouped_bootstrap_interval(
                    labels,
                    predictions,
                    groups,
                    macro_f1_value,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
                "hate_speech_recall": grouped_bootstrap_interval(
                    labels,
                    predictions,
                    groups,
                    hate_recall_value,
                    seed=bootstrap_seed,
                    resamples=bootstrap_resamples,
                ),
            },
            "slice_metrics": _slice_report_from_memberships(
                records, predictions, slice_configuration
            ),
        }
    disagreements = sum(
        int(record["fp32_prediction"]) != int(record["q8_prediction"])
        for record in records
    )
    return {
        "reports": reports,
        "comparison": {
            "prediction_disagreements": disagreements,
            "prediction_disagreement_rate": disagreements / len(records),
        },
    }


def _default_onnx_predictor(
    artifact_path: Path,
    tokenizer_path: Path,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
) -> list[int]:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    session = ort.InferenceSession(str(artifact_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    predictions: list[int] = []
    for offset in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[offset : offset + batch_size]),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feeds = {key: value for key, value in encoded.items() if key in input_names}
        predictions.extend(np.argmax(session.run(None, feeds)[0], axis=-1).tolist())
    return predictions


def score_frozen_test(
    *,
    manifest_path: str | Path,
    configuration_path: str | Path,
    receipt_path: str | Path,
    export_manifest_path: str | Path,
    quality_gate_path: str | Path,
    export_root: str | Path,
    split_secret_path: str | Path,
    ledger_path: str | Path,
    private_evidence_path: str | Path,
    public_result_path: str | Path,
    slices_path: str | Path,
    repository_root: str | Path,
    acknowledgement: str,
    dataset_loader: Callable[[str, str], Any],
    predictor: Callable[..., list[int]] = _default_onnx_predictor,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Consume the one-shot ledger, score exact fp32/q8 exports, and publish aggregates."""
    authorization = authorize_frozen_test(
        manifest_path,
        receipt_path,
        repository_root,
        configuration_path=configuration_path,
        split_secret_path=split_secret_path,
        export_manifest_path=export_manifest_path,
        quality_gate_path=quality_gate_path,
        export_root=export_root,
        dataset_loader=dataset_loader,
        ledger_path=ledger_path,
        acknowledgement=acknowledgement,
    )
    root = Path(repository_root).resolve()
    private_evidence_path = require_external_private_path(
        private_evidence_path, root, "private frozen-test evidence"
    )
    public_result_path = Path(public_result_path).resolve()
    try:
        public_result_path.relative_to(root)
    except ValueError as error:
        raise ValueError("public frozen-test result must live inside the repository") from error
    if bootstrap_resamples < 1000 or bootstrap_seed != 42:
        raise ValueError("frozen-test reporting requires >=1000 resamples at seed 42")
    if batch_size < 1:
        raise ValueError("batch size must be positive")

    manifest = authorization["manifest"]
    receipt = authorization["selection_receipt"]
    export_manifest = authorization["export_manifest"]
    quality_gate = authorization["quality_gate"]
    dataset = authorization["dataset"]
    test_records = authorization["test_records"]
    slice_configuration = json.loads(Path(slices_path).read_text(encoding="utf-8"))
    export_root = Path(export_root).resolve()
    fp32_relative = str(export_manifest["fp32_onnx_paths"][0])
    q8_relative = str(export_manifest["q8_onnx_paths"][0])
    max_length = int(export_manifest["max_length"])

    source_rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for test_record in test_records:
        row = dict(dataset[int(test_record["source_row_id"])])
        if (
            int(row["class"]) != int(test_record["label"])
            or int(row["count"]) != int(test_record["count"])
            or int(row["hate_speech_count"]) != int(test_record["hate_speech_votes"])
            or int(row["offensive_language_count"])
            != int(test_record["offensive_votes"])
            or int(row["neither_count"]) != int(test_record["neither_votes"])
        ):
            raise PermissionError("source dataset differs from the opened membership record")
        source_rows.append(row)
        texts.append(str(row["tweet"]))

    fp32_predictions = predictor(
        export_root / fp32_relative,
        export_root / Path(fp32_relative).parent,
        texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    q8_predictions = predictor(
        export_root / q8_relative,
        export_root / Path(q8_relative).parent,
        texts,
        batch_size=batch_size,
        max_length=max_length,
    )
    if len(fp32_predictions) != len(test_records) or len(q8_predictions) != len(test_records):
        raise RuntimeError("frozen-test artifact produced an incomplete prediction set")

    evidence_rows: list[list[Any]] = []
    for test_record, row, fp32, q8 in zip(
        test_records, source_rows, fp32_predictions, q8_predictions
    ):
        memberships = sorted(
            f"{category}/{name}"
            for category in ("ambiguity_slices", "hard_negative_slices")
            for name, specification in slice_configuration[category].items()
            if matches_slice(row, specification)
        )
        evidence_rows.append(
            [
                int(test_record["source_row_id"]),
                str(test_record["group_sha256"]),
                int(test_record["label"]),
                int(test_record["count"]),
                int(test_record["hate_speech_votes"]),
                int(test_record["offensive_votes"]),
                int(test_record["neither_votes"]),
                memberships,
                int(fp32),
                int(q8),
            ]
        )
    evidence_core: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": "frozen_test",
        "contains_raw_text": False,
        "columns": list(EVIDENCE_COLUMNS),
        "rows": evidence_rows,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": receipt["integrity"]["payload_sha256"],
        "export_manifest_payload_sha256": export_manifest["integrity"]["payload_sha256"],
        "quality_gate_payload_sha256": quality_gate["integrity"]["payload_sha256"],
        "one_shot_ledger_sha256": authorization["one_shot_ledger_sha256"],
    }
    evidence_records = _evidence_records(evidence_core)
    evidence_core["membership_records_sha256"] = canonical_json_sha256(
        _membership_records(evidence_records)
    )
    evidence_core["prediction_records_sha256"] = canonical_json_sha256(
        _prediction_records(evidence_records)
    )
    evidence_core["integrity"] = {"payload_sha256": payload_sha256(evidence_core)}
    _durable_create_json(private_evidence_path, evidence_core, 0o600)

    metric_evidence = build_frozen_metric_evidence(
        evidence_records,
        slice_configuration,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    provenance = export_manifest["quantization_provenance"]
    artifact_results = {
        "fp32": {
            "artifact_sha256": provenance["source_fp32_sha256"],
            **metric_evidence["reports"]["fp32"],
        },
        "q8": {
            "artifact_sha256": provenance["q8_sha256"],
            **metric_evidence["reports"]["q8"],
        },
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "evaluated_split": "frozen_test",
        "one_shot_opening_consumed": True,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": receipt["integrity"]["payload_sha256"],
        "export_manifest_payload_sha256": export_manifest["integrity"]["payload_sha256"],
        "quality_gate_payload_sha256": quality_gate["integrity"]["payload_sha256"],
        "opened_test_payload_sha256": evidence_core["integrity"]["payload_sha256"],
        "private_evidence_file_sha256": file_sha256(private_evidence_path),
        "one_shot_ledger_sha256": authorization["one_shot_ledger_sha256"],
        "opened_membership_sha256": evidence_core["membership_records_sha256"],
        "prediction_records_sha256": evidence_core["prediction_records_sha256"],
        "slices_sha256": file_sha256(slices_path),
        "slices_semantic_sha256": semantic_file_sha256(slices_path),
        "bootstrap": {"resamples": bootstrap_resamples, "seed": bootstrap_seed},
        "selected": {
            "candidate_id": receipt["selected"]["candidate_id"],
            "production_seed": receipt["selected"]["production_seed"],
            "production_run_id": receipt["selected"]["production_run_id"],
        },
        "artifact_results": artifact_results,
        "metrics": artifact_results["q8"]["metrics"],
        "fp32_q8_test_comparison": metric_evidence["comparison"],
        "contains_row_membership": False,
        "contains_raw_text": False,
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    validate_frozen_test_result(
        result,
        manifest=manifest,
        selection_receipt=receipt,
        export_manifest=export_manifest,
        quality_gate=quality_gate,
        slice_configuration=slice_configuration,
    )
    _durable_create_json(public_result_path, result, 0o644)
    return result


def validate_frozen_test_result(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    export_manifest: Mapping[str, Any],
    quality_gate: Mapping[str, Any],
    slice_configuration: Mapping[str, Any],
) -> None:
    for document in (result, selection_receipt, export_manifest, quality_gate):
        verify_payload_integrity(document)
    if result.get("schema_version") != 2 or result.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported frozen-test result")
    if result.get("evaluated_split") != "frozen_test" or result.get(
        "one_shot_opening_consumed"
    ) is not True:
        raise ValueError("result is not a consumed frozen-test evaluation")
    bindings = {
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": selection_receipt["integrity"]["payload_sha256"],
        "export_manifest_payload_sha256": export_manifest["integrity"]["payload_sha256"],
        "quality_gate_payload_sha256": quality_gate["integrity"]["payload_sha256"],
    }
    for field, expected in bindings.items():
        if _required_sha256(result, field) != expected:
            raise ValueError(f"frozen-test result binding differs: {field}")
    for field in (
        "opened_test_payload_sha256",
        "private_evidence_file_sha256",
        "one_shot_ledger_sha256",
        "opened_membership_sha256",
        "prediction_records_sha256",
    ):
        _required_sha256(result, field)
    if quality_gate.get("passed") is not True:
        raise PermissionError("frozen-test result requires a passing validation q8 gate")
    if export_manifest.get("selection_receipt_payload_sha256") != bindings[
        "selection_receipt_payload_sha256"
    ] or quality_gate.get("selection_receipt_payload_sha256") != bindings[
        "selection_receipt_payload_sha256"
    ]:
        raise ValueError("export/q8 evidence refers to another selection")
    if quality_gate.get("export_manifest_payload_sha256") != bindings[
        "export_manifest_payload_sha256"
    ]:
        raise ValueError("q8 gate refers to another export manifest")
    expected_selected = {
        "candidate_id": selection_receipt["selected"]["candidate_id"],
        "production_seed": selection_receipt["selected"]["production_seed"],
        "production_run_id": selection_receipt["selected"]["production_run_id"],
    }
    if result.get("selected") != expected_selected:
        raise ValueError("frozen-test result refers to another selected run")
    if result.get("contains_row_membership") is not False or result.get(
        "contains_raw_text"
    ) is not False:
        raise ValueError("public frozen-test result leaks private row data")
    forbidden = {"rows", "columns", "source_row_id", "tweet", "text", "predictions"}

    def inspect(value: Any) -> None:
        if isinstance(value, Mapping):
            if forbidden.intersection(str(key) for key in value):
                raise ValueError("public frozen-test result contains row-level data")
            for nested in value.values():
                inspect(nested)
        elif isinstance(value, list):
            for nested in value:
                inspect(nested)

    inspect(result)
    bootstrap = result.get("bootstrap", {})
    resamples = int(bootstrap.get("resamples", 0))
    seed = int(bootstrap.get("seed", -1))
    if resamples < 1000 or seed != 42:
        raise ValueError("frozen-test bootstrap plan differs from grouped_v1")
    support = manifest["sealed_test"]["aggregate_label_support"]
    artifact_results = result.get("artifact_results", {})
    if tuple(artifact_results) != ("fp32", "q8"):
        raise ValueError("frozen-test result must report exact fp32 and q8 artifacts")
    expected_hashes = {
        "fp32": export_manifest["quantization_provenance"]["source_fp32_sha256"],
        "q8": export_manifest["quantization_provenance"]["q8_sha256"],
    }
    for artifact in ("fp32", "q8"):
        report = artifact_results[artifact]
        if set(report) != {"artifact_sha256", "metrics", "grouped_bootstrap", "slice_metrics"}:
            raise ValueError("frozen artifact report schema is incomplete")
        if report["artifact_sha256"] != expected_hashes[artifact]:
            raise ValueError("frozen artifact hash differs from the exact export")
        validate_complete_metrics(
            report["metrics"],
            expected_n=int(support["rows"]),
            expected_support=support,
        )
        if set(report["grouped_bootstrap"]) != {"macro_f1", "hate_speech_recall"}:
            raise ValueError("frozen artifact bootstrap metrics are incomplete")
        for metric in ("macro_f1", "hate_speech_recall"):
            validate_grouped_bootstrap(
                report["grouped_bootstrap"][metric],
                estimate=float(report["metrics"][metric]),
                resamples=resamples,
                seed=seed,
            )
        validate_slice_report(report["slice_metrics"], slice_configuration)
    if result.get("metrics") != artifact_results["q8"]["metrics"]:
        raise ValueError("top-level metrics must be the exact q8 metric report")
    comparison = result.get("fp32_q8_test_comparison", {})
    if set(comparison) != {"prediction_disagreements", "prediction_disagreement_rate"}:
        raise ValueError("fp32/q8 frozen comparison schema is incomplete")
    disagreements = int(comparison["prediction_disagreements"])
    rate = float(comparison["prediction_disagreement_rate"])
    if disagreements < 0 or disagreements > int(support["rows"]) or not 0 <= rate <= 1:
        raise ValueError("fp32/q8 frozen disagreement report is invalid")
    if abs(rate - disagreements / int(support["rows"])) > 1e-12:
        raise ValueError("fp32/q8 frozen disagreement rate is inconsistent")


def _verify_evidence_result_bindings(
    evidence: Mapping[str, Any], result: Mapping[str, Any], evidence_path: str | Path
) -> list[dict[str, Any]]:
    records = validate_private_evidence(evidence)
    bindings = {
        "opened_test_payload_sha256": evidence["integrity"]["payload_sha256"],
        "private_evidence_file_sha256": file_sha256(evidence_path),
        "opened_membership_sha256": evidence["membership_records_sha256"],
        "prediction_records_sha256": evidence["prediction_records_sha256"],
        "one_shot_ledger_sha256": evidence["one_shot_ledger_sha256"],
    }
    for field, expected in bindings.items():
        if result.get(field) != expected:
            raise ValueError(f"result/private-evidence binding differs: {field}")
    return records


def create_post_test_reveal(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path,
    ledger_path: str | Path,
    opened_test_path: str | Path,
    test_result_path: str | Path,
    split_secret_path: str | Path,
    export_manifest_path: str | Path,
    quality_gate_path: str | Path,
    slices_path: str | Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    export_manifest = json.loads(Path(export_manifest_path).read_text(encoding="utf-8"))
    quality_gate = json.loads(Path(quality_gate_path).read_text(encoding="utf-8"))
    slice_configuration = json.loads(Path(slices_path).read_text(encoding="utf-8"))
    evidence = json.loads(Path(opened_test_path).read_text(encoding="utf-8"))
    test_result = json.loads(Path(test_result_path).read_text(encoding="utf-8"))
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    validate_frozen_test_result(
        test_result,
        manifest=manifest,
        selection_receipt=receipt,
        export_manifest=export_manifest,
        quality_gate=quality_gate,
        slice_configuration=slice_configuration,
    )
    if ledger.get("status") != "consumed_before_any_dataset_or_partition_access":
        raise ValueError("reveal requires the durable consumed opening ledger")
    if test_result["one_shot_ledger_sha256"] != file_sha256(ledger_path):
        raise ValueError("result refers to another one-shot ledger")
    records = _verify_evidence_result_bindings(evidence, test_result, opened_test_path)
    split_secret = read_split_secret(split_secret_path)
    observed_commitment = hmac.new(
        split_secret,
        b"grouped_v1\0sealed-test-v1\0" + canonical_json_bytes(_membership_records(records)),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(observed_commitment, str(manifest["sealed_test"]["commitment"])):
        raise PermissionError("split secret and evidence do not open the sealed partition")
    recomputed = build_frozen_metric_evidence(
        records,
        slice_configuration,
        bootstrap_resamples=int(test_result["bootstrap"]["resamples"]),
        bootstrap_seed=int(test_result["bootstrap"]["seed"]),
    )
    for artifact in ("fp32", "q8"):
        expected = {
            key: test_result["artifact_results"][artifact][key]
            for key in ("metrics", "grouped_bootstrap", "slice_metrics")
        }
        if recomputed["reports"][artifact] != expected:
            raise ValueError("private evidence does not recompute the published artifact report")
    if recomputed["comparison"] != test_result["fp32_q8_test_comparison"]:
        raise ValueError("private evidence does not recompute the published comparison")
    reveal: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "state": "post_test_revealed",
        "reveal_material": {"kind": "split_secret_hex", "value": split_secret.hex()},
        "sealed_test_commitment": manifest["sealed_test"]["commitment"],
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": receipt["integrity"]["payload_sha256"],
        "export_manifest_payload_sha256": export_manifest["integrity"]["payload_sha256"],
        "quality_gate_payload_sha256": quality_gate["integrity"]["payload_sha256"],
        "test_result_payload_sha256": test_result["integrity"]["payload_sha256"],
        "evidence_columns": list(EVIDENCE_COLUMNS),
        "evidence_rows": [
            [record[column] for column in EVIDENCE_COLUMNS] for record in records
        ],
        "membership_records_sha256": evidence["membership_records_sha256"],
        "prediction_records_sha256": evidence["prediction_records_sha256"],
    }
    reveal["integrity"] = {"payload_sha256": payload_sha256(reveal)}
    validate_reveal(
        reveal,
        result=test_result,
        manifest=manifest,
        slice_configuration=slice_configuration,
    )
    return reveal


def validate_reveal(
    reveal: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    manifest: Mapping[str, Any],
    slice_configuration: Mapping[str, Any],
) -> None:
    verify_payload_integrity(reveal)
    if reveal.get("schema_version") != 2 or reveal.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported reveal artifact")
    if reveal.get("state") != "post_test_revealed":
        raise ValueError("reveal artifact has the wrong state")
    secret_hex = str(reveal.get("reveal_material", {}).get("value", ""))
    if reveal.get("reveal_material", {}).get("kind") != "split_secret_hex" or not re.fullmatch(
        r"[0-9a-f]{64}", secret_hex
    ):
        raise ValueError("reveal artifact lacks a 256-bit split secret")
    if tuple(reveal.get("evidence_columns", ())) != EVIDENCE_COLUMNS:
        raise ValueError("reveal evidence columns differ")
    evidence = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": "frozen_test",
        "contains_raw_text": False,
        "columns": list(EVIDENCE_COLUMNS),
        "rows": reveal.get("evidence_rows"),
    }
    records = _evidence_records(evidence)
    membership_hash = canonical_json_sha256(_membership_records(records))
    prediction_hash = canonical_json_sha256(_prediction_records(records))
    if reveal.get("membership_records_sha256") != membership_hash or result.get(
        "opened_membership_sha256"
    ) != membership_hash:
        raise ValueError("revealed membership does not match the frozen-test result")
    if reveal.get("prediction_records_sha256") != prediction_hash or result.get(
        "prediction_records_sha256"
    ) != prediction_hash:
        raise ValueError("revealed predictions do not match the frozen-test result")
    observed_commitment = hmac.new(
        bytes.fromhex(secret_hex),
        b"grouped_v1\0sealed-test-v1\0" + canonical_json_bytes(_membership_records(records)),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(observed_commitment, str(manifest["sealed_test"]["commitment"])):
        raise PermissionError("revealed secret/membership do not open the public commitment")
    bindings = {
        "sealed_test_commitment": manifest["sealed_test"]["commitment"],
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "selection_receipt_payload_sha256": result["selection_receipt_payload_sha256"],
        "export_manifest_payload_sha256": result["export_manifest_payload_sha256"],
        "quality_gate_payload_sha256": result["quality_gate_payload_sha256"],
        "test_result_payload_sha256": result["integrity"]["payload_sha256"],
    }
    for field, expected in bindings.items():
        if reveal.get(field) != expected:
            raise ValueError(f"reveal binding differs: {field}")
    recomputed = build_frozen_metric_evidence(
        records,
        slice_configuration,
        bootstrap_resamples=int(result["bootstrap"]["resamples"]),
        bootstrap_seed=int(result["bootstrap"]["seed"]),
    )
    for artifact in ("fp32", "q8"):
        for field in ("metrics", "grouped_bootstrap", "slice_metrics"):
            if recomputed["reports"][artifact][field] != result["artifact_results"][artifact][
                field
            ]:
                raise ValueError("revealed evidence does not recompute the public metrics")
    if recomputed["comparison"] != result["fp32_q8_test_comparison"]:
        raise ValueError("revealed evidence does not recompute the public comparison")


def verify_repository_frozen_state(
    configuration_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    configuration = json.loads(Path(configuration_path).read_text(encoding="utf-8"))
    if configuration.get("schema_version") != 2 or configuration.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unsupported frozen-state configuration")
    root = Path(repository_root).resolve()

    def configured_path(name: str) -> Path:
        path = (root / str(configuration[name])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} escapes the repository") from error
        return path

    manifest = load_manifest(configured_path("manifest_path"))
    result_path = configured_path("result_path")
    reveal_path = configured_path("reveal_path")
    if reveal_path.exists() and not result_path.exists():
        raise ValueError("a reveal cannot exist before a completed frozen-test result")
    if not result_path.exists():
        state = "pre_test_sealed"
        result_payload = None
    else:
        receipt_path = configured_path("selection_receipt_path")
        export_manifest_path = configured_path("export_manifest_path")
        quality_gate_path = configured_path("quality_gate_path")
        slices_path = configured_path("slices_path")
        receipt, _ = verify_locked_selection_receipt(receipt_path, root)
        for path in (export_manifest_path, quality_gate_path, result_path):
            verify_file_is_committed_at_head(path, root)
        export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
        quality_gate = json.loads(quality_gate_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        slice_configuration = json.loads(slices_path.read_text(encoding="utf-8"))
        export_config_path = configured_path("export_config_path")
        from .artifact_evaluation import validate_quality_gate_result

        validate_quality_gate_result(
            quality_gate,
            export_manifest=export_manifest,
            selection_receipt=receipt,
            configuration=json.loads(export_config_path.read_text()),
            configuration_path=export_config_path,
            slice_configuration=slice_configuration,
            slices_path=slices_path,
        )
        validate_frozen_test_result(
            result,
            manifest=manifest,
            selection_receipt=receipt,
            export_manifest=export_manifest,
            quality_gate=quality_gate,
            slice_configuration=slice_configuration,
        )
        result_payload = result["integrity"]["payload_sha256"]
        if reveal_path.exists():
            verify_file_is_committed_at_head(reveal_path, root)
            reveal = json.loads(reveal_path.read_text(encoding="utf-8"))
            validate_reveal(
                reveal,
                result=result,
                manifest=manifest,
                slice_configuration=slice_configuration,
            )
            state = "post_test_revealed"
        else:
            state = "post_test_unrevealed"
    if state not in configuration.get("allowed_states", []):
        raise ValueError(f"repository frozen-test state is not allowed: {state}")
    return {
        "protocol_id": PROTOCOL_ID,
        "state": state,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "test_result_payload_sha256": result_payload,
    }
