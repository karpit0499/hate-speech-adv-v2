from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from .constants import (
    EXPECTED_ROWS,
    PROTOCOL_ID,
    PROTOCOL_LOCK_PATHS,
    SPLIT_CODES,
    TUNING_SPLITS,
)
from .integrity import (
    canonical_json_sha256,
    file_sha256,
    semantic_file_sha256,
    verify_payload_integrity,
)

PRIVATE_TUNING_COLUMNS = (
    "source_row_id",
    "row_token",
    "split_code",
    "group_token",
    "count",
    "hate_speech_count",
    "offensive_language_count",
    "neither_count",
    "class",
    "tweet",
)
FROZEN_TEST_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_GROUPED_V1_TEST_IS_ONE_WAY"


def require_external_private_path(
    path: str | Path,
    repository_root: str | Path,
    description: str,
) -> Path:
    resolved = Path(path).resolve()
    root = Path(repository_root).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(f"{description} must live outside the repository")


def load_manifest(path: str | Path, verify: bool = True) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if verify:
        validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported public manifest schema")
    if manifest.get("protocol", {}).get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol ID")
    if "rows" in manifest or "columns" in manifest:
        raise ValueError("public manifest must not disclose any row-level split membership")
    summary = manifest.get("summary", {})
    if int(summary.get("total_source_rows", -1)) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} source rows")
    tuning = manifest.get("sealed_tuning_membership", {})
    if tuning.get("commitment_scheme") != "hmac-sha256-v1":
        raise ValueError("unexpected tuning-membership commitment scheme")
    if tuning.get("membership_published") is not False:
        raise ValueError("exact tuning membership must remain external")
    if not re.fullmatch(r"[0-9a-f]{64}", str(tuning.get("commitment", ""))):
        raise ValueError("tuning-membership commitment is invalid")
    if int(tuning.get("rows", -1)) != int(summary.get("tuning_rows", -2)):
        raise ValueError("tuning-membership support does not match summary")

    sealed = manifest.get("sealed_test", {})
    if sealed.get("commitment_scheme") != "hmac-sha256-v1":
        raise ValueError("unexpected sealed-test commitment scheme")
    if not re.fullmatch(r"[0-9a-f]{64}", str(sealed.get("commitment", ""))):
        raise ValueError("sealed-test commitment is invalid")
    if int(sealed.get("rows", -1)) <= 0:
        raise ValueError("sealed-test support must be positive")
    if int(tuning["rows"]) + int(sealed["rows"]) != EXPECTED_ROWS:
        raise ValueError("tuning and sealed-test supports are incomplete")
    verify_payload_integrity(manifest)


def load_tuning_artifact(
    path: str | Path,
    manifest: Mapping[str, Any],
    split_secret: bytes,
) -> dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != 1:
        raise ValueError("unsupported private tuning-artifact schema")
    if artifact.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("private tuning artifact targets another protocol")
    if tuple(artifact.get("columns", ())) != PRIVATE_TUNING_COLUMNS:
        raise ValueError("private tuning artifact columns do not match")
    verify_payload_integrity(artifact)

    private = [dict(zip(PRIVATE_TUNING_COLUMNS, row)) for row in artifact["rows"]]
    if len(private) != int(manifest["sealed_tuning_membership"]["rows"]):
        raise ValueError("private tuning artifact has the wrong row count")
    source_ids = [int(row["source_row_id"]) for row in private]
    if len(source_ids) != len(set(source_ids)) or any(
        source_id < 0 or source_id >= EXPECTED_ROWS for source_id in source_ids
    ):
        raise ValueError("private tuning artifact contains invalid source IDs")
    row_tokens: set[str] = set()
    group_splits: dict[str, set[int]] = defaultdict(set)
    for row in private:
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["row_token"])):
            raise ValueError("private tuning artifact contains an invalid row token")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["group_token"])):
            raise ValueError("private tuning artifact contains an invalid group token")
        if str(row["row_token"]) in row_tokens:
            raise ValueError("private tuning artifact contains a duplicate row token")
        row_tokens.add(str(row["row_token"]))
        group_splits[str(row["group_token"])].add(int(row["split_code"]))
        if int(row["split_code"]) == SPLIT_CODES["test"]:
            raise ValueError("private tuning artifact must not contain frozen-test membership")
        if int(row["class"]) not in (0, 1, 2):
            raise ValueError("private tuning artifact contains an invalid label")
        if int(row["count"]) != (
            int(row["hate_speech_count"])
            + int(row["offensive_language_count"])
            + int(row["neither_count"])
        ):
            raise ValueError("private tuning artifact contains inconsistent vote counts")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("group leakage detected in private tuning artifact")
    commitment_payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"integrity", "membership_commitment"}
    }
    observed_commitment = hmac.new(
        split_secret,
        b"grouped_v1\0tuning-membership-v1\0"
        + json.dumps(
            commitment_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    expected_commitment = str(manifest["sealed_tuning_membership"]["commitment"])
    if not hmac.compare_digest(observed_commitment, expected_commitment):
        raise PermissionError("private tuning artifact does not open the public commitment")
    if not hmac.compare_digest(
        str(artifact.get("membership_commitment", "")), expected_commitment
    ):
        raise PermissionError("private tuning artifact commitment field differs")
    return artifact


def records_for_tuning(
    manifest: Mapping[str, Any],
    splits: Sequence[str],
    tuning_artifact: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    requested = frozenset(splits)
    if not requested or not requested.issubset(TUNING_SPLITS):
        raise PermissionError("tuning commands may load only train and validation; frozen test is sealed")
    if tuning_artifact is None:
        raise PermissionError("a private train/validation-only artifact is required")
    split_codes = {int(manifest["protocol"]["split_codes"][name]) for name in requested}
    output: list[dict[str, Any]] = []
    for private_row in (
        dict(zip(PRIVATE_TUNING_COLUMNS, row)) for row in tuning_artifact["rows"]
    ):
        if int(private_row["split_code"]) not in split_codes:
            continue
        output.append(dict(private_row, source_row_id=int(private_row["source_row_id"])))
    return output


def verify_protocol_commit_lock(
    receipt: Mapping[str, Any],
    repository_root: str | Path,
    required_paths: Sequence[str] = PROTOCOL_LOCK_PATHS,
) -> None:
    """Verify locked commit blobs and current bytes for the complete protocol gate."""
    lock = receipt.get("protocol_lock", {})
    commit = str(lock.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("receipt needs a full committed protocol SHA")
    expected_files = lock.get("files", [])
    if not expected_files:
        raise PermissionError("receipt has no frozen protocol file lock")
    locked_paths = tuple(str(item.get("path", "")) for item in expected_files)
    if locked_paths != tuple(required_paths) or len(locked_paths) != len(set(locked_paths)):
        raise PermissionError("receipt does not lock the complete ordered protocol path set")
    expected_tree = canonical_json_sha256(
        [{"path": item["path"], "sha256": item["sha256"]} for item in expected_files]
    )
    if lock.get("byte_tree_sha256") != expected_tree:
        raise PermissionError("protocol lock tree hash is internally inconsistent")
    expected_semantic_tree = canonical_json_sha256(
        [
            {"path": item["path"], "semantic_sha256": item["semantic_sha256"]}
            for item in expected_files
        ]
    )
    if lock.get("semantic_tree_sha256") != expected_semantic_tree:
        raise PermissionError("protocol semantic-lock tree hash is internally inconsistent")
    root = Path(repository_root).resolve()
    for expected in expected_files:
        relative_path = str(expected["path"])
        working_path = (root / relative_path).resolve()
        try:
            working_path.relative_to(root)
        except ValueError as error:
            raise PermissionError("protocol lock path escapes the repository") from error
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=root,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise PermissionError(f"protocol commit lacks locked path: {relative_path}")
        actual = hashlib.sha256(completed.stdout).hexdigest()
        if actual != expected["sha256"]:
            raise PermissionError(f"protocol commit lock mismatch: {relative_path}")
        try:
            working_bytes = working_path.read_bytes()
        except OSError as error:
            raise PermissionError(f"locked protocol file is unavailable: {relative_path}") from error
        if working_bytes != completed.stdout:
            raise PermissionError(f"current protocol bytes differ from locked commit: {relative_path}")
        if semantic_file_sha256(working_path) != expected["semantic_sha256"]:
            raise PermissionError(f"current protocol semantics differ from lock: {relative_path}")


def verify_result_commit_lock(receipt: Mapping[str, Any], repository_root: str | Path) -> None:
    lock = receipt.get("result_lock", {})
    commit = str(lock.get("commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PermissionError("receipt needs a full committed validation-results SHA")
    files = lock.get("files", [])
    expected_from_inputs = [
        {"path": str(item["path"]), "sha256": str(item["sha256"])}
        for item in receipt.get("result_inputs", [])
    ]
    if files != expected_from_inputs or not files:
        raise PermissionError("result commit lock does not match the selected result inputs")
    if lock.get("byte_tree_sha256") != canonical_json_sha256(files):
        raise PermissionError("result lock tree hash is internally inconsistent")
    root = Path(repository_root).resolve()
    for expected in files:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{expected['path']}"],
            cwd=root,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise PermissionError(f"results commit lacks locked path: {expected['path']}")
        if hashlib.sha256(completed.stdout).hexdigest() != expected["sha256"]:
            raise PermissionError(f"results commit lock mismatch: {expected['path']}")
        working_path = (root / str(expected["path"])).resolve()
        try:
            working_path.relative_to(root)
        except ValueError as error:
            raise PermissionError("result lock path escapes the repository") from error
        try:
            working_bytes = working_path.read_bytes()
        except OSError as error:
            raise PermissionError(f"locked validation result is unavailable: {expected['path']}") from error
        if working_bytes != completed.stdout:
            raise PermissionError(f"current validation result differs from locked commit: {expected['path']}")


def verify_file_is_committed_at_head(
    path: str | Path,
    repository_root: str | Path,
) -> str:
    """Require a file's current bytes to be the exact blob committed at ``HEAD``."""
    root = Path(repository_root).resolve()
    path = Path(path).resolve()
    try:
        relative = str(path.relative_to(root))
    except ValueError as error:
        raise PermissionError("committed evidence must be inside the repository") from error
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=root, capture_output=True
    )
    if completed.returncode != 0:
        raise PermissionError(f"evidence is not committed at HEAD: {relative}")
    committed = completed.stdout
    if committed != path.read_bytes():
        raise PermissionError(f"evidence differs from its committed HEAD version: {relative}")
    return head


def verify_receipt_is_committed(
    receipt_path: str | Path,
    repository_root: str | Path,
) -> str:
    """Compatibility wrapper with the selection-specific error boundary."""
    try:
        return verify_file_is_committed_at_head(receipt_path, repository_root)
    except PermissionError as error:
        raise PermissionError(
            "selection receipt must exactly match its committed HEAD version"
        ) from error


def verify_locked_selection_receipt(
    receipt_path: str | Path,
    repository_root: str | Path,
) -> tuple[dict[str, Any], str]:
    """Verify the committed receipt, both locks, and the complete ancestry chain."""
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    verify_payload_integrity(receipt)
    if receipt.get("schema_version") != 2 or receipt.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("unsupported selection receipt")
    if receipt.get("evaluated_split") != "validation" or not receipt.get(
        "selection_locked"
    ):
        raise PermissionError("selection must be locked from validation-only evidence")
    if receipt.get("frozen_test_evaluated") is not False:
        raise PermissionError("selection receipt must predate frozen-test evaluation")
    verify_protocol_commit_lock(receipt, repository_root)
    verify_result_commit_lock(receipt, repository_root)
    head = verify_receipt_is_committed(receipt_path, repository_root)
    from .selection import verify_commit_is_ancestor

    protocol_commit = str(receipt["protocol_lock"]["commit"])
    results_commit = str(receipt["result_lock"]["commit"])
    verify_commit_is_ancestor(repository_root, protocol_commit, results_commit)
    verify_commit_is_ancestor(repository_root, results_commit, head)
    verify_commit_is_ancestor(repository_root, protocol_commit, head)
    return receipt, head


def create_one_shot_ledger(path: str | Path, entry: Mapping[str, Any]) -> None:
    """Atomically consume the single opening before any test rows are returned."""
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(ledger, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(entry), handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The ledger deliberately remains after failure: an opening is one-way.
        raise
    directory_descriptor = os.open(ledger.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def authorize_frozen_test(
    manifest_path: str | Path,
    receipt_path: str | Path,
    repository_root: str | Path,
    *,
    configuration_path: str | Path,
    split_secret_path: str | Path,
    export_manifest_path: str | Path,
    quality_gate_path: str | Path,
    export_root: str | Path,
    dataset_loader: Callable[[str, str], Any],
    ledger_path: str | Path,
    acknowledgement: str | None = None,
) -> dict[str, Any]:
    """Consume the ledger first, then authorize and materialize in-memory test data."""
    if acknowledgement != FROZEN_TEST_ACKNOWLEDGEMENT:
        raise PermissionError("explicit frozen-test acknowledgement is required")
    create_one_shot_ledger(
        ledger_path,
        {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "consumed_before_any_dataset_or_partition_access",
            "manifest_input": str(manifest_path),
            "selection_receipt_input": str(receipt_path),
            "configuration_input": str(configuration_path),
            "split_secret_input": str(split_secret_path),
            "export_manifest_input": str(export_manifest_path),
            "quality_gate_input": str(quality_gate_path),
            "export_root_input": str(export_root),
        },
    )
    # All reads, hashes, commit checks, secret validation, dataset loading, and
    # partition reconstruction intentionally happen after the durable O_EXCL.
    manifest = load_manifest(manifest_path)
    receipt, receipt_head = verify_locked_selection_receipt(receipt_path, repository_root)
    if receipt.get("public_manifest_payload_sha256") != manifest["integrity"]["payload_sha256"]:
        raise PermissionError("selection receipt refers to a different public manifest")

    export_manifest = json.loads(Path(export_manifest_path).read_text(encoding="utf-8"))
    quality_gate = json.loads(Path(quality_gate_path).read_text(encoding="utf-8"))
    verify_file_is_committed_at_head(export_manifest_path, repository_root)
    verify_file_is_committed_at_head(quality_gate_path, repository_root)
    from .artifact_evaluation import validate_quality_gate_result
    from .export_pipeline import validate_export_manifest_and_tree

    configuration_export = json.loads(
        (Path(repository_root) / "configs/export/int8.json").read_text(encoding="utf-8")
    )
    slices_path = Path(repository_root) / "configs/evaluation/slices_v1.json"
    slice_configuration = json.loads(slices_path.read_text(encoding="utf-8"))
    validate_export_manifest_and_tree(
        export_manifest,
        receipt,
        configuration_export,
        export_root,
        configuration_path=Path(repository_root) / "configs/export/int8.json",
    )
    validate_quality_gate_result(
        quality_gate,
        export_manifest=export_manifest,
        selection_receipt=receipt,
        configuration=configuration_export,
        configuration_path=Path(repository_root) / "configs/export/int8.json",
        slice_configuration=slice_configuration,
        slices_path=slices_path,
    )

    from .protocol import load_json, read_split_secret, reconstruct_and_verify_sealed_partition

    configuration = load_json(configuration_path)
    split_secret = read_split_secret(split_secret_path)
    dataset = dataset_loader(
        str(configuration["dataset"]["repository"]),
        str(configuration["dataset"]["revision"]),
    )

    test_records = reconstruct_and_verify_sealed_partition(
        dataset,
        configuration,
        split_secret,
        manifest,
    )
    return {
        "manifest": manifest,
        "selection_receipt": receipt,
        "receipt_head": receipt_head,
        "export_manifest": export_manifest,
        "quality_gate": quality_gate,
        "dataset": dataset,
        "test_records": test_records,
        "one_shot_ledger_sha256": file_sha256(ledger_path),
    }
