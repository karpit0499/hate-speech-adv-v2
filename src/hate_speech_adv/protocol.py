from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import LABELS, PROTOCOL_ID, PROTOCOL_SOURCE_PATHS, SPLIT_CODES
from .data import stable_dataset_sha256, validate_dataset
from .grouping import (
    GROUP_VERSION,
    STRICT_VERSION,
    group_normalize,
    strict_normalize,
    template_group_hash,
)
from .integrity import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    payload_sha256,
    semantic_bundle,
)
from .manifest import PRIVATE_TUNING_COLUMNS
from .slices import matches_slice, vote_summary


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_split_secret(path: str | Path) -> bytes:
    """Read a 256-bit split secret without ever logging its value."""
    secret_path = Path(path)
    if os.name == "posix" and stat.S_IMODE(secret_path.stat().st_mode) & 0o077:
        raise PermissionError("split-secret file must not be readable by group or others")
    value = secret_path.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("split secret must be exactly 64 lowercase hexadecimal characters")
    return bytes.fromhex(value)


def _derive_integer(secret: bytes, purpose: str, modulus: int) -> int:
    digest = hmac.new(secret, purpose.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _opaque_token(secret: bytes, purpose: str, value: str) -> str:
    return hmac.new(
        secret,
        f"grouped_v1\0{purpose}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _row_identity(row: Mapping[str, Any], source_row_id: int) -> str:
    return canonical_json_sha256(
        [
            source_row_id,
            int(row["count"]),
            int(row["hate_speech_count"]),
            int(row["offensive_language_count"]),
            int(row["neither_count"]),
            int(row["class"]),
            str(row["tweet"]),
        ]
    )


def build_grouped_partition(
    dataset: Any,
    configuration: Mapping[str, Any],
    split_secret: bytes,
) -> list[str]:
    """Create the grouped partition from a non-public high-entropy secret."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    if len(split_secret) != 32:
        raise ValueError("grouped_v1 requires a 256-bit split secret")
    settings = configuration["partition"]
    if settings.get("algorithm") != "two_stage_stratified_group_kfold_secret_v1":
        raise ValueError("unexpected partition algorithm")
    labels = np.asarray(dataset["class"], dtype=np.int64)
    groups = np.asarray([template_group_hash(text) for text in dataset["tweet"]], dtype=object)
    placeholder = np.zeros(len(dataset), dtype=np.int8)

    test_folds = int(settings["test_folds"])
    outer = StratifiedGroupKFold(
        n_splits=test_folds,
        shuffle=True,
        random_state=_derive_integer(split_secret, "outer-random-state", 2**32 - 1),
    )
    outer_folds = list(outer.split(placeholder, labels, groups))
    test_fold = _derive_integer(split_secret, "outer-fold", test_folds)
    train_validation_indices, test_indices = outer_folds[test_fold]

    validation_folds = int(settings["validation_folds_within_remainder"])
    inner_labels = labels[train_validation_indices]
    inner_groups = groups[train_validation_indices]
    inner = StratifiedGroupKFold(
        n_splits=validation_folds,
        shuffle=True,
        random_state=_derive_integer(split_secret, "inner-random-state", 2**32 - 1),
    )
    inner_folds = list(
        inner.split(
            np.zeros(len(train_validation_indices), dtype=np.int8),
            inner_labels,
            inner_groups,
        )
    )
    validation_fold = _derive_integer(split_secret, "inner-fold", validation_folds)
    train_relative, validation_relative = inner_folds[validation_fold]
    train_indices = train_validation_indices[train_relative]
    validation_indices = train_validation_indices[validation_relative]

    split_names = [""] * len(dataset)
    for index in train_indices:
        split_names[int(index)] = "train"
    for index in validation_indices:
        split_names[int(index)] = "validation"
    for index in test_indices:
        split_names[int(index)] = "test"
    if any(not name for name in split_names):
        raise RuntimeError("partition did not assign every source row")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for group, split in zip(groups, split_names):
        group_splits[str(group)].add(split)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("group leakage detected while building partition")
    return split_names


def _support(rows: Sequence[list[Any]]) -> dict[str, int]:
    label_counts = Counter(int(row[3]) for row in rows)
    return {
        "rows": len(rows),
        **{LABELS[label]: label_counts[label] for label in range(len(LABELS))},
    }


def _sealed_record(dataset: Any, source_row_id: int) -> list[Any]:
    row = dataset[source_row_id]
    return [
        source_row_id,
        template_group_hash(str(row["tweet"])),
        int(row["class"]),
        int(row["count"]),
        int(row["hate_speech_count"]),
        int(row["offensive_language_count"]),
        int(row["neither_count"]),
    ]


def _sealed_test_commitment(
    dataset: Any,
    split_names: Sequence[str],
    split_secret: bytes,
) -> str:
    records = [
        _sealed_record(dataset, source_row_id)
        for source_row_id, split in enumerate(split_names)
        if split == "test"
    ]
    return hmac.new(
        split_secret,
        b"grouped_v1\0sealed-test-v1\0" + canonical_json_bytes(records),
        hashlib.sha256,
    ).hexdigest()


def _tuning_membership_commitment(
    private_artifact_without_commitment: Mapping[str, Any],
    split_secret: bytes,
) -> str:
    return hmac.new(
        split_secret,
        b"grouped_v1\0tuning-membership-v1\0"
        + canonical_json_bytes(private_artifact_without_commitment),
        hashlib.sha256,
    ).hexdigest()


def _protocol_source_provenance(
    config_path: str | Path,
    slices_path: str | Path,
) -> dict[str, Any]:
    root = Path(config_path).resolve().parents[2]
    pairs = []
    for logical in PROTOCOL_SOURCE_PATHS:
        physical = root / logical
        if logical == "configs/data/grouped_v1.json":
            physical = Path(config_path)
        elif logical == "configs/evaluation/slices_v1.json":
            physical = Path(slices_path)
        pairs.append((logical, physical))
    return semantic_bundle(pairs)


def build_protocol_artifacts(
    dataset: Any,
    configuration: Mapping[str, Any],
    config_path: str | Path,
    slice_configuration: Mapping[str, Any],
    slices_path: str | Path,
    split_secret: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a public opaque manifest, private TV map, and aggregate audit."""
    import numpy as np
    import sklearn

    validate_dataset(dataset)
    split_names = build_grouped_partition(dataset, configuration, split_secret)
    private_rows: list[list[Any]] = []
    support_rows: dict[str, list[list[Any]]] = {name: [] for name in SPLIT_CODES}
    group_hashes: set[str] = set()

    for source_row_id, (row, split_name) in enumerate(zip(dataset, split_names)):
        group_hash = template_group_hash(str(row["tweet"]))
        group_hashes.add(group_hash)
        row_token = _opaque_token(split_secret, "row", _row_identity(row, source_row_id))
        group_token = _opaque_token(split_secret, "group", group_hash)
        support_record = [
            row_token,
            SPLIT_CODES[split_name],
            group_token,
            int(row["class"]),
            int(row["count"]),
            int(row["hate_speech_count"]),
            int(row["offensive_language_count"]),
            int(row["neither_count"]),
        ]
        support_rows[split_name].append(support_record)
        if split_name != "test":
            private_rows.append(
                [
                    source_row_id,
                    row_token,
                    SPLIT_CODES[split_name],
                    group_token,
                    int(row["count"]),
                    int(row["hate_speech_count"]),
                    int(row["offensive_language_count"]),
                    int(row["neither_count"]),
                    int(row["class"]),
                    str(row["tweet"]),
                ]
            )

    private_rows.sort(key=lambda row: int(row[0]))
    private_core: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "scope": ["train", "validation"],
        "contains_frozen_test_membership": False,
        "columns": list(PRIVATE_TUNING_COLUMNS),
        "rows": private_rows,
    }
    tuning_commitment = _tuning_membership_commitment(private_core, split_secret)
    private: dict[str, Any] = {
        **private_core,
        "membership_commitment": tuning_commitment,
    }
    private["integrity"] = {"payload_sha256": payload_sha256(private)}
    source_provenance = _protocol_source_provenance(config_path, slices_path)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "protocol": {
            "protocol_id": PROTOCOL_ID,
            "dataset_repository": configuration["dataset"]["repository"],
            "dataset_revision": configuration["dataset"]["revision"],
            "dataset_split": configuration["dataset"]["split"],
            "grouping_version": GROUP_VERSION,
            "partition": configuration["partition"],
            "split_codes": configuration["partition"]["split_codes"],
            "policy": configuration["policy"],
            "generator": {
                "python": ".".join(map(str, __import__("sys").version_info[:3])),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "datasets": importlib.metadata.version("datasets"),
                "module": "hate_speech_adv.protocol",
                "source_semantic_tree_sha256": source_provenance["semantic_tree_sha256"],
            },
        },
        "summary": {
            "total_source_rows": len(dataset),
            "tuning_rows": len(private_rows),
            "total_groups": len(group_hashes),
            "split_support": {
                name: _support(rows) for name, rows in support_rows.items()
            },
        },
        "sealed_tuning_membership": {
            "commitment_scheme": "hmac-sha256-v1",
            "commitment_payload": "external train/validation-only artifact",
            "commitment": tuning_commitment,
            "rows": len(private_rows),
            "membership_published": False,
        },
        "sealed_test": {
            "commitment_scheme": "hmac-sha256-v1",
            "commitment_payload": "canonical ordered sealed-records-v1 payload",
            "commitment": _sealed_test_commitment(dataset, split_names, split_secret),
            "rows": len(support_rows["test"]),
            "aggregate_label_support": _support(support_rows["test"]),
            "membership_published": False,
        },
        "provenance": {
            "source_files": source_provenance,
            "config_sha256": file_sha256(config_path),
            "config_semantic_sha256": canonical_json_sha256(configuration),
            "slices_sha256": file_sha256(slices_path),
            "slices_semantic_sha256": canonical_json_sha256(slice_configuration),
            "dataset_content_sha256": stable_dataset_sha256(dataset),
        },
    }
    manifest["integrity"] = {"payload_sha256": payload_sha256(manifest)}
    audit = build_audit(dataset, manifest, split_names, slice_configuration)
    return manifest, private, audit


def reconstruct_and_verify_sealed_partition(
    dataset: Any,
    configuration: Mapping[str, Any],
    split_secret: bytes,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct test membership only inside the authorized one-shot path."""
    validate_dataset(dataset)
    split_names = build_grouped_partition(dataset, configuration, split_secret)
    commitment = _sealed_test_commitment(dataset, split_names, split_secret)
    if not hmac.compare_digest(commitment, str(manifest["sealed_test"]["commitment"])):
        raise PermissionError("split secret does not open the committed frozen partition")
    test_records = []
    for source_row_id, split_name in enumerate(split_names):
        if split_name != "test":
            continue
        row = dataset[source_row_id]
        test_records.append(
            {
                "source_row_id": source_row_id,
                "group_sha256": template_group_hash(str(row["tweet"])),
                "label": int(row["class"]),
                "count": int(row["count"]),
                "hate_speech_votes": int(row["hate_speech_count"]),
                "offensive_votes": int(row["offensive_language_count"]),
                "neither_votes": int(row["neither_count"]),
            }
        )
    if len(test_records) != int(manifest["sealed_test"]["rows"]):
        raise PermissionError("sealed-test support differs from commitment")
    return test_records


def _duplicate_summary(
    dataset: Any,
    split_names: Sequence[str],
    key_function: Callable[[str], str],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for source_row_id, text in enumerate(dataset["tweet"]):
        groups[key_function(str(text))].append(source_row_id)
    duplicates = [ids for ids in groups.values() if len(ids) > 1]
    conflicting = [
        ids for ids in duplicates if len({int(dataset[index]["class"]) for index in ids}) > 1
    ]
    cross_partition = [ids for ids in duplicates if len({split_names[index] for index in ids}) > 1]
    return {
        "unique_groups": len(groups),
        "duplicate_groups": len(duplicates),
        "rows_in_duplicate_groups": sum(len(ids) for ids in duplicates),
        "extra_duplicate_rows": sum(len(ids) - 1 for ids in duplicates),
        "conflicting_label_groups": len(conflicting),
        "rows_in_conflicting_groups": sum(len(ids) for ids in conflicting),
        "cross_partition_groups": len(cross_partition),
        "largest_group_sizes": sorted((len(ids) for ids in duplicates), reverse=True)[:10],
    }


def _ambiguity_summary(dataset: Any, split_names: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split_name in ("all", "train", "validation", "test"):
        indices = (
            range(len(dataset))
            if split_name == "all"
            else (index for index, value in enumerate(split_names) if value == split_name)
        )
        totals = Counter()
        by_class: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
        for index in indices:
            row = dataset[index]
            summary = vote_summary(row)
            label = LABELS[int(row["class"])]
            totals["rows"] += 1
            totals["unanimous"] += int(bool(summary["unanimous"]))
            totals["non_unanimous"] += int(not bool(summary["unanimous"]))
            totals["one_vote_margin"] += int(int(summary["vote_margin"]) <= 1)
            totals["low_consensus"] += int(float(summary["top_vote_fraction"]) < (2 / 3))
            by_class[label]["rows"] += 1
            by_class[label]["non_unanimous"] += int(not bool(summary["unanimous"]))
            by_class[label]["one_vote_margin"] += int(int(summary["vote_margin"]) <= 1)
        output[split_name] = {
            **dict(totals),
            "by_class": {label: dict(counts) for label, counts in by_class.items()},
        }
    return output


def build_audit(
    dataset: Any,
    manifest: Mapping[str, Any],
    split_names: Sequence[str],
    slice_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    slice_counts: dict[str, dict[str, int]] = {}
    for category in ("ambiguity_slices", "hard_negative_slices"):
        slice_counts[category] = {}
        for name, specification in slice_configuration[category].items():
            slice_counts[category][name] = {
                split_name: sum(
                    1
                    for index, row in enumerate(dataset)
                    if split_names[index] == split_name and matches_slice(row, specification)
                )
                for split_name in ("train", "validation", "test")
            }
    audit: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "contains_row_membership": False,
        "normalization": {"strict": STRICT_VERSION, "group": GROUP_VERSION},
        "duplicates": {
            "raw_exact": _duplicate_summary(dataset, split_names, lambda value: value),
            "strict_v1": _duplicate_summary(dataset, split_names, strict_normalize),
            "template_v1": _duplicate_summary(dataset, split_names, group_normalize),
        },
        "ambiguity": _ambiguity_summary(dataset, split_names),
        "slice_counts": slice_counts,
    }
    audit["integrity"] = {"payload_sha256": payload_sha256(audit)}
    return audit
