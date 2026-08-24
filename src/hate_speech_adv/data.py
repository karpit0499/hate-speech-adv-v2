from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .constants import DATASET_ID, DATASET_REVISION, EXPECTED_ROWS

REQUIRED_COLUMNS = (
    "count",
    "hate_speech_count",
    "offensive_language_count",
    "neither_count",
    "class",
    "tweet",
)


def load_pinned_dataset(repository: str = DATASET_ID, revision: str = DATASET_REVISION):
    """Load the one allowed dataset revision; imports datasets lazily."""
    if repository != DATASET_ID or revision != DATASET_REVISION:
        raise ValueError("grouped_v1 only permits the pinned Davidson dataset revision")
    from datasets import load_dataset

    dataset = load_dataset(repository, revision=revision)["train"]
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: Any) -> None:
    columns = tuple(dataset.column_names)
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(dataset)}")
    labels = set(int(value) for value in dataset["class"])
    if labels != {0, 1, 2}:
        raise ValueError(f"unexpected labels: {sorted(labels)}")


def stable_dataset_sha256(dataset: Any) -> str:
    """Hash all source fields, including text, without storing text in the manifest."""
    digest = hashlib.sha256()
    for source_row_id, row in enumerate(dataset):
        record = [
            source_row_id,
            int(row["count"]),
            int(row["hate_speech_count"]),
            int(row["offensive_language_count"]),
            int(row["neither_count"]),
            int(row["class"]),
            str(row["tweet"]),
        ]
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_source_rows(dataset: Any, source_row_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Return selected rows in manifest order and attach their immutable source IDs."""
    selected = dataset.select(list(source_row_ids))
    return [dict(row, source_row_id=int(source_row_id)) for source_row_id, row in zip(source_row_ids, selected)]

