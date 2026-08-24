from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .constants import LABELS


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, Any]:
    labels = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "hate_speech_precision": float(precision[0]),
        "hate_speech_recall": float(recall[0]),
        "hate_speech_f1": float(f1[0]),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
    }


def grouped_bootstrap_interval(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    group_ids: Sequence[str],
    metric: Callable[[Sequence[int], Sequence[int]], float],
    *,
    seed: int = 42,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if not (len(y_true) == len(y_pred) == len(group_ids)):
        raise ValueError("y_true, y_pred, and group_ids must have equal length")
    if resamples < 1000:
        raise ValueError("reported grouped bootstrap intervals require at least 1000 resamples")
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        members[str(group)].append(index)
    groups = sorted(members)
    if not groups:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in sampled_groups for index in members[str(group)]]
        estimates.append(float(metric(true[indices], pred[indices])))
    alpha = (1 - confidence) / 2
    values = np.asarray(estimates, dtype=np.float64)
    return {
        "estimate": float(metric(true, pred)),
        "confidence": confidence,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "bootstrap_mean": float(values.mean()),
        "bootstrap_standard_error": float(values.std(ddof=1)),
        "resamples": resamples,
        "seed": seed,
        "groups": len(groups),
    }


def macro_f1_value(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(classification_metrics(y_true, y_pred)["macro_f1"])


def hate_recall_value(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(classification_metrics(y_true, y_pred)["hate_speech_recall"])


def paired_grouped_bootstrap_delta(
    y_true: Sequence[int],
    baseline_pred: Sequence[int],
    candidate_pred: Sequence[int],
    group_ids: Sequence[str],
    metric: Callable[[Sequence[int], Sequence[int]], float] = macro_f1_value,
    *,
    seed: int = 42,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    if not (len(y_true) == len(baseline_pred) == len(candidate_pred) == len(group_ids)):
        raise ValueError("paired inputs must have equal length")
    if resamples < 1000:
        raise ValueError("reported paired grouped bootstrap requires at least 1000 resamples")
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        members[str(group)].append(index)
    groups = sorted(members)
    rng = np.random.default_rng(seed)
    true = np.asarray(y_true, dtype=np.int64)
    baseline = np.asarray(baseline_pred, dtype=np.int64)
    candidate = np.asarray(candidate_pred, dtype=np.int64)
    deltas: list[float] = []
    for _ in range(resamples):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in sampled for index in members[str(group)]]
        deltas.append(float(metric(true[indices], candidate[indices]) - metric(true[indices], baseline[indices])))
    alpha = (1 - confidence) / 2
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "estimate_delta": float(metric(true, candidate) - metric(true, baseline)),
        "confidence": confidence,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "probability_delta_gt_zero": float((values > 0).mean()),
        "resamples": resamples,
        "seed": seed,
        "groups": len(groups),
    }
