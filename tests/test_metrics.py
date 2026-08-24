import numpy as np

from hate_speech_adv.metrics import (
    classification_metrics,
    grouped_bootstrap_interval,
    macro_f1_value,
    paired_grouped_bootstrap_delta,
)


def test_metrics_have_fixed_three_class_semantics():
    metrics = classification_metrics([0, 1, 2, 2], [0, 1, 1, 2])
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.75
    assert set(metrics["per_class"]) == {"hate_speech", "offensive", "neither"}
    assert metrics["confusion_matrix_rows_true_columns_predicted"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]


def test_grouped_bootstrap_is_deterministic_and_group_aware():
    y_true = [0, 0, 1, 2, 2, 2]
    y_pred = [0, 1, 1, 2, 1, 2]
    groups = ["a", "a", "b", "c", "d", "d"]
    left = grouped_bootstrap_interval(y_true, y_pred, groups, macro_f1_value, seed=13, resamples=1000)
    right = grouped_bootstrap_interval(y_true, y_pred, groups, macro_f1_value, seed=13, resamples=1000)
    assert left == right
    assert left["groups"] == 4
    assert left["lower"] <= left["estimate"] <= left["upper"]


def test_paired_bootstrap_reports_positive_perfect_candidate_delta():
    result = paired_grouped_bootstrap_delta(
        [0, 1, 2, 2],
        [1, 1, 1, 2],
        [0, 1, 2, 2],
        ["a", "b", "c", "d"],
        seed=42,
        resamples=1000,
    )
    assert result["estimate_delta"] > 0
    assert result["probability_delta_gt_zero"] > 0.5
