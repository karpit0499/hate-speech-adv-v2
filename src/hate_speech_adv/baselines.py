from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

from .integrity import file_sha256, payload_sha256, semantic_file_sha256
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

BASELINE_SEEDS = (13, 42, 73)


def _select(records: list[dict[str, Any]]) -> tuple[list[str], list[int], list[str]]:
    return (
        [str(record["tweet"]) for record in records],
        [int(record["class"]) for record in records],
        [str(record["group_token"]) for record in records],
    )


def run_baselines(
    manifest_path: str | Path,
    tuning_artifact_path: str | Path,
    split_secret_path: str | Path,
    slices_path: str | Path,
    *,
    seeds: tuple[int, ...] = BASELINE_SEEDS,
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    if bootstrap_resamples < 1000:
        raise ValueError("reported baseline intervals require at least 1000 grouped resamples")
    manifest = load_manifest(manifest_path)
    tuning_artifact_path = require_external_private_path(
        tuning_artifact_path,
        Path(__file__).resolve().parents[2],
        "private train/validation artifact",
    )
    tuning_artifact = load_tuning_artifact(
        tuning_artifact_path, manifest, read_split_secret(split_secret_path)
    )
    train_records = records_for_tuning(manifest, ["train"], tuning_artifact)
    validation_records = records_for_tuning(manifest, ["validation"], tuning_artifact)
    train_text, y_train, train_groups = _select(train_records)
    validation_text, y_validation, validation_groups = _select(validation_records)
    validation_rows = validation_records
    slice_configuration = json.loads(Path(slices_path).read_text(encoding="utf-8"))

    majority_label = Counter(y_train).most_common(1)[0][0]
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    x_train = features.fit_transform(train_text)
    x_validation = features.transform(validation_text)
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        majority_train = np.full(len(y_train), majority_label, dtype=np.int64)
        majority_validation = np.full(len(y_validation), majority_label, dtype=np.int64)
        majority_result: dict[str, Any] = {
            "candidate_id": "majority",
            "seed": seed,
            "evaluated_split": "validation",
            "train_metrics": classification_metrics(y_train, majority_train),
            "metrics": classification_metrics(y_validation, majority_validation),
        }
        if bootstrap_resamples:
            majority_result["validation_group_bootstrap"] = {
                "macro_f1": grouped_bootstrap_interval(
                    y_validation,
                    majority_validation,
                    validation_groups,
                    macro_f1_value,
                    seed=seed,
                    resamples=bootstrap_resamples,
                ),
                "hate_speech_recall": grouped_bootstrap_interval(
                    y_validation,
                    majority_validation,
                    validation_groups,
                    hate_recall_value,
                    seed=seed,
                    resamples=bootstrap_resamples,
                ),
            }
        majority_result["slice_metrics"] = slice_metric_report(
            validation_rows,
            y_validation,
            majority_validation,
            slice_configuration,
        )
        runs.append(majority_result)

        # scikit-learn 1.9 correctly rejects direct multiclass liblinear.
        # The explicit wrapper preserves the prior one-vs-rest semantics.
        classifier = OneVsRestClassifier(
            LogisticRegression(
                C=4.0,
                class_weight="balanced",
                max_iter=500,
                random_state=seed,
                solver="liblinear",
                tol=1e-4,
            ),
            n_jobs=1,
        )
        classifier.fit(x_train, y_train)
        train_prediction = classifier.predict(x_train)
        validation_prediction = classifier.predict(x_validation)
        tfidf_result: dict[str, Any] = {
            "candidate_id": "tfidf_word_char_logistic",
            "seed": seed,
            "evaluated_split": "validation",
            "train_metrics": classification_metrics(y_train, train_prediction),
            "metrics": classification_metrics(y_validation, validation_prediction),
            "solver": "OneVsRestClassifier(LogisticRegression(liblinear))",
            "iterations": [
                int(estimator.n_iter_[0]) for estimator in classifier.estimators_
            ],
        }
        if bootstrap_resamples:
            tfidf_result["validation_group_bootstrap"] = {
                "macro_f1": grouped_bootstrap_interval(
                    y_validation,
                    validation_prediction,
                    validation_groups,
                    macro_f1_value,
                    seed=seed,
                    resamples=bootstrap_resamples,
                ),
                "hate_speech_recall": grouped_bootstrap_interval(
                    y_validation,
                    validation_prediction,
                    validation_groups,
                    hate_recall_value,
                    seed=seed,
                    resamples=bootstrap_resamples,
                ),
            }
        tfidf_result["slice_metrics"] = slice_metric_report(
            validation_rows,
            y_validation,
            validation_prediction,
            slice_configuration,
        )
        runs.append(tfidf_result)

    aggregates: dict[str, Any] = {}
    for candidate_id in sorted({run["candidate_id"] for run in runs}):
        candidate_runs = [run for run in runs if run["candidate_id"] == candidate_id]
        aggregates[candidate_id] = {
            "seeds": [run["seed"] for run in candidate_runs],
            "validation_macro_f1_mean": float(np.mean([run["metrics"]["macro_f1"] for run in candidate_runs])),
            "validation_macro_f1_std": float(np.std([run["metrics"]["macro_f1"] for run in candidate_runs], ddof=1)),
            "validation_hate_recall_mean": float(np.mean([run["metrics"]["hate_speech_recall"] for run in candidate_runs])),
            "validation_hate_precision_mean": float(np.mean([run["metrics"]["hate_speech_precision"] for run in candidate_runs])),
            "seed_variance": {
                metric: {
                    "mean": float(np.mean([run["metrics"][metric] for run in candidate_runs])),
                    "sample_std": float(np.std([run["metrics"][metric] for run in candidate_runs], ddof=1)),
                    "minimum": float(np.min([run["metrics"][metric] for run in candidate_runs])),
                    "maximum": float(np.max([run["metrics"][metric] for run in candidate_runs])),
                }
                for metric in ("macro_f1", "hate_speech_precision", "hate_speech_recall", "hate_speech_f1")
            },
        }

    result: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": manifest["protocol"]["protocol_id"],
        "public_manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
        "evaluated_splits": ["train", "validation"],
        "frozen_test_loaded": False,
        "seeds": list(seeds),
        "bootstrap_resamples": bootstrap_resamples,
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "provenance": {
            "slices_sha256": file_sha256(slices_path),
            "slices_semantic_sha256": semantic_file_sha256(slices_path),
        },
        "feature_configuration": {
            "word_ngrams": [1, 2],
            "word_max_features": 30_000,
            "character_ngrams": [3, 5],
            "character_max_features": 40_000,
            "class_weight": "balanced",
            "solver": "liblinear",
            "C": 4.0,
        },
        "runs": runs,
        "aggregates": aggregates,
    }
    result["integrity"] = {"payload_sha256": payload_sha256(result)}
    return result
