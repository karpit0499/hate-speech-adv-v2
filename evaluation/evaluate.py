#!/usr/bin/env python3
"""Reproduce the held-out metrics for both published classifier artifacts.

The repository originally published rounded fp32 metrics but did not include the
evaluation program.  This script pins the dataset and model revisions, rebuilds
the exact 80/10/10 stratified split, evaluates the fp32 PyTorch and q8 ONNX
artifacts, and optionally benchmarks q8 batch-1 CPU inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import huggingface_hub
import numpy as np
import onnxruntime as ort
import sklearn
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DATASET_ID = "tdavidson/hate_speech_offensive"
DATASET_REVISION = "adc5fb774614827695774f2dbe0ea8122f6a92b4"
FP32_MODEL_ID = "kmrarpit2704/hs-roberta-v2"
FP32_MODEL_REVISION = "bcaa0d20b4dd0d7af95ae732d9007893328206ae"
ONNX_MODEL_ID = "kmrarpit2704/hs-roberta-v2-onnx"
ONNX_MODEL_REVISION = "7c7c7c63c53f0e8373e5171c85c7e4a1a9a7e267"
SOURCE_REPOSITORY = "https://github.com/karpit0499/hate-speech-adv-v2"
SOURCE_BASE_COMMIT = "8e6cde0004b321e835979fcdeeab4576aeb1284c"

LABELS = ["hate_speech", "offensive", "neither"]
MAX_LENGTH = 256
SEED = 42
BENCHMARK_TEXT = "The service was slow but the staff were friendly and helpful."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        choices=("both", "fp32", "q8"),
        default="both",
        help="Artifact(s) to evaluate (default: both).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("metrics.json"),
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip the q8 batch-1 latency benchmark.",
    )
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_summary(milliseconds: list[float]) -> dict[str, float]:
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def hardware() -> dict[str, Any]:
    details: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        for key, name in (
            ("machdep.cpu.brand_string", "cpu_brand"),
            ("hw.memsize", "memory_bytes"),
        ):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
                ).strip()
                details[name] = int(value) if name == "memory_bytes" else value
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
    return details


def load_test_split():
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION)["train"]
    dataset = dataset.add_column("source_row_id", list(range(len(dataset))))

    # This two-stage split is load-bearing. A direct 10% split with the same
    # seed selects different rows and does not reproduce the published metrics.
    train_and_holdout = dataset.train_test_split(
        test_size=0.20,
        seed=SEED,
        stratify_by_column="class",
    )
    validation_and_test = train_and_holdout["test"].train_test_split(
        test_size=0.50,
        seed=SEED,
        stratify_by_column="class",
    )
    test = validation_and_test["test"]

    ordered_ids = ",".join(str(value) for value in test["source_row_id"])
    split_metadata = {
        "method": (
            "datasets.Dataset.train_test_split(test_size=0.20, seed=42, "
            "stratify_by_column='class'), then split the 20% holdout with "
            "test_size=0.50 using the same seed and stratification; evaluate "
            "the second split's test partition"
        ),
        "sizes": {
            "train": len(train_and_holdout["train"]),
            "validation": len(validation_and_test["train"]),
            "test": len(test),
        },
        "test_support": {
            LABELS[label]: int(sum(value == label for value in test["class"]))
            for label in range(len(LABELS))
        },
        "ordered_test_source_row_ids_sha256": hashlib.sha256(
            ordered_ids.encode("utf-8")
        ).hexdigest(),
    }
    return test, split_metadata


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    label_ids = list(range(len(LABELS)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "macro_avg": {
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
            "support": len(y_true),
        },
        "weighted_avg": {
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
            "support": len(y_true),
        },
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            y_true,
            y_pred,
            labels=label_ids,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_ids,
            target_names=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_fp32(test, batch_size: int) -> tuple[dict[str, Any], Path]:
    snapshot = Path(
        snapshot_download(FP32_MODEL_ID, revision=FP32_MODEL_REVISION)
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot,
        local_files_only=True,
    ).eval()

    predictions: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(test), batch_size):
            encoded = tokenizer(
                test[offset : offset + batch_size]["tweet"],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            predictions.extend(logits.argmax(dim=-1).tolist())
    elapsed = time.perf_counter() - started

    weights = snapshot / "model.safetensors"
    result = calculate_metrics(test["class"], predictions)
    result.update(
        {
            "artifact": {
                "repository": FP32_MODEL_ID,
                "revision": FP32_MODEL_REVISION,
                "file": "model.safetensors",
                "size_bytes": weights.stat().st_size,
                "sha256": sha256_file(weights),
            },
            "evaluation_runtime": {
                "device": "cpu",
                "batch_size": batch_size,
                "max_length": MAX_LENGTH,
                "elapsed_seconds": elapsed,
                "examples_per_second": len(test) / elapsed,
            },
        }
    )
    return result, snapshot


def make_ort_session(model_path: Path) -> ort.InferenceSession:
    # Thread values remain 0, which means ONNX Runtime chooses its default CPU
    # thread pools. The values are recorded in metrics.json with the hardware.
    options = ort.SessionOptions()
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_q8(
    session: ort.InferenceSession,
    tokenizer,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    input_names = {item.name for item in session.get_inputs()}
    encoded = tokenizer(
        BENCHMARK_TEXT,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    feeds = {key: value for key, value in encoded.items() if key in input_names}

    for _ in range(warmup):
        session.run(None, feeds)
    model_only: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        session.run(None, feeds)
        model_only.append((time.perf_counter_ns() - started) / 1_000_000)

    for _ in range(warmup):
        current = tokenizer(
            BENCHMARK_TEXT,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        session.run(
            None,
            {key: value for key, value in current.items() if key in input_names},
        )
    end_to_end: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        current = tokenizer(
            BENCHMARK_TEXT,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        session.run(
            None,
            {key: value for key, value in current.items() if key in input_names},
        )
        end_to_end.append((time.perf_counter_ns() - started) / 1_000_000)

    return {
        "scope": (
            "Native ONNX Runtime CPUExecutionProvider on this machine; this is "
            "not a browser/WASM benchmark"
        ),
        "batch_size": 1,
        "input_text": BENCHMARK_TEXT,
        "input_token_count_including_special_tokens": int(
            feeds["input_ids"].shape[1]
        ),
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "onnxruntime_session_threads": {
            "intra_op_num_threads": 0,
            "inter_op_num_threads": 0,
            "meaning": "ONNX Runtime defaults",
        },
        "model_only": percentile_summary(model_only),
        "tokenization_plus_model": percentile_summary(end_to_end),
    }


def evaluate_q8(
    test,
    batch_size: int,
    warmup: int,
    repeats: int,
    run_benchmark: bool,
) -> tuple[dict[str, Any], Path]:
    snapshot = Path(
        snapshot_download(
            ONNX_MODEL_ID,
            revision=ONNX_MODEL_REVISION,
            allow_patterns=[
                "config.json",
                "merges.txt",
                "onnx/model.onnx",
                "onnx/model_quantized.onnx",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    quantized = snapshot / "onnx" / "model_quantized.onnx"
    full_precision = snapshot / "onnx" / "model.onnx"
    session = make_ort_session(quantized)
    input_names = {item.name for item in session.get_inputs()}

    # Benchmark before the full test-set pass so sustained evaluation load does
    # not become an undocumented warm-up phase for the batch-1 measurement.
    latency = None
    if run_benchmark:
        latency = benchmark_q8(session, tokenizer, warmup, repeats)

    predictions: list[int] = []
    model_seconds = 0.0
    started = time.perf_counter()
    for offset in range(0, len(test), batch_size):
        encoded = tokenizer(
            test[offset : offset + batch_size]["tweet"],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        feeds = {key: value for key, value in encoded.items() if key in input_names}
        inference_started = time.perf_counter()
        logits = session.run(None, feeds)[0]
        model_seconds += time.perf_counter() - inference_started
        predictions.extend(np.argmax(logits, axis=-1).tolist())
    elapsed = time.perf_counter() - started

    result = calculate_metrics(test["class"], predictions)
    result.update(
        {
            "artifact": {
                "repository": ONNX_MODEL_ID,
                "revision": ONNX_MODEL_REVISION,
                "file": "onnx/model_quantized.onnx",
                "size_bytes": quantized.stat().st_size,
                "sha256": sha256_file(quantized),
                "fp32_onnx_size_bytes": full_precision.stat().st_size,
                "fp32_onnx_sha256": sha256_file(full_precision),
                "size_reduction_vs_fp32_onnx_percent": (
                    1 - quantized.stat().st_size / full_precision.stat().st_size
                )
                * 100,
                "fp32_onnx_to_q8_size_ratio": (
                    full_precision.stat().st_size / quantized.stat().st_size
                ),
            },
            "evaluation_runtime": {
                "provider": session.get_providers(),
                "batch_size": batch_size,
                "max_length": MAX_LENGTH,
                "elapsed_seconds_including_tokenization": elapsed,
                "model_elapsed_seconds": model_seconds,
                "examples_per_second_including_tokenization": len(test) / elapsed,
                "model_examples_per_second": len(test) / model_seconds,
            },
        }
    )
    if latency is not None:
        result["latency_benchmark"] = latency
    return result, snapshot


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("batch-size and repeats must be positive; warmup cannot be negative")

    test, split_metadata = load_test_split()
    results: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": {
            "url": SOURCE_REPOSITORY,
            "evaluated_base_commit": SOURCE_BASE_COMMIT,
            "note": (
                "This is the repository commit inspected before the "
                "reproducibility files were added."
            ),
        },
        "dataset": {
            "repository": DATASET_ID,
            "revision": DATASET_REVISION,
            "source_rows": 24_783,
        },
        "split": split_metadata,
        "labels_in_model_order": LABELS,
        "software": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hardware": hardware(),
        "results": {},
    }

    # q8 runs first because its latency benchmark should not follow the much
    # longer fp32 evaluation workload.
    if args.artifact in ("both", "q8"):
        q8, _ = evaluate_q8(
            test,
            args.batch_size,
            args.warmup,
            args.repeats,
            not args.no_benchmark,
        )
        results["results"]["q8_onnx"] = q8
    if args.artifact in ("both", "fp32"):
        fp32, _ = evaluate_fp32(test, args.batch_size)
        results["results"]["fp32_pytorch"] = fp32

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
