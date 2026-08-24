#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.artifact_evaluation import evaluate_validation_artifact
from hate_speech_adv.manifest import require_external_private_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an artifact on grouped_v1 validation only.")
    parser.add_argument("--runtime", required=True, choices=("pytorch", "onnx"))
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument(
        "--tuning-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--split-secret-file", required=True, type=Path)
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tuning_artifact = require_external_private_path(
        args.tuning_artifact,
        REPOSITORY_ROOT,
        "private train/validation artifact",
    )
    result = evaluate_validation_artifact(
        args.runtime,
        args.artifact,
        args.tokenizer,
        args.manifest,
        tuning_artifact,
        args.split_secret_file,
        args.slices,
        batch_size=args.batch_size,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
