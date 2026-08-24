#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.baselines import BASELINE_SEEDS, run_baselines
from hate_speech_adv.manifest import require_external_private_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run majority and TF-IDF baselines without opening frozen test.")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument(
        "--tuning-artifact",
        type=Path,
        required=True,
    )
    parser.add_argument("--split-secret-file", required=True, type=Path)
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "evaluation/results/grouped_v1_baselines.json")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tuning_artifact = require_external_private_path(
        args.tuning_artifact,
        REPOSITORY_ROOT,
        "private train/validation artifact",
    )
    result = run_baselines(
        args.manifest,
        tuning_artifact,
        args.split_secret_file,
        args.slices,
        seeds=BASELINE_SEEDS,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            raise SystemExit("baseline results differ from committed artifact")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps(result["aggregates"], indent=2))


if __name__ == "__main__":
    main()
