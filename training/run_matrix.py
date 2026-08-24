#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.training_plan import build_run_plan
from hate_speech_adv.manifest import require_external_private_path
from hate_speech_adv.transformer_pipeline import (
    prune_matrix_checkpoint_storage,
    require_production_seed_last,
    run_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan, smoke-test, or execute the frozen grouped_v1 matrix.")
    parser.add_argument("--matrix", type=Path, default=REPOSITORY_ROOT / "configs/train/experiment_matrix.json")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument(
        "--tuning-artifact",
        type=Path,
    )
    parser.add_argument("--split-secret-file", type=Path)
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    plan = build_run_plan(args.matrix, args.manifest)
    if args.plan or args.smoke:
        summary = {
            "status": "smoke_passed" if args.smoke else "planned",
            "runs": len(plan["runs"]),
            "tuning_rows": plan["tuning_rows"],
            "loss_smoke": plan["loss_smoke"],
            "offline_training_smoke": plan["offline_training_smoke"],
            "frozen_test_loaded": plan["frozen_test_loaded"],
            "plan_sha256": plan["integrity"]["payload_sha256"],
        }
        print(json.dumps(summary, indent=2))
        return
    if (
        not args.candidate
        or args.seed is None
        or args.output_dir is None
        or args.split_secret_file is None
        or args.tuning_artifact is None
    ):
        parser.error(
            "--execute requires --candidate, --seed, --output-dir, "
            "--tuning-artifact, and --split-secret-file"
        )
    tuning_artifact = require_external_private_path(
        args.tuning_artifact,
        REPOSITORY_ROOT,
        "private train/validation artifact",
    )
    run_root = require_production_seed_last(
        args.matrix,
        args.candidate,
        args.seed,
        args.output_dir,
    )
    result = run_training(
        args.matrix,
        args.manifest,
        tuning_artifact,
        args.split_secret_file,
        args.slices,
        args.candidate,
        args.seed,
        args.output_dir,
    )
    retention = prune_matrix_checkpoint_storage(args.matrix, run_root)
    print(json.dumps({"metrics": result["metrics"], "checkpoint_retention": retention}, indent=2))


if __name__ == "__main__":
    main()
