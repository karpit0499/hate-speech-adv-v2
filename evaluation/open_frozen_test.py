#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume the grouped_v1 one-shot ledger and score the frozen test once."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/data/grouped_v1.json",
    )
    parser.add_argument(
        "--slices",
        type=Path,
        default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json",
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--export-manifest", required=True, type=Path)
    parser.add_argument("--quality-gate", required=True, type=Path)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--split-secret-file", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--private-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--acknowledgement", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    # score_frozen_test consumes the durable ledger before it reads any supplied
    # evidence, secret, or dataset.
    from hate_speech_adv.data import load_pinned_dataset
    from hate_speech_adv.frozen_state import score_frozen_test
    from hate_speech_adv.manifest import FROZEN_TEST_ACKNOWLEDGEMENT

    if args.acknowledgement != FROZEN_TEST_ACKNOWLEDGEMENT:
        parser.error(f"--acknowledgement must be {FROZEN_TEST_ACKNOWLEDGEMENT}")
    result = score_frozen_test(
        manifest_path=args.manifest,
        configuration_path=args.config,
        receipt_path=args.receipt,
        export_manifest_path=args.export_manifest,
        quality_gate_path=args.quality_gate,
        export_root=args.export_root,
        split_secret_path=args.split_secret_file,
        ledger_path=args.ledger,
        private_evidence_path=args.private_evidence,
        public_result_path=args.output,
        slices_path=args.slices,
        repository_root=REPOSITORY_ROOT,
        acknowledgement=args.acknowledgement,
        dataset_loader=load_pinned_dataset,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=42,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "status": "one_shot_scoring_complete",
                "result_payload_sha256": result["integrity"]["payload_sha256"],
                "metrics": result["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
