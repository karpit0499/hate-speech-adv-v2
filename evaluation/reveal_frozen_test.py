#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.frozen_state import create_post_test_reveal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a publishable split-secret reveal after a complete one-shot test result."
    )
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--quality-gate", type=Path, required=True)
    parser.add_argument(
        "--slices",
        type=Path,
        default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--opened-test", type=Path, required=True)
    parser.add_argument("--test-result", type=Path, required=True)
    parser.add_argument("--split-secret-file", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "evaluation/reveals/grouped_v1_reveal.json",
    )
    args = parser.parse_args()
    reveal = create_post_test_reveal(
        manifest_path=args.manifest,
        receipt_path=args.receipt,
        ledger_path=args.ledger,
        opened_test_path=args.opened_test,
        test_result_path=args.test_result,
        split_secret_path=args.split_secret_file,
        export_manifest_path=args.export_manifest,
        quality_gate_path=args.quality_gate,
        slices_path=args.slices,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(reveal, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"state": "post_test_revealed", "output": str(args.output)}))


if __name__ == "__main__":
    main()
