#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.selection import create_selection_receipt, lock_selection_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock model selection from validation-only results.")
    parser.add_argument("results", nargs="*", type=Path)
    parser.add_argument("--matrix", type=Path, default=REPOSITORY_ROOT / "configs/train/experiment_matrix.json")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--lock-draft", type=Path)
    parser.add_argument("--protocol-commit")
    parser.add_argument("--results-commit")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "evaluation/selection/grouped_v1_selection.json")
    args = parser.parse_args()
    if args.lock_draft:
        if args.results or not args.protocol_commit or not args.results_commit:
            parser.error("--lock-draft requires --protocol-commit, --results-commit, and no result paths")
        receipt = lock_selection_receipt(
            json.loads(args.lock_draft.read_text(encoding="utf-8")),
            repository_root=REPOSITORY_ROOT,
            protocol_commit=args.protocol_commit,
            results_commit=args.results_commit,
        )
    else:
        if not args.results:
            parser.error("result paths are required when creating a selection draft")
        receipt = create_selection_receipt(
            args.results,
            matrix_path=args.matrix,
            manifest_path=args.manifest,
            slices_path=args.slices,
            repository_root=REPOSITORY_ROOT,
            protocol_commit=args.protocol_commit,
            results_commit=args.results_commit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_locked": receipt["selection_locked"], "selected": receipt["selected"]}, indent=2))


if __name__ == "__main__":
    main()
