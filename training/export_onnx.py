#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.export_pipeline import execute_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validation-selected checkpoint to fp32 ONNX and q8.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--selection-receipt", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/export/int8.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = execute_export(args.checkpoint, args.selection_receipt, args.config, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

