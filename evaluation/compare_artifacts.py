#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.artifact_evaluation import compare_validation_artifacts
from hate_speech_adv.integrity import file_sha256, semantic_file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the validation-only fp32-to-q8 quality gate.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--export-manifest", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/export/int8.json")
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    export_manifest = json.loads(args.export_manifest.read_text(encoding="utf-8"))
    slice_configuration = json.loads(args.slices.read_text(encoding="utf-8"))
    result = compare_validation_artifacts(
        reference,
        candidate,
        configuration,
        export_manifest,
        slice_configuration,
        configuration_sha256=file_sha256(args.config),
        slices_sha256=file_sha256(args.slices),
        slices_semantic_sha256=semantic_file_sha256(args.slices),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
