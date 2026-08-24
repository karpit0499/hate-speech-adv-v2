#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.data import load_pinned_dataset
from hate_speech_adv.manifest import load_manifest
from hate_speech_adv.protocol import build_grouped_partition, build_audit, load_json, read_split_secret


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute leakage, ambiguity, and hard-negative audits.")
    parser.add_argument("--manifest", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "evaluation/audits/grouped_v1_audit.json")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/data/grouped_v1.json")
    parser.add_argument("--split-secret-file", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    dataset = load_pinned_dataset(
        manifest["protocol"]["dataset_repository"],
        manifest["protocol"]["dataset_revision"],
    )
    split_names = build_grouped_partition(
        dataset,
        load_json(args.config),
        read_split_secret(args.split_secret_file),
    )
    audit = build_audit(dataset, manifest, split_names, load_json(args.slices))
    content = json.dumps(audit, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != content:
            raise SystemExit("data audit differs from committed artifact")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
