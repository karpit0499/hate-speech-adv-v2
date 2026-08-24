#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.data import load_pinned_dataset
from hate_speech_adv.manifest import require_external_private_path, validate_manifest
from hate_speech_adv.protocol import build_protocol_artifacts, load_json, read_split_secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the sealed grouped_v1 split manifest.")
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/data/grouped_v1.json")
    parser.add_argument("--slices", type=Path, default=REPOSITORY_ROOT / "configs/evaluation/slices_v1.json")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    parser.add_argument("--audit-output", type=Path, default=REPOSITORY_ROOT / "evaluation/audits/grouped_v1_audit.json")
    parser.add_argument(
        "--private-output",
        type=Path,
        required=True,
        help="Mode-0600 external artifact containing train/validation rows only.",
    )
    parser.add_argument("--split-secret-file", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Fail instead of overwriting when generated content differs.")
    return parser.parse_args()


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"generated artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_or_check_private(path: Path, content: str, check: bool) -> None:
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"generated artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def main() -> None:
    args = parse_args()
    private_output = require_external_private_path(
        args.private_output,
        REPOSITORY_ROOT,
        "private train/validation artifact",
    )
    configuration = load_json(args.config)
    slice_configuration = load_json(args.slices)
    dataset = load_pinned_dataset(
        configuration["dataset"]["repository"],
        configuration["dataset"]["revision"],
    )
    manifest, private_artifact, audit = build_protocol_artifacts(
        dataset,
        configuration,
        args.config,
        slice_configuration,
        args.slices,
        read_split_secret(args.split_secret_file),
    )
    validate_manifest(manifest)
    write_or_check(args.output, compact_json(manifest), args.check)
    write_or_check(args.audit_output, compact_json(audit), args.check)
    write_or_check_private(private_output, compact_json(private_artifact), args.check)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "private_tuning_artifact": str(private_output),
                "audit": str(args.audit_output),
                "summary": manifest["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
