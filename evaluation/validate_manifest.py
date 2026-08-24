#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate grouped_v1 manifest integrity and leakage gates.")
    parser.add_argument("manifest", nargs="?", type=Path, default=REPOSITORY_ROOT / "evaluation/manifests/grouped_v1.json")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    print(f"valid grouped_v1 manifest: {manifest['integrity']['payload_sha256']}")


if __name__ == "__main__":
    main()

