#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hate_speech_adv.frozen_state import verify_repository_frozen_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the pre- or post-test grouped_v1 state.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/evaluation/frozen_state.json",
    )
    args = parser.parse_args()
    print(json.dumps(verify_repository_frozen_state(args.config, REPOSITORY_ROOT), indent=2))


if __name__ == "__main__":
    main()
