#!/usr/bin/env python3
"""Compatibility entrypoint for the retired public-seed test evaluator.

The historical script reconstructed row-level held-out membership from a
public seed. That is incompatible with grouped_v1 and is intentionally not
retained as executable code.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Legacy evaluation is disabled. Complete validation-only selection, then use "
        "evaluation/open_frozen_test.py exactly once under the grouped_v1 protocol."
    )


if __name__ == "__main__":
    main()
