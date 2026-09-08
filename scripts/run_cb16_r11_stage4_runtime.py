#!/usr/bin/env python3
from __future__ import annotations

"""Stage-4 canonical R11 process entrypoint.

Concrete authority-bearing providers are intentionally not wired by S4B. The
unintegrated entrypoint therefore fails closed instead of falling back to any
Stage-2/Stage-3 qualification harness or legacy runtime.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_canonical_runtime_r11 import (  # noqa: E402
    Stage4RuntimeError,
    build_fail_closed_runtime,
    lifecycle_description,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CB16 R11 Stage-4 canonical runtime entrypoint")
    parser.add_argument(
        "--describe-lifecycle",
        action="store_true",
        help="print the canonical lifecycle without attempting authority acquisition",
    )
    args = parser.parse_args(argv)

    if args.describe_lifecycle:
        print(lifecycle_description())
        return 0

    runtime = build_fail_closed_runtime()
    try:
        runtime.start()
    except Stage4RuntimeError as exc:
        print(f"R11_STAGE4_RUNTIME_FAIL_CLOSED:{exc}", file=sys.stderr)
        return 2

    # S4B never supplies a provider set capable of reaching this point. Final
    # integration will construct the controller with qualified concrete providers.
    print("R11_STAGE4_UNEXPECTED_UNINTEGRATED_RUNNING", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
