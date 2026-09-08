#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_hostile_cutover_r11 import matrix_manifest, run_hostile_matrix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the independent CB16 R11 Stage-4 S4H hostile cutover harness"
    )
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    parser.add_argument(
        "--matrix-only", action="store_true", help="Emit the matrix without executing cases"
    )
    args = parser.parse_args()
    report = matrix_manifest() if args.matrix_only else run_hostile_matrix()
    text = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.matrix_only:
        return 0
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
