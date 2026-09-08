#!/usr/bin/env python3
"""CLI for fail-closed Stage-4 Wave-1 receipt aggregation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from cb16_local_opt.stage4_gate_compiler_r11 import (
    EXPECTED_GATEWORK_BASE,
    GateFailure,
    SubprocessRepositoryEvidence,
    compile_wave1_receipts,
    load_receipts,
)

MANIFEST = Path("authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json")
SCHEMA = Path("authority/rearchitecture_r11/CB16_R11_STAGE4_TASK_RECEIPT_SCHEMA_V1.json")
DEFAULT_RECEIPT_DIR = Path("authority/rearchitecture_r11/stage4_receipts")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--receipt-dir", default=str(DEFAULT_RECEIPT_DIR))
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    try:
        manifest = json.loads((root / MANIFEST).read_text())
        schema = json.loads((root / SCHEMA).read_text())
        receipt_dir = (root / args.receipt_dir).resolve()
        paths = sorted(receipt_dir.glob("S4*_RECEIPT_V1.json")) if receipt_dir.is_dir() else []
        receipts = load_receipts(paths, root)
        result = compile_wave1_receipts(
            manifest=manifest,
            receipt_schema=schema,
            receipts=receipts,
            repo=SubprocessRepositoryEvidence(root),
            repository_head="HEAD",
            expected_gatework_base=EXPECTED_GATEWORK_BASE,
        )
    except (GateFailure, OSError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, GateFailure) else "STAGE4_GATE_COMPILER_INPUT_ERROR"
        print(json.dumps({"schema": "CB16_R11_STAGE4_WAVE1_GATE_ADJUDICATION_V1", "status": "FAIL", "reason": code}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
