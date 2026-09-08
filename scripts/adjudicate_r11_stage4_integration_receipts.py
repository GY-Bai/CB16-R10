#!/usr/bin/env python3
"""Adjudicate Stage-4 Integration fork receipts from an explicit manifest.

This CLI never discovers sibling Integration branches.  The final adjudicator
must provide immutable receipt/commit metadata in --candidate-manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cb16_local_opt.stage4_integration_gate_compiler_r11 import (
    IntegrationGateFailure,
    ReceiptInput,
    SubprocessRepositoryEvidence,
    compile_integration_receipts,
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--candidate-manifest", type=Path, required=True)
    p.add_argument(
        "--gatework",
        type=Path,
        default=Path("authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_GATEWORK_V1.json"),
    )
    p.add_argument(
        "--receipt-schema",
        type=Path,
        default=Path("authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_RECEIPT_SCHEMA_V1.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    ns = p.parse_args(argv)
    root = ns.repo_root.resolve()

    try:
        gatework = _load(root / ns.gatework)
        schema = _load(root / ns.receipt_schema)
        manifest = _load(ns.candidate_manifest)
        rows = manifest.get("receipts")
        if not isinstance(rows, list):
            raise IntegrationGateFailure("INTEGRATION_CANDIDATE_MANIFEST_INVALID", "receipts")
        inputs: list[ReceiptInput] = []
        for row in rows:
            if not isinstance(row, dict):
                raise IntegrationGateFailure("INTEGRATION_CANDIDATE_MANIFEST_INVALID", "row")
            receipt_path = root / str(row["path"])
            inputs.append(
                ReceiptInput(
                    task_id=str(row["task_id"]),
                    path=str(row["path"]),
                    document=_load(receipt_path),
                    receipt_commit_sha=str(row["receipt_commit_sha"]),
                    branch_tip_sha=str(row["branch_tip_sha"]),
                )
            )
        result = compile_integration_receipts(
            gatework=gatework,
            receipt_schema=schema,
            receipts=inputs,
            repo=SubprocessRepositoryEvidence(root),
        )
    except (IntegrationGateFailure, KeyError, OSError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, IntegrationGateFailure) else type(exc).__name__
        result = {
            "schema": "CB16_R11_STAGE4_INTEGRATION_PRECONSOLIDATION_REPORT_V1",
            "verdict": "STAGE4_INTEGRATION_FORK_RECEIPTS_FAIL_CLOSED",
            "failure_code": code,
            "failure_detail": str(exc),
            "final_cutover_verdict_emitted": False,
        }
        ns.output.parent.mkdir(parents=True, exist_ok=True)
        ns.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 2

    ns.output.parent.mkdir(parents=True, exist_ok=True)
    ns.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
