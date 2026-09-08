#!/usr/bin/env python3
"""Audit CB16 R11 Stage-4 legacy-retirement declarations.

This script is intentionally a static/negative authority audit.  It never grants
runtime authority and never opens market data or scientific holdout content.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.stage4_legacy_retirement_r11 import (  # noqa: E402
    DeclaredCapabilityClaim,
    LegacyRetirementError,
    audit_declared_claims,
    self_audit_policy,
)


def _load_claims(path: Path) -> list[DeclaredCapabilityClaim]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        raise LegacyRetirementError("S4G_CLAIM_FILE_MUST_CONTAIN_CLAIMS_ARRAY")
    claims: list[DeclaredCapabilityClaim] = []
    for index, row in enumerate(raw["claims"]):
        if not isinstance(row, dict):
            raise LegacyRetirementError(f"S4G_CLAIM_ROW_NOT_OBJECT:{index}")
        if set(row) - {"subject_id", "role", "capability", "active"}:
            raise LegacyRetirementError(f"S4G_CLAIM_ROW_UNKNOWN_FIELD:{index}")
        try:
            claims.append(
                DeclaredCapabilityClaim(
                    subject_id=str(row["subject_id"]),
                    role=str(row["role"]),
                    capability=str(row["capability"]),
                    active=bool(row.get("active", True)),
                )
            )
        except KeyError as exc:
            raise LegacyRetirementError(
                f"S4G_CLAIM_ROW_MISSING_FIELD:{index}:{exc.args[0]}"
            ) from exc
    return claims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--claims-json",
        type=Path,
        help="Optional static claim declarations to audit. Declarations never establish authority.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run the complete synthetic role/capability denial matrix (also the default).",
    )
    args = parser.parse_args()

    try:
        report = self_audit_policy()
        if args.claims_json is not None:
            report["declared_claim_audit"] = audit_declared_claims(
                _load_claims(args.claims_json)
            )
        report["audit_status"] = "PASS"
        print(json.dumps(report, sort_keys=True, indent=2))
        return 0
    except (LegacyRetirementError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "schema": "CB16_R11_STAGE4_LEGACY_AUTHORITY_AUDIT_V1",
                    "audit_status": "FAIL",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
