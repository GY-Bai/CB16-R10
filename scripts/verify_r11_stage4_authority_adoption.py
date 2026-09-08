#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cb16_local_opt.stage4_authority_adoption_r11 import (  # noqa: E402
    AuthorityAdoptionContract,
    AuthorityAdoptionError,
    load_adoption_receipt,
)


def _load_contract(path: Path) -> AuthorityAdoptionContract:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityAdoptionError("STAGE4_ADOPTION_CONTRACT_READ_FAILED") from exc
    if not isinstance(obj, dict):
        raise AuthorityAdoptionError("STAGE4_ADOPTION_CONTRACT_NOT_OBJECT")
    return AuthorityAdoptionContract.from_mapping(obj)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a Stage-4 S4C authority-adoption receipt without mutating authority state."
    )
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        help="Optional JSON contract containing accepted_source and target_r11_authority_identity.",
    )
    args = parser.parse_args(argv)

    try:
        contract = _load_contract(args.contract) if args.contract is not None else None
        receipt = load_adoption_receipt(args.receipt, contract=contract)
    except AuthorityAdoptionError as exc:
        print(
            json.dumps(
                {
                    "schema": "CB16_R11_STAGE4_AUTHORITY_ADOPTION_VERIFY_V1",
                    "status": "FAIL",
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2

    print(
        json.dumps(
            {
                "schema": "CB16_R11_STAGE4_AUTHORITY_ADOPTION_VERIFY_V1",
                "status": "PASS",
                "canonical_content_hash": receipt["canonical_content_hash"],
                "source_generation": receipt["source"]["source_generation"],
                "adoption_generation": receipt["target"]["adoption_generation"],
                "new_evidence_created": receipt["semantic_guards"]["new_evidence_created"],
                "new_scientific_verdict": receipt["semantic_guards"]["new_scientific_verdict"],
                "scientific_history_rewritten": receipt["semantic_guards"]["scientific_history_rewritten"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
