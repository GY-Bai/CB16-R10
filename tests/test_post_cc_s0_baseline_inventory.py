from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _git_blob_sha(root: Path, path: str) -> str:
    return subprocess.check_output(
        ["git", "hash-object", path], cwd=root, text=True
    ).strip()


def test_s0_baseline_inventory_matches_all_frozen_files():
    root = Path(__file__).resolve().parents[1]
    baseline_path = (
        root
        / "authority/rearchitecture_r11/CB16_R11_POST_CC_S0_BASELINE_V1.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    records = (
        baseline["parent_cc"]["integration_spec"],
        baseline["parent_cc"]["integration_receipt"],
        baseline["parent_cc"]["thread_c_receipt"],
        baseline["legacy_economic_surfaces"]["evaluator"],
        baseline["legacy_economic_surfaces"]["promotion"],
        baseline["legacy_economic_surfaces"]["promotion_test"],
        baseline["planning_authority"]["economic_ordering_and_promotion"],
        baseline["planning_authority"]["post_cc_scientific_program"],
    )

    mismatches = []
    for record in records:
        observed = _git_blob_sha(root, record["path"])
        expected = record["blob_sha"]
        if observed != expected:
            mismatches.append(
                {
                    "path": record["path"],
                    "expected": expected,
                    "observed": observed,
                }
            )

    assert not mismatches, mismatches


def test_historical_receipts_are_inputs_not_edit_targets():
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads(
        (
            root
            / "authority/rearchitecture_r11/CB16_R11_POST_CC_S0_BASELINE_V1.json"
        ).read_text(encoding="utf-8")
    )
    assert baseline["parent_cc"]["integration_receipt"]["edit_target"] is False
    assert baseline["parent_cc"]["thread_c_receipt"]["edit_target"] is False
    assert baseline["historical_receipt_rule"]["rewrite_historical_cc_receipts"] is False
    assert baseline["historical_receipt_rule"]["rewrite_legacy_promotion_semantics"] is False
