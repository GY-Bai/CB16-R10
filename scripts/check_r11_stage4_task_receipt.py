#!/usr/bin/env python3
"""Fail-closed validator for CB16 R11 Stage-4 Wave-1 task receipts.

This is an infrastructure gate. It does not adjudicate scientific evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EXPECTED_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
EXPECTED_SCIENCE = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
FREEZE_PATH = "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"
MANIFEST_PATH = "authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json"
RECEIPT_SCHEMA = "CB16_R11_STAGE4_TASK_RECEIPT_V1"


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def require(cond: bool, code: str) -> None:
    if not cond:
        raise SystemExit(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    receipt_path = (root / args.receipt).resolve()
    require(receipt_path.is_file(), "STAGE4_RECEIPT_MISSING")

    manifest = json.loads((root / MANIFEST_PATH).read_text())
    receipt = json.loads(receipt_path.read_text())
    tasks = {x["id"]: x for x in manifest["wave1_tasks"]}

    require(receipt.get("schema") == RECEIPT_SCHEMA, "STAGE4_RECEIPT_SCHEMA_MISMATCH")
    task_id = receipt.get("task_id")
    require(task_id in tasks, "STAGE4_RECEIPT_UNKNOWN_TASK")
    expected_branch = tasks[task_id]["branch"]
    require(receipt.get("branch") == expected_branch, "STAGE4_RECEIPT_BRANCH_MISMATCH")
    require(receipt.get("status") in {"PASS", "FAIL", "BLOCKED", "NOT_RUN"}, "STAGE4_RECEIPT_BAD_STATUS")
    require(receipt.get("scientific_status") == EXPECTED_SCIENCE, "STAGE4_SCIENTIFIC_STATUS_CHANGED")

    guards = receipt.get("semantic_guards") or {}
    exact = {
        "semantic_freeze_unchanged": True,
        "final_holdout_untouched": True,
        "fresh_market_data_downloaded": False,
        "historical_market_data_mutated": False,
        "new_scientific_verdict": False,
        "scientific_semantics_changed": False,
        "replay_reinterpreted_as_new_evidence": False,
        "sibling_dependency_used": False,
    }
    for key, value in exact.items():
        require(guards.get(key) is value, f"STAGE4_GUARD_FAILED:{key}")

    freeze_blob = git("rev-parse", f"HEAD:{FREEZE_PATH}", cwd=root)
    require(freeze_blob == EXPECTED_FREEZE_BLOB, "STAGE4_SEMANTIC_FREEZE_BLOB_CHANGED")

    base = receipt.get("gatework_base_sha", "")
    head = receipt.get("task_head_sha", "")
    require(len(base) == 40 and len(head) == 40, "STAGE4_RECEIPT_BAD_SHA")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", base, head], cwd=root)
    subprocess.check_call(["git", "merge-base", "--is-ancestor", head, "HEAD"], cwd=root)

    merges = git("rev-list", "--merges", f"{base}..{head}", cwd=root).splitlines()
    require(not [x for x in merges if x], "STAGE4_SIBLING_OR_MERGE_COMMIT_FORBIDDEN")

    changed = {x for x in git("diff", "--name-only", f"{base}...{head}", cwd=root).splitlines() if x}
    declared = set(receipt.get("touched_paths") or [])
    require(changed == declared, f"STAGE4_TOUCHED_PATH_MISMATCH:actual={sorted(changed)}:declared={sorted(declared)}")

    forbidden = sorted(
        p for p in changed
        if p == FREEZE_PATH
        or "final_holdout" in p.lower()
        or "2025-09" in p.lower()
        or p.startswith("provision/assets/binance_usdm_1m_funding_2020_2026")
    )
    require(not forbidden, f"STAGE4_FORBIDDEN_PATH_CHANGE:{forbidden}")

    # Wave-1 is deliberately additive. Only S4E has one pre-authorized existing-file exception.
    existing_modified = []
    for p in sorted(changed):
        exists = subprocess.run(
            ["git", "cat-file", "-e", f"{base}:{p}"], cwd=root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
        if exists:
            existing_modified.append(p)
        else:
            require("stage4" in p.lower(), f"STAGE4_NEW_PATH_MUST_BE_NAMESPACED:{p}")

    allowed_existing = set(tasks[task_id].get("exclusive_existing_file_ownership") or [])
    require(set(existing_modified) <= allowed_existing,
            f"STAGE4_EXISTING_FILE_OWNERSHIP_VIOLATION:{sorted(set(existing_modified)-allowed_existing)}")

    if receipt["status"] == "PASS":
        require((receipt.get("tests") or {}).get("passed") is True, "STAGE4_PASS_WITHOUT_TEST_PASS")

    print(json.dumps({
        "schema": "CB16_R11_STAGE4_RECEIPT_VALIDATION_V1",
        "status": "PASS",
        "task_id": task_id,
        "qualified_implementation_head": head,
        "changed_paths": sorted(changed),
        "scientific_status": EXPECTED_SCIENCE,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
