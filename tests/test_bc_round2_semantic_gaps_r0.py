from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
GAPS_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_SEMANTIC_GAPS_V1.json"

REQUIRED_GAPS = {f"G-{index:02d}" for index in range(1, 11)}
TASK_RE = re.compile(r"^BC-(\d{3})$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_DEPENDENCIES = {
    "BC-001": {
        "merged_main_sha": "d69c8686a33e077669266a06550426d3e4f0d812",
        "artifact": "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_BASELINE_V1.json",
        "artifact_git_blob_sha": "9aec8635dd087d2e5c451c4d115ab257e8a2b973",
    },
    "BC-002": {
        "merged_main_sha": "f09492830ef2d5cda12ac055f329388f7fd43bc3",
        "artifact": "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_PRINCIPLE_TRACE_V1.json",
        "artifact_git_blob_sha": "f81fb0c344f14e584bca790666dac035cc1cccf4",
    },
}


def _load() -> dict:
    return json.loads(GAPS_PATH.read_text(encoding="utf-8"))


def _valid_task(value: object) -> bool:
    if not isinstance(value, str):
        return False
    match = TASK_RE.fullmatch(value)
    return bool(match and 1 <= int(match.group(1)) <= 105)


def _gap_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    gaps = payload.get("gaps")
    if not isinstance(gaps, list):
        return ["gaps must be a list"]

    ids = [entry.get("id") for entry in gaps if isinstance(entry, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate gap id")
    if set(ids) != REQUIRED_GAPS:
        errors.append("gap coverage must be exactly G-01 through G-10")

    for gap in gaps:
        if not isinstance(gap, dict):
            errors.append("gap entry must be an object")
            continue
        gid = str(gap.get("id", "<missing>"))
        if not str(gap.get("title", "")).strip():
            errors.append(f"{gid}: missing title")

        status = gap.get("status")
        if status not in {"OPEN", "RESOLVED"}:
            errors.append(f"{gid}: invalid status")

        refs = gap.get("source_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{gid}: missing source_refs")
        else:
            for ref in refs:
                if not isinstance(ref, dict):
                    errors.append(f"{gid}: invalid source_ref")
                    continue
                if not str(ref.get("path", "")).strip():
                    errors.append(f"{gid}: missing source path")
                if not str(ref.get("symbol", "")).strip():
                    errors.append(f"{gid}: missing source symbol")
                if not str(ref.get("observation", "")).strip():
                    errors.append(f"{gid}: missing source observation")
                blob = ref.get("git_blob_sha")
                if not isinstance(blob, str) or HEX40_RE.fullmatch(blob) is None:
                    errors.append(f"{gid}: invalid source blob sha")

        owners = gap.get("resolution_tasks")
        if not isinstance(owners, list) or not owners:
            errors.append(f"{gid}: missing resolution_tasks")
        elif any(not _valid_task(owner) for owner in owners):
            errors.append(f"{gid}: invalid resolution_tasks")

        gate = gap.get("required_resolution_gate")
        if not _valid_task(gate):
            errors.append(f"{gid}: invalid required_resolution_gate")
        receipt = gap.get("expected_gate_receipt")
        if not isinstance(receipt, str) or not receipt.strip():
            errors.append(f"{gid}: missing expected_gate_receipt")

        evidence = gap.get("resolution_evidence")
        if not isinstance(evidence, list):
            errors.append(f"{gid}: resolution_evidence must be a list")
            evidence = []

        if status == "RESOLVED":
            matching = [
                item
                for item in evidence
                if isinstance(item, dict)
                and item.get("task") == gate
                and item.get("artifact") == receipt
                and item.get("outcome") == "PASS"
            ]
            if not matching:
                errors.append(f"{gid}: stale RESOLVED without required gate PASS evidence")

    return errors


def test_bc003_registry_identity_and_dependencies_are_exact() -> None:
    payload = _load()

    assert payload["schema"] == "CB16_R11_BC_ROUND2_SEMANTIC_GAPS_V1"
    assert payload["status"] == "FROZEN_BASELINE_GAP_REGISTRY"
    assert payload["task_id"] == "BC-003"
    assert payload["dependencies"] == EXPECTED_DEPENDENCIES

    for dependency in EXPECTED_DEPENDENCIES.values():
        actual_blob = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{dependency['artifact']}"],
            cwd=ROOT,
            text=True,
        ).strip()
        assert actual_blob == dependency["artifact_git_blob_sha"]


def test_bc003_all_ten_baseline_gaps_have_sources_and_resolution_owners() -> None:
    payload = _load()

    assert _gap_errors(payload) == []
    assert {entry["id"] for entry in payload["gaps"]} == REQUIRED_GAPS
    assert all(entry["status"] == "OPEN" for entry in payload["gaps"])
    assert all(entry["resolution_evidence"] == [] for entry in payload["gaps"])


def test_bc003_missing_gap_or_resolution_owner_fails_closed() -> None:
    payload = _load()

    missing_gap = deepcopy(payload)
    missing_gap["gaps"] = missing_gap["gaps"][:-1]
    assert "gap coverage must be exactly G-01 through G-10" in _gap_errors(missing_gap)

    missing_owner = deepcopy(payload)
    missing_owner["gaps"][1]["resolution_tasks"] = []
    assert "G-02: missing resolution_tasks" in _gap_errors(missing_owner)


def test_bc003_stale_resolved_without_gate_pass_evidence_fails_closed() -> None:
    payload = _load()

    stale = deepcopy(payload)
    stale["gaps"][0]["status"] = "RESOLVED"
    errors = _gap_errors(stale)
    assert "G-01: stale RESOLVED without required gate PASS evidence" in errors

    wrong_evidence = deepcopy(payload)
    gap = wrong_evidence["gaps"][0]
    gap["status"] = "RESOLVED"
    gap["resolution_evidence"] = [
        {
            "task": "BC-019",
            "artifact": gap["expected_gate_receipt"],
            "outcome": "PASS",
        }
    ]
    assert "G-01: stale RESOLVED without required gate PASS evidence" in _gap_errors(wrong_evidence)


def test_bc003_exact_required_gate_pass_can_support_resolution_state() -> None:
    payload = _load()
    synthetic = deepcopy(payload)
    gap = synthetic["gaps"][0]
    gap["status"] = "RESOLVED"
    gap["resolution_evidence"] = [
        {
            "task": gap["required_resolution_gate"],
            "artifact": gap["expected_gate_receipt"],
            "outcome": "PASS",
        }
    ]

    assert _gap_errors(synthetic) == []
