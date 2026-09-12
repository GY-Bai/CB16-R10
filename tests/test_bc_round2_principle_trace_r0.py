from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_PRINCIPLE_TRACE_V1.json"
BASELINE_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_BASELINE_V1.json"

REQUIRED_IDS = {f"P-{index:02d}" for index in range(1, 11)}
TASK_RE = re.compile(r"^BC-(\d{3})$")
EXPECTED_BASELINE_MERGE_SHA = "d69c8686a33e077669266a06550426d3e4f0d812"
EXPECTED_BASELINE_BLOB_SHA = "9aec8635dd087d2e5c451c4d115ab257e8a2b973"
EXPECTED_SOURCE_SNAPSHOT = {
    "docs/PRINCIPLE_ALIGNMENT.md": "77012d60cb8fc3d34f58b865813e07ca9e81d2f3",
    "docs/COMPONENT_REQUIREMENTS.md": "8543bacecfdd6b929b9f404184725b75b7000648",
    "docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md": "d513b4be29eca4caed7d03e0b909d87d8bfa15f3",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_task(task_id: str) -> bool:
    match = TASK_RE.fullmatch(task_id)
    return bool(match and 1 <= int(match.group(1)) <= 105)


def _trace_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    principles = payload.get("principles")
    if not isinstance(principles, list):
        return ["principles must be a list"]

    ids = [entry.get("id") for entry in principles if isinstance(entry, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate principle id")
    if set(ids) != REQUIRED_IDS:
        errors.append("principle coverage must be exactly P-01 through P-10")

    for entry in principles:
        if not isinstance(entry, dict):
            errors.append("principle entry must be an object")
            continue
        pid = str(entry.get("id", "<missing>"))
        if not str(entry.get("requirement", "")).strip():
            errors.append(f"{pid}: missing requirement")

        for field in ("implementation_owners", "qualification_owners"):
            owners = entry.get(field)
            if not isinstance(owners, list) or not owners:
                errors.append(f"{pid}: missing {field}")
                continue
            if any(not isinstance(owner, str) or not _valid_task(owner) for owner in owners):
                errors.append(f"{pid}: invalid {field}")

        evidence = entry.get("observable_evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{pid}: missing observable_evidence")
        else:
            for item in evidence:
                if not isinstance(item, dict):
                    errors.append(f"{pid}: evidence entry must be an object")
                    continue
                if not _valid_task(str(item.get("task", ""))):
                    errors.append(f"{pid}: invalid evidence task")
                if not str(item.get("artifact", "")).strip():
                    errors.append(f"{pid}: missing evidence artifact")
                if item.get("status") != "EXPECTED":
                    errors.append(f"{pid}: future evidence must remain EXPECTED at BC-002")
                if not str(item.get("claim", "")).strip():
                    errors.append(f"{pid}: missing evidence claim")

        shortcuts = entry.get("forbidden_shortcuts")
        if not isinstance(shortcuts, list) or not shortcuts:
            errors.append(f"{pid}: missing forbidden_shortcuts")
        elif any(not isinstance(shortcut, str) or not shortcut.strip() for shortcut in shortcuts):
            errors.append(f"{pid}: invalid forbidden_shortcuts")

    return errors


def test_bc002_trace_manifest_identity_and_dependency_are_exact() -> None:
    trace = _load(TRACE_PATH)
    baseline = _load(BASELINE_PATH)

    assert trace["schema"] == "CB16_R11_BC_ROUND2_PRINCIPLE_TRACE_V1"
    assert trace["status"] == "FROZEN_TRACEABILITY_PLAN"
    assert trace["task_id"] == "BC-002"
    assert baseline["schema"] == "CB16_R11_BC_ROUND2_BASELINE_V1"
    assert baseline["task_id"] == "BC-001"

    dependency = trace["dependencies"]["BC-001"]
    assert dependency["required"] is True
    assert dependency["merged_main_sha"] == EXPECTED_BASELINE_MERGE_SHA
    assert dependency["artifact"] == str(BASELINE_PATH.relative_to(ROOT))
    assert dependency["artifact_git_blob_sha"] == EXPECTED_BASELINE_BLOB_SHA
    actual_blob = subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{dependency['artifact']}"], cwd=ROOT, text=True
    ).strip()
    assert actual_blob == EXPECTED_BASELINE_BLOB_SHA


def test_bc002_authority_sources_are_preregistered_snapshots() -> None:
    trace = _load(TRACE_PATH)
    sources = {entry["path"]: entry["git_blob_sha"] for entry in trace["authority_sources"]}

    assert sources == EXPECTED_SOURCE_SNAPSHOT
    assert trace["policy"]["future_artifact_status"] == "EXPECTED"
    assert trace["policy"]["rule"] == "A_PLANNED_OWNER_OR_EXPECTED_ARTIFACT_IS_NOT_EVIDENCE_OF_PASS"


def test_bc002_all_principles_have_implementation_qualification_evidence_and_shortcuts() -> None:
    trace = _load(TRACE_PATH)
    assert _trace_errors(trace) == []


def test_bc002_owner_and_evidence_mapping_fails_closed_when_missing() -> None:
    trace = _load(TRACE_PATH)

    no_qualification = deepcopy(trace)
    no_qualification["principles"][4]["qualification_owners"] = []
    assert any("P-05: missing qualification_owners" in error for error in _trace_errors(no_qualification))

    no_evidence = deepcopy(trace)
    no_evidence["principles"][9]["observable_evidence"] = []
    assert any("P-10: missing observable_evidence" in error for error in _trace_errors(no_evidence))

    no_shortcut_guard = deepcopy(trace)
    no_shortcut_guard["principles"][0]["forbidden_shortcuts"] = []
    assert any("P-01: missing forbidden_shortcuts" in error for error in _trace_errors(no_shortcut_guard))


def test_bc002_missing_or_duplicate_principle_fails_closed() -> None:
    trace = _load(TRACE_PATH)

    missing = deepcopy(trace)
    missing["principles"] = missing["principles"][:-1]
    assert "principle coverage must be exactly P-01 through P-10" in _trace_errors(missing)

    duplicate = deepcopy(trace)
    duplicate["principles"][9]["id"] = "P-09"
    errors = _trace_errors(duplicate)
    assert "duplicate principle id" in errors
    assert "principle coverage must be exactly P-01 through P-10" in errors
