from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_BASELINE_V1.json"
SEMANTIC_FREEZE_PATH = ROOT / "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json"

EXPECTED_BASE_MAIN_SHA = "c7938070d2a00c7f2898ebcd95ba55fac3a4736a"
EXPECTED_TODO_AUDIT_SHA = "c373d230e218e2fb83064d3a7fed529e174482b6"

EXPECTED_R0_BLOBS = {
    "cb16_local_opt/actor_critic_contract_r0.py": "ae76cd299be73ea822dea9adc54babc999833e16",
    "cb16_local_opt/actor_critic_runtime_router_r0.py": "e1ed865672766443dad4b3fa2928b0513df53a9d",
    "cb16_local_opt/action_contract_r0.py": "e978391559f55076fc1369e6e337395695d5361d",
    "cb16_local_opt/execution_record_r0.py": "ee9f6fbae6cdf291646163d0558570fcac53659e",
    "cb16_local_opt/actor_critic_supervisor_r0.py": "25846b96fb33140b8e80d9f22ef417cea86b1272",
    "cb16_local_opt/target_exposure_r0.py": "817705a041c8c9a59aedf6dcd68bb90e6adba5a2",
    "cb16_local_opt/actor_critic_physics_adapter_r0.py": "51a28efca8f592946ae2a69aec1291f8cac5e1f7",
}

EXPECTED_AUTHORITY_BLOBS = {
    "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json": "3c401a0a350984381912f7860181e3e96eb8d7cf",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json": "da8a5aa932eaa357280cb5e2a90c91c3e1301c54",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_GATEWORK_V1.json": "31bf4152a027f0a9c7fe5f4c184b3f9f43d302c4",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1.json": "7d975a0dfc66720628a461fa818866142b532ecd",
}

EXPECTED_DOC_SNAPSHOT = {
    "AGENTS.md": "223cd9208fd2ce017f43c79189c69ee9a654dd7d",
    "docs/VISION.md": "6991f42974d51db93fd19570b40e946a33672f49",
    "docs/PRINCIPLE_ALIGNMENT.md": "77012d60cb8fc3d34f58b865813e07ca9e81d2f3",
    "docs/COMPONENT_REQUIREMENTS.md": "8543bacecfdd6b929b9f404184725b75b7000648",
    "docs/CURRENT_STATE.md": "d714e14d8d0b9e2af931034a026851e3f9c9085f",
    "docs/R11_BC_ROUND2_CODE_ALIGNMENT_TODO.md": "d513b4be29eca4caed7d03e0b909d87d8bfa15f3",
}

EXPECTED_OBSERVED_BRANCH = {
    "name": "ai/r11-ac-015-actor-policy-interface-r0",
    "head_sha": "b21d826104830ecd80d4cbd096d86e012f4ec16f",
    "parent_sha": "c373d230e218e2fb83064d3a7fed529e174482b6",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries_to_map(entries: list[dict]) -> dict[str, str]:
    return {str(item["path"]): str(item["git_blob_sha"]) for item in entries}


def _git_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{path}"],
        cwd=ROOT,
        text=True,
    ).strip()


def _guard_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    guarded = payload.get("guarded_semantic_surfaces", {})
    declared_r0 = _entries_to_map(guarded.get("critical_r0_files", []))
    declared_authority = _entries_to_map(guarded.get("frozen_authorities", []))

    if declared_r0 != EXPECTED_R0_BLOBS:
        errors.append("critical R0 preregistration differs from BC-001 inventory")
    if declared_authority != EXPECTED_AUTHORITY_BLOBS:
        errors.append("frozen authority preregistration differs from BC-001 inventory")

    for path, expected_blob in {**EXPECTED_R0_BLOBS, **EXPECTED_AUTHORITY_BLOBS}.items():
        actual_blob = _git_blob(path)
        if actual_blob != expected_blob:
            errors.append(f"guarded blob drift: {path}: {actual_blob} != {expected_blob}")
    return errors


def test_bc001_baseline_identity_is_exact_live_main_at_task_start() -> None:
    baseline = _load_json(BASELINE_PATH)

    assert baseline["schema"] == "CB16_R11_BC_ROUND2_BASELINE_V1"
    assert baseline["status"] == "FROZEN"
    assert baseline["task_id"] == "BC-001"
    assert baseline["base"]["branch"] == "main"
    assert baseline["base"]["commit_sha"] == EXPECTED_BASE_MAIN_SHA
    assert baseline["base"]["todo_audit_baseline_sha"] == EXPECTED_TODO_AUDIT_SHA
    assert baseline["base"]["rule"] == "LIVE_MAIN_AT_TASK_START_IS_AUTHORITY"


def test_bc001_guarded_semantic_surfaces_are_byte_stable() -> None:
    baseline = _load_json(BASELINE_PATH)
    assert _guard_errors(baseline) == []


def test_bc001_verifier_fails_closed_on_preregistered_surface_drift() -> None:
    baseline = _load_json(BASELINE_PATH)
    tampered = deepcopy(baseline)
    tampered["guarded_semantic_surfaces"]["critical_r0_files"][0]["git_blob_sha"] = "0" * 40

    errors = _guard_errors(tampered)
    assert errors
    assert any("critical R0 preregistration differs" in error for error in errors)


def test_bc001_docs_are_snapshot_provenance_not_permanent_guards() -> None:
    baseline = _load_json(BASELINE_PATH)
    docs = _entries_to_map(baseline["snapshot_observations"]["current_docs"])

    assert docs == EXPECTED_DOC_SNAPSHOT
    assert "not permanent byte-stability guards" in baseline["snapshot_observations"]["note"]


def test_bc001_unmerged_branch_is_observation_only() -> None:
    baseline = _load_json(BASELINE_PATH)
    observed = baseline["snapshot_observations"]["unmerged_branches"]

    assert len(observed) == 1
    branch = observed[0]
    for key, expected in EXPECTED_OBSERVED_BRANCH.items():
        assert branch[key] == expected
    assert branch["authority"] is False
    assert branch["dependency"] is False
    assert baseline["dependency_policy"]["unmerged_branch_dependencies"] == "FORBIDDEN"
    assert baseline["dependency_policy"]["declared_dependencies"] == []


def test_bc001_final_and_fresh_data_boundaries_remain_closed() -> None:
    baseline = _load_json(BASELINE_PATH)
    semantic = _load_json(SEMANTIC_FREEZE_PATH)
    market = semantic["immutable"]["historical_market_dataset"]

    assert market["unopened_holdout_start"] == "2025-09-01T00:00:00Z"
    assert market["network_repair_or_redownload_during_scientific_runtime"] == "FORBIDDEN"

    boundaries = baseline["closed_boundaries"]
    assert boundaries["final_holdout"]["status"] == "SEALED"
    assert boundaries["final_holdout"]["unopened_holdout_start"] == market["unopened_holdout_start"]
    assert boundaries["final_holdout"]["payload_open_without_separate_authorization"] == "FORBIDDEN"
    assert boundaries["fresh_market_data"]["network_repair_or_redownload_during_scientific_runtime"] == "FORBIDDEN"
    assert boundaries["fresh_market_data"]["fresh_download_for_bc_round2"] == "FORBIDDEN"
    assert boundaries["round2_execution"]["canonical_scientific_replay_before_bc_a"] == "FORBIDDEN"
