from __future__ import annotations

import json
from pathlib import Path
import subprocess

from cb16_local_opt.actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_VERSION_FIELDS_R0,
    validate_actor_critic_science_contract_r0,
)
from cb16_local_opt.actor_critic_contract_r1 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
    SCIENCE_VERSION_FIELDS_R1,
)
from cb16_local_opt.actor_critic_runtime_router_r0 import (
    ACTOR_CRITIC_R0,
    LEGACY_R11,
    runtime_lane_contract_payload_r0,
    runtime_lane_contract_sha256_r0,
)
from cb16_local_opt.actor_critic_runtime_router_r1 import (
    BC_ROUND2_R1,
    RUNTIME_LANES_R1,
    runtime_lane_contract_payload_r1,
    runtime_lane_contract_sha256_r1,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_ROUND2_BASELINE_V1.json"
EXPECTED_BASELINE_BLOB = "9aec8635dd087d2e5c451c4d115ab257e8a2b973"

EXPECTED_LEGACY_FREEZE_BLOBS = {
    "cb16_local_opt/typed_central_brain_r10.py": "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff",
    "cb16_local_opt/r102_physics.py": "a9b3696dfa04bf8e5f72bff96f6480397407232d",
    "cb16_local_opt/training_runtime_r11.py": "56c227800d7a5c82ef88c691bb69d7550aee8b39",
    "tests/test_r11_trace_runtime.py": "d35bc1e2e10ceeb1c936522764bbf7d15739d341",
    "tests/test_actor_critic_legacy_freeze_r0.py": "5eb5186c531e5810f2172cc2122f22a55b7e0e29",
}

EXPECTED_R0_BLOBS = {
    "cb16_local_opt/actor_critic_contract_r0.py": "ae76cd299be73ea822dea9adc54babc999833e16",
    "cb16_local_opt/actor_critic_runtime_router_r0.py": "e1ed865672766443dad4b3fa2928b0513df53a9d",
    "cb16_local_opt/action_contract_r0.py": "e978391559f55076fc1369e6e337395695d5361d",
    "cb16_local_opt/execution_record_r0.py": "ee9f6fbae6cdf291646163d0558570fcac53659e",
    "cb16_local_opt/actor_critic_supervisor_r0.py": "25846b96fb33140b8e80d9f22ef417cea86b1272",
    "cb16_local_opt/target_exposure_r0.py": "817705a041c8c9a59aedf6dcd68bb90e6adba5a2",
    "cb16_local_opt/actor_critic_physics_adapter_r0.py": "51a28efca8f592946ae2a69aec1291f8cac5e1f7",
}

EXPECTED_FROZEN_AUTHORITY_BLOBS = {
    "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json": "3c401a0a350984381912f7860181e3e96eb8d7cf",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_GATEWORK_V1.json": "da8a5aa932eaa357280cb5e2a90c91c3e1301c54",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_INTEGRATION_GATEWORK_V1.json": "31bf4152a027f0a9c7fe5f4c184b3f9f43d302c4",
    "authority/rearchitecture_r11/CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1.json": "7d975a0dfc66720628a461fa818866142b532ecd",
}


def _git_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{path}"],
        cwd=ROOT,
        text=True,
    ).strip()


def _load_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _entry_map(entries: list[dict]) -> dict[str, str]:
    return {str(item["path"]): str(item["git_blob_sha"]) for item in entries}


def test_bc006_bc001_baseline_artifact_itself_is_byte_stable() -> None:
    assert _git_blob(str(BASELINE_PATH.relative_to(ROOT))) == EXPECTED_BASELINE_BLOB


def test_bc006_bc001_guarded_r0_and_frozen_authority_surfaces_remain_exact() -> None:
    baseline = _load_baseline()
    guarded = baseline["guarded_semantic_surfaces"]

    assert _entry_map(guarded["critical_r0_files"]) == EXPECTED_R0_BLOBS
    assert _entry_map(guarded["frozen_authorities"]) == EXPECTED_FROZEN_AUTHORITY_BLOBS

    for path, expected in {**EXPECTED_R0_BLOBS, **EXPECTED_FROZEN_AUTHORITY_BLOBS}.items():
        assert _git_blob(path) == expected


def test_bc006_existing_legacy_freeze_sources_and_fixture_remain_exact() -> None:
    actual = {path: _git_blob(path) for path in EXPECTED_LEGACY_FREEZE_BLOBS}
    assert actual == EXPECTED_LEGACY_FREEZE_BLOBS


def test_bc006_r0_science_contract_still_validates_without_round2_fields() -> None:
    validated = validate_actor_critic_science_contract_r0(ACTOR_CRITIC_SCIENCE_CONTRACT_R0)

    assert validated == ACTOR_CRITIC_SCIENCE_CONTRACT_R0
    assert validated.science_semantic_version == "CB16_R11_ACTOR_CRITIC_SCIENCE_R0"
    assert "account_economics_version" not in SCIENCE_VERSION_FIELDS_R0
    assert "environment_profile_version" not in SCIENCE_VERSION_FIELDS_R0


def test_bc006_round2_is_new_semantic_identity_not_a_silent_r0_edit() -> None:
    assert ACTOR_CRITIC_SCIENCE_CONTRACT_R1.science_semantic_version != ACTOR_CRITIC_SCIENCE_CONTRACT_R0.science_semantic_version

    shared_fields = set(SCIENCE_VERSION_FIELDS_R0) & set(SCIENCE_VERSION_FIELDS_R1)
    assert shared_fields
    for field_name in shared_fields:
        assert getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R1, field_name) != getattr(
            ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
            field_name,
        )


def test_bc006_r1_router_preserves_exact_legacy_and_r0_descriptors() -> None:
    assert RUNTIME_LANES_R1 == (LEGACY_R11, ACTOR_CRITIC_R0, BC_ROUND2_R1)

    for lane in (LEGACY_R11, ACTOR_CRITIC_R0):
        r1_payload = runtime_lane_contract_payload_r1(lane)
        assert r1_payload["upstream_r0_lane_contract"] == runtime_lane_contract_payload_r0(lane)
        assert r1_payload["upstream_r0_lane_contract_sha256"] == runtime_lane_contract_sha256_r0(lane)

    assert runtime_lane_contract_sha256_r1(BC_ROUND2_R1) not in {
        runtime_lane_contract_sha256_r1(LEGACY_R11),
        runtime_lane_contract_sha256_r1(ACTOR_CRITIC_R0),
    }
