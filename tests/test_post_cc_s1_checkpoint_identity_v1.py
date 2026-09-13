"""B4 R3/R4: versioned corrective checkpoint authority + exact-SHA identity artifact."""

from __future__ import annotations

import json
from pathlib import Path

from cb16_local_opt.post_cc_s1_checkpoint_identity_v1 import (
    effective_initial_checkpoint_contract_v1,
    initial_checkpoint_identity_v1,
    write_initial_checkpoint_identity_artifact_v1,
)

CORRECTIVE_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_INITIAL_CHECKPOINT_IDENTITY_V1.json"
RUN_SPEC_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json"


def test_identity_artifact_is_deterministic_and_machine_readable(tmp_path):
    first = write_initial_checkpoint_identity_artifact_v1(tmp_path / "identity-a.json")
    second = write_initial_checkpoint_identity_artifact_v1(tmp_path / "identity-b.json")
    assert first == second
    assert first["schema"] == "CB16_R11_POST_CC_S1_INITIAL_CHECKPOINT_IDENTITY_V1"
    assert first["status"] == "EMITTED"
    assert first["modules"] == ["actor", "critic"]
    identity = dict(initial_checkpoint_identity_v1())
    assert first["computed_actor_plus_critic_sha256"] == identity["computed_actor_plus_critic_sha256"]
    assert first["declared_hash_reproduced"] is False
    assert first["historical_v1_mutated"] is False
    assert first["environment"]["torch_version"]


def test_historical_v1_authority_is_not_mutated():
    spec = json.loads(Path(RUN_SPEC_PATH).read_text(encoding="utf-8"))
    assert (
        spec["model"]["initialization"]["initial_checkpoint_semantic_sha256"]
        == "32928a6b2fea2346d303c9b61e8f86dee66099e973e7391372836b4f8a706021"
    )
    corrective = json.loads(Path(CORRECTIVE_PATH).read_text(encoding="utf-8"))
    assert corrective["historical_v1_mutated"] is False
    assert corrective["legacy_declared_sha256"] == "32928a6b2fea2346d303c9b61e8f86dee66099e973e7391372836b4f8a706021"


def test_sol_frozen_corrective_authority_is_effective():
    contract = dict(effective_initial_checkpoint_contract_v1("."))
    assert contract["contract_state"] == "MATCH"
    assert contract["authority_status"] == "FROZEN_BY_SOL"
    assert contract["frozen_by_sol"] is True
    assert contract["binding_matches_computed_identity"] is True
    assert contract["computed_actor_plus_critic_sha256"] == (
        "adb3c6d4b52cd6bc04a8036702e28f45e0c888c43dbbbb4983f2e093a57eea1a"
    )


def test_sol_frozen_corrective_authority_with_matching_binding_resolves_match(tmp_path):
    payload = json.loads(Path(CORRECTIVE_PATH).read_text(encoding="utf-8"))
    payload["status"] = "FROZEN_BY_SOL"
    payload["frozen_actor_plus_critic_sha256"] = payload["computed_actor_plus_critic_sha256"]
    authority_path = tmp_path / CORRECTIVE_PATH
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = dict(effective_initial_checkpoint_contract_v1(tmp_path))
    assert contract["contract_state"] == "MATCH"
    assert contract["frozen_by_sol"] is True
    assert contract["binding_matches_computed_identity"] is True


def test_sol_frozen_corrective_authority_with_wrong_hash_stays_mismatch(tmp_path):
    payload = json.loads(Path(CORRECTIVE_PATH).read_text(encoding="utf-8"))
    payload["status"] = "FROZEN_BY_SOL"
    payload["frozen_actor_plus_critic_sha256"] = "0" * 64
    authority_path = tmp_path / CORRECTIVE_PATH
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contract = dict(effective_initial_checkpoint_contract_v1(tmp_path))
    assert contract["contract_state"] == "CONTRACT_MISMATCH"
