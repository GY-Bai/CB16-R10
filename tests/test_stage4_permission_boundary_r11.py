from __future__ import annotations

from types import SimpleNamespace

import pytest

from cb16_local_opt.stage4_permission_boundary_r11 import (
    FLAT,
    LONG,
    FrozenPermissionBoundaryR11,
    Stage4PermissionBoundaryError,
    brain_action_to_action_intent_r11,
    compatibility_action_to_action_intent_r11,
    recovery_action_to_action_intent_r11,
    validate_action_intent_r11,
)


def lineage():
    return {
        "generator": "TEST_BRAIN",
        "source_snapshot_sha256": "a" * 64,
        "market_packet_id": "b" * 64,
        "account_id": "acct-1",
        "intent_step": 7,
    }


def intent(direction=LONG, risk=0.25):
    return brain_action_to_action_intent_r11(direction=direction, requested_risk=risk, lineage=lineage())


class FakeFrozenSupervisor:
    def __init__(self): self.calls = []
    def supervise(self, action_intent, snapshot, risk_authority, physics_contract):
        self.calls.append("supervise")
        return {"schema": "RiskSupervisorDecisionV1", "decision": "ACCEPT", "executable_direction": action_intent["direction"], "executable_risk_multiplier": action_intent["requested_risk_multiplier"]}
    def executable_action(self, decision, physics_contract):
        self.calls.append("executable_action")
        return {"schema": "ExecutableActionV1", "direction": decision["executable_direction"], "risk_multiplier": decision["executable_risk_multiplier"]}
    def execute_physics(self, snapshot, executable, market_execution_input, physics_contract):
        self.calls.append("execute_physics")
        return {"snapshot_t1": {"ok": True}, "execution_metadata": {"path": "frozen"}}


def runtime(supervisor=None, *, physics=True):
    return SimpleNamespace(supervisor=FakeFrozenSupervisor() if supervisor is None else supervisor, physics=object() if physics else None, physics_contract={"contract_sha256": "frozen"})


def assert_rejected(raw, converter=compatibility_action_to_action_intent_r11):
    with pytest.raises(Stage4PermissionBoundaryError):
        converter(raw, lineage=lineage())


def test_brain_can_only_propose_action_intent_not_permission():
    x = intent()
    assert x == {"schema": "ActionIntentV1", "direction": LONG, "requested_risk_multiplier": 0.25, "lineage": lineage()}
    assert not ({"permission", "allowed_action", "executable_action"} & set(x))


def test_forged_direct_executable_action_from_brain_fails_closed():
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "executable_action": {"schema": "ExecutableActionV1"}})


def test_forged_legacy_allowed_action_fails_closed():
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "allowed_action": LONG})


def test_direct_account_transition_bypass_fails_closed():
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "account_transition": {"position": 1}})


def test_requested_risk_cannot_be_confidence_or_permission():
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "confidence": 0.99})
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "permission": True})


def test_recovery_cannot_reuse_stale_permission():
    assert_rejected({"direction": LONG, "requested_risk": 0.2, "permission_token": "stale"}, converter=recovery_action_to_action_intent_r11)


def test_compatibility_adapter_cannot_grant_authority():
    x = compatibility_action_to_action_intent_r11({"direction": LONG, "requested_risk": 0.2}, lineage=lineage())
    assert set(x) == {"schema", "direction", "requested_risk_multiplier", "lineage"}
    assert x["requested_risk_multiplier"] == 0.2


def test_missing_supervisor_or_physics_authority_fails_closed():
    with pytest.raises(Stage4PermissionBoundaryError, match="SUPERVISOR_REQUIRED"):
        FrozenPermissionBoundaryR11(SimpleNamespace(supervisor=None, physics=object(), physics_contract={}))
    with pytest.raises(Stage4PermissionBoundaryError, match="PHYSICS_REQUIRED"):
        FrozenPermissionBoundaryR11(runtime(physics=False))


def test_malformed_action_intent_fails_closed():
    bad = {"schema": "ActionIntentV1", "direction": 99, "requested_risk_multiplier": 0.2, "lineage": lineage()}
    with pytest.raises(Stage4PermissionBoundaryError): validate_action_intent_r11(bad)
    with pytest.raises(Stage4PermissionBoundaryError): brain_action_to_action_intent_r11(direction=FLAT, requested_risk=0.2, lineage=lineage())


def test_legacy_object_with_privileged_looking_fields_fails_closed():
    class Legacy:
        direction = LONG
        requested_risk = 0.2
        allowed_action = LONG
    with pytest.raises(Stage4PermissionBoundaryError, match="MAPPING_REQUIRED"):
        compatibility_action_to_action_intent_r11(Legacy(), lineage=lineage())


def test_execution_order_is_supervisor_then_executable_then_physics():
    rt = runtime()
    receipt = FrozenPermissionBoundaryR11(rt).execute_intent(intent(), snapshot={"s": 1}, risk_authority={"r": 1}, market_execution_input={"bar": {}})
    assert rt.supervisor.calls == ["supervise", "executable_action", "execute_physics"]
    assert receipt.permission_minted_by_stage4 is False
    assert receipt.scientific_semantics_changed is False
    assert receipt.physics_result["snapshot_t1"] == {"ok": True}


def test_boundary_has_no_api_for_caller_supplied_permission_or_transition():
    public = {x for x in dir(FrozenPermissionBoundaryR11) if not x.startswith("_")}
    assert "execute_intent" in public
    assert not ({"execute_action", "apply_transition", "grant_permission", "reuse_permission"} & public)
