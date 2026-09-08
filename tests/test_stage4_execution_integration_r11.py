from __future__ import annotations

from types import SimpleNamespace

import pytest

from cb16_local_opt.stage4_canonical_runtime_r11 import (
    AuthorityBinding,
    AuthorityLease,
    RecoveredRuntimeState,
    RuntimeContext,
)
from cb16_local_opt.stage4_execution_integration_r11 import (
    CanonicalExecutionBindingR11,
    CanonicalExecutionClosureR11,
    Stage4ExecutionIntegrationError,
)
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RuntimeRole,
)
from cb16_local_opt.stage4_permission_boundary_r11 import (
    LONG,
    Stage4PermissionBoundaryError,
    brain_action_to_action_intent_r11,
    compatibility_action_to_action_intent_r11,
    recovery_action_to_action_intent_r11,
)


AUTHORITY_ID = "canonical-runtime-authority-r11"
GENERATION = 11
SEMANTIC_FREEZE_ID = "CB16_SEMANTIC_FREEZE_V1"
STATE_ID = "sealed-state-r11"
HISTORY_ID = "scientific-history-r11"


def lineage():
    return {
        "generator": "R11_CANONICAL_BRAIN",
        "source_snapshot_sha256": "a" * 64,
        "market_packet_id": "b" * 64,
        "account_id": "acct-1",
        "intent_step": 7,
    }


def action_intent(risk=0.25):
    return brain_action_to_action_intent_r11(
        direction=LONG, requested_risk=risk, lineage=lineage()
    )


def runtime_context(*, generation=GENERATION, authority_id=AUTHORITY_ID):
    return RuntimeContext(
        binding=AuthorityBinding(
            authority_id=authority_id,
            generation=generation,
            semantic_freeze_id=SEMANTIC_FREEZE_ID,
        ),
        recovered=RecoveredRuntimeState(
            authority_id=authority_id,
            generation=generation,
            state_id=STATE_ID,
            scientific_history_id=HISTORY_ID,
        ),
        lease=AuthorityLease(
            authority_id=authority_id,
            fencing_token="opaque-s4b-handle-not-trusted-by-intd",
        ),
    )


class FakeFrozenSupervisor:
    def __init__(self):
        self.calls = []
        self.intent_seen = None

    def supervise(self, action_intent, snapshot, risk_authority, physics_contract):
        self.calls.append("supervise")
        self.intent_seen = dict(action_intent)
        return {
            "schema": "RiskSupervisorDecisionV1",
            "decision": "ACCEPT",
            "executable_direction": action_intent["direction"],
            "executable_risk_multiplier": action_intent[
                "requested_risk_multiplier"
            ],
        }

    def executable_action(self, decision, physics_contract):
        self.calls.append("executable_action")
        return {
            "schema": "ExecutableActionV1",
            "direction": decision["executable_direction"],
            "risk_multiplier": decision["executable_risk_multiplier"],
        }

    def execute_physics(
        self, snapshot, executable, market_execution_input, physics_contract
    ):
        self.calls.append("execute_physics")
        return {
            "snapshot_t1": {"account": "transitioned"},
            "account_observation_t1": {"position": executable["direction"]},
            "execution_metadata": {"path": "frozen"},
        }


def frozen_runtime(*, supervisor=None, physics=True):
    return SimpleNamespace(
        supervisor=FakeFrozenSupervisor() if supervisor is None else supervisor,
        physics=object() if physics else None,
        physics_contract={"contract_sha256": "frozen"},
    )


class StaleFence(RuntimeError):
    pass


class FakeLiveFence:
    def __init__(self, *, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.leases = []

    def assert_current(self, lease):
        self.calls += 1
        self.leases.append(lease)
        if self.fail_on_call == self.calls:
            raise StaleFence(f"STALE_FENCE_AT_ASSERT_{self.calls}")


def canonical_guard_and_binding(*, role=RuntimeRole.R11_CANONICAL_RUNTIME, subject=AUTHORITY_ID):
    guard = LegacyRetirementGuard(b"k" * 32, issuer_id="INTD_TEST_S4G")
    identity = guard.issue_identity(subject, role)

    # Non-canonical roles are deliberately paired with tokens from a canonical
    # identity so execution must fail on role before any positive fence check.
    token_identity = identity
    if role is not RuntimeRole.R11_CANONICAL_RUNTIME:
        token_identity = guard.issue_identity(
            AUTHORITY_ID, RuntimeRole.R11_CANONICAL_RUNTIME
        )
    permission_token = guard.issue_token(
        token_identity, Capability.GRANT_PERMISSION
    )
    physics_token = guard.issue_token(
        token_identity, Capability.EXECUTE_ACCOUNT_PHYSICS_TRANSITION
    )
    binding = CanonicalExecutionBindingR11(
        authority_id=AUTHORITY_ID,
        generation=GENERATION,
        semantic_freeze_id=SEMANTIC_FREEZE_ID,
        state_id=STATE_ID,
        scientific_history_id=HISTORY_ID,
        runtime_identity=identity,
        grant_permission_admission=permission_token,
        execute_physics_admission=physics_token,
    )
    return guard, binding


def closure(
    *,
    fence=None,
    runtime=None,
    role=RuntimeRole.R11_CANONICAL_RUNTIME,
    subject=AUTHORITY_ID,
):
    guard, binding = canonical_guard_and_binding(role=role, subject=subject)
    fence = FakeLiveFence() if fence is None else fence
    runtime = frozen_runtime() if runtime is None else runtime
    return (
        CanonicalExecutionClosureR11(
            frozen_runtime=runtime,
            fence_assertion=fence,
            legacy_guard=guard,
            binding=binding,
        ),
        fence,
        runtime,
    )


def execute(c, *, intent=None, context=None):
    return c.execute_action_intent(
        action_intent() if intent is None else intent,
        runtime_context=runtime_context() if context is None else context,
        snapshot={"snapshot": "t"},
        risk_authority={"risk": "frozen-authority"},
        market_execution_input={"market": "execution-input"},
    )


def test_happy_path_preserves_s4e_order_and_requested_risk():
    c, fence, rt = closure()
    receipt = execute(c, intent=action_intent(0.37))

    assert rt.supervisor.calls == [
        "supervise",
        "executable_action",
        "execute_physics",
    ]
    assert fence.calls == 2
    assert rt.supervisor.intent_seen["requested_risk_multiplier"] == 0.37
    assert "confidence" not in rt.supervisor.intent_seen
    assert "permission" not in rt.supervisor.intent_seen
    assert receipt.permission_minted_by_stage4 is False
    assert receipt.scientific_semantics_changed is False
    assert receipt.physics_result["snapshot_t1"] == {"account": "transitioned"}


@pytest.mark.parametrize(
    "privileged_field,value",
    [
        ("executable_action", {"schema": "ExecutableActionV1"}),
        ("allowed_action", LONG),
        ("supervisor_decision", {"schema": "RiskSupervisorDecisionV1"}),
        ("permission", True),
        ("confidence", 0.99),
    ],
)
def test_privileged_brain_fields_fail_closed_before_authority(
    privileged_field, value
):
    c, fence, rt = closure()
    bad = dict(action_intent())
    bad[privileged_field] = value
    with pytest.raises(Stage4PermissionBoundaryError):
        execute(c, intent=bad)
    assert fence.calls == 0
    assert rt.supervisor.calls == []


def test_direct_physics_executable_object_cannot_enter_as_action_intent():
    c, fence, rt = closure()
    forged = {
        "schema": "ExecutableActionV1",
        "direction": LONG,
        "risk_multiplier": 0.4,
    }
    with pytest.raises(Stage4PermissionBoundaryError):
        execute(c, intent=forged)
    assert fence.calls == 0
    assert rt.supervisor.calls == []
    public = {x for x in dir(CanonicalExecutionClosureR11) if not x.startswith("_")}
    assert public == {"execute_action_intent"}


def test_requested_risk_is_passed_unchanged_not_reinterpreted_as_permission():
    c, _, rt = closure()
    receipt = execute(c, intent=action_intent(0.91))
    assert rt.supervisor.intent_seen["requested_risk_multiplier"] == 0.91
    assert receipt.intent["requested_risk_multiplier"] == 0.91
    assert receipt.permission_minted_by_stage4 is False


def test_stale_fence_before_supervisor_fails_closed():
    fence = FakeLiveFence(fail_on_call=1)
    c, _, rt = closure(fence=fence)
    with pytest.raises(StaleFence, match="ASSERT_1"):
        execute(c)
    assert rt.supervisor.calls == []


def test_fence_that_goes_stale_after_supervisor_cannot_reach_physics():
    fence = FakeLiveFence(fail_on_call=2)
    c, _, rt = closure(fence=fence)
    with pytest.raises(StaleFence, match="ASSERT_2"):
        execute(c)
    assert rt.supervisor.calls == ["supervise", "executable_action"]


@pytest.mark.parametrize(
    "role",
    [
        RuntimeRole.LEGACY_REFERENCE,
        RuntimeRole.LEGACY_REPLAY,
        RuntimeRole.TEST,
        RuntimeRole.DIAGNOSTIC,
    ],
)
def test_noncanonical_runtime_identity_cannot_escalate(role):
    c, fence, rt = closure(role=role)
    with pytest.raises(
        Stage4ExecutionIntegrationError, match="RUNTIME_ROLE_NOT_CANONICAL"
    ):
        execute(c)
    assert fence.calls == 0
    assert rt.supervisor.calls == []


def test_s4g_subject_must_bind_exact_runtime_authority():
    c, fence, rt = closure(subject="different-canonical-subject")
    with pytest.raises(
        Stage4ExecutionIntegrationError, match="S4G_SUBJECT_AUTHORITY_MISMATCH"
    ):
        execute(c)
    assert fence.calls == 0
    assert rt.supervisor.calls == []


def test_wrong_generation_or_runtime_context_fails_before_fence():
    c, fence, rt = closure()
    with pytest.raises(
        Stage4ExecutionIntegrationError, match="RUNTIME_GENERATION_MISMATCH"
    ):
        execute(c, context=runtime_context(generation=GENERATION + 1))
    assert fence.calls == 0
    assert rt.supervisor.calls == []

    with pytest.raises(
        Stage4ExecutionIntegrationError, match="RUNTIME_AUTHORITY_MISMATCH"
    ):
        execute(c, context=runtime_context(authority_id="wrong-authority"))
    assert fence.calls == 0
    assert rt.supervisor.calls == []


def test_recovery_cannot_reuse_stale_permission():
    with pytest.raises(Stage4PermissionBoundaryError, match="PRIVILEGED_FIELD"):
        recovery_action_to_action_intent_r11(
            {
                "direction": LONG,
                "requested_risk": 0.2,
                "permission_token": "stale",
            },
            lineage=lineage(),
        )


def test_malformed_action_intent_fails_closed_before_authority():
    c, fence, rt = closure()
    bad = dict(action_intent())
    bad["direction"] = 99
    with pytest.raises(Stage4PermissionBoundaryError):
        execute(c, intent=bad)
    assert fence.calls == 0
    assert rt.supervisor.calls == []


def test_missing_frozen_supervisor_or_physics_authority_fails_closed():
    guard, binding = canonical_guard_and_binding()
    fence = FakeLiveFence()
    with pytest.raises(Stage4PermissionBoundaryError, match="SUPERVISOR_REQUIRED"):
        CanonicalExecutionClosureR11(
            frozen_runtime=SimpleNamespace(
                supervisor=None, physics=object(), physics_contract={}
            ),
            fence_assertion=fence,
            legacy_guard=guard,
            binding=binding,
        )
    with pytest.raises(Stage4PermissionBoundaryError, match="PHYSICS_REQUIRED"):
        CanonicalExecutionClosureR11(
            frozen_runtime=frozen_runtime(physics=False),
            fence_assertion=fence,
            legacy_guard=guard,
            binding=binding,
        )


def test_compatibility_conversion_is_representation_only_and_privilege_rejecting():
    compat = compatibility_action_to_action_intent_r11(
        {"direction": LONG, "requested_risk": 0.44}, lineage=lineage()
    )
    assert set(compat) == {
        "schema",
        "direction",
        "requested_risk_multiplier",
        "lineage",
    }
    c, _, rt = closure()
    receipt = execute(c, intent=compat)
    assert receipt.intent["requested_risk_multiplier"] == 0.44
    assert rt.supervisor.calls == [
        "supervise",
        "executable_action",
        "execute_physics",
    ]

    with pytest.raises(Stage4PermissionBoundaryError, match="PRIVILEGED_FIELD"):
        compatibility_action_to_action_intent_r11(
            {
                "direction": LONG,
                "requested_risk": 0.44,
                "allowed_action": LONG,
            },
            lineage=lineage(),
        )


def test_s4g_admissions_are_explicitly_non_authoritative():
    guard, binding = canonical_guard_and_binding()
    permission = guard.assert_token(
        binding.runtime_identity,
        Capability.GRANT_PERMISSION,
        binding.grant_permission_admission,
    )
    physics = guard.assert_token(
        binding.runtime_identity,
        Capability.EXECUTE_ACCOUNT_PHYSICS_TRANSITION,
        binding.execute_physics_admission,
    )
    assert permission.authority_granted is False
    assert physics.authority_granted is False
    assert permission.requires_independent_runtime_authority is True
    assert physics.requires_independent_runtime_authority is True
