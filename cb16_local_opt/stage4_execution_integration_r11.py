from __future__ import annotations

"""Stage-4 INTD canonical Permission / execution closure for CB16 R11.

This module composes already-qualified Stage-4 boundaries. It does not define
Permission, Supervisor policy, Physics semantics, lease/fencing mechanics, or
legacy-retirement authority. Production callers may submit only ActionIntent
plus legitimate frozen-authority state/context inputs.

The authoritative path is:

    ActionIntent
      -> canonical runtime identity / S4G eligibility / live fence
      -> Frozen Supervisor
      -> ExecutableAction
      -> live fence re-assertion
      -> Frozen Physics
      -> account transition

A Stage-4 retirement admission token is explicitly not positive runtime
authority and is never treated as Permission.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from cb16_local_opt.stage4_canonical_runtime_r11 import AuthorityLease, RuntimeContext
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyRetirementGuard,
    RetirementCapabilityToken,
    RuntimeIdentity,
    RuntimeRole,
)
from cb16_local_opt.stage4_permission_boundary_r11 import (
    FrozenPermissionBoundaryR11,
    PermissionChainReceiptR11,
    validate_action_intent_r11,
)


class Stage4ExecutionIntegrationError(RuntimeError):
    """Fail-closed INTD integration invariant violation."""


@runtime_checkable
class LiveFenceAssertionBoundaryR11(Protocol):
    """Positive authority boundary supplied by the canonical runtime spine.

    The opaque S4B lease is not interpreted here. Implementations must retain
    whatever S4D live-fence binding is needed and must ultimately validate the
    currently live fence. A copied opaque fencing string is not sufficient.
    """

    def assert_current(self, lease: AuthorityLease) -> None: ...


@dataclass(frozen=True)
class CanonicalExecutionBindingR11:
    """Expected canonical runtime identity for one execution closure.

    This is identity binding only; it is not Permission and does not grant
    runtime/fencing authority.
    """

    authority_id: str
    generation: int
    semantic_freeze_id: str
    state_id: str
    scientific_history_id: str
    runtime_identity: RuntimeIdentity
    grant_permission_admission: RetirementCapabilityToken
    execute_physics_admission: RetirementCapabilityToken

    def __post_init__(self) -> None:
        if not isinstance(self.authority_id, str) or not self.authority_id:
            raise ValueError("INTD_AUTHORITY_ID_REQUIRED")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int) or self.generation < 0:
            raise ValueError("INTD_GENERATION_INVALID")
        for name in ("semantic_freeze_id", "state_id", "scientific_history_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"INTD_{name.upper()}_REQUIRED")


class _FenceCheckedSupervisorProxy:
    """Private representation-only proxy around the frozen Supervisor surface."""

    def __init__(self, delegate: Any, assert_before_physics: Any) -> None:
        self.__delegate = delegate
        self.__assert_before_physics = assert_before_physics

    def supervise(
        self,
        action_intent: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        risk_authority: Mapping[str, Any],
        physics_contract: Mapping[str, Any],
    ) -> Any:
        return self.__delegate.supervise(
            action_intent, snapshot, risk_authority, physics_contract
        )

    def executable_action(
        self,
        decision: Mapping[str, Any],
        physics_contract: Mapping[str, Any],
    ) -> Any:
        return self.__delegate.executable_action(decision, physics_contract)

    def execute_physics(
        self,
        snapshot: Mapping[str, Any],
        executable: Mapping[str, Any],
        market_execution_input: Mapping[str, Any],
        physics_contract: Mapping[str, Any],
    ) -> Any:
        # Close the stale-owner race at the authoritative mutation boundary.
        self.__assert_before_physics()
        return self.__delegate.execute_physics(
            snapshot, executable, market_execution_input, physics_contract
        )


@dataclass(frozen=True)
class _FrozenRuntimeExecutionView:
    """Minimal S4E-compatible view; no semantic field is converted."""

    supervisor: Any
    physics: Any
    physics_contract: Mapping[str, Any]


class CanonicalExecutionClosureR11:
    """Canonical production action ingress for INTD.

    The class intentionally exposes no direct Physics, Permission, Supervisor
    decision, ExecutableAction, or account-transition API.
    """

    def __init__(
        self,
        *,
        frozen_runtime: Any,
        fence_assertion: LiveFenceAssertionBoundaryR11,
        legacy_guard: LegacyRetirementGuard,
        binding: CanonicalExecutionBindingR11,
    ) -> None:
        if fence_assertion is None or not callable(getattr(fence_assertion, "assert_current", None)):
            raise Stage4ExecutionIntegrationError("INTD_LIVE_FENCE_ASSERTION_REQUIRED")
        if not isinstance(legacy_guard, LegacyRetirementGuard):
            raise Stage4ExecutionIntegrationError("INTD_S4G_LEGACY_GUARD_REQUIRED")
        if not isinstance(binding, CanonicalExecutionBindingR11):
            raise Stage4ExecutionIntegrationError("INTD_EXECUTION_BINDING_REQUIRED")

        # Reuse the qualified S4E constructor as the only authority-shape check
        # for the frozen Supervisor / Physics boundary.
        FrozenPermissionBoundaryR11(frozen_runtime)

        self.__fence_assertion = fence_assertion
        self.__legacy_guard = legacy_guard
        self.__binding = binding
        self.__supervisor = getattr(frozen_runtime, "supervisor")
        self.__physics = getattr(frozen_runtime, "physics")
        self.__physics_contract = getattr(frozen_runtime, "physics_contract")

    def _assert_runtime_context(self, context: RuntimeContext) -> None:
        if not isinstance(context, RuntimeContext):
            raise Stage4ExecutionIntegrationError("INTD_RUNTIME_CONTEXT_TYPE_INVALID")

        expected = self.__binding
        binding = context.binding
        recovered = context.recovered
        lease = context.lease

        if binding.authority_id != expected.authority_id:
            raise Stage4ExecutionIntegrationError("INTD_RUNTIME_AUTHORITY_MISMATCH")
        if recovered.authority_id != expected.authority_id or lease.authority_id != expected.authority_id:
            raise Stage4ExecutionIntegrationError("INTD_RUNTIME_COMPONENT_AUTHORITY_MISMATCH")
        if binding.generation != expected.generation or recovered.generation != expected.generation:
            raise Stage4ExecutionIntegrationError("INTD_RUNTIME_GENERATION_MISMATCH")
        if binding.semantic_freeze_id != expected.semantic_freeze_id:
            raise Stage4ExecutionIntegrationError("INTD_SEMANTIC_FREEZE_IDENTITY_MISMATCH")
        if recovered.state_id != expected.state_id:
            raise Stage4ExecutionIntegrationError("INTD_RECOVERED_STATE_IDENTITY_MISMATCH")
        if recovered.scientific_history_id != expected.scientific_history_id:
            raise Stage4ExecutionIntegrationError("INTD_SCIENTIFIC_HISTORY_IDENTITY_MISMATCH")

    def _assert_s4g_execution_eligibility(self) -> None:
        expected = self.__binding
        role = self.__legacy_guard.validate_identity(expected.runtime_identity)
        if role is not RuntimeRole.R11_CANONICAL_RUNTIME:
            raise Stage4ExecutionIntegrationError(
                f"INTD_RUNTIME_ROLE_NOT_CANONICAL:{role.value}"
            )
        if expected.runtime_identity.subject_id != expected.authority_id:
            raise Stage4ExecutionIntegrationError("INTD_S4G_SUBJECT_AUTHORITY_MISMATCH")

        checks = (
            (
                Capability.GRANT_PERMISSION,
                expected.grant_permission_admission,
            ),
            (
                Capability.EXECUTE_ACCOUNT_PHYSICS_TRANSITION,
                expected.execute_physics_admission,
            ),
        )
        for capability, token in checks:
            decision = self.__legacy_guard.assert_token(
                expected.runtime_identity, capability, token
            )
            # S4G is only a negative eligibility/admission guard. If it ever
            # claimed positive authority, integration must fail closed.
            if decision.authority_granted:
                raise Stage4ExecutionIntegrationError(
                    "INTD_S4G_TOKEN_MUST_NOT_GRANT_AUTHORITY"
                )
            if not decision.eligible_under_legacy_retirement_policy:
                raise Stage4ExecutionIntegrationError("INTD_S4G_ELIGIBILITY_REQUIRED")
            if not decision.requires_independent_runtime_authority:
                raise Stage4ExecutionIntegrationError(
                    "INTD_S4G_INDEPENDENT_RUNTIME_AUTHORITY_REQUIRED"
                )

    def _assert_live_fence(self, context: RuntimeContext) -> None:
        # Never interpret or trust context.lease.fencing_token here. The
        # injected provider owns the live S4D binding and must prove it current.
        self.__fence_assertion.assert_current(context.lease)

    def execute_action_intent(
        self,
        intent: Any,
        *,
        runtime_context: RuntimeContext,
        snapshot: Mapping[str, Any],
        risk_authority: Mapping[str, Any],
        market_execution_input: Mapping[str, Any],
    ) -> PermissionChainReceiptR11:
        """Execute one already-formed ActionIntent through frozen authorities."""

        # Input validation is representation/shape checking only and does not
        # constitute authoritative progress.
        canonical_intent = validate_action_intent_r11(intent)

        self._assert_runtime_context(runtime_context)
        self._assert_s4g_execution_eligibility()
        self._assert_live_fence(runtime_context)

        # S4E owns the semantic ordering. The private proxy is representation
        # only and inserts no decision; it merely re-asserts the same live
        # positive authority immediately before Physics/account mutation.
        proxy = _FenceCheckedSupervisorProxy(
            self.__supervisor,
            lambda: self._assert_live_fence(runtime_context),
        )
        frozen_view = _FrozenRuntimeExecutionView(
            supervisor=proxy,
            physics=self.__physics,
            physics_contract=self.__physics_contract,
        )
        boundary = FrozenPermissionBoundaryR11(frozen_view)
        return boundary.execute_intent(
            canonical_intent,
            snapshot=snapshot,
            risk_authority=risk_authority,
            market_execution_input=market_execution_input,
        )
