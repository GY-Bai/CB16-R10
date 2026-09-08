from __future__ import annotations

"""Canonical Stage-4 R11 process lifecycle boundary.

This module owns process lifecycle ordering only. It deliberately does not own
scientific adjudication, authority adoption semantics, lease/fencing mechanics,
persistent-root semantics, legacy retirement policy, recovery semantics, or R11
work orchestration. Those are injected through narrow Protocol boundaries.

The default construction is intentionally unusable for production writes: every
integration boundary fails closed until final Stage-4 integration supplies a
qualified implementation. There is no legacy-runtime fallback.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class Stage4RuntimeError(RuntimeError):
    """Base error for Stage-4 canonical runtime lifecycle failures."""


class LifecycleTransitionError(Stage4RuntimeError):
    """Raised when a caller requests an illegal lifecycle transition."""


class StartupInvariantError(Stage4RuntimeError):
    """Raised when an injected provider violates the S4B startup contract."""


class ShutdownInvariantError(Stage4RuntimeError):
    """Raised when shutdown would rewrite recovered authority identity."""


class IntegrationRequiredError(Stage4RuntimeError):
    """Raised by fail-closed defaults until final Stage-4 integration wires them."""


class LifecyclePhase(str, Enum):
    BOOT = "BOOT"
    VERIFY_AUTHORITY = "VERIFY_AUTHORITY"
    RECOVER_STATE = "RECOVER_STATE"
    ACQUIRE_AUTHORITY = "ACQUIRE_AUTHORITY"
    START_WORKERS = "START_WORKERS"
    RUNNING = "RUNNING"
    DRAIN = "DRAIN"
    SEAL = "SEAL"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


CANONICAL_LIFECYCLE = (
    LifecyclePhase.BOOT,
    LifecyclePhase.VERIFY_AUTHORITY,
    LifecyclePhase.RECOVER_STATE,
    LifecyclePhase.ACQUIRE_AUTHORITY,
    LifecyclePhase.START_WORKERS,
    LifecyclePhase.RUNNING,
    LifecyclePhase.DRAIN,
    LifecyclePhase.SEAL,
    LifecyclePhase.STOPPED,
)


@dataclass(frozen=True)
class AuthorityBinding:
    """Opaque authority identity returned by the adoption/verification boundary."""

    authority_id: str
    generation: int
    semantic_freeze_id: str


@dataclass(frozen=True)
class RecoveredRuntimeState:
    """Identity S4B must preserve while recovering an already accepted generation."""

    authority_id: str
    generation: int
    state_id: str
    scientific_history_id: str


@dataclass(frozen=True)
class AuthorityLease:
    """Opaque lease/fencing identity. S4B never interprets fencing mechanics."""

    authority_id: str
    fencing_token: str


@dataclass(frozen=True)
class RuntimeContext:
    binding: AuthorityBinding
    recovered: RecoveredRuntimeState
    lease: AuthorityLease


@dataclass(frozen=True)
class RuntimeSealObservation:
    """Post-drain identity observation; it is not a scientific checkpoint verdict."""

    authority_id: str
    generation: int
    scientific_history_id: str
    seal_id: str


@runtime_checkable
class AuthorityAdoptionProvider(Protocol):
    """Later adoption integration: bind/verify already accepted authority."""

    def adopt_or_verify_existing_authority(self) -> AuthorityBinding: ...


@runtime_checkable
class PersistentStateRootProvider(Protocol):
    """Later storage integration: verify canonical persistent roots are usable."""

    def verify_state_roots(self, binding: AuthorityBinding) -> None: ...


@runtime_checkable
class LegacyRetirementGuard(Protocol):
    """Later retirement integration: reject legacy authoritative-write fallback."""

    def assert_canonical_only(self, binding: AuthorityBinding) -> None: ...


@runtime_checkable
class RecoveryProvider(Protocol):
    """Upstream-qualified recovery boundary; S4B only checks identity preservation."""

    def recover_existing_state(self, binding: AuthorityBinding) -> RecoveredRuntimeState: ...


@runtime_checkable
class AuthorityLeaseProvider(Protocol):
    """Later lease integration. Token format/epoch semantics stay opaque to S4B."""

    def acquire_authority(self, binding: AuthorityBinding) -> AuthorityLease: ...

    def assert_current(self, lease: AuthorityLease) -> None: ...

    def release_authority(self, lease: AuthorityLease) -> None: ...


@runtime_checkable
class OrchestrationProvider(Protocol):
    """Qualified R11 orchestration boundary; S4B does not reinterpret its work."""

    def start_workers(self, context: RuntimeContext) -> None: ...

    def drain_workers(self, context: RuntimeContext) -> None: ...

    def seal_runtime_state(self, context: RuntimeContext) -> RuntimeSealObservation: ...

    def stop_workers(self, context: RuntimeContext) -> None: ...


_LEGAL_FORWARD = {
    LifecyclePhase.BOOT: LifecyclePhase.VERIFY_AUTHORITY,
    LifecyclePhase.VERIFY_AUTHORITY: LifecyclePhase.RECOVER_STATE,
    LifecyclePhase.RECOVER_STATE: LifecyclePhase.ACQUIRE_AUTHORITY,
    LifecyclePhase.ACQUIRE_AUTHORITY: LifecyclePhase.START_WORKERS,
    LifecyclePhase.START_WORKERS: LifecyclePhase.RUNNING,
    LifecyclePhase.RUNNING: LifecyclePhase.DRAIN,
    LifecyclePhase.DRAIN: LifecyclePhase.SEAL,
    LifecyclePhase.SEAL: LifecyclePhase.STOPPED,
}


class CanonicalRuntimeControllerR11:
    """Fail-closed controller for one canonical R11 runtime process lifecycle."""

    def __init__(
        self,
        *,
        authority_adoption: AuthorityAdoptionProvider,
        state_roots: PersistentStateRootProvider,
        legacy_guard: LegacyRetirementGuard,
        recovery: RecoveryProvider,
        authority_lease: AuthorityLeaseProvider,
        orchestration: OrchestrationProvider,
    ) -> None:
        self._authority_adoption = authority_adoption
        self._state_roots = state_roots
        self._legacy_guard = legacy_guard
        self._recovery = recovery
        self._authority_lease = authority_lease
        self._orchestration = orchestration
        self.phase = LifecyclePhase.BOOT
        self._binding: AuthorityBinding | None = None
        self._recovered: RecoveredRuntimeState | None = None
        self._lease: AuthorityLease | None = None
        self._context: RuntimeContext | None = None
        self._worker_start_attempted = False
        self._workers_stopped = False

    @property
    def context(self) -> RuntimeContext | None:
        return self._context

    def start(self) -> RuntimeContext:
        if self.phase is not LifecyclePhase.BOOT:
            raise LifecycleTransitionError(f"R11_STAGE4_DOUBLE_OR_ILLEGAL_START:{self.phase.value}")

        try:
            self._advance(LifecyclePhase.VERIFY_AUTHORITY)
            binding = self._authority_adoption.adopt_or_verify_existing_authority()
            self._validate_binding(binding)
            self._state_roots.verify_state_roots(binding)
            self._legacy_guard.assert_canonical_only(binding)
            self._binding = binding

            self._advance(LifecyclePhase.RECOVER_STATE)
            recovered = self._recovery.recover_existing_state(binding)
            self._validate_recovery(binding, recovered)
            self._recovered = recovered

            self._advance(LifecyclePhase.ACQUIRE_AUTHORITY)
            lease = self._authority_lease.acquire_authority(binding)
            if isinstance(lease, AuthorityLease):
                # Record a structurally valid returned lease before deeper checks so
                # fencing/assertion failure can still unwind acquired authority.
                self._lease = lease
            self._validate_lease(binding, lease)
            self._authority_lease.assert_current(lease)

            context = RuntimeContext(binding=binding, recovered=recovered, lease=lease)
            self._context = context
            self._advance(LifecyclePhase.START_WORKERS)
            self._worker_start_attempted = True
            self._orchestration.start_workers(context)
            self._authority_lease.assert_current(lease)

            self._advance(LifecyclePhase.RUNNING)
            return context
        except BaseException:
            self._fail_closed_cleanup()
            raise

    def shutdown(self) -> RuntimeSealObservation:
        if self.phase is not LifecyclePhase.RUNNING:
            raise LifecycleTransitionError(f"R11_STAGE4_ILLEGAL_SHUTDOWN:{self.phase.value}")
        assert self._context is not None and self._lease is not None and self._recovered is not None

        try:
            self._authority_lease.assert_current(self._lease)
            self._advance(LifecyclePhase.DRAIN)
            self._orchestration.drain_workers(self._context)

            self._authority_lease.assert_current(self._lease)
            self._advance(LifecyclePhase.SEAL)
            observation = self._orchestration.seal_runtime_state(self._context)
            self._validate_shutdown_observation(self._recovered, observation)

            self._orchestration.stop_workers(self._context)
            self._workers_stopped = True
            self._authority_lease.release_authority(self._lease)
            self._lease = None
            self._advance(LifecyclePhase.STOPPED)
            return observation
        except BaseException:
            self._fail_closed_cleanup()
            raise

    def _advance(self, target: LifecyclePhase) -> None:
        expected = _LEGAL_FORWARD.get(self.phase)
        if expected is not target:
            raise LifecycleTransitionError(
                f"R11_STAGE4_ILLEGAL_TRANSITION:{self.phase.value}->{target.value}"
            )
        self.phase = target

    @staticmethod
    def _validate_binding(binding: AuthorityBinding) -> None:
        if not isinstance(binding, AuthorityBinding):
            raise StartupInvariantError("R11_STAGE4_INVALID_AUTHORITY_BINDING_TYPE")
        if not binding.authority_id or not binding.semantic_freeze_id or binding.generation < 0:
            raise StartupInvariantError("R11_STAGE4_INVALID_AUTHORITY_BINDING")

    @staticmethod
    def _validate_recovery(binding: AuthorityBinding, recovered: RecoveredRuntimeState) -> None:
        if not isinstance(recovered, RecoveredRuntimeState):
            raise StartupInvariantError("R11_STAGE4_INVALID_RECOVERY_TYPE")
        if recovered.authority_id != binding.authority_id:
            raise StartupInvariantError("R11_STAGE4_RECOVERY_AUTHORITY_MISMATCH")
        if recovered.generation != binding.generation:
            raise StartupInvariantError("R11_STAGE4_RECOVERY_GENERATION_MANUFACTURE_REFUSED")
        if not recovered.state_id or not recovered.scientific_history_id:
            raise StartupInvariantError("R11_STAGE4_RECOVERY_IDENTITY_MISSING")

    @staticmethod
    def _validate_lease(binding: AuthorityBinding, lease: AuthorityLease) -> None:
        if not isinstance(lease, AuthorityLease):
            raise StartupInvariantError("R11_STAGE4_INVALID_LEASE_TYPE")
        if lease.authority_id != binding.authority_id or not lease.fencing_token:
            raise StartupInvariantError("R11_STAGE4_LEASE_AUTHORITY_MISMATCH")

    @staticmethod
    def _validate_shutdown_observation(
        recovered: RecoveredRuntimeState, observation: RuntimeSealObservation
    ) -> None:
        if not isinstance(observation, RuntimeSealObservation):
            raise ShutdownInvariantError("R11_STAGE4_INVALID_SEAL_OBSERVATION_TYPE")
        if observation.authority_id != recovered.authority_id:
            raise ShutdownInvariantError("R11_STAGE4_SHUTDOWN_AUTHORITY_REWRITE_REFUSED")
        if observation.generation != recovered.generation:
            raise ShutdownInvariantError("R11_STAGE4_SHUTDOWN_GENERATION_REWRITE_REFUSED")
        if observation.scientific_history_id != recovered.scientific_history_id:
            raise ShutdownInvariantError("R11_STAGE4_SHUTDOWN_HISTORY_REWRITE_REFUSED")
        if not observation.seal_id:
            raise ShutdownInvariantError("R11_STAGE4_SHUTDOWN_SEAL_ID_MISSING")

    def _fail_closed_cleanup(self) -> None:
        # Cleanup never authorizes work and never falls back to legacy runtime.
        context = self._context
        if self._worker_start_attempted and not self._workers_stopped and context is not None:
            try:
                self._orchestration.stop_workers(context)
                self._workers_stopped = True
            except BaseException:
                pass
        if self._lease is not None:
            lease = self._lease
            try:
                self._authority_lease.release_authority(lease)
            except BaseException:
                pass
            self._lease = None
        if self.phase not in {LifecyclePhase.STOPPED, LifecyclePhase.FAILED}:
            self.phase = LifecyclePhase.FAILED


class _FailClosedProvider:
    """Structural provider implementing every boundary by refusing integration."""

    @staticmethod
    def _blocked(boundary: str) -> None:
        raise IntegrationRequiredError(f"R11_STAGE4_INTEGRATION_REQUIRED:{boundary}")

    def adopt_or_verify_existing_authority(self) -> AuthorityBinding:
        self._blocked("authority_adoption")
        raise AssertionError("unreachable")

    def verify_state_roots(self, binding: AuthorityBinding) -> None:
        self._blocked("persistent_state_roots")

    def assert_canonical_only(self, binding: AuthorityBinding) -> None:
        self._blocked("legacy_retirement_guard")

    def recover_existing_state(self, binding: AuthorityBinding) -> RecoveredRuntimeState:
        self._blocked("recovery")
        raise AssertionError("unreachable")

    def acquire_authority(self, binding: AuthorityBinding) -> AuthorityLease:
        self._blocked("authority_lease")
        raise AssertionError("unreachable")

    def assert_current(self, lease: AuthorityLease) -> None:
        self._blocked("authority_lease")

    def release_authority(self, lease: AuthorityLease) -> None:
        return None

    def start_workers(self, context: RuntimeContext) -> None:
        self._blocked("orchestration")

    def drain_workers(self, context: RuntimeContext) -> None:
        self._blocked("orchestration")

    def seal_runtime_state(self, context: RuntimeContext) -> RuntimeSealObservation:
        self._blocked("orchestration")
        raise AssertionError("unreachable")

    def stop_workers(self, context: RuntimeContext) -> None:
        return None


def build_fail_closed_runtime() -> CanonicalRuntimeControllerR11:
    """Return the canonical entrypoint with no authority-bearing integration wired."""

    blocked = _FailClosedProvider()
    return CanonicalRuntimeControllerR11(
        authority_adoption=blocked,
        state_roots=blocked,
        legacy_guard=blocked,
        recovery=blocked,
        authority_lease=blocked,
        orchestration=blocked,
    )


def lifecycle_description() -> str:
    return " -> ".join(phase.value for phase in CANONICAL_LIFECYCLE)
