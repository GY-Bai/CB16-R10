from __future__ import annotations

"""Stage-4 INTF worker lifecycle / orchestration provider.

This module implements the S4B OrchestrationProvider boundary only.  Authority
adoption, fencing mechanics, persistence, scientific engines, Permission and
Physics stay outside INTF and are injected through narrow boundaries.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from threading import Condition, RLock
from typing import Callable, Iterator, Protocol, Sequence, runtime_checkable

from .stage4_canonical_runtime_r11 import RuntimeContext, RuntimeSealObservation


SCIENTIFIC_STATUS_R11 = (
    "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
)
REPLAY_SEMANTICS_R11 = "ENGINEERING_RECOVERY_ONLY__NOT_NEW_SCIENTIFIC_EVIDENCE"
NUMERIC_MODE_R11 = "FP32"
AMP_ENABLED_R11 = False


class Stage4OrchestrationError(RuntimeError):
    """Base fail-closed INTF lifecycle error."""


class WorkerLifecycleTransitionError(Stage4OrchestrationError):
    """Illegal worker lifecycle transition."""


class WorkerAuthorityError(Stage4OrchestrationError):
    """Runtime context or live-authority assertion failed."""


class WorkerDrainError(Stage4OrchestrationError):
    """Bounded owned work did not drain correctly."""


class WorkerPhase(str, Enum):
    NEW = "NEW"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    DRAINED = "DRAINED"
    SEALED = "SEALED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class Stage4WorkerRuntimeConfigR11:
    """Canonical short-smoke/runtime topology; topology is not scientific identity."""

    teacher_workers: int = 8
    trace_workers: int = 8
    queue_depth: int = 8
    buffer_mib: int = 64
    nested_numeric_threads: int = 1
    numeric_mode: str = NUMERIC_MODE_R11
    amp: bool = AMP_ENABLED_R11

    def __post_init__(self) -> None:
        expected = (8, 8, 8, 64, 1, "FP32", False)
        actual = (
            int(self.teacher_workers),
            int(self.trace_workers),
            int(self.queue_depth),
            int(self.buffer_mib),
            int(self.nested_numeric_threads),
            str(self.numeric_mode),
            bool(self.amp),
        )
        if actual != expected:
            raise ValueError(
                "R11_INTF_CANONICAL_RUNTIME_DEFAULTS_REQUIRED:"
                f"EXPECTED={expected}:ACTUAL={actual}"
            )


@dataclass(frozen=True)
class AuthoritativeWorkTicketR11:
    """Opaque ownership ticket for already-admitted bounded authoritative work."""

    ticket_id: int
    authority_id: str
    generation: int
    scientific_history_id: str


@runtime_checkable
class LiveAuthorityAssertionR11(Protocol):
    """Injected final-integration boundary; must validate the current live fence."""

    def assert_current(self, context: RuntimeContext) -> None: ...


@runtime_checkable
class WorkerComponentR11(Protocol):
    """Narrow lifecycle adapter around an already-qualified worker facility."""

    @property
    def name(self) -> str: ...

    def start(self, context: RuntimeContext, config: Stage4WorkerRuntimeConfigR11) -> None: ...

    def begin_drain(self, context: RuntimeContext) -> None: ...

    def await_drained(self, context: RuntimeContext) -> None: ...

    def stop(self, context: RuntimeContext) -> None: ...

    def has_live_workers(self) -> bool: ...


@runtime_checkable
class EngineeringSealObserverR11(Protocol):
    """Observes engineering/runtime identity only; never scientific Evidence."""

    def observe(
        self,
        context: RuntimeContext,
        config: Stage4WorkerRuntimeConfigR11,
        component_names: tuple[str, ...],
    ) -> str: ...


class DeterministicEngineeringSealObserverR11:
    """Content-addressed engineering seal over already-existing runtime identity."""

    def observe(
        self,
        context: RuntimeContext,
        config: Stage4WorkerRuntimeConfigR11,
        component_names: tuple[str, ...],
    ) -> str:
        payload = {
            "schema": "CB16_R11_INTF_ENGINEERING_RUNTIME_SEAL_V1",
            "authority_id": context.binding.authority_id,
            "generation": int(context.binding.generation),
            "state_id": context.recovered.state_id,
            "scientific_history_id": context.recovered.scientific_history_id,
            "semantic_freeze_id": context.binding.semantic_freeze_id,
            "components": list(component_names),
            "numeric_mode": config.numeric_mode,
            "amp": config.amp,
            "replay_semantics": REPLAY_SEMANTICS_R11,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"engineering-runtime-seal:{digest}"


class ManagedCloseableWorkerComponentR11:
    """Lifecycle adapter for a qualified runtime whose cleanup method is ``close``."""

    def __init__(
        self,
        *,
        name: str,
        factory: Callable[[RuntimeContext, Stage4WorkerRuntimeConfigR11], object],
    ) -> None:
        if not name:
            raise ValueError("R11_INTF_COMPONENT_NAME_REQUIRED")
        self._name = str(name)
        self._factory = factory
        self._resource: object | None = None
        self._lock = RLock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def resource(self) -> object:
        with self._lock:
            if self._resource is None:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_COMPONENT_NOT_RUNNING:{self._name}"
                )
            return self._resource

    def start(self, context: RuntimeContext, config: Stage4WorkerRuntimeConfigR11) -> None:
        with self._lock:
            if self._resource is not None:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_COMPONENT_DOUBLE_START:{self._name}"
                )
            resource = self._factory(context, config)
            close = getattr(resource, "close", None)
            if not callable(close):
                raise TypeError(f"R11_INTF_COMPONENT_NOT_CLOSEABLE:{self._name}")
            self._resource = resource

    def begin_drain(self, context: RuntimeContext) -> None:
        # Admission is closed by the provider. Existing qualified pools own their
        # internal bounded futures; their close() remains the final waiting cleanup.
        return None

    def await_drained(self, context: RuntimeContext) -> None:
        return None

    def stop(self, context: RuntimeContext) -> None:
        with self._lock:
            resource = self._resource
        if resource is None:
            return
        close = getattr(resource, "close")
        close()
        with self._lock:
            if self._resource is resource:
                self._resource = None

    def has_live_workers(self) -> bool:
        with self._lock:
            return self._resource is not None


def build_teacher_worker_component_r11(
    *, max_workers: int = 8
) -> ManagedCloseableWorkerComponentR11:
    """Adapt the Stage-2-qualified reusable Teacher worker pool without rewriting it."""

    if int(max_workers) != 8:
        raise ValueError("R11_INTF_TEACHER_WORKERS_MUST_REMAIN_8")

    def factory(
        context: RuntimeContext, config: Stage4WorkerRuntimeConfigR11
    ) -> object:
        del context
        from .teacher_scheduler_r11 import TeacherWorkerPoolR11

        return TeacherWorkerPoolR11(max_workers=config.teacher_workers)

    return ManagedCloseableWorkerComponentR11(name="teacher", factory=factory)


def build_fork_trace_worker_component_r11(
    *,
    physics: object,
    market_cache: object,
    max_workers: int = 8,
) -> ManagedCloseableWorkerComponentR11:
    """Adapt the qualified lazy fork H72 runtime; do not force a new scientific run."""

    if int(max_workers) != 8:
        raise ValueError("R11_INTF_TRACE_WORKERS_MUST_REMAIN_8")

    def factory(
        context: RuntimeContext, config: Stage4WorkerRuntimeConfigR11
    ) -> object:
        del context
        from .trace_process_runtime_r11 import ForkProcessTraceRuntimeR11

        return ForkProcessTraceRuntimeR11(
            physics=physics,
            market_cache=market_cache,
            max_workers=config.trace_workers,
        )

    return ManagedCloseableWorkerComponentR11(name="trace", factory=factory)


class Stage4OrchestrationProviderR11:
    """Concrete S4B orchestration provider with fail-closed authority admission."""

    def __init__(
        self,
        *,
        live_authority: LiveAuthorityAssertionR11,
        components: Sequence[WorkerComponentR11],
        config: Stage4WorkerRuntimeConfigR11 | None = None,
        seal_observer: EngineeringSealObserverR11 | None = None,
    ) -> None:
        if live_authority is None:
            raise WorkerAuthorityError("R11_INTF_LIVE_AUTHORITY_BOUNDARY_REQUIRED")
        self._live_authority = live_authority
        self._components = tuple(components)
        names = tuple(component.name for component in self._components)
        if len(names) != len(set(names)):
            raise ValueError("R11_INTF_DUPLICATE_COMPONENT_NAME")
        self._config = config or Stage4WorkerRuntimeConfigR11()
        self._seal_observer = seal_observer or DeterministicEngineeringSealObserverR11()

        self._condition = Condition(RLock())
        self._phase = WorkerPhase.NEW
        self._accepting = False
        self._started_components: list[WorkerComponentR11] = []
        self._started_context_key: tuple[object, ...] | None = None
        self._next_ticket = 1
        self._active_tickets: dict[int, AuthoritativeWorkTicketR11] = {}
        self._seal: RuntimeSealObservation | None = None

    @property
    def phase(self) -> WorkerPhase:
        with self._condition:
            return self._phase

    @property
    def config(self) -> Stage4WorkerRuntimeConfigR11:
        return self._config

    @property
    def accepting_authoritative_work(self) -> bool:
        with self._condition:
            return self._accepting

    @property
    def in_flight_authoritative_work(self) -> int:
        with self._condition:
            return len(self._active_tickets)

    def _context_key(self, context: RuntimeContext) -> tuple[object, ...]:
        binding = context.binding
        recovered = context.recovered
        lease = context.lease
        if recovered.authority_id != binding.authority_id:
            raise WorkerAuthorityError("R11_INTF_RECOVERED_AUTHORITY_MISMATCH")
        if lease.authority_id != binding.authority_id:
            raise WorkerAuthorityError("R11_INTF_LEASE_AUTHORITY_MISMATCH")
        if int(recovered.generation) != int(binding.generation):
            raise WorkerAuthorityError("R11_INTF_GENERATION_MISMATCH")
        return (
            binding.authority_id,
            int(binding.generation),
            binding.semantic_freeze_id,
            recovered.state_id,
            recovered.scientific_history_id,
            lease.authority_id,
            lease.fencing_token,
        )

    def _require_started_context(self, context: RuntimeContext) -> tuple[object, ...]:
        key = self._context_key(context)
        with self._condition:
            if self._started_context_key != key:
                raise WorkerAuthorityError("R11_INTF_RUNTIME_CONTEXT_IDENTITY_MISMATCH")
        return key

    def _assert_live(self, context: RuntimeContext) -> None:
        try:
            self._live_authority.assert_current(context)
        except Exception as exc:
            if isinstance(exc, WorkerAuthorityError):
                raise
            raise WorkerAuthorityError("R11_INTF_LIVE_AUTHORITY_ASSERTION_FAILED") from exc

    def _cleanup_started(self, context: RuntimeContext) -> None:
        first_error: Exception | None = None
        failed: list[WorkerComponentR11] = []
        for component in reversed(tuple(self._started_components)):
            try:
                component.stop(context)
            except Exception as exc:  # cleanup must continue through every component
                failed.append(component)
                if first_error is None:
                    first_error = exc
        # Keep failed components registered so S4B cleanup re-entry can retry.
        self._started_components = list(reversed(failed))
        with self._condition:
            self._accepting = False
            self._active_tickets.clear()
            self._condition.notify_all()
        if first_error is not None:
            raise Stage4OrchestrationError(
                "R11_INTF_WORKER_CLEANUP_INCOMPLETE"
            ) from first_error

    def _fail_and_cleanup(self, context: RuntimeContext) -> None:
        with self._condition:
            self._phase = WorkerPhase.FAILED
            self._accepting = False
            self._condition.notify_all()
        self._cleanup_started(context)

    def start_workers(self, context: RuntimeContext) -> None:
        key = self._context_key(context)
        with self._condition:
            if self._phase is not WorkerPhase.NEW:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_DOUBLE_OR_ILLEGAL_START:{self._phase.value}"
                )
            self._phase = WorkerPhase.STARTING
            self._started_context_key = key

        try:
            self._assert_live(context)
            for component in self._components:
                self._assert_live(context)
                # Register before start so a component that fails after partial
                # allocation is still included in fail-closed cleanup.
                self._started_components.append(component)
                component.start(context, self._config)
                self._assert_live(context)
            with self._condition:
                self._phase = WorkerPhase.RUNNING
                self._accepting = True
        except Exception:
            try:
                self._fail_and_cleanup(context)
            except Exception as cleanup_exc:
                raise Stage4OrchestrationError(
                    "R11_INTF_PARTIAL_START_CLEANUP_FAILED"
                ) from cleanup_exc
            raise

    def admit_authoritative_work(
        self, context: RuntimeContext
    ) -> AuthoritativeWorkTicketR11:
        self._require_started_context(context)
        self._assert_live(context)
        with self._condition:
            if self._phase is not WorkerPhase.RUNNING or not self._accepting:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_AUTHORITATIVE_WORK_ADMISSION_CLOSED:{self._phase.value}"
                )
            ticket = AuthoritativeWorkTicketR11(
                ticket_id=self._next_ticket,
                authority_id=context.binding.authority_id,
                generation=int(context.binding.generation),
                scientific_history_id=context.recovered.scientific_history_id,
            )
            self._next_ticket += 1
            self._active_tickets[ticket.ticket_id] = ticket
            return ticket

    def complete_authoritative_work(self, ticket: AuthoritativeWorkTicketR11) -> None:
        with self._condition:
            active = self._active_tickets.get(int(ticket.ticket_id))
            if active != ticket:
                raise WorkerDrainError(
                    f"R11_INTF_UNKNOWN_OR_ALREADY_COMPLETED_TICKET:{ticket.ticket_id}"
                )
            del self._active_tickets[int(ticket.ticket_id)]
            self._condition.notify_all()

    @contextmanager
    def owned_authoritative_work(
        self, context: RuntimeContext
    ) -> Iterator[AuthoritativeWorkTicketR11]:
        ticket = self.admit_authoritative_work(context)
        try:
            yield ticket
        finally:
            self.complete_authoritative_work(ticket)

    def drain_workers(self, context: RuntimeContext) -> None:
        self._require_started_context(context)
        with self._condition:
            if self._phase is not WorkerPhase.RUNNING:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_ILLEGAL_DRAIN:{self._phase.value}"
                )
            self._accepting = False
            self._phase = WorkerPhase.DRAINING

        try:
            self._assert_live(context)
            for component in self._started_components:
                component.begin_drain(context)
            with self._condition:
                while self._active_tickets:
                    self._condition.wait()
            for component in self._started_components:
                component.await_drained(context)
            self._assert_live(context)
            with self._condition:
                self._phase = WorkerPhase.DRAINED
        except Exception:
            try:
                self._fail_and_cleanup(context)
            except Exception as cleanup_exc:
                raise Stage4OrchestrationError(
                    "R11_INTF_DRAIN_CLEANUP_FAILED"
                ) from cleanup_exc
            raise

    def seal_runtime_state(self, context: RuntimeContext) -> RuntimeSealObservation:
        self._require_started_context(context)
        with self._condition:
            if self._phase is not WorkerPhase.DRAINED:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_SEAL_REQUIRES_DRAIN:{self._phase.value}"
                )
        try:
            self._assert_live(context)
            seal_id = self._seal_observer.observe(
                context,
                self._config,
                tuple(component.name for component in self._started_components),
            )
            if not isinstance(seal_id, str) or not seal_id:
                raise Stage4OrchestrationError("R11_INTF_EMPTY_ENGINEERING_SEAL")
            self._assert_live(context)
            observation = RuntimeSealObservation(
                authority_id=context.binding.authority_id,
                generation=int(context.binding.generation),
                scientific_history_id=context.recovered.scientific_history_id,
                seal_id=seal_id,
            )
            with self._condition:
                self._seal = observation
                self._phase = WorkerPhase.SEALED
            return observation
        except Exception:
            try:
                self._fail_and_cleanup(context)
            except Exception as cleanup_exc:
                raise Stage4OrchestrationError(
                    "R11_INTF_SEAL_FAILURE_CLEANUP_FAILED"
                ) from cleanup_exc
            raise

    def stop_workers(self, context: RuntimeContext) -> None:
        self._require_started_context(context)
        with self._condition:
            phase = self._phase
            if phase is WorkerPhase.STOPPED:
                return
            if phase is WorkerPhase.FAILED:
                # S4B may call stop again after provider-local failure cleanup.
                # Retry any component whose cleanup previously raised.
                self._cleanup_started(context)
                return
            if phase is not WorkerPhase.SEALED:
                raise WorkerLifecycleTransitionError(
                    f"R11_INTF_STOP_REQUIRES_SEAL:{phase.value}"
                )
        try:
            self._cleanup_started(context)
        except Exception:
            with self._condition:
                self._phase = WorkerPhase.FAILED
            raise
        with self._condition:
            self._phase = WorkerPhase.STOPPED

    def assert_no_orphan_workers(self) -> None:
        live = tuple(
            component.name
            for component in self._components
            if component.has_live_workers()
        )
        if live:
            raise Stage4OrchestrationError(
                f"R11_INTF_ORPHAN_WORKERS_DETECTED:{','.join(live)}"
            )


__all__ = [
    "AMP_ENABLED_R11",
    "AuthoritativeWorkTicketR11",
    "DeterministicEngineeringSealObserverR11",
    "EngineeringSealObserverR11",
    "LiveAuthorityAssertionR11",
    "ManagedCloseableWorkerComponentR11",
    "NUMERIC_MODE_R11",
    "REPLAY_SEMANTICS_R11",
    "SCIENTIFIC_STATUS_R11",
    "Stage4OrchestrationError",
    "Stage4OrchestrationProviderR11",
    "Stage4WorkerRuntimeConfigR11",
    "WorkerAuthorityError",
    "WorkerComponentR11",
    "WorkerDrainError",
    "WorkerLifecycleTransitionError",
    "WorkerPhase",
    "build_fork_trace_worker_component_r11",
    "build_teacher_worker_component_r11",
]
