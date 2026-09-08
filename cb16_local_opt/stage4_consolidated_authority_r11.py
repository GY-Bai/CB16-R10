from __future__ import annotations

"""Final Stage-4 cross-task authority reconciliation.

This module is additive final-adjudicator plumbing.  It does not mint scientific
Evidence, alter Permission/Physics semantics, or replace any qualified child gate.
It keeps the process-lifetime S4B/INTA authority stable while deriving a separate,
strictly monotone generation authority only from the already-qualified R11 durable
commit/checkpoint/NEXT_GENERATION_RELEASED barrier.
"""

from dataclasses import dataclass
from typing import Protocol

from .runtime_events_r11 import AuthorityStamp, CheckpointSeal, CommitReceipt, EventKind, RuntimeEvent
from .stage4_authority_lease_r11 import FencingTokenR11, Stage4AuthorityLeaseR11
from .stage4_canonical_runtime_r11 import AuthorityBinding, AuthorityLease, RecoveredRuntimeState, RuntimeContext
from .stage4_runtime_spine_integration_r11 import Stage4AuthorityLeaseProviderR11

SCHEMA = "CB16_R11_STAGE4_CONSOLIDATED_AUTHORITY_V1"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


class ConsolidatedAuthorityError(RuntimeError):
    """Fail-closed cross-task authority reconciliation error."""


@dataclass(frozen=True)
class ResolvedNativeFenceR11:
    """The exact native S4D lease/token backing one live INTA opaque lease."""

    lease: Stage4AuthorityLeaseR11
    token: FencingTokenR11


def resolve_native_fence_r11(
    provider: Stage4AuthorityLeaseProviderR11,
    opaque_lease: AuthorityLease,
) -> ResolvedNativeFenceR11:
    """Resolve INTA's opaque handle back to its existing live S4D fence.

    No token is created here.  Final consolidation deliberately binds to the exact
    INTA implementation class and fails if its qualified internal binding shape
    drifts instead of guessing a replacement authority.
    """

    if not isinstance(provider, Stage4AuthorityLeaseProviderR11):
        raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INTA_PROVIDER_REQUIRED")
    provider.assert_current(opaque_lease)
    try:
        record = provider._record_for(opaque_lease)  # final-adjudicator bridge to accepted INTA
        native = provider._lease
    except AttributeError as exc:
        raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INTA_NATIVE_FENCE_UNAVAILABLE") from exc
    if not isinstance(native, Stage4AuthorityLeaseR11) or not isinstance(record.fencing_token, FencingTokenR11):
        raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INTA_NATIVE_FENCE_TYPE_INVALID")
    native.assert_fencing_token(record.fencing_token)
    return ResolvedNativeFenceR11(native, record.fencing_token)


class RuntimeContextLiveAuthorityBridgeR11:
    """Adapt INTF's process RuntimeContext assertion to INTA's opaque lease fence."""

    def __init__(self, *, process_context: RuntimeContext, lease_provider: Stage4AuthorityLeaseProviderR11) -> None:
        if not isinstance(process_context, RuntimeContext):
            raise TypeError("STAGE4_CONSOLIDATED_PROCESS_CONTEXT_REQUIRED")
        self._context = process_context
        self._provider = lease_provider
        self._validate_context(process_context)

    @staticmethod
    def _validate_context(context: RuntimeContext) -> None:
        authority = context.binding.authority_id
        if not authority or context.recovered.authority_id != authority or context.lease.authority_id != authority:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_PROCESS_AUTHORITY_IDENTITY_MISMATCH")
        if context.binding.generation != context.recovered.generation:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_PROCESS_GENERATION_MISMATCH")

    @property
    def process_context(self) -> RuntimeContext:
        return self._context

    def assert_current(self, context: RuntimeContext) -> None:
        self._validate_context(context)
        if context != self._context:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_PROCESS_CONTEXT_SUBSTITUTION_REFUSED")
        self._provider.assert_current(context.lease)


class GenerationJournalReaderR11(Protocol):
    def read_generation(self, generation: int) -> list[RuntimeEvent]: ...


class GenerationCheckpointReaderR11(Protocol):
    def read_commit(self, generation: int) -> CommitReceipt | None: ...
    def read_checkpoint(self, generation: int) -> CheckpointSeal | None: ...


@dataclass(frozen=True)
class GenerationAuthorityStateR11:
    stamp: AuthorityStamp
    state_id: str
    scientific_history_id: str
    release_event_id: str | None


class GenerationAuthorityCoordinatorR11:
    """Advance generation authority only after the existing durable R11 release barrier.

    The process RuntimeContext and its lease never advance.  A generation-scoped
    RuntimeContext is a read-only admission projection for INTC only; it is never
    passed to S4B process shutdown or INTF worker lifecycle methods.
    """

    def __init__(
        self,
        *,
        process_context: RuntimeContext,
        initial_stamp: AuthorityStamp,
        process_authority: RuntimeContextLiveAuthorityBridgeR11,
        journal: GenerationJournalReaderR11,
        checkpoints: GenerationCheckpointReaderR11,
    ) -> None:
        if not isinstance(process_context, RuntimeContext) or not isinstance(initial_stamp, AuthorityStamp):
            raise TypeError("STAGE4_CONSOLIDATED_GENERATION_INIT_TYPE_INVALID")
        if process_authority.process_context != process_context:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_PROCESS_BRIDGE_CONTEXT_MISMATCH")
        if initial_stamp.generation != process_context.binding.generation:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INITIAL_GENERATION_MISMATCH")
        if not initial_stamp.champion_id or initial_stamp.champion_id != initial_stamp.champion_hash:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INITIAL_CHAMPION_IDENTITY_INVALID")
        if not initial_stamp.teacher_authority_id or not initial_stamp.physics_authority_id:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_INITIAL_UPSTREAM_AUTHORITY_MISSING")
        process_authority.assert_current(process_context)
        self._process_context = process_context
        self._process_authority = process_authority
        self._journal = journal
        self._checkpoints = checkpoints
        self._state = GenerationAuthorityStateR11(
            stamp=initial_stamp,
            state_id=process_context.recovered.state_id,
            scientific_history_id=process_context.recovered.scientific_history_id,
            release_event_id=None,
        )

    @property
    def process_context(self) -> RuntimeContext:
        return self._process_context

    @property
    def state(self) -> GenerationAuthorityStateR11:
        return self._state

    @property
    def current_generation(self) -> int:
        return int(self._state.stamp.generation)

    def assert_stamp_current(self, stamp: AuthorityStamp) -> None:
        self._process_authority.assert_current(self._process_context)
        if stamp != self._state.stamp:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_STALE_OR_SUBSTITUTED_GENERATION_STAMP")

    def generation_runtime_context(self) -> RuntimeContext:
        """Project current generation authority without mutating process authority."""

        self._process_authority.assert_current(self._process_context)
        stamp = self._state.stamp
        return RuntimeContext(
            binding=AuthorityBinding(
                authority_id=self._process_context.binding.authority_id,
                generation=int(stamp.generation),
                semantic_freeze_id=self._process_context.binding.semantic_freeze_id,
            ),
            recovered=RecoveredRuntimeState(
                authority_id=self._process_context.recovered.authority_id,
                generation=int(stamp.generation),
                state_id=self._state.state_id,
                scientific_history_id=self._state.scientific_history_id,
            ),
            lease=self._process_context.lease,
        )

    def advance(self, released_stamp: AuthorityStamp) -> GenerationAuthorityStateR11:
        self._process_authority.assert_current(self._process_context)
        current = self._state.stamp
        if not isinstance(released_stamp, AuthorityStamp):
            raise TypeError("STAGE4_CONSOLIDATED_RELEASE_STAMP_TYPE_INVALID")
        if released_stamp.generation != current.generation + 1:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_GENERATION_MUST_ADVANCE_EXACTLY_ONE")
        if released_stamp.teacher_authority_id != current.teacher_authority_id:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_TEACHER_AUTHORITY_DRIFT")
        if released_stamp.physics_authority_id != current.physics_authority_id:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_PHYSICS_AUTHORITY_DRIFT")
        if not released_stamp.champion_id or released_stamp.champion_id != released_stamp.champion_hash:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_RELEASED_CHAMPION_IDENTITY_INVALID")

        generation = int(current.generation)
        commit = self._checkpoints.read_commit(generation)
        seal = self._checkpoints.read_checkpoint(generation)
        if commit is None or seal is None:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_COMMIT_AND_CHECKPOINT_REQUIRED")
        if not commit.atomic or not commit.durable or not seal.durable:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_DURABILITY_UNPROVEN")
        if commit.generation != generation or seal.generation != generation:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_DURABLE_GENERATION_MISMATCH")
        if seal.commit_id != commit.commit_id:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_CHECKPOINT_COMMIT_MISMATCH")
        if seal.checkpoint_id != commit.next_champion_hash:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_CHECKPOINT_CHAMPION_MISMATCH")
        if commit.next_champion_id != released_stamp.champion_id or commit.next_champion_hash != released_stamp.champion_hash:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_RELEASE_CHAMPION_MISMATCH")

        events = self._journal.read_generation(generation)
        releases = [event for event in events if event.kind is EventKind.NEXT_GENERATION_RELEASED]
        if len(releases) != 1:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_EXACTLY_ONE_GENERATION_RELEASE_REQUIRED")
        event = releases[0]
        if event.generation != generation:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_RELEASE_EVENT_GENERATION_MISMATCH")
        meta = event.meta()
        expected = {
            "next_generation": str(released_stamp.generation),
            "next_champion_id": released_stamp.champion_id,
            "next_champion_hash": released_stamp.champion_hash,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ConsolidatedAuthorityError(f"STAGE4_CONSOLIDATED_RELEASE_METADATA_MISMATCH:{key}")

        self._state = GenerationAuthorityStateR11(
            stamp=released_stamp,
            state_id=seal.checkpoint_id,
            # Engineering generation advance does not rewrite scientific history.
            scientific_history_id=self._process_context.recovered.scientific_history_id,
            release_event_id=event.event_id,
        )
        return self._state


class GenerationBoundLiveFenceR11:
    """INTC live fence that also invalidates every old generation projection."""

    def __init__(
        self,
        *,
        coordinator: GenerationAuthorityCoordinatorR11,
        expected_generation: int,
    ) -> None:
        self._coordinator = coordinator
        self._expected_generation = int(expected_generation)

    def assert_current(self, lease: AuthorityLease) -> None:
        if lease != self._coordinator.process_context.lease:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_ENGINE_LEASE_SUBSTITUTION_REFUSED")
        if self._coordinator.current_generation != self._expected_generation:
            raise ConsolidatedAuthorityError("STAGE4_CONSOLIDATED_STALE_GENERATION_FENCE")
        self._coordinator._process_authority.assert_current(self._coordinator.process_context)


__all__ = [
    "ConsolidatedAuthorityError",
    "GenerationAuthorityCoordinatorR11",
    "GenerationAuthorityStateR11",
    "GenerationBoundLiveFenceR11",
    "ResolvedNativeFenceR11",
    "RuntimeContextLiveAuthorityBridgeR11",
    "SCHEMA",
    "SCIENTIFIC_STATUS",
    "resolve_native_fence_r11",
]
