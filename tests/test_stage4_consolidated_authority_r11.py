from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    CheckpointSeal,
    CommitReceipt,
    EventKind,
    GenerationState,
    RuntimeEvent,
    TournamentDecision,
)
from cb16_local_opt.stage4_authority_lease_r11 import Stage4AuthorityLeaseR11
from cb16_local_opt.stage4_canonical_runtime_r11 import (
    AuthorityBinding,
    AuthorityLease,
    RecoveredRuntimeState,
    RuntimeContext,
)
from cb16_local_opt.stage4_consolidated_authority_r11 import (
    ConsolidatedAuthorityError,
    GenerationAuthorityCoordinatorR11,
    GenerationBoundLiveFenceR11,
    RuntimeContextLiveAuthorityBridgeR11,
    resolve_native_fence_r11,
)
from cb16_local_opt.stage4_runtime_spine_integration_r11 import Stage4AuthorityLeaseProviderR11


AUTHORITY = "R11-CANONICAL"
FREEZE = "freeze"
CHAMPION_G7 = "a" * 64
CHAMPION_G8 = "b" * 64


class _FakeLeaseProvider:
    def __init__(self, expected: AuthorityLease) -> None:
        self.expected = expected
        self.live = True
        self.assertions = 0

    def assert_current(self, lease: AuthorityLease) -> None:
        self.assertions += 1
        if not self.live or lease != self.expected:
            raise RuntimeError("fake-stale")


class _Journal:
    def __init__(self, events: list[RuntimeEvent]) -> None:
        self.events = events

    def read_generation(self, generation: int) -> list[RuntimeEvent]:
        return [event for event in self.events if event.generation == generation]


class _Checkpoints:
    def __init__(self, commit: CommitReceipt | None, seal: CheckpointSeal | None) -> None:
        self.commit = commit
        self.seal = seal

    def read_commit(self, generation: int) -> CommitReceipt | None:
        return self.commit if self.commit is not None and self.commit.generation == generation else None

    def read_checkpoint(self, generation: int) -> CheckpointSeal | None:
        return self.seal if self.seal is not None and self.seal.generation == generation else None


def _context() -> RuntimeContext:
    lease = AuthorityLease(AUTHORITY, "opaque-live")
    return RuntimeContext(
        binding=AuthorityBinding(AUTHORITY, 7, FREEZE),
        recovered=RecoveredRuntimeState(AUTHORITY, 7, "state-g7", "scientific-history-frozen"),
        lease=lease,
    )


def _stamp7() -> AuthorityStamp:
    return AuthorityStamp(7, CHAMPION_G7, CHAMPION_G7, "teacher", "physics")


def _stamp8() -> AuthorityStamp:
    return AuthorityStamp(8, CHAMPION_G8, CHAMPION_G8, "teacher", "physics")


def _commit() -> CommitReceipt:
    return CommitReceipt(
        generation=7,
        decision=TournamentDecision.PROMOTE,
        parent_champion_id=CHAMPION_G7,
        next_champion_id=CHAMPION_G8,
        next_champion_hash=CHAMPION_G8,
        commit_id="commit-g7",
        atomic=True,
        durable=True,
    )


def _seal() -> CheckpointSeal:
    return CheckpointSeal(7, CHAMPION_G8, "state-digest", "commit-g7", True)


def _release_event(stamp: AuthorityStamp | None = None) -> RuntimeEvent:
    stamp = stamp or _stamp8()
    return RuntimeEvent.build(
        kind=EventKind.NEXT_GENERATION_RELEASED,
        generation=7,
        state_after=GenerationState.COMMITTED,
        payload=stamp,
        metadata={
            "next_generation": stamp.generation,
            "next_champion_id": stamp.champion_id,
            "next_champion_hash": stamp.champion_hash,
        },
    )


def _coordinator(*, events=None, commit=None, seal=None):
    process = _context()
    fake = _FakeLeaseProvider(process.lease)
    bridge = RuntimeContextLiveAuthorityBridgeR11(process_context=process, lease_provider=fake)  # type: ignore[arg-type]
    coord = GenerationAuthorityCoordinatorR11(
        process_context=process,
        initial_stamp=_stamp7(),
        process_authority=bridge,
        journal=_Journal(events if events is not None else [_release_event()]),
        checkpoints=_Checkpoints(commit if commit is not None else _commit(), seal if seal is not None else _seal()),
    )
    return coord, bridge, fake


def test_process_authority_stays_stable_while_generation_advances() -> None:
    coord, bridge, _ = _coordinator()
    original_process_context = coord.process_context
    old_fence = GenerationBoundLiveFenceR11(coordinator=coord, expected_generation=7)
    old_fence.assert_current(original_process_context.lease)

    state = coord.advance(_stamp8())
    assert state.stamp == _stamp8()
    assert coord.process_context == original_process_context
    assert coord.process_context.binding.generation == 7
    assert bridge.process_context.binding.generation == 7

    engine_context = coord.generation_runtime_context()
    assert engine_context.binding.generation == 8
    assert engine_context.recovered.generation == 8
    assert engine_context.lease == original_process_context.lease
    assert engine_context.recovered.state_id == CHAMPION_G8
    assert engine_context.recovered.scientific_history_id == "scientific-history-frozen"
    assert state.release_event_id is not None

    with pytest.raises(ConsolidatedAuthorityError, match="STALE_GENERATION_FENCE"):
        old_fence.assert_current(original_process_context.lease)
    GenerationBoundLiveFenceR11(coordinator=coord, expected_generation=8).assert_current(
        original_process_context.lease
    )


def test_generation_advance_requires_exact_existing_release_barrier() -> None:
    coord, _, _ = _coordinator(events=[])
    with pytest.raises(ConsolidatedAuthorityError, match="EXACTLY_ONE_GENERATION_RELEASE"):
        coord.advance(_stamp8())

    coord, _, _ = _coordinator(events=[_release_event(), _release_event()])
    with pytest.raises(ConsolidatedAuthorityError, match="EXACTLY_ONE_GENERATION_RELEASE"):
        coord.advance(_stamp8())

    bad_commit = replace(_commit(), next_champion_hash="c" * 64)
    coord, _, _ = _coordinator(commit=bad_commit)
    with pytest.raises(ConsolidatedAuthorityError, match="CHECKPOINT_CHAMPION_MISMATCH"):
        coord.advance(_stamp8())


def test_skip_rollback_and_upstream_authority_drift_fail_closed() -> None:
    coord, _, _ = _coordinator()
    skip = AuthorityStamp(9, "c" * 64, "c" * 64, "teacher", "physics")
    with pytest.raises(ConsolidatedAuthorityError, match="ADVANCE_EXACTLY_ONE"):
        coord.advance(skip)

    drift = AuthorityStamp(8, CHAMPION_G8, CHAMPION_G8, "teacher-drift", "physics")
    with pytest.raises(ConsolidatedAuthorityError, match="TEACHER_AUTHORITY_DRIFT"):
        coord.advance(drift)

    coord.advance(_stamp8())
    with pytest.raises(ConsolidatedAuthorityError, match="ADVANCE_EXACTLY_ONE"):
        coord.advance(_stamp8())
    with pytest.raises(ConsolidatedAuthorityError, match="STALE_OR_SUBSTITUTED"):
        coord.assert_stamp_current(_stamp7())


def test_process_context_bridge_rejects_generation_projection_substitution() -> None:
    coord, bridge, _ = _coordinator()
    coord.advance(_stamp8())
    with pytest.raises(ConsolidatedAuthorityError, match="PROCESS_CONTEXT_SUBSTITUTION_REFUSED"):
        bridge.assert_current(coord.generation_runtime_context())


def test_native_fence_resolution_reuses_exact_inta_s4d_token(tmp_path) -> None:
    root = tmp_path / "lease"
    Stage4AuthorityLeaseR11.initialize(root)
    native = Stage4AuthorityLeaseR11(root, owner_id="runtime-a")
    provider = Stage4AuthorityLeaseProviderR11(
        lease=native,
        expected_authority_id=AUTHORITY,
        expected_generation=7,
    )
    binding = AuthorityBinding(AUTHORITY, 7, FREEZE)
    opaque = provider.acquire_authority(binding)
    resolved = resolve_native_fence_r11(provider, opaque)
    assert resolved.lease is native
    assert resolved.token == native.token
    native.assert_fencing_token(resolved.token)
    provider.release_authority(opaque)
    with pytest.raises(Exception):
        resolve_native_fence_r11(provider, opaque)
