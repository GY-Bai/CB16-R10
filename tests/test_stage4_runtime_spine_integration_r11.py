from __future__ import annotations

import json
from pathlib import Path

import pytest

from cb16_local_opt.stage4_authority_adoption_r11 import (
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    SourceAuthorityMismatch,
    adopt_authority,
)
from cb16_local_opt.stage4_authority_lease_r11 import (
    AuthorityBusyError,
    AuthorityRecoveryRequiredError,
    FencingTokenR11,
    SCHEMA as LEASE_STATE_SCHEMA,
    STATE_FILE,
    Stage4AuthorityLeaseR11,
    StaleFencingTokenError,
)
from cb16_local_opt.stage4_canonical_runtime_r11 import (
    AuthorityBinding,
    LifecyclePhase,
    LifecycleTransitionError,
    RecoveredRuntimeState,
    RuntimeSealObservation,
    ShutdownInvariantError,
)
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyAuthorityDenied,
    LegacyRetirementGuard,
    RuntimeRole,
)
from cb16_local_opt.stage4_runtime_spine_integration_r11 import (
    CanonicalRuntimeAuthoritySpineR11,
    LeaseAcquisitionMode,
    RuntimeSpineIntegrationError,
    Stage4AuthorityAdoptionProviderR11,
    Stage4AuthorityLeaseProviderR11,
    Stage4LegacyRetirementProviderR11,
    Stage4StateRootProviderR11,
)
from cb16_local_opt.stage4_state_roots_r11 import (
    SEMANTIC_FREEZE_BLOB_SHA,
    Stage4RootOverlapError,
    Stage4RootSetR11,
    Stage4RootTypeError,
    Stage4StateRootsR11,
    Stage4UnclaimedRootError,
)


SEED = "86a4ac8a9080cd8382600cb998059e9585495a11"
AUTHORITY_ID = "CB16_R11_CANONICAL_RUNTIME"
CHAMPION_HASH = "a" * 64
CHECKPOINT_HASH = "b" * 64


def _source(*, generation: int = 7) -> SourceAuthorityIdentity:
    return SourceAuthorityIdentity(
        source_repo="GY-Bai/CB16-R10",
        source_sha=SEED,
        semantic_freeze_identity=SEMANTIC_FREEZE_BLOB_SHA,
        source_generation=generation,
        champion_identity="champion-g7",
        champion_hash=CHAMPION_HASH,
        checkpoint_identity="checkpoint-g7",
        checkpoint_hash=CHECKPOINT_HASH,
        evidence_root_identity="evidence-root-g7",
        journal_head_identity="journal-head-g7",
        checkpoint_root_identity="checkpoint-root-g7",
    )


def _adoption(tmp_path: Path, *, source: SourceAuthorityIdentity | None = None):
    source = source or _source()
    contract = AuthorityAdoptionContract(source, AUTHORITY_ID)
    receipt = tmp_path / "adoption.json"
    adopt_authority(
        source,
        contract=contract,
        receipt_path=receipt,
        adoption_timestamp_utc="2026-09-08T00:00:00Z",
    )
    return contract, receipt


def _roots(tmp_path: Path):
    control = tmp_path / "control"
    data = tmp_path / "data"
    frozen = tmp_path / "frozen"
    frozen.mkdir(parents=True)
    frozen.chmod(0o555)
    roots = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control,
            data_root=data,
            frozen_raw_root=frozen,
            frozen_raw_identity="FROZEN_RAW_TEST_ID",
        )
    )
    receipt = roots.initialize()
    return roots, receipt


def _legacy(authority_id: str = AUTHORITY_ID):
    guard = LegacyRetirementGuard(b"k" * 32, issuer_id="INTA_TEST_ISSUER")
    identity = guard.issue_identity(authority_id, RuntimeRole.R11_CANONICAL_RUNTIME)
    token = guard.issue_token(identity, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)
    return Stage4LegacyRetirementProviderR11(
        guard=guard,
        identity=identity,
        token=token,
    )


class RecoveryStub:
    def __init__(self, events: list[str] | None = None, *, generation_delta: int = 0):
        self.events = events
        self.generation_delta = generation_delta

    def recover_existing_state(self, binding: AuthorityBinding) -> RecoveredRuntimeState:
        if self.events is not None:
            self.events.append("recovery")
        return RecoveredRuntimeState(
            authority_id=binding.authority_id,
            generation=binding.generation + self.generation_delta,
            state_id="state-g7",
            scientific_history_id="history-g7",
        )


class OrchestrationStub:
    def __init__(
        self,
        events: list[str] | None = None,
        *,
        fail_start: bool = False,
        seal_generation_delta: int = 0,
        rewrite_history: bool = False,
    ):
        self.events = events
        self.fail_start = fail_start
        self.seal_generation_delta = seal_generation_delta
        self.rewrite_history = rewrite_history
        self.started = False
        self.stopped = False

    def start_workers(self, context):
        if self.events is not None:
            self.events.append("worker_start")
        self.started = True
        if self.fail_start:
            raise RuntimeError("TEST_WORKER_START_FAILURE")

    def drain_workers(self, context):
        if self.events is not None:
            self.events.append("drain")

    def seal_runtime_state(self, context):
        if self.events is not None:
            self.events.append("seal")
        return RuntimeSealObservation(
            authority_id=context.recovered.authority_id,
            generation=context.recovered.generation + self.seal_generation_delta,
            scientific_history_id=(
                "rewritten-history"
                if self.rewrite_history
                else context.recovered.scientific_history_id
            ),
            seal_id="runtime-seal",
        )

    def stop_workers(self, context):
        if self.events is not None:
            self.events.append("worker_stop")
        self.stopped = True


def _spine(tmp_path: Path, *, owner_id: str = "owner-1", mode=LeaseAcquisitionMode.ORDINARY):
    contract, adoption_receipt = _adoption(tmp_path)
    roots, root_receipt = _roots(tmp_path)
    lease_root = roots.control_root / "runtime_lease_fencing_state"
    Stage4AuthorityLeaseR11.initialize(lease_root)
    concrete_lease = Stage4AuthorityLeaseR11(lease_root, owner_id=owner_id)
    adoption = Stage4AuthorityAdoptionProviderR11(
        contract=contract, receipt_path=adoption_receipt
    )
    state_roots = Stage4StateRootProviderR11(
        state_roots=roots,
        expected_control_root_id=root_receipt.control_root_id,
        expected_data_root_id=root_receipt.data_root_id,
        expected_frozen_raw_identity=root_receipt.frozen_raw_identity,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=contract.accepted_source.source_generation,
    )
    lease_provider = Stage4AuthorityLeaseProviderR11(
        lease=concrete_lease,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=contract.accepted_source.source_generation,
        acquisition_mode=mode,
    )
    return (
        CanonicalRuntimeAuthoritySpineR11(
            authority_adoption=adoption,
            state_roots=state_roots,
            legacy_guard=_legacy(),
            authority_lease=lease_provider,
        ),
        concrete_lease,
        roots,
        root_receipt,
        contract,
        adoption_receipt,
    )


def test_canonical_lifecycle_order_and_clean_shutdown(tmp_path):
    events: list[str] = []
    spine, lease, *_ = _spine(tmp_path)

    original_adopt = spine.authority_adoption.adopt_or_verify_existing_authority
    original_roots = spine.state_roots.verify_state_roots
    original_legacy = spine.legacy_guard.assert_canonical_only
    original_acquire = spine.authority_lease.acquire_authority
    original_assert = spine.authority_lease.assert_current

    spine.authority_adoption.adopt_or_verify_existing_authority = lambda: (
        events.append("adoption") or original_adopt()
    )
    spine.state_roots.verify_state_roots = lambda binding: (
        events.append("roots") or original_roots(binding)
    )
    spine.legacy_guard.assert_canonical_only = lambda binding: (
        events.append("legacy_eligibility") or original_legacy(binding)
    )
    spine.authority_lease.acquire_authority = lambda binding: (
        events.append("authority_acquire") or original_acquire(binding)
    )
    spine.authority_lease.assert_current = lambda opaque: (
        events.append("fence_assert") or original_assert(opaque)
    )

    orchestration = OrchestrationStub(events)
    controller = spine.build_controller(
        recovery=RecoveryStub(events), orchestration=orchestration
    )
    context = controller.start()
    assert controller.phase is LifecyclePhase.RUNNING
    assert context.binding.generation == 7
    assert events[:8] == [
        "adoption",
        "roots",
        "legacy_eligibility",
        "recovery",
        "authority_acquire",
        "fence_assert",
        "worker_start",
        "fence_assert",
    ]
    controller.shutdown()
    assert controller.phase is LifecyclePhase.STOPPED
    assert lease.inspect_current_authority().state == "RELEASED"


def test_wrong_adoption_identity_and_generation_fail_closed(tmp_path):
    contract, receipt = _adoption(tmp_path)
    wrong_target = AuthorityAdoptionContract(contract.accepted_source, "WRONG_TARGET")
    with pytest.raises(SourceAuthorityMismatch):
        Stage4AuthorityAdoptionProviderR11(
            contract=wrong_target, receipt_path=receipt
        ).adopt_or_verify_existing_authority()

    wrong_source = _source(generation=8)
    wrong_generation = AuthorityAdoptionContract(wrong_source, AUTHORITY_ID)
    with pytest.raises(SourceAuthorityMismatch):
        Stage4AuthorityAdoptionProviderR11(
            contract=wrong_generation, receipt_path=receipt
        ).adopt_or_verify_existing_authority()


def test_root_provider_rejects_wrong_swapped_unclaimed_and_overlapping_roots(tmp_path):
    contract, _ = _adoption(tmp_path)
    binding = AuthorityBinding(
        AUTHORITY_ID, contract.accepted_source.source_generation, SEMANTIC_FREEZE_BLOB_SHA
    )
    roots, receipt = _roots(tmp_path / "good")

    wrong_id = Stage4StateRootProviderR11(
        state_roots=roots,
        expected_control_root_id="0" * 64,
        expected_data_root_id=receipt.data_root_id,
        expected_frozen_raw_identity=receipt.frozen_raw_identity,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=7,
    )
    with pytest.raises(Stage4RootTypeError):
        wrong_id.verify_state_roots(binding)

    swapped = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=roots.data_root,
            data_root=roots.control_root,
            frozen_raw_root=roots.frozen_raw_root,
            frozen_raw_identity=receipt.frozen_raw_identity,
        )
    )
    swapped_provider = Stage4StateRootProviderR11(
        state_roots=swapped,
        expected_control_root_id=receipt.control_root_id,
        expected_data_root_id=receipt.data_root_id,
        expected_frozen_raw_identity=receipt.frozen_raw_identity,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=7,
    )
    with pytest.raises(Stage4RootTypeError):
        swapped_provider.verify_state_roots(binding)

    base = tmp_path / "unclaimed"
    frozen = base / "frozen"
    control = base / "control"
    data = base / "data"
    frozen.mkdir(parents=True)
    frozen.chmod(0o555)
    control.mkdir()
    data.mkdir()
    (control / "legacy.txt").write_text("not claimed")
    unclaimed_roots = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control,
            data_root=data,
            frozen_raw_root=frozen,
            frozen_raw_identity="FROZEN_RAW_TEST_ID",
        )
    )
    with pytest.raises((Stage4UnclaimedRootError, Stage4RootTypeError)):
        unclaimed_roots.initialize()

    base2 = tmp_path / "overlap"
    frozen2 = base2 / "frozen"
    control2 = base2 / "control"
    frozen2.mkdir(parents=True)
    frozen2.chmod(0o555)
    overlapping = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control2,
            data_root=control2 / "data",
            frozen_raw_root=frozen2,
            frozen_raw_identity="FROZEN_RAW_TEST_ID",
        )
    )
    with pytest.raises(Stage4RootOverlapError):
        overlapping.initialize()


def test_legacy_identity_cannot_request_canonical_runtime_authority(tmp_path):
    guard = LegacyRetirementGuard(b"x" * 32)
    legacy = guard.issue_identity("legacy", RuntimeRole.LEGACY_REFERENCE)
    with pytest.raises(LegacyAuthorityDenied):
        guard.issue_token(legacy, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)

    canonical = guard.issue_identity(AUTHORITY_ID, RuntimeRole.R11_CANONICAL_RUNTIME)
    token = guard.issue_token(canonical, Capability.ACQUIRE_CANONICAL_RUNTIME_AUTHORITY)
    provider = Stage4LegacyRetirementProviderR11(
        guard=guard, identity=legacy, token=token
    )
    with pytest.raises(RuntimeSpineIntegrationError):
        provider.assert_canonical_only(
            AuthorityBinding(AUTHORITY_ID, 7, SEMANTIC_FREEZE_BLOB_SHA)
        )


def test_duplicate_start_and_second_simultaneous_owner_fail_closed(tmp_path):
    spine1, lease1, roots, root_receipt, contract, receipt = _spine(tmp_path, owner_id="one")
    controller1 = spine1.build_controller(
        recovery=RecoveryStub(), orchestration=OrchestrationStub()
    )
    controller1.start()

    with pytest.raises(LifecycleTransitionError):
        controller1.start()

    lease_root = roots.control_root / "runtime_lease_fencing_state"
    lease2 = Stage4AuthorityLeaseR11(lease_root, owner_id="two")
    spine2 = CanonicalRuntimeAuthoritySpineR11(
        authority_adoption=Stage4AuthorityAdoptionProviderR11(
            contract=contract, receipt_path=receipt
        ),
        state_roots=Stage4StateRootProviderR11(
            state_roots=roots,
            expected_control_root_id=root_receipt.control_root_id,
            expected_data_root_id=root_receipt.data_root_id,
            expected_frozen_raw_identity=root_receipt.frozen_raw_identity,
            expected_authority_id=AUTHORITY_ID,
            expected_generation=7,
        ),
        legacy_guard=_legacy(),
        authority_lease=Stage4AuthorityLeaseProviderR11(
            lease=lease2,
            expected_authority_id=AUTHORITY_ID,
            expected_generation=7,
        ),
    )
    orchestration2 = OrchestrationStub()
    controller2 = spine2.build_controller(
        recovery=RecoveryStub(), orchestration=orchestration2
    )
    with pytest.raises(AuthorityBusyError):
        controller2.start()
    assert orchestration2.started is False
    assert controller2.phase is LifecyclePhase.FAILED

    controller1.shutdown()
    assert lease1.inspect_current_authority().state == "RELEASED"


def test_stale_and_copied_but_not_live_opaque_lease_fail(tmp_path):
    spine, concrete, *_ = _spine(tmp_path)
    binding = spine.authority_adoption.adopt_or_verify_existing_authority()
    opaque = spine.authority_lease.acquire_authority(binding)
    spine.authority_lease.assert_current(opaque)

    concrete.release_authority(concrete.token)
    with pytest.raises(StaleFencingTokenError):
        spine.authority_lease.assert_current(opaque)

    other = Stage4AuthorityLeaseR11(concrete.root, owner_id="copy-check")
    copied_provider = Stage4AuthorityLeaseProviderR11(
        lease=other,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=7,
    )
    with pytest.raises(RuntimeSpineIntegrationError):
        copied_provider.assert_current(opaque)


class _FailAssertAfterAcquireLease:
    def __init__(self):
        self.token = FencingTokenR11(1, "c" * 32)
        self.released = False

    def acquire_authority(self):
        return self.token

    def recover_after_dead_owner(self):
        return self.token

    def assert_fencing_token(self, token):
        raise StaleFencingTokenError("TEST_ASSERT_CURRENT_FAILURE")

    def release_authority(self, token):
        assert token == self.token
        self.released = True


def test_lease_failure_after_acquisition_releases_authority(tmp_path):
    contract, receipt = _adoption(tmp_path)
    roots, rr = _roots(tmp_path)
    fake = _FailAssertAfterAcquireLease()
    spine = CanonicalRuntimeAuthoritySpineR11(
        authority_adoption=Stage4AuthorityAdoptionProviderR11(
            contract=contract, receipt_path=receipt
        ),
        state_roots=Stage4StateRootProviderR11(
            state_roots=roots,
            expected_control_root_id=rr.control_root_id,
            expected_data_root_id=rr.data_root_id,
            expected_frozen_raw_identity=rr.frozen_raw_identity,
            expected_authority_id=AUTHORITY_ID,
            expected_generation=7,
        ),
        legacy_guard=_legacy(),
        authority_lease=Stage4AuthorityLeaseProviderR11(
            lease=fake,
            expected_authority_id=AUTHORITY_ID,
            expected_generation=7,
        ),
    )
    controller = spine.build_controller(
        recovery=RecoveryStub(), orchestration=OrchestrationStub()
    )
    with pytest.raises(StaleFencingTokenError):
        controller.start()
    assert fake.released is True
    assert controller.phase is LifecyclePhase.FAILED


def test_worker_start_failure_stops_workers_and_releases_live_authority(tmp_path):
    spine, concrete, *_ = _spine(tmp_path)
    orchestration = OrchestrationStub(fail_start=True)
    controller = spine.build_controller(
        recovery=RecoveryStub(), orchestration=orchestration
    )
    with pytest.raises(RuntimeError, match="TEST_WORKER_START_FAILURE"):
        controller.start()
    assert orchestration.started is True
    assert orchestration.stopped is True
    assert concrete.inspect_current_authority().state == "RELEASED"
    assert controller.phase is LifecyclePhase.FAILED


def _write_crashed_active_state(root: Path, *, epoch: int = 5):
    state = {
        "schema": LEASE_STATE_SCHEMA,
        "state": "ACTIVE",
        "epoch": epoch,
        "owner_id": "dead-owner",
        "owner_pid": 424242,
        "owner_nonce": "d" * 32,
    }
    (root / STATE_FILE).write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return FencingTokenR11(epoch, "d" * 32)


def test_crashed_active_requires_explicit_dead_owner_recovery(tmp_path):
    root = tmp_path / "lease"
    Stage4AuthorityLeaseR11.initialize(root)
    stale = _write_crashed_active_state(root)

    ordinary = Stage4AuthorityLeaseProviderR11(
        lease=Stage4AuthorityLeaseR11(root, owner_id="ordinary"),
        expected_authority_id=AUTHORITY_ID,
        expected_generation=7,
    )
    binding = AuthorityBinding(AUTHORITY_ID, 7, SEMANTIC_FREEZE_BLOB_SHA)
    with pytest.raises(AuthorityRecoveryRequiredError):
        ordinary.acquire_authority(binding)

    explicit_lease = Stage4AuthorityLeaseR11(root, owner_id="recovered")
    explicit = Stage4AuthorityLeaseProviderR11(
        lease=explicit_lease,
        expected_authority_id=AUTHORITY_ID,
        expected_generation=7,
        acquisition_mode=LeaseAcquisitionMode.EXPLICIT_DEAD_OWNER_RECOVERY,
    )
    opaque = explicit.acquire_authority(binding)
    explicit.assert_current(opaque)
    assert explicit_lease.token.epoch == stale.epoch + 1
    assert explicit_lease.validate_fencing_token(stale) is False
    explicit.release_authority(opaque)


@pytest.mark.parametrize(
    "seal_generation_delta,rewrite_history",
    [(1, False), (0, True)],
)
def test_shutdown_generation_or_history_rewrite_refused_and_cleanup(
    tmp_path, seal_generation_delta, rewrite_history
):
    spine, concrete, *_ = _spine(tmp_path)
    orchestration = OrchestrationStub(
        seal_generation_delta=seal_generation_delta,
        rewrite_history=rewrite_history,
    )
    controller = spine.build_controller(
        recovery=RecoveryStub(), orchestration=orchestration
    )
    controller.start()
    with pytest.raises(ShutdownInvariantError):
        controller.shutdown()
    assert orchestration.stopped is True
    assert concrete.inspect_current_authority().state == "RELEASED"
    assert controller.phase is LifecyclePhase.FAILED


def test_wrong_recovery_generation_never_reaches_workers(tmp_path):
    spine, concrete, *_ = _spine(tmp_path)
    orchestration = OrchestrationStub()
    controller = spine.build_controller(
        recovery=RecoveryStub(generation_delta=1), orchestration=orchestration
    )
    with pytest.raises(Exception, match="RECOVERY_GENERATION_MANUFACTURE_REFUSED"):
        controller.start()
    assert orchestration.started is False
    assert concrete.inspect_current_authority().state == "RELEASED"
