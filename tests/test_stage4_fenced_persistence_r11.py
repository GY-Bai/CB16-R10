from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cb16_local_opt.evidence_store_r11 import EvidenceItemR11
from cb16_local_opt.integration_adapters_r11 import EvidenceStoreProtocolAdapterR11
from cb16_local_opt.orchestrator_r11 import R11Orchestrator
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    CheckpointSeal,
    EvidenceRef,
    EventKind,
    GenerationState,
    RuntimeEvent,
    SnapshotSeal,
    TournamentCommitProposal,
    TournamentDecision,
)
from cb16_local_opt.stage4_authority_lease_r11 import (
    FencingTokenR11,
    Stage4AuthorityLeaseR11,
    StaleFencingTokenError,
)
from cb16_local_opt.stage4_fenced_persistence_r11 import (
    CanonicalWriteAuthorityR11,
    Stage4CanonicalRoleRequired,
    Stage4PersistenceRootBindingError,
    open_canonical_persistence_r11,
)
from cb16_local_opt.stage4_legacy_retirement_r11 import (
    Capability,
    LegacyAuthorityDenied,
    LegacyRetirementGuard,
    RuntimeRole,
)
from cb16_local_opt.stage4_state_roots_r11 import (
    Stage4ContentIdentityConflict,
    Stage4IncompleteObjectError,
    Stage4PathEscapeError,
    Stage4RootSetR11,
    Stage4RootTypeError,
    Stage4StateRootsR11,
    Stage4SymlinkError,
    Stage4UnclaimedRootError,
    canonical_json_bytes,
    sha256_bytes,
)


def _make_roots(tmp_path: Path, name: str = "canonical"):
    base = tmp_path / name
    frozen = base / "frozen_raw"
    frozen.mkdir(parents=True)
    frozen.chmod(0o555)
    roots = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=base / "control",
            data_root=base / "data",
            frozen_raw_root=frozen,
            frozen_raw_identity="FROZEN_RAW_TEST_IDENTITY",
        )
    )
    receipt = roots.initialize()
    return roots, receipt


def _make_live_authority(tmp_path: Path, name: str = "canonical"):
    roots, receipt = _make_roots(tmp_path, name)
    lease_root = roots.control_path("runtime_lease_fencing_state/authority")
    Stage4AuthorityLeaseR11.initialize(lease_root)
    lease = Stage4AuthorityLeaseR11(lease_root, owner_id=f"{name}-owner")
    token = lease.acquire_authority()
    retirement = LegacyRetirementGuard(b"I" * 32, issuer_id=f"INTB-{name}")
    identity = retirement.issue_identity(
        f"{name}-canonical-runtime", RuntimeRole.R11_CANONICAL_RUNTIME
    )
    authority = CanonicalWriteAuthorityR11.bind(
        roots=roots,
        root_receipt=receipt,
        lease=lease,
        fencing_token=token,
        retirement_guard=retirement,
        runtime_identity=identity,
    )
    return roots, receipt, lease, token, retirement, identity, authority


def _evidence(evidence_id: str = "E1", payload_hash: str = "payload-hash-1") -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        payload_hash=payload_hash,
        source_generation=0,
        producer_champion_id="champion-0",
        teacher_authority_id="teacher-v1",
        physics_authority_id="physics-v1",
        teacher_generation=0,
    )


def _event(event_id: str = "event-1") -> RuntimeEvent:
    return RuntimeEvent(
        event_id=event_id,
        kind=EventKind.INPUTS_VERIFIED,
        generation=1,
        state_after=GenerationState.PREPARING,
        payload_hash="event-payload-hash",
    )


def test_canonical_bundle_routes_control_and_data_roots_and_preserves_types(tmp_path: Path):
    roots, _receipt, lease, token, _retirement, _identity, authority = _make_live_authority(tmp_path)
    bundle = open_canonical_persistence_r11(authority)
    try:
        assert roots.control_path("hot_indexes") in bundle.evidence_payloads.metadata_root.parents
        assert roots.data_path("evidence_payloads") in bundle.evidence_payloads.payload_roots[0].parents
        assert roots.control_path("authoritative_journal_metadata") in bundle._journal_backend.root.parents
        assert roots.control_path("checkpoint_metadata") in bundle.checkpoint_objects.root.parents
        assert roots.data_path("cold_content_artifacts") in bundle.checkpoint_objects.object_root.parents

        payload_item = EvidenceItemR11(
            evidence_id="PAYLOAD-E1",
            parent_snapshot_hash="snapshot-0",
            lineage_hash="lineage-0",
            teacher_protocol_hash="teacher-protocol-0",
            payload={"utility": 1.25, "engineering_replay": False},
        )
        payload_refs, payload_receipt = bundle.evidence_payloads.put_evidence([payload_item])
        assert payload_receipt.created_payload_count == 1
        assert roots.data_path("evidence_payloads") in Path(payload_refs[0].segment_path).parents
        assert bundle.evidence_payloads.get_payload(payload_item.content_hash) == payload_item.payload

        first = bundle.evidence_store.put_once(_evidence())
        second = bundle.evidence_store.put_once(_evidence())
        assert first.inserted is True and second.inserted is False
        assert bundle.evidence_store.get("E1") == _evidence()

        snapshot = SnapshotSeal("S1", "snapshot-hash", 1, "champion-0", ("E1",), True)
        assert bundle.evidence_store.seal_snapshot(snapshot) == snapshot
        assert bundle.evidence_store.get_snapshot("S1") == snapshot

        journal_receipt = bundle.journal.append_once(_event())
        assert journal_receipt.durable is True
        assert bundle.journal.read_generation(1) == [_event()]

        parent = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([1.0])})
        challenger = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([2.0])})
        proposal = TournamentCommitProposal(
            generation=1,
            decision=TournamentDecision.PROMOTE,
            parent_champion_id=parent.semantic_sha256,
            challenger_id=challenger.semantic_sha256,
            challenger_hash=challenger.semantic_sha256,
        )
        commit = bundle.checkpoint_store.atomic_commit(proposal)
        seal = CheckpointSeal(1, challenger.semantic_sha256, "state-digest", commit.commit_id)
        assert bundle.checkpoint_store.seal_checkpoint(seal) == seal
        assert bundle.checkpoint_store.read_checkpoint(1) == seal
    finally:
        bundle.close()
        lease.release_authority(token)


def test_stale_token_blocks_evidence_journal_and_checkpoint_commit_after_clean_handoff(tmp_path: Path):
    roots, _receipt, lease1, token1, _retirement, _identity, authority = _make_live_authority(tmp_path)
    bundle = open_canonical_persistence_r11(authority)
    parent = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([1.0])})
    challenger = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([2.0])})
    proposal = TournamentCommitProposal(
        generation=1,
        decision=TournamentDecision.PROMOTE,
        parent_champion_id=parent.semantic_sha256,
        challenger_id=challenger.semantic_sha256,
        challenger_hash=challenger.semantic_sha256,
    )

    lease1.release_authority(token1)
    lease2 = Stage4AuthorityLeaseR11(lease1.root, owner_id="handoff-owner")
    token2 = lease2.acquire_authority()
    try:
        with pytest.raises(StaleFencingTokenError):
            bundle.evidence_store.put_once(_evidence("STALE-E"))
        with pytest.raises(StaleFencingTokenError):
            bundle.journal.append_once(_event("stale-journal"))
        with pytest.raises(StaleFencingTokenError):
            bundle.checkpoint_store.atomic_commit(proposal)
        with pytest.raises(StaleFencingTokenError):
            bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([3.0])})
        assert bundle.evidence_store.get("STALE-E") is None
        assert bundle.journal.read_generation(1) == []
        assert bundle.checkpoint_store.read_commit(1) is None
    finally:
        lease2.release_authority(token2)
        bundle.close()


def _crash_owner(lease_root: str, connection) -> None:
    lease = Stage4AuthorityLeaseR11(lease_root, owner_id="crashing-owner")
    token = lease.acquire_authority()
    connection.send((token.epoch, token.owner_nonce))
    connection.close()
    os._exit(0)


def test_stale_token_after_explicit_crash_recovery_cannot_mutate(tmp_path: Path):
    roots, receipt = _make_roots(tmp_path, "crash")
    lease_root = roots.control_path("runtime_lease_fencing_state/authority")
    Stage4AuthorityLeaseR11.initialize(lease_root)
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_crash_owner, args=(str(lease_root), child_conn))
    proc.start()
    old_epoch, old_nonce = parent_conn.recv()
    proc.join(timeout=10)
    assert proc.exitcode == 0

    recovered = Stage4AuthorityLeaseR11(lease_root, owner_id="recovered-owner")
    recovered_token = recovered.recover_after_dead_owner()
    try:
        retirement = LegacyRetirementGuard(b"C" * 32, issuer_id="INTB-crash")
        identity = retirement.issue_identity("canonical-runtime", RuntimeRole.R11_CANONICAL_RUNTIME)
        tokens = {cap: retirement.issue_token(identity, cap) for cap in (
            Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
            Capability.APPEND_AUTHORITATIVE_EVENT_JOURNAL,
            Capability.SEAL_TRAINING_SNAPSHOT,
            Capability.TRANSITION_CHALLENGER_CHAMPION,
            Capability.SEAL_CHECKPOINT,
            Capability.RELEASE_GENERATION,
        )}
        copied_handle = Stage4AuthorityLeaseR11(lease_root, owner_id="copied-stale-owner")
        stale_authority = CanonicalWriteAuthorityR11(
            roots=roots,
            root_receipt=receipt,
            lease=copied_handle,
            fencing_token=FencingTokenR11(old_epoch, old_nonce),
            retirement_guard=retirement,
            runtime_identity=identity,
            capability_tokens=tokens,
        )

        writes = 0
        with pytest.raises(StaleFencingTokenError):
            stale_authority.assert_mutation(
                Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE,
                mutation="hostile.crash_recovery.stale_evidence_write",
            )
            writes += 1
        assert writes == 0
        assert recovered_token.epoch == old_epoch + 1
    finally:
        recovered.release_authority(recovered_token)


@pytest.mark.parametrize(
    "role",
    [
        RuntimeRole.LEGACY_REFERENCE,
        RuntimeRole.LEGACY_REPLAY,
        RuntimeRole.DIAGNOSTIC,
        RuntimeRole.TEST,
    ],
)
def test_noncanonical_roles_cannot_bind_writer_authority(tmp_path: Path, role: RuntimeRole):
    roots, receipt = _make_roots(tmp_path, f"role-{role.value}")
    lease_root = roots.control_path("runtime_lease_fencing_state/authority")
    Stage4AuthorityLeaseR11.initialize(lease_root)
    lease = Stage4AuthorityLeaseR11(lease_root, owner_id="owner")
    token = lease.acquire_authority()
    retirement = LegacyRetirementGuard(b"R" * 32)
    identity = retirement.issue_identity("noncanonical", role)
    try:
        with pytest.raises(Stage4CanonicalRoleRequired):
            CanonicalWriteAuthorityR11.bind(
                roots=roots,
                root_receipt=receipt,
                lease=lease,
                fencing_token=token,
                retirement_guard=retirement,
                runtime_identity=identity,
            )
    finally:
        lease.release_authority(token)


def test_replay_cannot_be_offered_as_new_evidence_even_with_valid_identity_token_key():
    retirement = LegacyRetirementGuard(b"P" * 32)
    replay = retirement.issue_identity("replay-worker", RuntimeRole.LEGACY_REPLAY)
    with pytest.raises(LegacyAuthorityDenied):
        retirement.issue_token(replay, Capability.MINT_OR_ADMIT_AUTHORITATIVE_EVIDENCE)


def test_wrong_root_role_unclaimed_root_and_lease_namespace_fail_closed(tmp_path: Path):
    roots, receipt = _make_roots(tmp_path, "roots")
    bad = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=roots.data_root,
            data_root=roots.control_root,
            frozen_raw_root=roots.frozen_raw_root,
            frozen_raw_identity=receipt.frozen_raw_identity,
        )
    )
    with pytest.raises(Stage4RootTypeError):
        bad.verify_startup()

    frozen = tmp_path / "legacy" / "frozen"
    frozen.mkdir(parents=True)
    frozen.chmod(0o555)
    control = tmp_path / "legacy" / "control"
    control.mkdir(parents=True)
    (control / "legacy.sqlite").write_text("legacy", encoding="utf-8")
    unclaimed = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control,
            data_root=tmp_path / "legacy" / "data",
            frozen_raw_root=frozen,
            frozen_raw_identity="LEGACY-TEST",
        )
    )
    with pytest.raises(Stage4UnclaimedRootError):
        unclaimed.initialize()

    external_lease = tmp_path / "outside-lease"
    Stage4AuthorityLeaseR11.initialize(external_lease)
    lease = Stage4AuthorityLeaseR11(external_lease, owner_id="outside")
    token = lease.acquire_authority()
    retirement = LegacyRetirementGuard(b"L" * 32)
    identity = retirement.issue_identity("canonical", RuntimeRole.R11_CANONICAL_RUNTIME)
    try:
        with pytest.raises(Stage4PersistenceRootBindingError):
            CanonicalWriteAuthorityR11.bind(
                roots=roots,
                root_receipt=receipt,
                lease=lease,
                fencing_token=token,
                retirement_guard=retirement,
                runtime_identity=identity,
            )
    finally:
        lease.release_authority(token)


def test_path_escape_and_symlink_escape_fail_closed(tmp_path: Path):
    roots, _receipt = _make_roots(tmp_path, "escape")
    with pytest.raises(Stage4PathEscapeError):
        roots.control_path("../outside")
    outside = tmp_path / "outside"
    outside.mkdir()
    link = roots.control_root / "hot_indexes" / "escape-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Stage4SymlinkError):
        roots.verify_startup()


@pytest.mark.parametrize("mode", ["orphan_payload", "incomplete_seal", "content_conflict"])
def test_s4f_incomplete_or_conflicting_payloads_never_become_authoritative(tmp_path: Path, mode: str):
    roots, _receipt = _make_roots(tmp_path, mode)
    raw = b"canonical-bytes"
    digest = sha256_bytes(raw)
    payload_path = roots.data_path(
        Path("evidence_payloads") / "sha256" / digest[:2] / f"{digest}.blob"
    )
    seal_path = roots.control_path(
        Path("object_seals") / "evidence_payloads" / digest[:2] / f"{digest}.json"
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "orphan_payload":
        payload_path.write_bytes(raw)
        with pytest.raises(Stage4IncompleteObjectError):
            roots.verify_startup()
    elif mode == "incomplete_seal":
        seal = roots._storage_seal(object_class="evidence_payloads", digest=digest, byte_count=len(raw))
        seal_path.write_bytes(canonical_json_bytes(seal) + b"\n")
        with pytest.raises(Stage4IncompleteObjectError):
            roots.verify_startup()
    else:
        ref = roots.materialize_immutable_payload("evidence_payloads", raw)
        path = Path(ref.payload_path)
        path.chmod(0o644)
        path.write_bytes(b"tampered")
        with pytest.raises(Stage4ContentIdentityConflict):
            roots.verify_startup()


def test_checkpoint_generation_seal_is_fenced_and_preserves_content_ids(tmp_path: Path):
    _roots, _receipt, lease1, token1, _retirement, _identity, authority = _make_live_authority(tmp_path, "gseal")
    bundle = open_canonical_persistence_r11(authority)
    parent = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([1.0])})
    challenger = bundle.checkpoint_objects.put_state_dict({"w": torch.tensor([2.0])})
    sealed = bundle.checkpoint_objects.seal_generation_checkpoint(
        generation=1,
        parent_champion=parent.semantic_sha256,
        challenger=challenger.semantic_sha256,
        decision="PROMOTE",
        champion_after=challenger.semantic_sha256,
        trace_batch_id="trace-1",
        trace_batch_hash="trace-hash-1",
    )
    assert sealed.parent_champion == parent.semantic_sha256
    assert sealed.champion_after == challenger.semantic_sha256

    lease1.release_authority(token1)
    lease2 = Stage4AuthorityLeaseR11(lease1.root, owner_id="gseal-next")
    token2 = lease2.acquire_authority()
    try:
        with pytest.raises(StaleFencingTokenError):
            bundle.checkpoint_objects.seal_generation_checkpoint(
                generation=2,
                parent_champion=challenger.semantic_sha256,
                challenger=challenger.semantic_sha256,
                decision="REJECT",
                champion_after=challenger.semantic_sha256,
                trace_batch_id="trace-2",
                trace_batch_hash="trace-hash-2",
            )
    finally:
        lease2.release_authority(token2)
        bundle.close()


def test_duplicate_evidence_is_idempotent_but_conflicting_duplicate_does_not_replace(tmp_path: Path):
    _roots, _receipt, lease, token, _retirement, _identity, authority = _make_live_authority(tmp_path, "dupe")
    bundle = open_canonical_persistence_r11(authority)
    try:
        original = _evidence("DUP", "hash-A")
        conflict = _evidence("DUP", "hash-B")
        first = bundle.evidence_store.put_once(original)
        same = bundle.evidence_store.put_once(original)
        conflicting = bundle.evidence_store.put_once(conflict)
        assert first.inserted is True
        assert same.inserted is False and same.payload_hash == "hash-A"
        assert conflicting.inserted is False and conflicting.payload_hash == "hash-A"
        assert bundle.evidence_store.get("DUP") == original
    finally:
        bundle.close()
        lease.release_authority(token)


def test_adapter_mutation_guard_is_reasserted_at_concrete_insert_boundary():
    conn = sqlite3.connect(":memory:", isolation_level=None)
    calls: list[str] = []

    def guard(operation: str) -> None:
        calls.append(operation)
        if operation.startswith("evidence.put_once.insert:"):
            raise RuntimeError("HOSTILE_FENCE_CHANGED_BEFORE_INSERT")

    adapter = EvidenceStoreProtocolAdapterR11(
        SimpleNamespace(conn=conn), mutation_guard=guard
    )
    with pytest.raises(RuntimeError, match="HOSTILE_FENCE_CHANGED_BEFORE_INSERT"):
        adapter.put_once(_evidence("RACE-WINDOW"))
    assert adapter.get("RACE-WINDOW") is None
    assert conn.in_transaction is False
    assert any(x.startswith("evidence.put_once.begin:") for x in calls)
    assert any(x.startswith("evidence.put_once.insert:") for x in calls)
    conn.close()


def test_actual_orchestrator_append_path_is_fenced(tmp_path: Path):
    _roots, _receipt, lease1, token1, _retirement, _identity, authority = _make_live_authority(tmp_path, "orch")
    bundle = open_canonical_persistence_r11(authority)
    stamp = AuthorityStamp(1, "champion-1", "champion-1", "teacher-v1", "physics-v1")
    R11Orchestrator(
        authority=stamp,
        evidence_store=bundle.evidence_store,
        journal=bundle.journal,
        checkpoint_store=bundle.checkpoint_store,
    )
    lease1.release_authority(token1)
    lease2 = Stage4AuthorityLeaseR11(lease1.root, owner_id="orch-next")
    token2 = lease2.acquire_authority()
    try:
        with pytest.raises(StaleFencingTokenError):
            R11Orchestrator(
                authority=AuthorityStamp(2, "champion-2", "champion-2", "teacher-v1", "physics-v1"),
                evidence_store=bundle.evidence_store,
                journal=bundle.journal,
                checkpoint_store=bundle.checkpoint_store,
            )
    finally:
        lease2.release_authority(token2)
        bundle.close()
