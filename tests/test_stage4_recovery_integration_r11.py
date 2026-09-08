from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cb16_local_opt.stage4_authority_adoption_r11 import (
    FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC,
    AuthorityAdoptionContract,
    AuthorityAdoptionConflict,
    SourceAuthorityIdentity,
    SourceAuthorityMismatch,
    adopt_authority,
)
from cb16_local_opt.stage4_authority_lease_r11 import (
    STATE_FILE,
    AuthorityRecoveryRequiredError,
    FencingTokenR11,
    Stage4AuthorityLeaseR11,
    StaleFencingTokenError,
)
from cb16_local_opt.stage4_canonical_runtime_r11 import AuthorityBinding
from cb16_local_opt.stage4_recovery_integration_r11 import (
    CanonicalRootExpectationR11,
    DeadOwnerRecoveryMode,
    ExplicitDeadOwnerRecoveryRequired,
    RecoveryAuthorityMismatch,
    RecoverySemanticViolation,
    RecoveryStateIncomplete,
    Stage3RecoveryObservationR11,
    Stage4RecoveryProviderR11,
)
from cb16_local_opt.stage4_state_roots_r11 import (
    CONTROL_ROOT_MARKER,
    Stage4IncompleteObjectError,
    Stage4RootSetR11,
    Stage4RootTypeError,
    Stage4StateRootsR11,
)

FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"


def _h(ch: str) -> str:
    return ch * 64


def _source(**updates: object) -> SourceAuthorityIdentity:
    base = SourceAuthorityIdentity(
        source_repo="GY-Bai/CB16-R10",
        source_sha="8" * 40,
        semantic_freeze_identity=FREEZE_BLOB,
        source_generation=17,
        champion_identity="champion:G00000017",
        champion_hash=_h("a"),
        checkpoint_identity="checkpoint:G00000017",
        checkpoint_hash=_h("b"),
        evidence_root_identity="evidence-root:accepted-17",
        journal_head_identity="journal-head:accepted-17",
        checkpoint_root_identity="checkpoint-root:accepted-17",
    )
    return replace(base, **updates)


def _contract(source: SourceAuthorityIdentity | None = None) -> AuthorityAdoptionContract:
    return AuthorityAdoptionContract(source or _source(), "cb16-r11-canonical-authority:v1")


class Inspector:
    def __init__(self, observation: Stage3RecoveryObservationR11):
        self.observation = observation
        self.calls = 0

    def inspect_recovery_state(self, accepted_source: SourceAuthorityIdentity) -> Stage3RecoveryObservationR11:
        self.calls += 1
        return self.observation


def _observation(source: SourceAuthorityIdentity | None = None, **updates: object) -> Stage3RecoveryObservationR11:
    base = Stage3RecoveryObservationR11(
        observed_source=source or _source(),
        fully_sealed=True,
        journal_transition_complete=True,
        checkpoint_transition_complete=True,
        journal_audit_passed=True,
        checkpoint_audit_passed=True,
        replay_engineering_only=True,
        new_evidence_created=False,
        scientific_history_rewritten=False,
        new_scientific_verdict=False,
    )
    return replace(base, **updates)


def _roots(tmp_path: Path) -> tuple[Stage4StateRootsR11, CanonicalRootExpectationR11, Path]:
    frozen = tmp_path / "frozen_raw"
    frozen.mkdir()
    frozen.chmod(0o555)
    roots = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=tmp_path / "control",
            data_root=tmp_path / "data",
            frozen_raw_root=frozen,
            frozen_raw_identity="frozen-raw-qualified-identity-v1",
        )
    )
    startup = roots.initialize()
    expectation = CanonicalRootExpectationR11(
        startup.control_root_id, startup.data_root_id, startup.frozen_raw_identity
    )
    lease_root = roots.control_path("runtime_lease_fencing_state")
    Stage4AuthorityLeaseR11.initialize(lease_root)
    return roots, expectation, lease_root


def _binding(source: SourceAuthorityIdentity | None = None) -> AuthorityBinding:
    src = source or _source()
    return AuthorityBinding(
        authority_id="cb16-r11-canonical-authority:v1",
        generation=src.source_generation,
        semantic_freeze_id=src.semantic_freeze_identity,
    )


def _setup(
    tmp_path: Path,
    *,
    source: SourceAuthorityIdentity | None = None,
    observation: Stage3RecoveryObservationR11 | None = None,
    adopt: bool = True,
    mode: DeadOwnerRecoveryMode = DeadOwnerRecoveryMode.ORDINARY_RESTART,
):
    src = source or _source()
    roots, expectation, lease_root = _roots(tmp_path)
    receipt = roots.control_path("adoption_control_metadata/adoption.json")
    contract = _contract(src)
    if adopt:
        adopt_authority(src, contract=contract, receipt_path=receipt)
    inspector = Inspector(observation or _observation(src))
    provider = Stage4RecoveryProviderR11(
        adoption_contract=contract,
        adoption_receipt_path=receipt,
        state_roots=roots,
        root_expectation=expectation,
        stage3_recovery=inspector,
        lease_root=lease_root,
        recovery_owner_id="inte-recovery-owner",
        dead_owner_mode=mode,
    )
    return provider, inspector, roots, expectation, lease_root, receipt


def _crash_active_owner(lease_root: Path, token_path: Path) -> FencingTokenR11:
    code = r'''
import json, os, sys
from cb16_local_opt.stage4_authority_lease_r11 import Stage4AuthorityLeaseR11
lease = Stage4AuthorityLeaseR11(sys.argv[1], owner_id="crashed-owner")
token = lease.acquire_authority()
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump({"epoch": token.epoch, "owner_nonce": token.owner_nonce}, f)
    f.flush(); os.fsync(f.fileno())
os._exit(0)
'''
    proc = subprocess.run(
        [sys.executable, "-c", code, str(lease_root), str(token_path)],
        check=False,
        env=dict(os.environ),
    )
    assert proc.returncode == 0
    raw = json.loads(token_path.read_text(encoding="utf-8"))
    return FencingTokenR11(int(raw["epoch"]), str(raw["owner_nonce"]))


def test_restart_from_fully_sealed_state_preserves_exact_accepted_identity(tmp_path: Path) -> None:
    provider, inspector, *_ = _setup(tmp_path)
    recovered = provider.recover_existing_state(_binding())
    assert recovered.authority_id == "cb16-r11-canonical-authority:v1"
    assert recovered.generation == _source().source_generation
    assert recovered.state_id == _source().checkpoint_identity
    assert recovered.scientific_history_id == _source().journal_head_identity
    assert inspector.calls == 1


def test_crash_before_adoption_publication_fails_before_stage3_inspection(tmp_path: Path) -> None:
    provider, inspector, *_ = _setup(tmp_path, adopt=False)
    with pytest.raises(Exception, match="STAGE4_ADOPTION_RECEIPT_READ_FAILED"):
        provider.recover_existing_state(_binding())
    assert inspector.calls == 0


def test_crash_after_atomic_adoption_publication_keeps_one_valid_identity(tmp_path: Path) -> None:
    roots, expectation, lease_root = _roots(tmp_path)
    receipt = roots.control_path("adoption_control_metadata/adoption.json")
    contract = _contract()

    def crash(stage: str) -> None:
        if stage == FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC:
            raise RuntimeError("synthetic-crash-after-publish")

    with pytest.raises(RuntimeError, match="synthetic-crash-after-publish"):
        adopt_authority(_source(), contract=contract, receipt_path=receipt, _fault_hook=crash)

    provider = Stage4RecoveryProviderR11(
        adoption_contract=contract,
        adoption_receipt_path=receipt,
        state_roots=roots,
        root_expectation=expectation,
        stage3_recovery=Inspector(_observation()),
        lease_root=lease_root,
        recovery_owner_id="inte-recovery-owner",
    )
    assert provider.recover_existing_state(_binding()).generation == 17


def test_duplicate_identical_adoption_is_idempotent_but_conflict_is_refused(tmp_path: Path) -> None:
    provider, _inspector, _roots_obj, _expectation, _lease_root, receipt = _setup(tmp_path)
    before = receipt.read_bytes()
    result = adopt_authority(_source(), contract=_contract(), receipt_path=receipt)
    assert result.status == "ALREADY_ADOPTED"
    assert receipt.read_bytes() == before
    other = _source(source_generation=18)
    with pytest.raises((SourceAuthorityMismatch, AuthorityAdoptionConflict)):
        adopt_authority(other, contract=_contract(other), receipt_path=receipt)
    assert receipt.read_bytes() == before
    assert provider.recover_existing_state(_binding()).generation == 17


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("journal_head_identity", "journal-head:wrong"),
        ("checkpoint_identity", "checkpoint:wrong"),
        ("checkpoint_hash", _h("c")),
        ("champion_identity", "champion:wrong"),
        ("champion_hash", _h("d")),
        ("source_generation", 18),
        ("source_sha", "9" * 40),
        ("semantic_freeze_identity", "1" * 40),
        ("evidence_root_identity", "evidence-root:wrong"),
        ("checkpoint_root_identity", "checkpoint-root:wrong"),
    ],
)
def test_any_observed_source_identity_drift_fails_closed(tmp_path: Path, field: str, value: object) -> None:
    observed = replace(_source(), **{field: value})
    provider, *_ = _setup(tmp_path, observation=_observation(observed))
    with pytest.raises(RecoveryAuthorityMismatch, match="ACCEPTED_SOURCE_MISMATCH"):
        provider.recover_existing_state(_binding())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"fully_sealed": False}, "UNSEALED_STATE"),
        ({"journal_transition_complete": False}, "JOURNAL_TRANSITION_INCOMPLETE"),
        ({"checkpoint_transition_complete": False}, "CHECKPOINT_TRANSITION_INCOMPLETE"),
        ({"journal_audit_passed": False}, "JOURNAL_AUDIT_FAILED"),
        ({"checkpoint_audit_passed": False}, "CHECKPOINT_AUDIT_FAILED"),
    ],
)
def test_crash_or_incomplete_journal_checkpoint_transition_is_non_authoritative(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    provider, *_ = _setup(tmp_path, observation=_observation(**updates))
    with pytest.raises(RecoveryStateIncomplete, match=message):
        provider.recover_existing_state(_binding())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"replay_engineering_only": False}, "REPLAY_EVIDENCE_REINTERPRETATION"),
        ({"new_evidence_created": True}, "NEW_EVIDENCE"),
        ({"scientific_history_rewritten": True}, "HISTORY_REWRITE"),
        ({"new_scientific_verdict": True}, "NEW_SCIENTIFIC_VERDICT"),
    ],
)
def test_recovery_cannot_mint_evidence_history_or_verdict(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    provider, *_ = _setup(tmp_path, observation=_observation(**updates))
    with pytest.raises(RecoverySemanticViolation, match=message):
        provider.recover_existing_state(_binding())


def test_wrong_binding_generation_or_runtime_identity_fails_before_recovery(tmp_path: Path) -> None:
    provider, inspector, *_ = _setup(tmp_path)
    with pytest.raises(RecoveryAuthorityMismatch, match="BINDING_GENERATION_MISMATCH"):
        provider.recover_existing_state(replace(_binding(), generation=18))
    with pytest.raises(RecoveryAuthorityMismatch, match="TARGET_AUTHORITY_IDENTITY_MISMATCH"):
        provider.recover_existing_state(replace(_binding(), authority_id="legacy-runtime"))
    assert inspector.calls == 0


def test_wrong_canonical_root_identity_fails_before_stage3_inspection(tmp_path: Path) -> None:
    _provider, inspector, roots, expectation, lease_root, receipt = _setup(tmp_path)
    bad = replace(expectation, control_root_id="0" * 64)
    provider = Stage4RecoveryProviderR11(
        adoption_contract=_contract(), adoption_receipt_path=receipt, state_roots=roots,
        root_expectation=bad, stage3_recovery=inspector, lease_root=lease_root,
        recovery_owner_id="inte-recovery-owner",
    )
    with pytest.raises(Stage4RootTypeError, match="CONTROL_ROOT_ID_DRIFT"):
        provider.recover_existing_state(_binding())
    assert inspector.calls == 0


def test_corrupt_root_marker_fails_closed(tmp_path: Path) -> None:
    provider, _inspector, roots, *_ = _setup(tmp_path)
    marker = roots.control_root / CONTROL_ROOT_MARKER
    marker.chmod(0o644)
    marker.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Stage4RootTypeError, match="ROOT_MARKER_MISMATCH"):
        provider.recover_existing_state(_binding())


def test_orphan_data_object_after_crash_is_never_reconstructed_as_authority(tmp_path: Path) -> None:
    provider, _inspector, roots, *_ = _setup(tmp_path)
    digest = "e" * 64
    orphan = roots.data_root / "evidence_payloads" / "sha256" / digest[:2] / f"{digest}.blob"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"orphan")
    with pytest.raises(Stage4IncompleteObjectError, match="ORPHAN_PAYLOAD"):
        provider.recover_existing_state(_binding())


def test_ordinary_restart_cannot_steal_crashed_active_authority(tmp_path: Path) -> None:
    provider, _inspector, _roots_obj, _expectation, lease_root, _receipt = _setup(tmp_path)
    old_token = _crash_active_owner(lease_root, tmp_path / "crashed-token.json")
    ordinary = Stage4AuthorityLeaseR11(lease_root, owner_id="ordinary-owner")
    with pytest.raises(AuthorityRecoveryRequiredError, match="EXPLICIT_DEAD_OWNER_RECOVERY_REQUIRED"):
        ordinary.acquire_authority()
    assert ordinary.inspect_current_authority().fencing_token == old_token
    with pytest.raises(ExplicitDeadOwnerRecoveryRequired):
        provider.recover_existing_state(_binding())
    assert ordinary.inspect_current_authority().fencing_token == old_token


def test_explicit_dead_owner_recovery_advances_epoch_and_stales_old_token(tmp_path: Path) -> None:
    _provider, _inspector, roots, expectation, lease_root, receipt = _setup(tmp_path)
    old_token = _crash_active_owner(lease_root, tmp_path / "crashed-token.json")
    explicit = Stage4RecoveryProviderR11(
        adoption_contract=_contract(), adoption_receipt_path=receipt, state_roots=roots,
        root_expectation=expectation, stage3_recovery=Inspector(_observation()), lease_root=lease_root,
        recovery_owner_id="explicit-recovery-owner",
        dead_owner_mode=DeadOwnerRecoveryMode.EXPLICIT_DEAD_OWNER_RECOVERY,
    )
    assert explicit.recover_existing_state(_binding()).generation == 17
    probe = Stage4AuthorityLeaseR11(lease_root, owner_id="post-recovery-probe")
    snap = probe.inspect_current_authority()
    assert snap.state == "RELEASED"
    assert snap.epoch > old_token.epoch
    assert probe.validate_fencing_token(old_token) is False
    with pytest.raises(StaleFencingTokenError):
        probe.assert_fencing_token(old_token)


def test_second_restart_preserves_identity_without_recovering_released_state_again(tmp_path: Path) -> None:
    _provider, _inspector, roots, expectation, lease_root, receipt = _setup(tmp_path)
    _crash_active_owner(lease_root, tmp_path / "crashed-token.json")
    inspector = Inspector(_observation())
    explicit = Stage4RecoveryProviderR11(
        adoption_contract=_contract(), adoption_receipt_path=receipt, state_roots=roots,
        root_expectation=expectation, stage3_recovery=inspector, lease_root=lease_root,
        recovery_owner_id="explicit-recovery-owner",
        dead_owner_mode=DeadOwnerRecoveryMode.EXPLICIT_DEAD_OWNER_RECOVERY,
    )
    first = explicit.recover_existing_state(_binding())
    epoch1 = Stage4AuthorityLeaseR11(lease_root, owner_id="probe-a").inspect_current_authority().epoch
    second = explicit.recover_existing_state(_binding())
    epoch2 = Stage4AuthorityLeaseR11(lease_root, owner_id="probe-b").inspect_current_authority().epoch
    assert first == second
    assert epoch1 == epoch2
    assert inspector.calls == 2


def test_corrupt_lease_metadata_fails_closed(tmp_path: Path) -> None:
    provider, _inspector, _roots_obj, _expectation, lease_root, _receipt = _setup(tmp_path)
    (lease_root / STATE_FILE).write_text("{corrupt", encoding="utf-8")
    with pytest.raises(Exception, match="STAGE4_LEASE_JSON_CORRUPT"):
        provider.recover_existing_state(_binding())


def test_adoption_and_lease_must_live_in_s4f_control_namespaces(tmp_path: Path) -> None:
    _provider, inspector, roots, expectation, lease_root, receipt = _setup(tmp_path)
    with pytest.raises(RecoveryAuthorityMismatch, match="ADOPTION_RECEIPT_OUTSIDE"):
        Stage4RecoveryProviderR11(
            adoption_contract=_contract(), adoption_receipt_path=tmp_path / "outside-adoption.json",
            state_roots=roots, root_expectation=expectation, stage3_recovery=inspector,
            lease_root=lease_root, recovery_owner_id="owner",
        )
    with pytest.raises(RecoveryAuthorityMismatch, match="LEASE_OUTSIDE"):
        Stage4RecoveryProviderR11(
            adoption_contract=_contract(), adoption_receipt_path=receipt, state_roots=roots,
            root_expectation=expectation, stage3_recovery=inspector,
            lease_root=tmp_path / "outside-lease", recovery_owner_id="owner",
        )
