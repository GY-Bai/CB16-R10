from __future__ import annotations

import pytest

from cb16_local_opt.checkpoint_store_r11 import CheckpointStoreR11
from cb16_local_opt.event_journal_r11 import EventJournalR11
from cb16_local_opt.integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from cb16_local_opt.orchestrator_r11 import IllegalTransition, R11Orchestrator
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    FrozenInputReceipt,
    PolicyResult,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
    semantic_hash,
)
from cb16_local_opt.stage3_e2_lineage_soak_r11 import (
    PHYSICS_AUTHORITY_ID,
    TEACHER_AUTHORITY_ID,
    _frozen_replay_evidence,
    _make_policy_state,
    _snapshot,
)


def test_stage3_e2_commit_without_checkpoint_cannot_release_next_generation(tmp_path):
    journal_store = EventJournalR11(tmp_path / "journal", synchronous="OFF")
    checkpoint_store = CheckpointStoreR11(tmp_path / "checkpoints", synchronous="OFF")
    journal = EventJournalProtocolAdapterR11(journal_store)
    evidence_store = EvidenceStoreProtocolAdapterR11(journal_store)
    checkpoints = CheckpointStoreProtocolAdapterR11(checkpoint_store)

    try:
        seed = checkpoint_store.put_state_dict(_make_policy_state(-1, seed=True))
        authority = AuthorityStamp(
            generation=0,
            champion_id=seed.semantic_sha256,
            champion_hash=seed.semantic_sha256,
            teacher_authority_id=TEACHER_AUTHORITY_ID,
            physics_authority_id=PHYSICS_AUTHORITY_ID,
        )
        core = R11Orchestrator(
            authority=authority,
            evidence_store=evidence_store,
            journal=journal,
            checkpoint_store=checkpoints,
        )
        evidence = _frozen_replay_evidence()
        core.verify_frozen_inputs(
            FrozenInputReceipt(
                input_identity="CB16_R11_STAGE3_E2_FROZEN_REPLAY_ONLY",
                seal_hash=evidence.payload_hash,
                verified=True,
            )
        )
        core.accept_teacher_evidence(evidence)
        snapshot = _snapshot(authority, evidence)
        core.seal_training_snapshot(snapshot)

        core.accept_policy_result(
            PolicyResult(
                work_id="policy-g0-release-boundary",
                generation=0,
                champion_id=authority.champion_id,
                payload_hash=semantic_hash({"kind": "policy", "generation": 0}),
            )
        )
        core.start_trace("trace-g0-release-boundary")
        core.start_challenger_training(work_id="train-g0-release-boundary")

        challenger = checkpoint_store.put_state_dict(_make_policy_state(0))
        challenger_id = challenger.semantic_sha256
        training = TrainingResult(
            work_id="train-g0-release-boundary",
            generation=0,
            challenger_id=challenger_id,
            challenger_hash=challenger_id,
            parent_champion_id=authority.champion_id,
            snapshot_id=snapshot.snapshot_id,
            payload_hash=semantic_hash({"kind": "training", "generation": 0, "challenger": challenger_id}),
        )
        core.complete_training(training)

        validation = ValidationResult(
            work_id="validation-g0-release-boundary",
            generation=0,
            challenger_id=challenger_id,
            parent_champion_id=authority.champion_id,
            snapshot_id=snapshot.snapshot_id,
            validation_id="validation-g0-release-boundary",
            payload_hash=semantic_hash({"kind": "validation", "generation": 0, "challenger": challenger_id}),
        )
        core.complete_validation(validation)

        tournament = TournamentResult(
            work_id="tournament-g0-release-boundary",
            generation=0,
            challenger_id=challenger_id,
            parent_champion_id=authority.champion_id,
            validation_id=validation.validation_id,
            decision=TournamentDecision.PROMOTE,
            payload_hash=semantic_hash({"kind": "tournament", "generation": 0, "decision": "PROMOTE"}),
        )
        core.decide_tournament(tournament)
        commit = core.commit_tournament()

        # Atomic commit alone is not sufficient. The checkpoint seal is an
        # authority-changing durability boundary and release must fail closed until it exists.
        with pytest.raises(IllegalTransition, match="R11_STATE_REQUIRED:COMMITTED"):
            core.release_next_generation()

        checkpoint = core.seal_checkpoint(commit.next_champion_hash)
        assert checkpoint.commit_id == commit.commit_id
        assert checkpoint.checkpoint_id == commit.next_champion_hash

        next_authority = core.release_next_generation()
        assert next_authority.generation == 1
        assert next_authority.champion_id == challenger_id
        assert next_authority.champion_hash == challenger_id
        assert next_authority.teacher_authority_id == TEACHER_AUTHORITY_ID
        assert next_authority.physics_authority_id == PHYSICS_AUTHORITY_ID
        assert journal_store.audit()["pass"] is True
    finally:
        checkpoint_store.close()
        journal_store.close()
