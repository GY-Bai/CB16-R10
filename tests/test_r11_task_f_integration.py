from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cb16_local_opt.checkpoint_store_r11 import CheckpointStoreR11
from cb16_local_opt.evidence_store_r11 import EvidenceStoreR11
from cb16_local_opt.event_journal_r11 import EventJournalR11
from cb16_local_opt.integration_adapters_r11 import (
    CheckpointStoreProtocolAdapterR11,
    EvidenceStoreProtocolAdapterR11,
    EventJournalProtocolAdapterR11,
)
from cb16_local_opt.orchestrator_r11 import LineageViolation, R11Orchestrator
from cb16_local_opt.runtime_events_r11 import (
    AuthorityStamp,
    EvidenceRef,
    FrozenInputReceipt,
    GenerationState,
    PolicyResult,
    SnapshotSeal,
    TournamentDecision,
    TournamentResult,
    TrainingResult,
    ValidationResult,
)
from cb16_local_opt.runtime_protocols_r11 import CheckpointStore, EvidenceStore, EventJournal


class TestR11TaskFIntegration(unittest.TestCase):
    def _run_generation(self, decision: TournamentDecision) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence_impl = EvidenceStoreR11(
                metadata_root=root / "meta",
                payload_roots=[root / "payload"],
                codec="none",
                segment_target_bytes=4096,
                read_only_source_roots=[root / "frozen_source"],
            )
            journal_impl = EventJournalR11(root / "journal")
            checkpoint_impl = CheckpointStoreR11(root / "checkpoints")
            try:
                parent = checkpoint_impl.put_state_dict({"w": torch.tensor([1.0], dtype=torch.float32)})
                challenger = checkpoint_impl.put_state_dict({"w": torch.tensor([2.0], dtype=torch.float32)})
                evidence = EvidenceStoreProtocolAdapterR11(evidence_impl)
                journal = EventJournalProtocolAdapterR11(journal_impl)
                checkpoints = CheckpointStoreProtocolAdapterR11(checkpoint_impl)
                self.assertIsInstance(evidence, EvidenceStore)
                self.assertIsInstance(journal, EventJournal)
                self.assertIsInstance(checkpoints, CheckpointStore)

                authority = AuthorityStamp(
                    generation=1,
                    champion_id=parent.semantic_sha256,
                    champion_hash=parent.semantic_sha256,
                    teacher_authority_id="teacher-v7-qualified",
                    physics_authority_id="frozen-physics-v1",
                )
                runtime = R11Orchestrator(
                    authority=authority,
                    evidence_store=evidence,
                    journal=journal,
                    checkpoint_store=checkpoints,
                )
                runtime.verify_frozen_inputs(FrozenInputReceipt("r104-cache", "frozen-seal", True))
                old = EvidenceRef(
                    evidence_id="teacher-e0",
                    payload_hash="teacher-payload-content-address",
                    source_generation=0,
                    producer_champion_id="historical-champion",
                    teacher_authority_id=authority.teacher_authority_id,
                    physics_authority_id=authority.physics_authority_id,
                    teacher_generation=0,
                )
                runtime.accept_teacher_evidence(old)
                snapshot = SnapshotSeal(
                    snapshot_id="snapshot-g1",
                    snapshot_hash="snapshot-g1-hash",
                    generation=1,
                    parent_champion_id=authority.champion_id,
                    evidence_ids=(old.evidence_id,),
                )
                runtime.seal_training_snapshot(snapshot)
                runtime.accept_policy_result(
                    PolicyResult("policy-g1", 1, authority.champion_id, "policy-payload-hash")
                )
                runtime.start_trace("trace-g1")
                trace = EvidenceRef(
                    evidence_id="trace-e1",
                    payload_hash="trace-payload-content-address",
                    source_generation=1,
                    producer_champion_id=authority.champion_id,
                    teacher_authority_id=authority.teacher_authority_id,
                    physics_authority_id=authority.physics_authority_id,
                    teacher_generation=1,
                )
                runtime.materialize_trace_evidence(trace, work_id="trace-complete-g1")
                runtime.start_challenger_training(work_id="train-g1")
                runtime.complete_training(
                    TrainingResult(
                        work_id="train-g1",
                        generation=1,
                        challenger_id=challenger.semantic_sha256,
                        challenger_hash=challenger.semantic_sha256,
                        parent_champion_id=authority.champion_id,
                        snapshot_id=snapshot.snapshot_id,
                        payload_hash="training-result-hash",
                    )
                )
                runtime.complete_validation(
                    ValidationResult(
                        work_id="validation-g1",
                        generation=1,
                        challenger_id=challenger.semantic_sha256,
                        parent_champion_id=authority.champion_id,
                        snapshot_id=snapshot.snapshot_id,
                        validation_id="validation-id-g1",
                        payload_hash="validation-result-hash",
                    )
                )
                runtime.decide_tournament(
                    TournamentResult(
                        work_id="tournament-g1",
                        generation=1,
                        challenger_id=challenger.semantic_sha256,
                        parent_champion_id=authority.champion_id,
                        validation_id="validation-id-g1",
                        decision=decision,
                        payload_hash="tournament-result-hash-" + decision.value,
                    )
                )
                commit = runtime.commit_tournament()
                expected = (
                    challenger.semantic_sha256
                    if decision is TournamentDecision.PROMOTE
                    else parent.semantic_sha256
                )
                self.assertEqual(commit.next_champion_id, expected)
                self.assertEqual(commit.next_champion_hash, expected)
                runtime.seal_checkpoint(expected)
                nxt = runtime.release_next_generation()
                self.assertEqual(nxt.generation, 2)
                self.assertEqual(nxt.champion_id, expected)
                self.assertEqual(nxt.champion_hash, expected)
                self.assertEqual(runtime.record.state, GenerationState.COMMITTED)

                # Durable recovery must rebuild the same generation authority/barriers.
                recovered = R11Orchestrator(
                    authority=authority,
                    evidence_store=EvidenceStoreProtocolAdapterR11(evidence_impl),
                    journal=EventJournalProtocolAdapterR11(journal_impl),
                    checkpoint_store=CheckpointStoreProtocolAdapterR11(checkpoint_impl),
                    recover=True,
                )
                self.assertEqual(recovered.record.state, GenerationState.COMMITTED)
                self.assertEqual(recovered.record.commit, runtime.record.commit)
                self.assertEqual(recovered.record.checkpoint, runtime.record.checkpoint)
                self.assertEqual(recovered.record.snapshot, runtime.record.snapshot)

                # Future evidence remains fail-closed after concrete-store integration.
                with self.assertRaises(LineageViolation):
                    recovered.materialize_trace_evidence(
                        EvidenceRef(
                            evidence_id="future",
                            payload_hash="future-hash",
                            source_generation=2,
                            producer_champion_id=authority.champion_id,
                            teacher_authority_id=authority.teacher_authority_id,
                            physics_authority_id=authority.physics_authority_id,
                            teacher_generation=2,
                        ),
                        work_id="future-work",
                    )
            finally:
                checkpoint_impl.close()
                journal_impl.close()
                evidence_impl.close()

    def test_promote_durable_bridge(self):
        self._run_generation(TournamentDecision.PROMOTE)

    def test_reject_durable_bridge(self):
        self._run_generation(TournamentDecision.REJECT)

    def test_checkpoint_adapter_rejects_mutable_policy_alias(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStoreR11(Path(td) / "cp")
            try:
                parent = store.put_state_dict({"w": torch.tensor([1.0])})
                challenger = store.put_state_dict({"w": torch.tensor([2.0])})
                adapter = CheckpointStoreProtocolAdapterR11(store)
                from cb16_local_opt.runtime_events_r11 import TournamentCommitProposal
                with self.assertRaisesRegex(RuntimeError, "POLICY_ID_MUST_EQUAL_CONTENT_HASH"):
                    adapter.atomic_commit(
                        TournamentCommitProposal(
                            generation=0,
                            decision=TournamentDecision.PROMOTE,
                            parent_champion_id=parent.semantic_sha256,
                            challenger_id="mutable-alias",
                            challenger_hash=challenger.semantic_sha256,
                        )
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
