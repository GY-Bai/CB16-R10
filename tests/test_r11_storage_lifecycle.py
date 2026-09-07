from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cb16_local_opt.checkpoint_store_r11 import CheckpointStoreR11
from cb16_local_opt.evidence_store_r11 import EvidenceItemR11, EvidenceStoreR11
from cb16_local_opt.event_journal_r11 import (
    EventItemR11,
    EventJournalR11,
    FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL,
)


class TestR11StorageLifecycle(unittest.TestCase):
    def test_payload_survives_prejournal_crash_then_replay_is_zero_new_payload_and_seals_generation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evidence = EvidenceStoreR11(
                metadata_root=root / "evidence_meta", payload_roots=[root / "payload"], codec="none"
            )
            ev = EvidenceItemR11("E", "parent-snapshot", "causal-trace", "teacher", {"u": 0.25})
            evidence.put_evidence([ev])

            journal = EventJournalR11(root / "journal")
            event = EventItemR11(
                "trace-1", "TRACE", 0, "P", "snapshot", "causal-trace", {"decision": "FLAT"}
            )

            def crash(phase: str):
                if phase == FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL:
                    raise RuntimeError("crash-after-payload-before-journal-commit")

            with self.assertRaisesRegex(RuntimeError, "crash-after-payload-before-journal-commit"):
                journal.seal_trace_batch([event], trace_batch_id="tb0", fail_hook=crash)
            journal.close()
            evidence.close()

            evidence = EvidenceStoreR11(
                metadata_root=root / "evidence_meta", payload_roots=[root / "payload"], codec="none"
            )
            _, replay = evidence.put_evidence([ev])
            self.assertEqual(replay.created_payload_count, 0)
            journal = EventJournalR11(root / "journal")
            batch = journal.seal_trace_batch([event], trace_batch_id="tb0")

            checkpoints = CheckpointStoreR11(root / "checkpoints")
            parent = checkpoints.put_state_dict({"w": torch.tensor([1.0])}).semantic_sha256
            challenger = checkpoints.put_state_dict({"w": torch.tensor([2.0])}).semantic_sha256
            snapshot = checkpoints.seal_generation_checkpoint(
                generation=0, parent_champion=parent, challenger=challenger, decision="REJECT",
                champion_after=parent, trace_batch_id=batch.trace_batch_id,
                trace_batch_hash=batch.batch_hash, evidence_set_hash="evidence-set",
            )
            journal.seal_generation_outcome(
                generation=0, parent_champion=parent, challenger=challenger, decision="REJECT",
                champion_after=parent, trace_batch_id=batch.trace_batch_id,
                snapshot_hash=snapshot.snapshot_hash,
            )
            self.assertTrue(journal.audit()["pass"])
            self.assertEqual(journal.pending_trace_batches(), [])
            evidence.close(); journal.close(); checkpoints.close()


if __name__ == "__main__":
    unittest.main()
