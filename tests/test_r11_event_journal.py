from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.event_journal_r11 import (
    EventItemR11,
    EventJournalR11,
    FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL,
    FAIL_AFTER_JOURNAL_COMMIT_BEFORE_SNAPSHOT_SEAL,
)


def event(name: str, generation: int = 0, value: int = 1) -> EventItemR11:
    return EventItemR11(
        event_id=name,
        event_type="TRACE",
        generation=generation,
        policy_weight_hash=f"policy-{generation}",
        snapshot_hash=f"snapshot-{generation}",
        lineage_hash=f"lineage-{name}",
        payload={"value": value},
    )


class TestEventJournalR11(unittest.TestCase):
    def test_duplicate_and_reordered_completion_have_one_canonical_batch(self):
        with tempfile.TemporaryDirectory() as td:
            journal = EventJournalR11(Path(td))
            a, b = event("a"), event("b")
            first = journal.seal_trace_batch([b, a, a], trace_batch_id="batch-0")
            self.assertEqual(first.event_count, 2)
            self.assertEqual(first.created_event_count, 2)
            replay = journal.seal_trace_batch([a, b], trace_batch_id="batch-0")
            self.assertTrue(replay.reused_batch)
            self.assertEqual(first.batch_hash, replay.batch_hash)
            self.assertEqual([x["event_id"] for x in journal.canonical_events("batch-0")], ["a", "b"])
            self.assertEqual(journal.audit()["events"], 2)
            journal.close()

    def test_crash_before_batch_seal_rolls_back_all_events(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            journal = EventJournalR11(root)

            def fail(phase: str):
                if phase == FAIL_AFTER_EVENTS_BEFORE_BATCH_SEAL:
                    raise RuntimeError("crash-before-seal")

            with self.assertRaisesRegex(RuntimeError, "crash-before-seal"):
                journal.seal_trace_batch([event("a")], trace_batch_id="batch-0", fail_hook=fail)
            self.assertEqual(journal.audit()["events"], 0)
            self.assertEqual(journal.audit()["trace_batches"], 0)
            journal.close()

    def test_crash_after_journal_commit_surfaces_pending_batch_without_dropping_event(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            journal = EventJournalR11(root)

            def fail(phase: str):
                if phase == FAIL_AFTER_JOURNAL_COMMIT_BEFORE_SNAPSHOT_SEAL:
                    raise RuntimeError("power-loss")

            with self.assertRaisesRegex(RuntimeError, "power-loss"):
                journal.seal_trace_batch([event("a")], trace_batch_id="batch-0", fail_hook=fail)
            journal.close()

            reopened = EventJournalR11(root)
            recovery = reopened.recover()
            self.assertEqual(recovery["events_dropped"], 0)
            self.assertEqual(recovery["pending_count"], 1)
            replay = reopened.seal_trace_batch([event("a")], trace_batch_id="batch-0")
            self.assertTrue(replay.reused_batch)
            reopened.close()

    def test_reject_cannot_become_next_parent_and_promote_can(self):
        with tempfile.TemporaryDirectory() as td:
            journal = EventJournalR11(Path(td))
            b0 = journal.seal_trace_batch([event("g0", 0)], trace_batch_id="b0")
            journal.seal_generation_outcome(
                generation=0, parent_champion="P", challenger="C0", decision="REJECT",
                champion_after="P", trace_batch_id=b0.trace_batch_id, snapshot_hash="s0",
            )
            b1 = journal.seal_trace_batch([event("g1", 1)], trace_batch_id="b1")
            with self.assertRaisesRegex(RuntimeError, "REJECTED_OR_STALE_CHALLENGER_PARENT_FORBIDDEN"):
                journal.seal_generation_outcome(
                    generation=1, parent_champion="C0", challenger="C1", decision="PROMOTE",
                    champion_after="C1", trace_batch_id=b1.trace_batch_id, snapshot_hash="bad",
                )
            journal.seal_generation_outcome(
                generation=1, parent_champion="P", challenger="C1", decision="PROMOTE",
                champion_after="C1", trace_batch_id=b1.trace_batch_id, snapshot_hash="s1",
            )
            b2 = journal.seal_trace_batch([event("g2", 2)], trace_batch_id="b2")
            journal.seal_generation_outcome(
                generation=2, parent_champion="C1", challenger="C2", decision="REJECT",
                champion_after="C1", trace_batch_id=b2.trace_batch_id, snapshot_hash="s2",
            )
            self.assertTrue(journal.audit()["pass"])
            journal.close()

    def test_same_event_id_different_content_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            journal = EventJournalR11(Path(td))
            journal.seal_trace_batch([event("a", value=1)], trace_batch_id="b0")
            with self.assertRaisesRegex(RuntimeError, "EVENT_ID_CONTENT_CONFLICT"):
                journal.seal_trace_batch([event("a", value=2)], trace_batch_id="b1")
            journal.close()


if __name__ == "__main__":
    unittest.main()
