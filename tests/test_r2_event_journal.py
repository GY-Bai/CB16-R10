import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.r2_event_journal import R2EventItem, R2EventJournal


def _event(i: int, *, generation: int = 3, extra=None):
    payload = {"schema":"CB16_R2_EVENT_TEST","i":i,"value":float(i)}
    if extra:
        payload.update(extra)
    return R2EventItem(
        event_id=f"EV:{generation}:{i}", event_type="DECISION_EVENT" if i % 2 == 0 else "OUTCOME_SAMPLE",
        generation=generation, policy_weight_hash="P", snapshot_hash=f"S{i}",
        lineage_hash=f"L{i}", payload=payload,
    )


class R2EventJournalTests(unittest.TestCase):
    def test_generation_batch_exactly_once(self):
        with tempfile.TemporaryDirectory() as td:
            journal = R2EventJournal(Path(td) / "events")
            rows = [_event(i) for i in range(48)]
            refs, first = journal.put_events(rows)
            self.assertEqual(len(refs), 48)
            self.assertEqual(first.created_count, 48)
            self.assertEqual(first.reused_count, 0)
            self.assertEqual(journal.count(generation=3), 48)
            _refs, second = journal.put_events(rows)
            self.assertEqual(second.created_count, 0)
            self.assertEqual(second.reused_count, 48)
            self.assertTrue(journal.audit()["pass"])
            journal.close()

    def test_same_event_id_different_content_conflicts(self):
        with tempfile.TemporaryDirectory() as td:
            journal = R2EventJournal(Path(td) / "events")
            journal.put_events([_event(1)])
            with self.assertRaisesRegex(RuntimeError, "R2_EVENT_ID_CONTENT_CONFLICT"):
                journal.put_events([_event(1, extra={"changed":True})])
            journal.close()

    def test_batch_must_be_single_generation(self):
        with tempfile.TemporaryDirectory() as td:
            journal = R2EventJournal(Path(td) / "events")
            with self.assertRaisesRegex(RuntimeError, "R2_EVENT_BATCH_MUST_BE_SINGLE_GENERATION"):
                journal.put_events([_event(1, generation=1), _event(2, generation=2)])
            journal.close()


if __name__ == "__main__":
    unittest.main()
