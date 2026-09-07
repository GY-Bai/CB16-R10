from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.r2_evidence_storage import (
    FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX,
    R2EvidenceItem,
    R2EvidenceStore,
)
from cb16_local_opt.r2_evidence_storage_incremental import R2IncrementalEvidenceStore


def _item(i: int) -> R2EvidenceItem:
    return R2EvidenceItem(
        evidence_id=f"E{i}",
        parent_snapshot_hash=f"P{i}",
        lineage_hash=f"L{i}",
        teacher_protocol_hash="T",
        payload={"schema": "TEST", "i": i, "x": [i, i + 1, i + 2]},
    )


class IncrementalPayloadRecoveryTest(unittest.TestCase):
    def test_clean_reopen_skips_indexed_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, payload = root / "meta", root / "payload"
            store = R2EvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            store.materialize_evidence_set(evidence_set_id="S", items=[_item(0), _item(1)])
            store.close()

            reopened = R2IncrementalEvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            receipt = reopened.recover_payload_index()
            self.assertEqual(receipt["discovered_payloads"], 0)
            self.assertEqual(receipt["tail_bytes_scanned"], 0)
            self.assertGreater(receipt["indexed_prefix_bytes_skipped"], 0)
            reopened.close()

    def test_fsynced_orphan_tail_is_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, payload = root / "meta", root / "payload"
            store = R2EvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            store.materialize_evidence_set(evidence_set_id="S0", items=[_item(0)])

            def crash(point: str):
                if point == FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX:
                    raise RuntimeError("INJECTED_CRASH")

            with self.assertRaisesRegex(RuntimeError, "INJECTED_CRASH"):
                store.materialize_evidence_set(
                    evidence_set_id="S1", items=[_item(1)], fail_hook=crash
                )
            store.close()

            reopened = R2IncrementalEvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            receipt = reopened.recover_payload_index()
            self.assertEqual(receipt["discovered_payloads"], 1)
            self.assertGreater(receipt["tail_bytes_scanned"], 0)
            self.assertIsNotNone(reopened.get_payload(_item(1).content_hash))
            reopened.close()

    def test_incomplete_tail_is_truncated_after_verified_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meta, payload = root / "meta", root / "payload"
            store = R2EvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            store.materialize_evidence_set(evidence_set_id="S", items=[_item(0)])
            segment = store._segments(0)[0]
            clean_size = segment.stat().st_size
            store.close()
            with segment.open("ab") as handle:
                handle.write(b"\x04zli")
            self.assertGreater(segment.stat().st_size, clean_size)

            reopened = R2IncrementalEvidenceStore(
                metadata_root=meta,
                payload_roots=[payload],
                codec="zlib",
                recover_on_open=False,
            )
            receipt = reopened.recover_payload_index()
            self.assertEqual(len(receipt["truncated_tails"]), 1)
            self.assertEqual(segment.stat().st_size, clean_size)
            reopened.close()


if __name__ == "__main__":
    unittest.main()
