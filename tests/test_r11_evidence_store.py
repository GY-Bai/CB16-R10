from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.evidence_store_r11 import (
    EvidenceItemR11,
    EvidenceStoreR11,
    FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX,
)


def item(i: int, payload=None) -> EvidenceItemR11:
    return EvidenceItemR11(
        evidence_id=f"E{i}",
        parent_snapshot_hash="parent",
        lineage_hash=f"lineage-{i}",
        teacher_protocol_hash="teacher",
        payload=payload if payload is not None else {"i": i, "blob": "x" * 220},
    )


class TestEvidenceStoreR11(unittest.TestCase):
    def make_store(self, root: Path, **kwargs) -> EvidenceStoreR11:
        return EvidenceStoreR11(
            metadata_root=root / "ssd",
            payload_roots=[root / "hdd"],
            segment_target_bytes=512,
            codec="none",
            **kwargs,
        )

    def test_payload_fsync_before_index_recovers_and_replay_writes_zero_new_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = self.make_store(root)

            def fail(phase: str):
                if phase == FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX:
                    raise RuntimeError("crash")

            with self.assertRaisesRegex(RuntimeError, "crash"):
                store.put_evidence([item(1)], fail_hook=fail)
            store.close()

            reopened = self.make_store(root)
            self.assertEqual(reopened.startup_stats["recovered_payloads"], 1)
            _, receipt = reopened.put_evidence([item(1)])
            self.assertEqual(receipt.created_payload_count, 0)
            self.assertEqual(receipt.created_evidence_count, 1)
            _, replay = reopened.put_evidence([item(1)])
            self.assertEqual(replay.created_payload_count, 0)
            self.assertEqual(replay.created_evidence_count, 0)
            reopened.close()

    def test_truncated_active_tail_is_only_recovered_region_and_is_truncated(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = self.make_store(root)
            store.put_evidence([item(1)])
            segment_id, path, committed_end, _ = store._active_row(0)
            with path.open("ab") as handle:
                handle.write(b"\x04zlibBROKEN_PARTIAL_FRAME")
                handle.flush()
            before = path.stat().st_size
            store.close()

            reopened = self.make_store(root)
            self.assertEqual(path.stat().st_size, committed_end)
            self.assertGreater(before, committed_end)
            self.assertGreater(reopened.startup_stats["truncated_tail_bytes"], 0)
            self.assertEqual(reopened.startup_stats["sealed_payload_bytes_read"], 0)
            reopened.close()

    def test_fast_open_skips_sealed_payload_and_full_audit_detects_historical_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = self.make_store(root)
            store.put_evidence([item(1)])
            store.seal_all_active_segments()
            sealed_path = Path(store.conn.execute(
                "SELECT path FROM segments WHERE state='SEALED'"
            ).fetchone()[0])
            store.close()

            fast = self.make_store(root)
            self.assertEqual(fast.startup_stats["sealed_segments"], 1)
            self.assertEqual(fast.startup_stats["sealed_payload_bytes_read"], 0)
            fast.close()

            data = bytearray(sealed_path.read_bytes())
            data[-1] ^= 0x01
            sealed_path.write_bytes(data)

            still_fast = self.make_store(root)
            self.assertEqual(still_fast.startup_stats["sealed_payload_bytes_read"], 0)
            audit = still_fast.full_forensic_audit()
            self.assertFalse(audit["pass"])
            self.assertIn("SEALED_SEGMENT_SHA256_MISMATCH", audit["problems"][0]["error"])
            still_fast.close()

    def test_same_payload_across_distinct_evidence_ids_is_stored_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = self.make_store(root)
            payload = {"same": [1, 2, 3]}
            a = item(1, payload)
            b = EvidenceItemR11("E2", "parent", "lineage-2", "teacher", payload)
            _, receipt = store.put_evidence([a, b])
            self.assertEqual(receipt.unique_payload_count, 1)
            self.assertEqual(receipt.created_payload_count, 1)
            self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM payloads").fetchone()[0], 1)
            store.close()

    def test_source_market_root_is_guarded_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "market"
            source.mkdir()
            with self.assertRaisesRegex(RuntimeError, "READ_ONLY_SOURCE_ROOT_OVERLAP"):
                EvidenceStoreR11(
                    metadata_root=source / "metadata",
                    payload_roots=[root / "hdd"],
                    read_only_source_roots=[source],
                )


if __name__ == "__main__":
    unittest.main()
