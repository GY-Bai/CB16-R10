import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cb16_local_opt.r2_evidence_storage import (
    FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX,
    R2EvidenceItem,
    R2EvidenceStore,
)


def _item(i: int, *, payload_extra=None):
    payload = {
        "schema": "CB16_R2_TEST_EVIDENCE",
        "parent_id": f"P{i}",
        "operator48": [float(i), 1.0],
        "direction_target_probs": [0.2, 0.3, 0.5],
    }
    if payload_extra:
        payload.update(payload_extra)
    return R2EvidenceItem(
        evidence_id=f"E{i}",
        parent_snapshot_hash=f"S{i}",
        lineage_hash=f"L{i}",
        teacher_protocol_hash="T",
        payload=payload,
    )


class R2EvidenceStorageTests(unittest.TestCase):
    def _store(self, td: str, *, recover=True):
        root = Path(td)
        return R2EvidenceStore(
            metadata_root=root / "ssd",
            payload_roots=[root / "hdd"],
            codec="zlib",
            recover_on_open=recover,
            segment_target_bytes=1024 * 1024,
        )

    def test_package_root_is_storage_lightweight(self):
        repo = Path(__file__).resolve().parents[1]
        code = (
            "import sys; import cb16_local_opt; "
            "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if k.startswith('torch'))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo)
        subprocess.run([sys.executable, "-c", code], env=env, check=True)

    def test_materialize_once_reuse_across_100_generations(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            rows = [_item(i) for i in range(8)]
            evidence_set, receipt = store.materialize_evidence_set(
                evidence_set_id="TRAIN_V1", items=rows
            )
            self.assertEqual(receipt.created_payload_count, 8)
            for g in range(100):
                snap = store.seal_generation_snapshot(
                    snapshot_id=f"G{g}",
                    generation=g,
                    parent_policy_hash=f"POLICY{g}",
                    evidence_set=evidence_set,
                )
                self.assertEqual(snap.evidence_set_hash, evidence_set.content_hash)
            stats = store.stats()
            self.assertEqual(stats["payload_objects"], 8)
            self.assertEqual(stats["evidence_catalog_objects"], 8)
            self.assertEqual(stats["evidence_sets"], 1)
            self.assertEqual(stats["generation_snapshots"], 100)
            evidence_set_2, receipt_2 = store.materialize_evidence_set(
                evidence_set_id="TRAIN_V1", items=rows
            )
            self.assertEqual(evidence_set_2.content_hash, evidence_set.content_hash)
            self.assertEqual(receipt_2.created_payload_count, 0)
            self.assertTrue(store.audit()["pass"])
            store.close()

    def test_same_evidence_id_different_content_is_conflict(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.materialize_evidence_set(evidence_set_id="A", items=[_item(1)])
            with self.assertRaisesRegex(RuntimeError, "R2_EVIDENCE_ID_CONTENT_CONFLICT"):
                store.materialize_evidence_set(
                    evidence_set_id="B", items=[_item(1, payload_extra={"changed": True})]
                )
            store.close()

    def test_fsync_before_index_crash_recovers_orphan_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)

            def fail(stage):
                if stage == FAIL_AFTER_PACK_FSYNC_BEFORE_INDEX:
                    raise RuntimeError("SIMULATED_CRASH")

            with self.assertRaisesRegex(RuntimeError, "SIMULATED_CRASH"):
                store.materialize_evidence_set(
                    evidence_set_id="A", items=[_item(i) for i in range(4)], fail_hook=fail
                )
            store.close()

            recovered = self._store(td, recover=True)
            self.assertEqual(recovered.stats()["payload_objects"], 4)
            _set_ref, receipt = recovered.materialize_evidence_set(
                evidence_set_id="A", items=[_item(i) for i in range(4)]
            )
            self.assertEqual(receipt.created_payload_count, 0)
            self.assertTrue(recovered.audit()["pass"])
            recovered.close()

    def test_incomplete_pack_tail_is_truncated_to_last_valid_record(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            store.materialize_evidence_set(
                evidence_set_id="A", items=[_item(i) for i in range(3)]
            )
            segment = next((Path(td) / "hdd" / "cb16_r2_lane_00").glob("segment_*.pack"))
            valid_size = segment.stat().st_size
            with segment.open("ab") as f:
                f.write(b"\x04oops")
                f.flush()
                os.fsync(f.fileno())
            store.close()

            recovered = self._store(td, recover=True)
            self.assertEqual(segment.stat().st_size, valid_size)
            self.assertTrue(recovered.audit()["pass"])
            recovered.close()

    def test_snapshot_id_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = self._store(td)
            evidence_set, _ = store.materialize_evidence_set(
                evidence_set_id="A", items=[_item(1), _item(2)]
            )
            store.seal_generation_snapshot(
                snapshot_id="G0", generation=0, parent_policy_hash="P0", evidence_set=evidence_set
            )
            with self.assertRaisesRegex(RuntimeError, "R2_SNAPSHOT_ID_CONTENT_CONFLICT"):
                store.seal_generation_snapshot(
                    snapshot_id="G0", generation=0, parent_policy_hash="P1", evidence_set=evidence_set
                )
            store.close()


if __name__ == "__main__":
    unittest.main()
