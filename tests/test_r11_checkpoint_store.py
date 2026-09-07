from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from cb16_local_opt.checkpoint_store_r11 import (
    CheckpointStoreR11,
    tensor_mapping_semantic_sha256,
)


def state(value: float):
    return {
        "layer.bias": torch.tensor([value], dtype=torch.float32),
        "layer.weight": torch.tensor([[value, value + 1]], dtype=torch.float32),
    }


class TestCheckpointStoreR11(unittest.TestCase):
    def test_tensor_semantic_identity_is_order_independent_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = CheckpointStoreR11(root)
            a = state(1.0)
            b = OrderedDict(reversed(list(a.items())))
            self.assertEqual(tensor_mapping_semantic_sha256(a), tensor_mapping_semantic_sha256(b))
            first = store.put_state_dict(a)
            second = store.put_state_dict(b)
            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(first.semantic_sha256, second.semantic_sha256)
            self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM checkpoint_objects").fetchone()[0], 1)
            store.close()

    def test_missing_checkpoint_object_fails_closed_on_fast_open(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = CheckpointStoreR11(root)
            ref = store.put_state_dict(state(1.0))
            store.close()
            Path(ref.object_path).unlink()
            with self.assertRaisesRegex(RuntimeError, "CHECKPOINT_FAST_VALIDATION_FAILED"):
                CheckpointStoreR11(root)

    def test_reject_reuses_parent_object_and_promote_advances_parent(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStoreR11(Path(td))
            parent = store.put_state_dict(state(1.0)).semantic_sha256
            c0 = store.put_state_dict(state(2.0)).semantic_sha256
            c1 = store.put_state_dict(state(3.0)).semantic_sha256
            g0 = store.seal_generation_checkpoint(
                generation=0, parent_champion=parent, challenger=c0, decision="REJECT",
                champion_after=parent, trace_batch_id="b0", trace_batch_hash="bh0",
            )
            self.assertEqual(g0.champion_after, parent)
            count_before = store.conn.execute("SELECT COUNT(*) FROM checkpoint_objects").fetchone()[0]
            again = store.put_state_dict(state(1.0))
            self.assertFalse(again.created)
            self.assertEqual(store.conn.execute("SELECT COUNT(*) FROM checkpoint_objects").fetchone()[0], count_before)
            with self.assertRaisesRegex(RuntimeError, "STALE_OR_REJECTED_PARENT_FORBIDDEN"):
                store.seal_generation_checkpoint(
                    generation=1, parent_champion=c0, challenger=c1, decision="PROMOTE",
                    champion_after=c1, trace_batch_id="bad", trace_batch_hash="bad",
                )
            g1 = store.seal_generation_checkpoint(
                generation=1, parent_champion=parent, challenger=c1, decision="PROMOTE",
                champion_after=c1, trace_batch_id="b1", trace_batch_hash="bh1",
            )
            self.assertEqual(g1.champion_after, c1)
            store.close()

    def test_runtime_metadata_does_not_change_generation_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as td:
            store = CheckpointStoreR11(Path(td))
            p = store.put_state_dict(state(1.0)).semantic_sha256
            c = store.put_state_dict(state(2.0)).semantic_sha256
            first = store.seal_generation_checkpoint(
                generation=0, parent_champion=p, challenger=c, decision="REJECT",
                champion_after=p, trace_batch_id="b", trace_batch_hash="bh",
                extra_metadata={"worker": 1, "ssd": "/fast"},
            )
            second = store.seal_generation_checkpoint(
                generation=0, parent_champion=p, challenger=c, decision="REJECT",
                champion_after=p, trace_batch_id="b", trace_batch_hash="bh",
                extra_metadata={"worker": 12, "ssd": "/other"},
            )
            self.assertEqual(first.snapshot_hash, second.snapshot_hash)
            store.close()

    def test_full_audit_detects_semantic_tensor_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = CheckpointStoreR11(root)
            ref = store.put_state_dict(state(1.0))
            torch.save({
                "schema": "CB16_R11_CHECKPOINT_TENSOR_OBJECT_V1",
                "semantic_sha256": ref.semantic_sha256,
                "state_dict": state(99.0),
            }, ref.object_path)
            audit = store.full_forensic_audit()
            self.assertFalse(audit["pass"])
            self.assertIn("SEMANTIC_HASH_MISMATCH", audit["problems"][0]["error"])
            store.close()


if __name__ == "__main__":
    unittest.main()
