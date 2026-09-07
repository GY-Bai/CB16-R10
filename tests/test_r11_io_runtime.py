from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cb16_local_opt.checkpoint_store_r11 import tensor_mapping_semantic_sha256
from cb16_local_opt.evidence_store_r11 import (
    EvidenceItemR11,
    EvidenceStoreR11,
    FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX,
)
from cb16_local_opt.event_journal_r11 import EventItemR11, EventJournalR11
from cb16_local_opt.io_runtime_r11 import (
    FAIL_AFTER_METADATA_COMMIT_BEFORE_EVENT_SEAL,
    FAIL_DURING_BATCH_APPEND,
    IOThroughputConfigR11,
    IOThroughputRuntimeR11,
    IORuntimeBudgetExceededR11,
    IORuntimeBudgetR11,
    ImmutableEvidenceObjectR11,
    MarketCacheBudgetGuardR11,
    StorageWriterFailedR11,
    put_evidence_batched_r11,
)


def item(i: int, *, blob_bytes: int = 512) -> EvidenceItemR11:
    return EvidenceItemR11(
        evidence_id=f"E{i}",
        parent_snapshot_hash="parent",
        lineage_hash=f"lineage-{i}",
        teacher_protocol_hash="teacher",
        payload={"i": i, "blob": "x" * blob_bytes},
    )


def event(i: int, generation: int = 0) -> EventItemR11:
    return EventItemR11(
        event_id=f"EV{i}",
        event_type="TRACE",
        generation=generation,
        policy_weight_hash=f"policy-{generation}",
        snapshot_hash=f"snapshot-{generation}",
        lineage_hash=f"lineage-{i}",
        payload={"i": i, "decision": "FLAT"},
    )


def config(root: Path, **kwargs) -> IOThroughputConfigR11:
    return IOThroughputConfigR11(
        evidence_metadata_root=root / "ssd" / "evidence",
        evidence_payload_roots=(root / "hdd",),
        journal_root=root / "ssd" / "journal",
        checkpoint_root=root / "ssd" / "checkpoints",
        segment_target_bytes=1 << 20,
        evidence_codec="none",
        queue_max_items=8,
        writer_batch_max_objects=64,
        writer_batch_max_bytes=1 << 20,
        **kwargs,
    )


class TestR11IOThroughput(unittest.TestCase):
    def test_batched_path_preserves_evidence_identity_and_amortizes_fsync(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [item(i) for i in range(8)]

            baseline = EvidenceStoreR11(
                metadata_root=root / "base-meta",
                payload_roots=[root / "base-payload"],
                segment_target_bytes=1 << 20,
                codec="none",
            )
            base_refs, base_receipt = baseline.put_evidence(rows)
            base_set = baseline.seal_evidence_set("S", [x.evidence_id for x in rows])
            baseline.close()

            optimized = EvidenceStoreR11(
                metadata_root=root / "opt-meta",
                payload_roots=[root / "opt-payload"],
                segment_target_bytes=1 << 20,
                codec="none",
            )
            frozen = tuple(ImmutableEvidenceObjectR11.from_item(x) for x in rows)
            result = put_evidence_batched_r11(optimized, frozen)
            opt_set = optimized.seal_evidence_set("S", [x.evidence_id for x in rows])
            self.assertEqual([x.content_hash for x in base_refs], [x.content_hash for x in result.refs])
            self.assertEqual(base_receipt, result.receipt)
            self.assertEqual(base_set["evidence_set_hash"], opt_set["evidence_set_hash"])
            self.assertEqual(result.payload_fsync_count, 1)
            self.assertEqual(result.metadata_transaction_count, 1)
            for original, immutable in zip(rows, frozen):
                self.assertEqual(original.content_hash, immutable.content_hash)
                self.assertEqual(original.identity_hash, immutable.identity_hash)
                self.assertEqual(optimized.get_payload(original.content_hash), original.payload)
            optimized.close()

    def test_batch_append_mid_crash_recovers_prefix_and_replay_finishes_exactly_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = tuple(ImmutableEvidenceObjectR11.from_item(item(i)) for i in range(3))
            store = EvidenceStoreR11(
                metadata_root=root / "meta", payload_roots=[root / "payload"], codec="none"
            )

            def fail(phase: str):
                if phase == FAIL_DURING_BATCH_APPEND:
                    raise RuntimeError("mid-append")

            with self.assertRaisesRegex(RuntimeError, "mid-append"):
                put_evidence_batched_r11(store, rows, fail_hook=fail)
            store.close()

            reopened = EvidenceStoreR11(
                metadata_root=root / "meta", payload_roots=[root / "payload"], codec="none"
            )
            completed = put_evidence_batched_r11(reopened, rows)
            self.assertLessEqual(completed.receipt.created_payload_count, len(rows))
            replay = put_evidence_batched_r11(reopened, rows)
            self.assertEqual(replay.receipt.created_payload_count, 0)
            self.assertEqual(replay.receipt.created_evidence_count, 0)
            self.assertTrue(reopened.full_forensic_audit()["pass"])
            reopened.close()

    def test_fsync_before_metadata_crash_recovers_durable_batch_without_duplicate_payload(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = tuple(ImmutableEvidenceObjectR11.from_item(item(i)) for i in range(4))
            store = EvidenceStoreR11(
                metadata_root=root / "meta", payload_roots=[root / "payload"], codec="none"
            )

            def fail(phase: str):
                if phase == FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX:
                    raise RuntimeError("power-loss")

            with self.assertRaisesRegex(RuntimeError, "power-loss"):
                put_evidence_batched_r11(store, rows, fail_hook=fail)
            store.close()

            reopened = EvidenceStoreR11(
                metadata_root=root / "meta", payload_roots=[root / "payload"], codec="none"
            )
            self.assertEqual(reopened.startup_stats["recovered_payloads"], len(rows))
            completed = put_evidence_batched_r11(reopened, rows)
            self.assertEqual(completed.receipt.created_payload_count, 0)
            self.assertEqual(completed.receipt.created_evidence_count, len(rows))
            replay = put_evidence_batched_r11(reopened, rows)
            self.assertEqual(replay.receipt.created_payload_count, 0)
            self.assertEqual(replay.receipt.created_evidence_count, 0)
            reopened.close()

    def test_metadata_commit_before_event_seal_is_replayable_without_payload_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = config(root)
            runtime = IOThroughputRuntimeR11(cfg)

            def fail(phase: str):
                if phase == FAIL_AFTER_METADATA_COMMIT_BEFORE_EVENT_SEAL:
                    raise RuntimeError("crash-before-event-seal")

            ticket = runtime.submit_trace_commit(
                [item(1)], [event(1)], trace_batch_id="tb0", fail_hook=fail
            )
            with self.assertRaises(StorageWriterFailedR11):
                runtime.wait(ticket, timeout=5)
            runtime.close(drain=False, raise_on_failure=False)

            evidence = EvidenceStoreR11(
                metadata_root=cfg.evidence_metadata_root,
                payload_roots=cfg.evidence_payload_roots,
                codec="none",
            )
            self.assertEqual(evidence.conn.execute("SELECT COUNT(*) FROM payloads").fetchone()[0], 1)
            self.assertEqual(evidence.conn.execute("SELECT COUNT(*) FROM evidence_catalog").fetchone()[0], 1)
            evidence.close()
            journal = EventJournalR11(cfg.journal_root)
            self.assertEqual(journal.audit()["events"], 0)
            self.assertEqual(journal.audit()["trace_batches"], 0)
            journal.close()

            runtime2 = IOThroughputRuntimeR11(cfg)
            replay_ticket = runtime2.submit_trace_commit(
                [item(1)], [event(1)], trace_batch_id="tb0"
            )
            result = runtime2.wait(replay_ticket, timeout=5)
            self.assertEqual(result.evidence.receipt.created_payload_count, 0)
            self.assertEqual(result.evidence.receipt.created_evidence_count, 0)
            self.assertEqual(result.trace_batch.created_event_count, 1)
            runtime2.close()

    def test_queue_full_applies_backpressure_without_drop_or_sequence_gap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = config(root, queue_max_items=1)
            runtime = IOThroughputRuntimeR11(cfg)
            entered = threading.Event()
            release = threading.Event()

            def block_after_fsync(phase: str):
                if phase == FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX:
                    entered.set()
                    if not release.wait(3):
                        raise RuntimeError("test-release-timeout")

            first = runtime.submit_evidence([item(1)], fail_hook=block_after_fsync)
            self.assertTrue(entered.wait(3))
            with self.assertRaisesRegex(TimeoutError, "BACKPRESSURE"):
                runtime.submit_evidence([item(2)], timeout=0.05)
            release.set()
            runtime.wait(first, timeout=5)
            second = runtime.submit_evidence([item(2)], timeout=1)
            runtime.wait(second, timeout=5)
            self.assertEqual(runtime.durable_barrier(second, timeout=5), second.sequence)
            stats = runtime.stats()
            self.assertEqual(stats.submitted_requests, 2)
            self.assertEqual(stats.durable_requests, 2)
            self.assertLessEqual(stats.queue_peak_items, 1)
            self.assertLessEqual(stats.queue_peak_bytes, cfg.budgets.evidence_queue_bytes)
            runtime.close()

    def test_writer_crash_fails_closed_for_wait_submit_and_barrier(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = IOThroughputRuntimeR11(config(Path(td)))

            def fail(phase: str):
                if phase == FAIL_AFTER_PAYLOAD_FSYNC_BEFORE_INDEX:
                    raise RuntimeError("writer-boom")

            ticket = runtime.submit_evidence([item(1)], fail_hook=fail)
            with self.assertRaises(StorageWriterFailedR11):
                runtime.wait(ticket, timeout=5)
            with self.assertRaises(StorageWriterFailedR11):
                runtime.submit_evidence([item(2)])
            with self.assertRaises(StorageWriterFailedR11):
                runtime.durable_barrier(ticket, timeout=1)
            runtime.close(drain=False, raise_on_failure=False)

    def test_sealed_fast_startup_reads_zero_payload_and_full_audit_detects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = config(root)
            runtime = IOThroughputRuntimeR11(cfg)
            runtime.wait(runtime.submit_evidence([item(1), item(2)]), timeout=5)
            runtime.close()

            store = EvidenceStoreR11(
                metadata_root=cfg.evidence_metadata_root,
                payload_roots=cfg.evidence_payload_roots,
                codec="none",
            )
            store.seal_all_active_segments()
            sealed_path = Path(
                store.conn.execute("SELECT path FROM segments WHERE state='SEALED'").fetchone()[0]
            )
            store.close()

            fast = EvidenceStoreR11(
                metadata_root=cfg.evidence_metadata_root,
                payload_roots=cfg.evidence_payload_roots,
                codec="none",
            )
            self.assertGreaterEqual(fast.startup_stats["sealed_segments"], 1)
            self.assertEqual(fast.startup_stats["sealed_payload_bytes_read"], 0)
            fast.close()

            data = bytearray(sealed_path.read_bytes())
            data[-1] ^= 0x01
            sealed_path.write_bytes(data)
            audit_store = EvidenceStoreR11(
                metadata_root=cfg.evidence_metadata_root,
                payload_roots=cfg.evidence_payload_roots,
                codec="none",
            )
            audit = audit_store.full_forensic_audit()
            self.assertFalse(audit["pass"])
            self.assertTrue(any("SEALED_SEGMENT_SHA256_MISMATCH" in x["error"] for x in audit["problems"]))
            audit_store.close()

    def test_ram_budget_market_cache_queue_and_checkpoint_staging_are_bounded(self):
        with self.assertRaises(IORuntimeBudgetExceededR11):
            IORuntimeBudgetR11(
                total_ram_bytes=100,
                market_cache_bytes=40,
                evidence_queue_bytes=40,
                checkpoint_staging_bytes=40,
                process_reserve_bytes=40,
            )

        guard = MarketCacheBudgetGuardR11(2000)
        fake = SimpleNamespace(
            symbol="BTCUSDT",
            open_time_ms=np.zeros(4, dtype=np.int64),
            ohlcv=np.zeros((4, 5), dtype=np.float32),
            funding_rate=np.zeros(4, dtype=np.float32),
            index_by_time_ms={0: 0, 1: 1, 2: 2, 3: 3},
        )
        charge = guard.admit(fake)
        self.assertEqual(guard.admit(fake), charge)
        self.assertEqual(guard.used_bytes, charge)
        too_large = SimpleNamespace(
            symbol="ETHUSDT",
            open_time_ms=np.zeros(100, dtype=np.int64),
            ohlcv=np.zeros((100, 5), dtype=np.float64),
            funding_rate=np.zeros(100, dtype=np.float64),
            index_by_time_ms={i: i for i in range(100)},
        )
        with self.assertRaises(IORuntimeBudgetExceededR11):
            guard.admit(too_large)

        with tempfile.TemporaryDirectory() as td:
            budgets = IORuntimeBudgetR11(
                total_ram_bytes=8 << 20,
                market_cache_bytes=1 << 20,
                evidence_queue_bytes=2048,
                checkpoint_staging_bytes=1024,
                process_reserve_bytes=1 << 20,
            )
            cfg = config(
                Path(td),
                budgets=budgets,
                writer_batch_max_bytes=2048,
            )
            runtime = IOThroughputRuntimeR11(cfg)
            with self.assertRaises(IORuntimeBudgetExceededR11):
                runtime.submit_evidence([item(9, blob_bytes=4096)])
            with self.assertRaises(IORuntimeBudgetExceededR11):
                runtime.put_checkpoint_durable({"w": torch.ones(1024, dtype=torch.float32)})
            runtime.close()

    def test_checkpoint_writer_preserves_semantic_hash_and_deduplicates_transport(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = IOThroughputRuntimeR11(config(Path(td)))
            state = {
                "layer.bias": torch.tensor([1.0], dtype=torch.float32),
                "layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            }
            expected = tensor_mapping_semantic_sha256(state)
            first = runtime.put_checkpoint_durable(state, timeout=5)
            second = runtime.put_checkpoint_durable(state, timeout=5)
            self.assertEqual(first.semantic_sha256, expected)
            self.assertEqual(second.semantic_sha256, expected)
            self.assertTrue(first.created)
            self.assertFalse(second.created)
            stats = runtime.stats()
            self.assertEqual(stats.checkpoint_created, 1)
            self.assertEqual(stats.checkpoint_reused, 1)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
