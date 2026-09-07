from __future__ import annotations

"""Task C synthetic real-machine micro-benchmark.

No market data is read.  The benchmark compares the pre-existing per-payload fsync path
against the R11 bounded writer/batched durability path using identical immutable evidence
objects and reports scientific/storage identity before any performance conclusion.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from cb16_local_opt.evidence_store_r11 import EvidenceItemR11, EvidenceStoreR11
from cb16_local_opt.io_runtime_r11 import IOThroughputConfigR11, IOThroughputRuntimeR11


def make_items(count: int, payload_bytes: int) -> list[EvidenceItemR11]:
    pad = "x" * max(1, int(payload_bytes) - 128)
    return [
        EvidenceItemR11(
            evidence_id=f"E{i:06d}",
            parent_snapshot_hash="task-c-parent",
            lineage_hash=f"task-c-lineage-{i:06d}",
            teacher_protocol_hash="task-c-teacher",
            payload={"ordinal": i, "payload": pad},
        )
        for i in range(int(count))
    ]


def clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", required=True, type=Path)
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--payload-bytes", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    metadata_root = args.metadata_root.resolve()
    payload_root = args.payload_root.resolve()
    baseline_meta = metadata_root / "baseline"
    optimized_meta = metadata_root / "optimized"
    baseline_payload = payload_root / "baseline"
    optimized_payload = payload_root / "optimized"
    for path in (baseline_meta, optimized_meta, baseline_payload, optimized_payload):
        clean(path)

    items = make_items(args.count, args.payload_bytes)
    evidence_ids = [x.evidence_id for x in items]
    expected_hashes = [x.content_hash for x in items]

    baseline = EvidenceStoreR11(
        metadata_root=baseline_meta,
        payload_roots=[baseline_payload],
        segment_target_bytes=256 * 1024 * 1024,
        codec="none",
        sqlite_synchronous="FULL",
    )
    t0 = time.perf_counter()
    baseline_refs, baseline_receipt = baseline.put_evidence(items)
    baseline_set = baseline.seal_evidence_set("TASK_C_EQUIVALENCE", evidence_ids)
    baseline_seconds = time.perf_counter() - t0
    baseline_hashes = [x.content_hash for x in baseline_refs]
    baseline_audit = baseline.full_forensic_audit()
    baseline.close()

    cfg = IOThroughputConfigR11(
        evidence_metadata_root=optimized_meta,
        evidence_payload_roots=(optimized_payload,),
        journal_root=optimized_meta / "journal",
        checkpoint_root=None,
        segment_target_bytes=256 * 1024 * 1024,
        evidence_codec="none",
        sqlite_synchronous="FULL",
        queue_max_items=8,
        writer_batch_max_objects=args.batch_size,
        writer_batch_max_bytes=max(1 << 20, args.batch_size * (args.payload_bytes + 1024)),
    )
    runtime = IOThroughputRuntimeR11(cfg)
    tickets = []
    t1 = time.perf_counter()
    for start in range(0, len(items), args.batch_size):
        tickets.append(runtime.submit_evidence(items[start : start + args.batch_size]))
    runtime.durable_barrier(tickets[-1], timeout=240)
    optimized_results = [runtime.wait(ticket, timeout=5) for ticket in tickets]
    optimized_set = runtime.seal_evidence_set("TASK_C_EQUIVALENCE", evidence_ids, timeout=30)
    optimized_seconds = time.perf_counter() - t1
    optimized_stats = runtime.stats()
    runtime.close()

    optimized_store = EvidenceStoreR11(
        metadata_root=optimized_meta,
        payload_roots=[optimized_payload],
        segment_target_bytes=256 * 1024 * 1024,
        codec="none",
        sqlite_synchronous="FULL",
    )
    optimized_hashes = [
        optimized_store.conn.execute(
            "SELECT content_hash FROM evidence_catalog WHERE evidence_id=?", (evidence_id,)
        ).fetchone()[0]
        for evidence_id in evidence_ids
    ]
    optimized_audit = optimized_store.full_forensic_audit()
    for index in (0, len(items) // 2, len(items) - 1):
        if optimized_store.get_payload(expected_hashes[index]) != items[index].payload:
            raise RuntimeError(f"TASK_C_PAYLOAD_IDENTITY_MISMATCH:{index}")
    optimized_store.close()

    replay = IOThroughputRuntimeR11(cfg)
    replay_tickets = []
    for start in range(0, len(items), args.batch_size):
        replay_tickets.append(replay.submit_evidence(items[start : start + args.batch_size]))
    replay.durable_barrier(replay_tickets[-1], timeout=240)
    replay_results = [replay.wait(ticket, timeout=5) for ticket in replay_tickets]
    replay.close()
    replay_new_payloads = sum(x.receipt.created_payload_count for x in replay_results)
    replay_new_evidence = sum(x.receipt.created_evidence_count for x in replay_results)

    identity_ok = all(
        (
            expected_hashes == baseline_hashes,
            expected_hashes == optimized_hashes,
            baseline_set["evidence_set_hash"] == optimized_set["evidence_set_hash"],
            baseline_receipt.created_payload_count == args.count,
            sum(x.receipt.created_payload_count for x in optimized_results) == args.count,
            replay_new_payloads == 0,
            replay_new_evidence == 0,
            bool(baseline_audit["pass"]),
            bool(optimized_audit["pass"]),
        )
    )
    speedup = baseline_seconds / optimized_seconds if optimized_seconds > 0 else float("inf")
    report = {
        "schema": "CB16_R11_TASK_C_IO_MICROBENCH_V1",
        "count": args.count,
        "payload_bytes_each": args.payload_bytes,
        "logical_payload_mib": args.count * args.payload_bytes / (1024 * 1024),
        "batch_size": args.batch_size,
        "baseline_seconds": baseline_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup_x": speedup,
        "baseline_expected_payload_fsyncs": baseline_receipt.created_payload_count,
        "optimized_payload_fsyncs": optimized_stats.payload_fsyncs,
        "optimized_metadata_transactions": optimized_stats.evidence_metadata_transactions,
        "optimized_queue_peak_items": optimized_stats.queue_peak_items,
        "optimized_queue_peak_bytes": optimized_stats.queue_peak_bytes,
        "replay_new_payloads": replay_new_payloads,
        "replay_new_evidence": replay_new_evidence,
        "baseline_evidence_set_hash": baseline_set["evidence_set_hash"],
        "optimized_evidence_set_hash": optimized_set["evidence_set_hash"],
        "baseline_audit_pass": bool(baseline_audit["pass"]),
        "optimized_audit_pass": bool(optimized_audit["pass"]),
        "correctness_identity": "PASS" if identity_ok else "FAIL",
        "raw_1m_read": False,
        "final_holdout_read": False,
        "scientific_verdict_produced": False,
    }
    print("TASK_C_MICROBENCH_JSON=" + json.dumps(report, sort_keys=True))
    if not identity_ok:
        raise RuntimeError("TASK_C_CORRECTNESS_IDENTITY_FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
