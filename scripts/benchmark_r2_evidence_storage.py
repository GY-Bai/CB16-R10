from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from cb16_local_opt.r2_evidence_storage import R2EvidenceItem, R2EvidenceStore


def make_items(n: int, payload_bytes: int) -> list[R2EvidenceItem]:
    pad = "x" * max(0, int(payload_bytes) - 256)
    out = []
    for i in range(int(n)):
        payload = {
            "schema": "CB16_R2_SYNTHETIC_BENCH_EVIDENCE_V1",
            "parent_id": f"P{i:08d}",
            "dependence_group_id": f"D{i % 64:03d}",
            "operator48": [float((i + j) % 17) / 17.0 for j in range(48)],
            "medium48": [float((i + j) % 19) / 19.0 for j in range(48)],
            "account6": [float((i + j) % 7) / 7.0 for j in range(6)],
            "direction_target_probs": [0.2, 0.3, 0.5],
            "requested_risk_target": 0.25,
            "padding": pad,
        }
        out.append(R2EvidenceItem(
            evidence_id=f"E{i:08d}",
            parent_snapshot_hash=f"{i:064x}"[-64:],
            lineage_hash=f"{i + 1:064x}"[-64:],
            teacher_protocol_hash="R2_BENCH_TEACHER",
            payload=payload,
        ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-root", required=True)
    ap.add_argument("--payload-root", action="append", required=True,
                    help="repeat once per PHYSICAL payload device; one HDD => one value")
    ap.add_argument("--objects", type=int, default=9714)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--payload-bytes", type=int, default=3072)
    ap.add_argument("--codec", choices=["zstd", "zlib", "none"], default="zstd")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    meta = Path(args.metadata_root).resolve()
    payloads = [Path(x).resolve() for x in args.payload_root]
    forbidden = Path("/data/cb16_hdd/cb16_runtime/R10_4").resolve()
    for path in [meta, *payloads]:
        if path == forbidden or forbidden in path.parents:
            raise SystemExit("REFUSE_CANONICAL_R10_4_ROOT")

    meta.mkdir(parents=True, exist_ok=True)
    for p in payloads:
        p.mkdir(parents=True, exist_ok=True)

    items = make_items(args.objects, args.payload_bytes)
    store = R2EvidenceStore(
        metadata_root=meta,
        payload_roots=payloads,
        codec=args.codec,
        sqlite_synchronous="FULL",
    )
    try:
        t0 = time.perf_counter()
        evidence_set, first = store.materialize_evidence_set(
            evidence_set_id="R2_BENCH_TRAIN_SET", items=items
        )
        first_wall = time.perf_counter() - t0

        t1 = time.perf_counter()
        _same, second = store.materialize_evidence_set(
            evidence_set_id="R2_BENCH_TRAIN_SET", items=items
        )
        reuse_wall = time.perf_counter() - t1

        snap_times = []
        for g in range(args.generations):
            s0 = time.perf_counter()
            store.seal_generation_snapshot(
                snapshot_id=f"R2_BENCH_G{g:04d}",
                generation=g,
                parent_policy_hash=f"POLICY_{g:04d}",
                evidence_set=evidence_set,
            )
            snap_times.append(time.perf_counter() - s0)

        audit = store.audit(verify_payloads=True)
        stats = store.stats()
        logical_memberships = args.objects * args.generations
        result = {
            "schema": "CB16_R2_STORAGE_BENCHMARK_R0",
            "status": "PASS" if audit["pass"] else "FAIL",
            "metadata_root": str(meta),
            "payload_roots": [str(x) for x in payloads],
            "payload_lanes": len(payloads),
            "objects": args.objects,
            "generations": args.generations,
            "logical_generation_evidence_memberships": logical_memberships,
            "physical_unique_payload_objects": stats["payload_objects"],
            "materialization_amplification_reduction": (
                logical_memberships / max(stats["payload_objects"], 1)
            ),
            "first_materialize_wall_seconds": first_wall,
            "first_created_payload_count": first.created_payload_count,
            "first_pack_write_seconds": first.pack_write_seconds,
            "first_index_commit_seconds": first.index_commit_seconds,
            "reuse_materialize_wall_seconds": reuse_wall,
            "reuse_created_payload_count": second.created_payload_count,
            "generation_snapshot_total_seconds": sum(snap_times),
            "generation_snapshot_mean_seconds": sum(snap_times) / max(len(snap_times), 1),
            "stats": stats,
            "audit": audit,
            "writes_to_canonical_run_root": False,
            "scientific_semantics_changed": False,
            "final_holdout_2025_09_accessed": False,
        }
        text = json.dumps(result, indent=2, sort_keys=True)
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n")
    finally:
        store.checkpoint("TRUNCATE")
        store.close()

    if not args.keep:
        for p in [meta, *payloads]:
            if "r2" in p.name.lower() or "qual" in p.name.lower() or "bench" in p.name.lower():
                shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    main()
