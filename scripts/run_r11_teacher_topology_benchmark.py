#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Pre-import protection for BLAS runtimes not loaded yet.  The production scheduler also
# enforces one BLAS thread dynamically with threadpoolctl during multi-worker compilation.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_common import sha256_obj
from cb16_local_opt.r102_evidence_cache import load_teacher_samples
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.teacher_runtime_r11 import R11_TEACHER_RUNTIME, compile_teacher_evidence_r11


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _evidence_hash(train, val) -> str:
    return sha256_obj({
        "train": [asdict(x) for x in train],
        "validation": [asdict(x) for x in val],
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-targets", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.legacy_r104_root).resolve()
    manifest_path = root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_TOPOLOGY_EXPECTED_R104_STRIDE_256")
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_TOPOLOGY_REFUSES_FINAL_HOLDOUT_CACHE")
    parents, samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])

    # Symmetric order reduces first/last cache-temperature bias while keeping runtime small.
    # This script intentionally benchmarks runtime topology only; the evidence hash must be
    # identical across every run and worker count.
    schedule = (1, 4, 8, 12, 12, 8, 4, 1)
    rows = []
    reference_hash = None
    for workers in schedule:
        started = time.perf_counter()
        train, val, stats = compile_teacher_evidence_r11(
            samples=samples,
            parents=parents,
            train_config=TRAIN_TEACHER_CONFIG_R102,
            val_config=VAL_TEACHER_CONFIG_R102,
            workers=workers,
            block_targets=int(args.block_targets),
        )
        seconds = time.perf_counter() - started
        content_hash = _evidence_hash(train, val)
        if reference_hash is None:
            reference_hash = content_hash
        if content_hash != reference_hash:
            raise RuntimeError(
                f"R11_THREAD_SCHEDULER_CHANGED_TEACHER_OUTPUT:{workers}:{content_hash}!={reference_hash}"
            )
        row = {
            "workers": workers,
            "seconds": seconds,
            "evidence_hash": content_hash,
            "train_count": len(train),
            "validation_count": len(val),
            "support_regimes": stats.core.support_regimes,
            "geometry_blocks": stats.core.geometry_blocks,
            "scheduler": stats.scheduler,
            "memory_model": stats.memory_model,
            "nested_blas_threads": stats.nested_blas_threads_required,
            "nested_blas_limit_enforced": stats.nested_blas_limit_enforced,
            "topology_in_scientific_identity": stats.topology_in_scientific_identity,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    by_workers = {}
    for workers in sorted(set(schedule)):
        values = [r["seconds"] for r in rows if r["workers"] == workers]
        by_workers[str(workers)] = {
            "runs": values,
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    best = min(by_workers, key=lambda w: by_workers[w]["median_seconds"])
    cpu_affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    result = {
        "schema": "CB16_R11_TEACHER_TOPOLOGY_BENCHMARK_V2",
        "status": "PASS",
        "runtime": R11_TEACHER_RUNTIME,
        "scientific_semantics_changed": False,
        "evidence_hash_identical_across_all_runs": True,
        "evidence_hash": reference_hash,
        "block_targets": int(args.block_targets),
        "nested_blas_threads": 1,
        "nested_blas_limit_enforced": True,
        "logical_cpus_available": len(cpu_affinity),
        "cpu_affinity": cpu_affinity,
        "schedule": list(schedule),
        "measurements": rows,
        "summary": by_workers,
        "best_workers_by_median": int(best),
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "forbidden_work": {
            "raw_1m_archive_read": False,
            "final_holdout_payload_read": False,
            "h72_physics_execution": False,
            "central_brain_training": False,
        },
    }
    _atomic_json(Path(args.out).resolve(), result)
    print(json.dumps({
        "status": "PASS",
        "best_workers": int(best),
        "summary": by_workers,
        "peak_process_rss_kib": result["peak_process_rss_kib"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
