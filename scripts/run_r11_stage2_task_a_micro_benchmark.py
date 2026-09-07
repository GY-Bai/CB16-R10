#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

from cb16_local_opt.cpu_runtime_r11 import CpuExecutionBudgetR11
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_common import sha256_obj
from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r102_physics import LONG, FrozenPhysicsRuntimeR102
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.teacher_scheduler_r11 import TeacherWorkerPoolR11
from cb16_local_opt.trace_runtime_r11 import H72TraceWorkItemR11, TraceRuntimeR11


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _teacher_hash(train, val) -> str:
    return sha256_obj({
        "train": [asdict(x) for x in train],
        "validation": [asdict(x) for x in val],
    })


def _trace_hash(rows) -> str:
    return sha256_obj([
        {
            "ordinal": int(x.ordinal),
            "causal_trace_id": x.causal_trace_id,
            "account_id": x.account_id,
            "parent_id": x.parent_id,
            "symbol": x.symbol,
            "branch": x.branch,
            "snapshot_sha256_sequence": list(x.snapshot_sha256_sequence),
        }
        for x in rows
    ])


def _cpu_model() -> str:
    p = Path("/proc/cpuinfo")
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "UNKNOWN"


def _teacher_run(*, samples, parents, workers: int, block_targets: int, worker_pool=None):
    started = time.perf_counter()
    train, val, stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=int(workers),
        block_targets=int(block_targets),
        worker_pool=worker_pool,
    )
    seconds = time.perf_counter() - started
    return {
        "workers": int(workers),
        "seconds": float(seconds),
        "evidence_hash": _teacher_hash(train, val),
        "train_count": len(train),
        "validation_count": len(val),
        "support_regimes": int(stats.core.support_regimes),
        "geometry_blocks": int(stats.core.geometry_blocks),
        "persistent_pool_supplied": bool(stats.persistent_pool_supplied),
        "max_in_flight": int(stats.max_in_flight),
        "nested_blas_threads": int(stats.nested_blas_threads_required),
        "topology_in_scientific_identity": bool(stats.topology_in_scientific_identity),
    }


def _choose_trace_items(parents, parent_states, max_groups: int):
    eligible = [p for p in parents.values() if p.eligible_for_economic_evidence]
    eligible.sort(key=lambda p: (0 if p.split == "VALIDATION" else 1, p.decision_time_ms, p.symbol, p.scenario, p.parent_id))
    chosen = []
    seen_dep = set()
    for p in eligible:
        if p.scenario != "CLEAN_FLAT_FULL" or p.dependence_group_id in seen_dep:
            continue
        state = dict(parent_states[p.parent_id])
        state.setdefault("account_id", state["risk_authority"]["account_id"])
        chosen.append((p, state))
        seen_dep.add(p.dependence_group_id)
        if len(chosen) >= int(max_groups):
            break
    if len(chosen) < int(max_groups):
        raise RuntimeError(f"R11_TASK_A_TRACE_GROUP_SUPPORT_TOO_SMALL:{len(chosen)}<{max_groups}")
    account_ids = [str(state["account_id"]) for _, state in chosen]
    if len(account_ids) != len(set(account_ids)):
        raise RuntimeError("R11_TASK_A_TRACE_ACCOUNT_IDS_NOT_INDEPENDENT")
    return [
        H72TraceWorkItemR11(
            ordinal=i,
            causal_trace_id=f"R11TASKA:TRACE:{i}",
            account_id=str(state["account_id"]),
            parent_id=p.parent_id,
            symbol=p.symbol,
            decision_time_ms=int(p.decision_time_ms),
            parent_state=state,
            direction_v55=LONG,
            requested_risk=0.25,
        )
        for i, (p, state) in enumerate(chosen)
    ]


def _trace_run(*, physics, market_cache, items, workers: int, runtime=None):
    own = runtime is None
    rt = runtime or TraceRuntimeR11(physics=physics, market_cache=market_cache, max_workers=int(workers))
    try:
        started = time.perf_counter()
        rows = rt.run(items)
        seconds = time.perf_counter() - started
    finally:
        if own:
            rt.close()
    return {
        "workers": int(workers),
        "seconds": float(seconds),
        "trace_hash": _trace_hash(rows),
        "trace_count": len(rows),
        "matured": sum(x.branch.get("status") == "MATURED" for x in rows),
        "runtime_reused": not own,
    }


def _summary(rows, hash_key: str):
    out = {}
    for workers in sorted({int(x["workers"]) for x in rows}):
        values = [float(x["seconds"]) for x in rows if int(x["workers"]) == workers]
        hashes = {x[hash_key] for x in rows if int(x["workers"]) == workers}
        out[str(workers)] = {
            "runs": values,
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "identity_hashes": sorted(hashes),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R11 Stage-2 Task A Shanxi micro-benchmark")
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-targets", type=int, default=64)
    ap.add_argument("--trace-groups", type=int, default=24)
    args = ap.parse_args()

    legacy_root = Path(args.legacy_r104_root).resolve()
    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_TASK_A_EXPECTED_R104_STRIDE_256")
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_TASK_A_REFUSES_FINAL_HOLDOUT_CACHE")

    parents, samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = load_parent_physics_states(manifest["parent_states_file"])
    budget = CpuExecutionBudgetR11()

    teacher_schedule = (1, 4, 8, 12, 8, 4)
    teacher_rows = []
    teacher_reference_hash = None
    for workers in teacher_schedule:
        row = _teacher_run(
            samples=samples,
            parents=parents,
            workers=workers,
            block_targets=int(args.block_targets),
        )
        if teacher_reference_hash is None:
            teacher_reference_hash = row["evidence_hash"]
        if row["evidence_hash"] != teacher_reference_hash:
            raise RuntimeError(
                f"R11_TASK_A_TEACHER_IDENTITY_FAIL:{workers}:{row['evidence_hash']}!={teacher_reference_hash}"
            )
        teacher_rows.append(row)
        print(json.dumps({"phase": "teacher_topology", **row}, sort_keys=True), flush=True)

    teacher_pool_rows = []
    with TeacherWorkerPoolR11(max_workers=8) as pool:
        pool_identity = pool.executor_identity
        for repetition in range(2):
            row = _teacher_run(
                samples=samples,
                parents=parents,
                workers=8,
                block_targets=int(args.block_targets),
                worker_pool=pool,
            )
            row["repetition"] = repetition
            row["executor_identity"] = pool.executor_identity
            if row["evidence_hash"] != teacher_reference_hash:
                raise RuntimeError("R11_TASK_A_PERSISTENT_TEACHER_POOL_IDENTITY_FAIL")
            if pool.executor_identity != pool_identity:
                raise RuntimeError("R11_TASK_A_PERSISTENT_TEACHER_POOL_RECREATED")
            teacher_pool_rows.append(row)
            print(json.dumps({"phase": "teacher_pool_reuse", **row}, sort_keys=True), flush=True)

    physics = FrozenPhysicsRuntimeR102.load(ROOT)
    market_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    try:
        trace_items = _choose_trace_items(parents, parent_states, int(args.trace_groups))
        for symbol in sorted({x.symbol for x in trace_items}):
            market_cache.get(symbol)
        market_cache.assert_read_only()

        trace_schedule = (1, 4, 8, 8, 4, 1)
        trace_rows = []
        trace_reference_hash = None
        for workers in trace_schedule:
            row = _trace_run(
                physics=physics,
                market_cache=market_cache,
                items=trace_items,
                workers=workers,
            )
            if trace_reference_hash is None:
                trace_reference_hash = row["trace_hash"]
            if row["trace_hash"] != trace_reference_hash:
                raise RuntimeError(
                    f"R11_TASK_A_TRACE_IDENTITY_FAIL:{workers}:{row['trace_hash']}!={trace_reference_hash}"
                )
            trace_rows.append(row)
            print(json.dumps({"phase": "trace_topology", **row}, sort_keys=True), flush=True)

        trace_pool_rows = []
        with TraceRuntimeR11(physics=physics, market_cache=market_cache, max_workers=8) as trace_pool:
            for repetition in range(2):
                row = _trace_run(
                    physics=physics,
                    market_cache=market_cache,
                    items=trace_items,
                    workers=8,
                    runtime=trace_pool,
                )
                row["repetition"] = repetition
                if row["trace_hash"] != trace_reference_hash:
                    raise RuntimeError("R11_TASK_A_PERSISTENT_TRACE_POOL_IDENTITY_FAIL")
                trace_pool_rows.append(row)
                print(json.dumps({"phase": "trace_pool_reuse", **row}, sort_keys=True), flush=True)

        cache_stats = asdict(market_cache.stats())
    finally:
        market_cache.close()

    teacher_summary = _summary(teacher_rows, "evidence_hash")
    trace_summary = _summary(trace_rows, "trace_hash")
    teacher_best = min(teacher_summary, key=lambda w: teacher_summary[w]["median_seconds"])
    trace_best = min(trace_summary, key=lambda w: trace_summary[w]["median_seconds"])

    result = {
        "schema": "CB16_R11_STAGE2_TASK_A_SHANXI_MICRO_BENCHMARK_V1",
        "status": "PASS",
        "scientific_semantics_changed": False,
        "final_holdout_payload_read": False,
        "raw_1m_archive_read": False,
        "new_scientific_verdict": False,
        "cpu_model": _cpu_model(),
        "cpu_budget": budget.runtime_receipt(),
        "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "teacher": {
            "schedule": list(teacher_schedule),
            "measurements": teacher_rows,
            "summary": teacher_summary,
            "best_workers_by_median": int(teacher_best),
            "evidence_hash": teacher_reference_hash,
            "identity_exact_across_all_runs": True,
            "persistent_pool_measurements": teacher_pool_rows,
        },
        "trace": {
            "groups": int(args.trace_groups),
            "schedule": list(trace_schedule),
            "measurements": trace_rows,
            "summary": trace_summary,
            "best_workers_by_median": int(trace_best),
            "trace_hash": trace_reference_hash,
            "identity_exact_across_all_runs": True,
            "persistent_pool_measurements": trace_pool_rows,
            "market_cache_stats": cache_stats,
        },
    }
    _atomic_json(Path(args.out).resolve(), result)
    print(json.dumps({
        "status": "PASS",
        "teacher_best_workers": int(teacher_best),
        "trace_best_workers": int(trace_best),
        "teacher_identity": True,
        "trace_identity": True,
        "cpu_budget": result["cpu_budget"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
