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

for _name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11, MarketRuntimeCacheStatsR11
from cb16_local_opt.r102_common import sha256_obj
from cb16_local_opt.r102_evidence_cache import load_parent_physics_states, load_teacher_samples
from cb16_local_opt.r102_physics import LONG, FrozenPhysicsRuntimeR102
from cb16_local_opt.trace_process_runtime_r11 import ForkProcessTraceRuntimeR11
from cb16_local_opt.trace_runtime_r11 import H72TraceWorkItemR11, TraceRuntimeR11


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _cache_stats(stats: MarketRuntimeCacheStatsR11) -> dict:
    return {
        "compressed_loads_total": int(stats.compressed_loads_total),
        "index_builds_total": int(stats.index_builds_total),
        "loaded_symbols": list(stats.loaded_symbols),
        "compressed_loads_by_symbol": dict(stats.compressed_loads_by_symbol),
    }


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


def _choose_items(parents, parent_states, max_groups: int):
    eligible = [p for p in parents.values() if p.eligible_for_economic_evidence]
    eligible.sort(key=lambda p: (
        0 if p.split == "VALIDATION" else 1,
        int(p.decision_time_ms), p.symbol, p.scenario, p.parent_id,
    ))
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
        raise RuntimeError(f"R11_TASK_A_PROCESS_TRACE_SUPPORT_TOO_SMALL:{len(chosen)}<{max_groups}")
    items = []
    for ordinal, (p, state) in enumerate(chosen):
        items.append(H72TraceWorkItemR11(
            ordinal=ordinal,
            causal_trace_id=f"R11TASKA:PROC:{ordinal}",
            account_id=str(state["account_id"]),
            parent_id=p.parent_id,
            symbol=p.symbol,
            decision_time_ms=int(p.decision_time_ms),
            parent_state=state,
            direction_v55=LONG,
            requested_risk=0.25,
        ))
    return items


def _run(runtime, items):
    t0 = time.perf_counter()
    rows = runtime.run(items)
    dt = time.perf_counter() - t0
    return float(dt), _trace_hash(rows), len(rows), sum(x.branch.get("status") == "MATURED" for x in rows)


def _cpu_model() -> str:
    p = Path("/proc/cpuinfo")
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "UNKNOWN"


def main() -> int:
    ap = argparse.ArgumentParser(description="R11 Task A real H72 thread-vs-fork micro benchmark")
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--groups", type=int, default=96)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    legacy_root = Path(args.legacy_r104_root).resolve()
    manifest = json.loads((legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json").read_text(encoding="utf-8"))
    if int(manifest.get("stride_hours", -1)) != 256:
        raise RuntimeError("R11_TASK_A_PROCESS_EXPECTED_R104_STRIDE_256")
    if bool(manifest.get("final_holdout_2025_09_accessed", False)):
        raise RuntimeError("R11_TASK_A_PROCESS_REFUSES_FINAL_HOLDOUT_CACHE")

    parents, _samples = load_teacher_samples(manifest["parents_file"], manifest["branches_file"])
    parent_states = load_parent_physics_states(manifest["parent_states_file"])
    items = _choose_items(parents, parent_states, int(args.groups))
    symbols = sorted({x.symbol for x in items})
    physics = FrozenPhysicsRuntimeR102.load(ROOT)

    reference_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    reference_cache.preload(symbols)
    reference_cache.assert_read_only()
    with TraceRuntimeR11(physics=physics, market_cache=reference_cache, max_workers=1) as serial_rt:
        serial_seconds, reference_hash, count, matured = _run(serial_rt, items)
    if count != len(items) or matured != len(items):
        raise RuntimeError("R11_TASK_A_PROCESS_REFERENCE_NOT_FULLY_MATURED")
    reference_stats = _cache_stats(reference_cache.stats())
    reference_cache.close()
    print(json.dumps({
        "phase": "thread_serial_reference", "seconds": serial_seconds,
        "trace_hash": reference_hash, "groups": len(items), "symbols": symbols,
    }, sort_keys=True), flush=True)

    thread8_cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
    thread8_cache.preload(symbols)
    with TraceRuntimeR11(physics=physics, market_cache=thread8_cache, max_workers=8) as thread8_rt:
        thread8_seconds, thread8_hash, _, _ = _run(thread8_rt, items)
    if thread8_hash != reference_hash:
        raise RuntimeError("R11_TASK_A_PROCESS_THREAD8_IDENTITY_FAIL")
    thread8_cache.close()
    print(json.dumps({
        "phase": "thread8", "seconds": thread8_seconds, "trace_hash": thread8_hash,
    }, sort_keys=True), flush=True)

    process_rows = []
    for workers in (1, 2, 4, 8):
        cache = MarketRuntimeCacheR11(legacy_root / "evidence_cache")
        cache.preload(symbols)
        cache.assert_read_only()
        with ForkProcessTraceRuntimeR11(physics=physics, market_cache=cache, max_workers=workers) as rt:
            executor_before = rt.executor_identity
            first_seconds, first_hash, first_count, first_matured = _run(rt, items)
            executor_after_first = rt.executor_identity
            second_seconds, second_hash, second_count, second_matured = _run(rt, list(reversed(items)))
            executor_after_second = rt.executor_identity
            stats = asdict(rt.stats())
        if first_hash != reference_hash or second_hash != reference_hash:
            raise RuntimeError(f"R11_TASK_A_PROCESS_IDENTITY_FAIL:W{workers}")
        if first_count != len(items) or second_count != len(items):
            raise RuntimeError(f"R11_TASK_A_PROCESS_COUNT_FAIL:W{workers}")
        if first_matured != len(items) or second_matured != len(items):
            raise RuntimeError(f"R11_TASK_A_PROCESS_MATURITY_FAIL:W{workers}")
        if executor_before is not None or executor_after_first != executor_after_second:
            raise RuntimeError(f"R11_TASK_A_PROCESS_POOL_REUSE_FAIL:W{workers}")
        cache_stats = _cache_stats(cache.stats())
        if cache_stats["compressed_loads_total"] != len(symbols) or cache_stats["index_builds_total"] != len(symbols):
            raise RuntimeError(f"R11_TASK_A_PROCESS_CACHE_RELOAD_FAIL:W{workers}:{cache_stats}")
        cache.assert_read_only()
        cache.close()
        row = {
            "workers": workers,
            "first_seconds_including_pool_start": first_seconds,
            "reuse_seconds": second_seconds,
            "trace_hash": first_hash,
            "pool_reused": True,
            "runtime_stats": stats,
            "market_cache_stats": cache_stats,
        }
        process_rows.append(row)
        print(json.dumps({"phase": "fork_process", **row}, sort_keys=True), flush=True)

    best_reuse = min(process_rows, key=lambda x: x["reuse_seconds"])
    result = {
        "schema": "CB16_R11_STAGE2_TASK_A_TRACE_PROCESS_MICRO_BENCHMARK_V1",
        "status": "PASS",
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
        "final_holdout_payload_read": False,
        "raw_1m_archive_read": False,
        "cpu_model": _cpu_model(),
        "groups": len(items),
        "symbols": symbols,
        "reference": {
            "thread_workers": 1,
            "seconds": serial_seconds,
            "trace_hash": reference_hash,
            "market_cache_stats": reference_stats,
        },
        "thread8": {"seconds": thread8_seconds, "trace_hash": thread8_hash},
        "fork_process": process_rows,
        "best_process_workers_by_reuse_seconds": int(best_reuse["workers"]),
        "best_process_reuse_seconds": float(best_reuse["reuse_seconds"]),
        "speedup_vs_thread_serial": float(serial_seconds / best_reuse["reuse_seconds"]),
        "speedup_vs_thread8": float(thread8_seconds / best_reuse["reuse_seconds"]),
        "identity_exact_across_thread_and_process": True,
        "peak_parent_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    _atomic_json(Path(args.out).resolve(), result)
    print(json.dumps({
        "status": "PASS",
        "identity_exact": True,
        "best_process_workers": result["best_process_workers_by_reuse_seconds"],
        "speedup_vs_thread_serial": result["speedup_vs_thread_serial"],
        "speedup_vs_thread8": result["speedup_vs_thread8"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
