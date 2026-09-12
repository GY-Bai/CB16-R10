from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np

from .cc_fast_benchmark_contract_r0 import BENCHMARK_SCHEMA, BenchmarkReport
from .cc_fast_collector_r0 import CollectorConfig, run_collector
from .cc_fast_metrics_r0 import FastMetrics, query_nvidia_smi
from .cc_fast_policy_benchmark_r0 import benchmark_cpu, benchmark_cuda_if_available, select_policy_mode
from .cc_fast_policy_broker_r0 import PolicyRequest, PolicySpec
from .cc_fast_semantic_harness_r0 import run_semantic_harness
from .cc_fast_wire_r0 import SCIENCE_SEMANTIC_VERSION, semantic_sha256
from .cc_fast_workload_r0 import WorkloadConfig, build_workload

RUNNER_SCHEMA = "CB16_R11_CC_FAST_BENCHMARK_RUN_V1"


def _hardware_identity() -> dict[str, object]:
    ram = None
    try:
        ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        pass
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": ram,
        "gpu": query_nvidia_smi(),
    }


def _policy_fixture() -> PolicySpec:
    return PolicySpec(
        policy_generation=7,
        policy_id="cc-fast-benchmark-fixture",
        policy_sha256="7" * 64,
        direction_logits=(-0.25, -0.85, 0.55),
        risk_loc_short=-0.20,
        risk_scale_short=0.70,
        risk_loc_long=0.20,
        risk_scale_long=0.70,
    )


def _policy_requests(count: int) -> list[PolicyRequest]:
    out: list[PolicyRequest] = []
    for i in range(count):
        obs = (100.0 + i * 0.001, 0.0, 0.0, 10000.0, 0.0, 10000.0, 0.0, 0.0, 0.0)
        out.append(PolicyRequest(
            account_lineage_id=f"bench-{i:05d}",
            decision_index=i,
            environment_time_ns=i,
            policy_generation=7,
            policy_sha256="7" * 64,
            observation=obs,
            observation_hash=hashlib.sha256(np.asarray(obs, dtype=np.float64).tobytes()).hexdigest(),
            normalizer_id="cc-fast-bench-normalizer-v1",
            rng_stream_id=f"bench-stream-{i:05d}",
            rng_counter=i,
            stochastic=True,
        ))
    return out


def _batch_distribution(batch_sizes: tuple[int, ...]) -> dict[str, float]:
    if not batch_sizes:
        return {"count": 0.0, "min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(int(x) for x in batch_sizes)
    def q(frac: float) -> float:
        idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * frac)))
        return float(ordered[idx])
    return {
        "count": float(len(ordered)),
        "min": float(ordered[0]),
        "p50": q(0.50),
        "p95": q(0.95),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


def _collector_benchmark(*, worker_count: int, accounts: int, market_steps: int, chunk_facts: int) -> tuple[BenchmarkReport, dict[str, object]]:
    workload = build_workload(WorkloadConfig(seed=230912, market_steps=market_steps, account_count=accounts))
    policy = _policy_fixture()
    warm = build_workload(WorkloadConfig(seed=230912, market_steps=64, account_count=min(accounts, 4)))
    with tempfile.TemporaryDirectory(prefix="cc-fast-warm-") as warm_dir:
        run_collector(warm, policy, output_root=warm_dir, config=CollectorConfig(worker_count=2, writer_chunk_facts=16))

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cc-fast-measured-") as out_dir:
        result = run_collector(
            workload,
            policy,
            output_root=out_dir,
            config=CollectorConfig(
                worker_count=worker_count,
                queue_max_bytes=8 * 1024 * 1024,
                queue_max_age_s=30.0,
                writer_chunk_facts=chunk_facts,
            ),
        )
    wall = time.perf_counter() - t0
    system = result.metrics
    gpu = system.get("gpu") or {}
    cpu = tuple(float(x) for x in (system.get("cpu_per_core_utilization") or (0.0,)))
    report = BenchmarkReport(
        schema_version=BENCHMARK_SCHEMA,
        workload_identity=workload.workload_id,
        code_identity=os.environ.get("GITHUB_SHA", "LOCAL_UNPINNED"),
        science_identity=SCIENCE_SEMANTIC_VERSION,
        topology={
            "worker_count": worker_count,
            "active_accounts": accounts,
            "writer_chunk_facts": chunk_facts,
            "market_reuse": True,
            "policy_execution": "CPU_COLLECTOR",
            "spawn_workers": True,
        },
        account_count=accounts,
        decision_rate=(result.policy_decisions / wall) if wall else 0.0,
        batch_size_distribution=_batch_distribution(result.batch_sizes),
        compliant_transitions_per_s=(result.transitions / wall) if wall else 0.0,
        policy_decisions_per_s=(result.policy_decisions / wall) if wall else 0.0,
        wall_clock_s=wall,
        cpu_per_core_utilization=cpu,
        pss_bytes=int(system.get("pss_bytes") or 0),
        cgroup_memory_bytes=system.get("cgroup_memory_bytes"),
        page_faults=int(system.get("minor_faults") or 0) + int(system.get("major_faults") or 0),
        swap_in_bytes=int(system.get("swap_in_bytes") or 0),
        swap_out_bytes=int(system.get("swap_out_bytes") or 0),
        disk_read_bytes_per_s=float(system.get("disk_read_bytes_per_s") or 0.0),
        disk_write_bytes_per_s=float(system.get("disk_write_bytes_per_s") or 0.0),
        io_wait_fraction=float(system.get("io_wait_fraction") or 0.0),
        gpu_vram_bytes=int(gpu["vram_used_bytes"]) if "vram_used_bytes" in gpu else None,
        gpu_kernel_ms=None,
        gpu_transfer_ms=None,
        queue_bytes=result.queue_peak_bytes,
        queue_depth=result.queue_peak_depth,
        queue_oldest_age_s=0.0,
        correctness_checksum=result.semantic_checksum,
        semantic_verdict="PASS",
        notes=("Synthetic-only; no FINAL/fresh data.", "Warm-up excluded from measured wall time."),
    ).validate()
    extra = {
        "final_account_checksum": result.final_account_checksum,
        "terminal_transitions": result.terminal_transitions,
        "worker_pids": list(result.worker_pids),
        "worker_cuda_initialized": list(result.worker_cuda_initialized),
        "chunk_count": len(result.chunk_receipts),
    }
    return report, extra


def run(*, worker_count: int, accounts: int, market_steps: int, chunk_facts: int, policy_batch: int) -> dict[str, object]:
    semantic = run_semantic_harness()
    if semantic.verdict != "PASS":
        raise RuntimeError("SEMANTIC_HARNESS_FAILED")
    requests = _policy_requests(policy_batch)
    policy = _policy_fixture()
    cpu = benchmark_cpu(policy, requests, repeat=8)
    gpu = benchmark_cuda_if_available(policy, requests, repeat=8)
    selected_policy = select_policy_mode(cpu, gpu)
    report, extra = _collector_benchmark(
        worker_count=worker_count,
        accounts=accounts,
        market_steps=market_steps,
        chunk_facts=chunk_facts,
    )
    return {
        "schema_version": RUNNER_SCHEMA,
        "semantic": {
            "verdict": semantic.verdict,
            "checksum": semantic.checksum,
            "numeric_tolerance": semantic.numeric_tolerance,
            "checks": semantic.checks,
        },
        "hardware": _hardware_identity(),
        "policy_mode_benchmark": {
            "cpu": asdict(cpu),
            "gpu": asdict(gpu),
            "selected_for_policy_microbenchmark": selected_policy,
            "selection_rule": "GPU only if available, semantic checksum identical, and measured decisions/s is higher; otherwise CPU.",
        },
        "collector_report": report.to_dict(),
        "collector_extra": extra,
        "accesses_final_or_fresh_data": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, choices=(2, 4, 6), default=2)
    parser.add_argument("--accounts", type=int, choices=(16, 32, 64), default=16)
    parser.add_argument("--market-steps", type=int, default=512)
    parser.add_argument("--chunk-facts", type=int, default=128)
    parser.add_argument("--policy-batch", type=int, default=32)
    args = parser.parse_args()
    payload = run(worker_count=args.workers, accounts=args.accounts, market_steps=args.market_steps, chunk_facts=args.chunk_facts, policy_batch=args.policy_batch)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": payload["semantic"]["verdict"],
        "selected_policy": payload["policy_mode_benchmark"]["selected_for_policy_microbenchmark"],
        "transitions_per_s": payload["collector_report"]["compliant_transitions_per_s"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
