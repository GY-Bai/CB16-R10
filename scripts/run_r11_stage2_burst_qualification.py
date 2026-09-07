#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 Stage-2 Shanxi short-burst qualification supervisor.

This module is infrastructure-only. It supervises an integration workload, samples
host resources, validates a narrow runtime telemetry contract, and emits the
machine-readable CB16_R11_STAGE2_BURST_QUALIFICATION_V1 report. It does not own
Teacher, Training, Trace, Storage, Champion/Challenger, or scientific semantics.
"""

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.rearchitecture_authority_r11 import verify_static_semantic_contracts

REPORT_SCHEMA = "CB16_R11_STAGE2_BURST_QUALIFICATION_V1"
TRIGGER_SCHEMA = "CB16_R11_STAGE2_BURST_TRIGGER_V1"
TELEMETRY_SCHEMA = "CB16_R11_STAGE2_BURST_TELEMETRY_V1"
CORRECTNESS_SCHEMA = "CB16_R11_STAGE2_BURST_CORRECTNESS_V1"
EXACT_INTEGRATION_BRANCH = "ai/r11-stage2-integration-r1"

DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_generations_per_min": 0.10,
    "min_teacher_evidence_per_sec": 1.0,
    "min_traces_per_sec": 0.10,
    "min_training_steps_per_sec": 0.10,
    "min_fp32_training_examples_per_sec": 1.0,
    "min_cpu_avg_pct": 55.0,
    "min_gpu_avg_pct": 45.0,
    "max_gpu_idle_ratio": 0.40,
    "min_cpu_gpu_overlap_pct": 35.0,
    "max_barrier_block_ratio": 0.15,
    "min_available_ram_gib": 1.5,
    "max_swap_out_mib": 64.0,
    "max_major_faults": 500.0,
    "max_hdd_avg_queue_depth": 3.0,
    "max_fsync_stall_ratio": 0.05,
}
REQUIRED_CORRECTNESS = (
    "semantic_freeze_pass",
    "final_holdout_untouched",
    "frozen_authority_unchanged",
    "teacher_identity_unchanged",
    "h72_receipt_identity_unchanged",
    "training_differential_pass",
    "no_forbidden_gradient",
    "replay_zero_new_payload",
    "drain_or_crash_restart_receipt_valid",
    "champion_challenger_lifecycle_correct",
)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON_OBJECT_REQUIRED:{path}")
    return obj


def load_trigger(path: Path) -> dict[str, Any]:
    x = _read_json(path)
    if x.get("schema") != TRIGGER_SCHEMA:
        raise RuntimeError("R11_STAGE2_TRIGGER_SCHEMA_MISMATCH")
    if x.get("qualification_branch") != EXACT_INTEGRATION_BRANCH:
        raise RuntimeError("R11_STAGE2_TRIGGER_BRANCH_ALLOWLIST_DRIFT")
    if not bool(x.get("armed", False)):
        raise RuntimeError("R11_STAGE2_TRIGGER_NOT_ARMED")
    warmup = int(x.get("warmup_seconds", 30))
    measured = int(x.get("measured_seconds", 420))
    hard_timeout = int(x.get("hard_timeout_minutes", 15))
    if not (5 <= warmup <= 120):
        raise RuntimeError("R11_STAGE2_WARMUP_OUT_OF_RANGE")
    if not (360 <= measured <= 600):
        raise RuntimeError("R11_STAGE2_MEASURED_BURST_MUST_BE_6_TO_10_MINUTES")
    if not (10 <= hard_timeout <= 15):
        raise RuntimeError("R11_STAGE2_HARD_TIMEOUT_MUST_BE_10_TO_15_MINUTES")
    argv = x.get("workload_argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and v for v in argv):
        raise RuntimeError("R11_STAGE2_WORKLOAD_ARGV_REQUIRED")
    if len(argv) < 2 or argv[0] != "{python}" or argv[1] != "scripts/run_r11_stage2_integration_burst.py":
        raise RuntimeError("R11_STAGE2_WORKLOAD_ENTRYPOINT_NOT_ALLOWLISTED")
    forbidden = x.get("forbidden", {})
    required_forbidden = (
        "fresh_market_data_download",
        "final_holdout_open",
        "oci_compute",
        "daemon_mode",
        "long_endurance",
    )
    if not isinstance(forbidden, dict) or not all(bool(forbidden.get(k)) for k in required_forbidden):
        raise RuntimeError("R11_STAGE2_FORBIDDEN_POLICY_NOT_EXPLICIT")
    return x


def _proc_stat() -> dict[str, int]:
    cpu = None
    ctxt = None
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu "):
            values = [int(x) for x in line.split()[1:]]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            cpu = {"total": total, "idle": idle}
        elif line.startswith("ctxt "):
            ctxt = int(line.split()[1])
    if cpu is None or ctxt is None:
        raise RuntimeError("PROC_STAT_UNAVAILABLE")
    return {"cpu_total": cpu["total"], "cpu_idle": cpu["idle"], "ctxt": ctxt}


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        token = rest.strip().split()[0]
        if token.isdigit():
            result[key] = int(token) * 1024
    return result


def _vmstat() -> dict[str, int]:
    wanted = {"pgmajfault", "pswpin", "pswpout"}
    result: dict[str, int] = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        key, value = line.split()
        if key in wanted:
            result[key] = int(value)
    return result


def _loadavg() -> tuple[float, float, float]:
    fields = Path("/proc/loadavg").read_text().split()
    return float(fields[0]), float(fields[1]), float(fields[2])


def _descendant_pids(root_pid: int) -> list[int]:
    ppid: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            tail = stat.rsplit(")", 1)[1].strip().split()
            ppid[int(entry.name)] = int(tail[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in ppid.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sorted(descendants)


def _process_tree_stats(root_pid: int) -> dict[str, int]:
    rss = threads = major_faults = 0
    for pid in _descendant_pids(root_pid):
        try:
            status = (Path("/proc") / str(pid) / "status").read_text().splitlines()
            for line in status:
                if line.startswith("VmRSS:"):
                    rss += int(line.split()[1]) * 1024
                elif line.startswith("Threads:"):
                    threads += int(line.split()[1])
            stat = (Path("/proc") / str(pid) / "stat").read_text()
            tail = stat.rsplit(")", 1)[1].strip().split()
            major_faults += int(tail[9])
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return {"rss_bytes": rss, "threads": threads, "major_faults_total": major_faults}


def _device_for_path(path: Path) -> tuple[int, int, str] | None:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    sysdev = Path(f"/sys/dev/block/{major}:{minor}")
    name = sysdev.resolve().name if sysdev.exists() else f"{major}:{minor}"
    return major, minor, name


def _diskstats(dev: tuple[int, int, str] | None) -> dict[str, int] | None:
    if dev is None:
        return None
    major, minor, name = dev
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) < 14:
            continue
        if int(f[0]) == major and int(f[1]) == minor:
            values = [int(v) for v in f[3:]]
            return {
                "name": name,
                "reads_completed": values[0],
                "sectors_read": values[2],
                "read_ms": values[3],
                "writes_completed": values[4],
                "sectors_written": values[6],
                "write_ms": values[7],
                "ios_in_progress": values[8],
                "io_ms": values[9],
                "weighted_io_ms": values[10],
            }
    return None


def _gpu_sample() -> dict[str, float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5).strip().splitlines()
    if len(out) != 1:
        raise RuntimeError("R11_STAGE2_EXPECTS_ONE_GPU")
    util, used, total = [float(v.strip()) for v in out[0].split(",")]
    return {"util_pct": util, "vram_used_mib": used, "vram_total_mib": total}


def verify_hardware() -> dict[str, Any]:
    logical = os.cpu_count()
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    import torch

    cap = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else ()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    checks = {
        "cpu_model": "AMD Ryzen 7 3700X" in cpu_model,
        "logical_cpus": logical == 16,
        "torch": torch.__version__ == "2.8.0+cu126",
        "cuda_runtime": torch.version.cuda == "12.6",
        "gpu_name": "GTX 1060" in gpu and "6GB" in gpu,
        "capability": cap == (6, 1),
        "default_dtype_fp32": torch.get_default_dtype() == torch.float32,
    }
    if not all(checks.values()):
        raise RuntimeError("R11_STAGE2_HARDWARE_GUARD_FAIL:" + json.dumps(checks, sort_keys=True))
    return {
        "status": "PASS",
        "cpu_model": cpu_model,
        "logical_cpus": logical,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "capability": list(cap),
        "canonical_arithmetic": "FP32",
        "amp": False,
        "checks": checks,
    }


@dataclass
class HostSample:
    t: float
    cpu_total: int
    cpu_idle: int
    ctxt: int
    load1: float
    load5: float
    load15: float
    mem_available: int
    swap_free: int
    swap_total: int
    pswpin: int
    pswpout: int
    pgmajfault: int
    rss_bytes: int
    threads: int
    proc_major_faults: int
    gpu_util_pct: float
    vram_used_mib: float
    ssd: dict[str, int] | None
    hdd: dict[str, int] | None


def sample_host(pid: int, ssd_dev: tuple[int, int, str] | None, hdd_dev: tuple[int, int, str] | None) -> HostSample:
    ps = _proc_stat()
    mem = _meminfo()
    vm = _vmstat()
    load = _loadavg()
    proc = _process_tree_stats(pid)
    gpu = _gpu_sample()
    return HostSample(
        t=time.monotonic(),
        cpu_total=ps["cpu_total"],
        cpu_idle=ps["cpu_idle"],
        ctxt=ps["ctxt"],
        load1=load[0],
        load5=load[1],
        load15=load[2],
        mem_available=mem.get("MemAvailable", 0),
        swap_free=mem.get("SwapFree", 0),
        swap_total=mem.get("SwapTotal", 0),
        pswpin=vm.get("pswpin", 0),
        pswpout=vm.get("pswpout", 0),
        pgmajfault=vm.get("pgmajfault", 0),
        rss_bytes=proc["rss_bytes"],
        threads=proc["threads"],
        proc_major_faults=proc["major_faults_total"],
        gpu_util_pct=gpu["util_pct"],
        vram_used_mib=gpu["vram_used_mib"],
        ssd=_diskstats(ssd_dev),
        hdd=_diskstats(hdd_dev),
    )


def _cpu_pct(a: HostSample, b: HostSample) -> float:
    dt = b.cpu_total - a.cpu_total
    if dt <= 0:
        return 0.0
    idle = b.cpu_idle - a.cpu_idle
    return max(0.0, min(100.0, 100.0 * (dt - idle) / dt))


def _rate_delta(a: int | float, b: int | float, seconds: float) -> float:
    return max(0.0, float(b) - float(a)) / max(seconds, 1e-9)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def load_runtime_telemetry(path: Path, start_t: float, end_t: float) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict) or row.get("schema") != TELEMETRY_SCHEMA:
            raise RuntimeError("R11_STAGE2_TELEMETRY_SCHEMA_MISMATCH")
        t = float(row["monotonic_seconds"])
        if start_t <= t <= end_t:
            rows.append(row)
    rows.sort(key=lambda x: float(x["monotonic_seconds"]))
    return rows


def _counter_delta(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    if len(rows) < 2:
        return None
    first = rows[0].get("counters", {})
    last = rows[-1].get("counters", {})
    if key not in first or key not in last:
        return None
    return max(0.0, float(last[key]) - float(first[key]))


def _gauge_values(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        gauges = row.get("gauges", {})
        if key in gauges:
            out.append(float(gauges[key]))
    return out


def summarize_host(samples: Sequence[HostSample], measured_seconds: float) -> dict[str, Any]:
    if len(samples) < 2:
        raise RuntimeError("R11_STAGE2_INSUFFICIENT_HOST_SAMPLES")
    cpu = [_cpu_pct(a, b) for a, b in zip(samples, samples[1:])]
    gpu = [s.gpu_util_pct for s in samples]
    overlap = [
        1.0 if c >= 50.0 and g >= 10.0 else 0.0
        for c, g in zip(cpu, [s.gpu_util_pct for s in samples[1:]])
    ]
    first, last = samples[0], samples[-1]

    def disk_summary(attr: str) -> dict[str, Any] | None:
        a = getattr(first, attr)
        b = getattr(last, attr)
        if a is None or b is None:
            return None
        elapsed_ms = max((last.t - first.t) * 1000.0, 1.0)
        read_bps = _rate_delta(a["sectors_read"] * 512, b["sectors_read"] * 512, measured_seconds)
        write_bps = _rate_delta(a["sectors_written"] * 512, b["sectors_written"] * 512, measured_seconds)
        avg_q = max(0.0, float(b["weighted_io_ms"] - a["weighted_io_ms"]) / elapsed_ms)
        return {
            "device": a["name"],
            "read_bytes_per_sec": read_bps,
            "write_bytes_per_sec": write_bps,
            "avg_queue_depth": avg_q,
            "reads_per_sec": _rate_delta(a["reads_completed"], b["reads_completed"], measured_seconds),
            "writes_per_sec": _rate_delta(a["writes_completed"], b["writes_completed"], measured_seconds),
        }

    return {
        "cpu": {
            "avg_utilization_pct": statistics.fmean(cpu),
            "p95_utilization_pct": _percentile(cpu, 0.95),
            "context_switches_per_sec": _rate_delta(first.ctxt, last.ctxt, measured_seconds),
            "load_average_peak_1m": max(s.load1 for s in samples),
            "load_average_end": [last.load1, last.load5, last.load15],
        },
        "gpu": {
            "avg_utilization_pct": statistics.fmean(gpu),
            "p95_utilization_pct": _percentile(gpu, 0.95),
            "idle_ratio": statistics.fmean(1.0 if g <= 5.0 else 0.0 for g in gpu),
            "vram_peak_mib": max(s.vram_used_mib for s in samples),
        },
        "memory": {
            "rss_peak_bytes": max(s.rss_bytes for s in samples),
            "available_ram_min_bytes": min(s.mem_available for s in samples),
            "swap_total_bytes": max(s.swap_total for s in samples),
            "swap_used_peak_bytes": max(s.swap_total - s.swap_free for s in samples),
            "swap_in_pages": max(0, last.pswpin - first.pswpin),
            "swap_out_pages": max(0, last.pswpout - first.pswpout),
            "major_faults_system": max(0, last.pgmajfault - first.pgmajfault),
            "major_faults_process_tree": max(0, last.proc_major_faults - first.proc_major_faults),
            "thread_count_peak": max(s.threads for s in samples),
        },
        "io": {"ssd": disk_summary("ssd"), "hdd": disk_summary("hdd")},
        "pipeline": {
            "sampled_cpu_gpu_overlap_pct": 100.0 * statistics.fmean(overlap) if overlap else 0.0,
        },
    }


def summarize_runtime(rows: Sequence[Mapping[str, Any]], measured_seconds: float) -> dict[str, Any]:
    required_counters = (
        "generations_committed",
        "teacher_evidence",
        "traces",
        "training_steps",
        "fp32_training_examples",
        "barrier_block_seconds",
        "fsync_stall_seconds",
        "ssd_metadata_ops",
        "hdd_read_bytes",
        "hdd_write_bytes",
    )
    deltas = {k: _counter_delta(rows, k) for k in required_counters}
    missing = [k for k, v in deltas.items() if v is None]
    teacher_util = _gauge_values(rows, "teacher_worker_utilization_pct")
    trace_util = _gauge_values(rows, "trace_worker_utilization_pct")
    fsync_p95 = _gauge_values(rows, "fsync_latency_p95_ms")
    queue_depth = _gauge_values(rows, "pipeline_queue_depth")
    if not teacher_util:
        missing.append("teacher_worker_utilization_pct")
    if not trace_util:
        missing.append("trace_worker_utilization_pct")
    if not fsync_p95:
        missing.append("fsync_latency_p95_ms")
    if missing:
        raise RuntimeError("R11_STAGE2_RUNTIME_TELEMETRY_INCOMPLETE:" + ",".join(sorted(set(missing))))

    return {
        "generations_per_min": 60.0 * deltas["generations_committed"] / measured_seconds,
        "teacher_evidence_per_sec": deltas["teacher_evidence"] / measured_seconds,
        "traces_per_sec": deltas["traces"] / measured_seconds,
        "training_steps_per_sec": deltas["training_steps"] / measured_seconds,
        "fp32_training_examples_per_sec": deltas["fp32_training_examples"] / measured_seconds,
        "barrier_block_seconds": deltas["barrier_block_seconds"],
        "barrier_block_ratio": deltas["barrier_block_seconds"] / measured_seconds,
        "teacher_worker_utilization_avg_pct": statistics.fmean(teacher_util),
        "trace_worker_utilization_avg_pct": statistics.fmean(trace_util),
        "ssd_metadata_ops_per_sec": deltas["ssd_metadata_ops"] / measured_seconds,
        "hdd_read_bytes_per_sec_runtime": deltas["hdd_read_bytes"] / measured_seconds,
        "hdd_write_bytes_per_sec_runtime": deltas["hdd_write_bytes"] / measured_seconds,
        "fsync_stall_seconds": deltas["fsync_stall_seconds"],
        "fsync_stall_ratio": deltas["fsync_stall_seconds"] / measured_seconds,
        "fsync_latency_p95_ms_peak": max(fsync_p95),
        "pipeline_queue_depth_avg": statistics.fmean(queue_depth) if queue_depth else None,
        "counter_deltas": deltas,
    }


def validate_correctness(
    path: Path,
    *,
    static_guard: Mapping[str, Any],
    manifest_before: Mapping[str, Any],
    manifest_after: Mapping[str, Any],
    frozen_before: Mapping[str, Any],
    frozen_after: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    x = _read_json(path)
    if x.get("schema") != CORRECTNESS_SCHEMA:
        raise RuntimeError("R11_STAGE2_CORRECTNESS_SCHEMA_MISMATCH")
    merged = dict(x)
    merged["semantic_freeze_pass"] = bool(static_guard.get("status") == "PASS" or static_guard.get("pass") is True)
    merged["final_holdout_untouched"] = bool(
        not manifest_before.get("final_holdout_2025_09_accessed", False)
        and not manifest_after.get("final_holdout_2025_09_accessed", False)
    )
    merged["frozen_authority_unchanged"] = frozen_before == frozen_after
    failures = [k for k in REQUIRED_CORRECTNESS if merged.get(k) is not True]
    if merged.get("scientific_semantics_changed") is not False:
        failures.append("scientific_semantics_changed")
    if merged.get("training_dtype") != "FP32":
        failures.append("training_dtype")
    if merged.get("amp") is not False:
        failures.append("amp")
    forbidden = merged.get("forbidden_work", {})
    if not isinstance(forbidden, dict) or any(bool(v) for v in forbidden.values()):
        failures.append("forbidden_work")
    return merged, failures


def evaluate_thresholds(
    host: Mapping[str, Any],
    runtime: Mapping[str, Any],
    thresholds: Mapping[str, float],
    baseline: Mapping[str, float] | None,
) -> list[str]:
    failures: list[str] = []
    checks = {
        "generations_per_min": (runtime["generations_per_min"], ">=", thresholds["min_generations_per_min"]),
        "teacher_evidence_per_sec": (runtime["teacher_evidence_per_sec"], ">=", thresholds["min_teacher_evidence_per_sec"]),
        "traces_per_sec": (runtime["traces_per_sec"], ">=", thresholds["min_traces_per_sec"]),
        "training_steps_per_sec": (runtime["training_steps_per_sec"], ">=", thresholds["min_training_steps_per_sec"]),
        "fp32_training_examples_per_sec": (runtime["fp32_training_examples_per_sec"], ">=", thresholds["min_fp32_training_examples_per_sec"]),
        "cpu_avg_pct": (host["cpu"]["avg_utilization_pct"], ">=", thresholds["min_cpu_avg_pct"]),
        "gpu_avg_pct": (host["gpu"]["avg_utilization_pct"], ">=", thresholds["min_gpu_avg_pct"]),
        "gpu_idle_ratio": (host["gpu"]["idle_ratio"], "<=", thresholds["max_gpu_idle_ratio"]),
        "cpu_gpu_overlap_pct": (host["pipeline"]["sampled_cpu_gpu_overlap_pct"], ">=", thresholds["min_cpu_gpu_overlap_pct"]),
        "barrier_block_ratio": (runtime["barrier_block_ratio"], "<=", thresholds["max_barrier_block_ratio"]),
        "available_ram_gib": (host["memory"]["available_ram_min_bytes"] / (1024**3), ">=", thresholds["min_available_ram_gib"]),
        "swap_out_mib": (host["memory"]["swap_out_pages"] * os.sysconf("SC_PAGE_SIZE") / (1024**2), "<=", thresholds["max_swap_out_mib"]),
        "major_faults": (max(host["memory"]["major_faults_process_tree"], host["memory"]["major_faults_system"]), "<=", thresholds["max_major_faults"]),
        "fsync_stall_ratio": (runtime["fsync_stall_ratio"], "<=", thresholds["max_fsync_stall_ratio"]),
    }
    hdd = host["io"].get("hdd")
    if hdd is not None:
        checks["hdd_avg_queue_depth"] = (hdd["avg_queue_depth"], "<=", thresholds["max_hdd_avg_queue_depth"])
    for name, (value, op, limit) in checks.items():
        ok = value >= limit if op == ">=" else value <= limit
        if not ok:
            failures.append(f"{name}:{value:.6g}{op}{limit:.6g}:FAIL")

    if baseline:
        for metric in (
            "generations_per_min",
            "teacher_evidence_per_sec",
            "traces_per_sec",
            "training_steps_per_sec",
            "fp32_training_examples_per_sec",
        ):
            if metric in baseline and float(baseline[metric]) > 0:
                ratio = float(runtime[metric]) / float(baseline[metric])
                if ratio < 0.90:
                    failures.append(f"{metric}_baseline_ratio:{ratio:.4f}>=0.90:FAIL")
    return failures


def bottlenecks(host: Mapping[str, Any], runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cpu = float(host["cpu"]["avg_utilization_pct"])
    gpu = float(host["gpu"]["avg_utilization_pct"])
    idle = float(host["gpu"]["idle_ratio"])
    barrier = float(runtime["barrier_block_ratio"])
    avail = float(host["memory"]["available_ram_min_bytes"]) / (1024**3)
    hdd_q = float((host["io"].get("hdd") or {}).get("avg_queue_depth", 0.0))
    fsync = float(runtime["fsync_stall_ratio"])
    if avail < 2.0:
        out.append({"cause": "MEMORY_PRESSURE", "severity": "HIGH", "evidence": {"available_ram_min_gib": avail}})
    if hdd_q > 2.0 or fsync > 0.05:
        out.append({"cause": "STORAGE_BACKPRESSURE", "severity": "HIGH" if hdd_q > 3.0 else "MEDIUM", "evidence": {"hdd_avg_queue_depth": hdd_q, "fsync_stall_ratio": fsync}})
    if barrier > 0.15:
        out.append({"cause": "TOURNAMENT_BARRIER", "severity": "MEDIUM", "evidence": {"barrier_block_ratio": barrier}})
    if idle > 0.35 and cpu >= 70.0:
        out.append({"cause": "GPU_STARVED_BY_CPU_PIPELINE", "severity": "MEDIUM", "evidence": {"gpu_idle_ratio": idle, "cpu_avg_pct": cpu}})
    elif gpu >= 85.0 and idle < 0.10:
        out.append({"cause": "GPU_TRAINING_BOUND", "severity": "INFO", "evidence": {"gpu_avg_pct": gpu}})
    if cpu < 55.0 and gpu < 45.0:
        out.append({"cause": "PIPELINE_UNDERFILL_OR_SERIAL_WAIT", "severity": "HIGH", "evidence": {"cpu_avg_pct": cpu, "gpu_avg_pct": gpu}})
    if not out:
        out.append({"cause": "BALANCED_NO_SINGLE_DOMINANT_BOTTLENECK", "severity": "INFO", "evidence": {"cpu_avg_pct": cpu, "gpu_avg_pct": gpu}})
    return out


def recommend_defaults(config: Mapping[str, Any], host: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    teacher = int(config["teacher_workers"])
    trace = int(config["trace_workers"])
    queue = int(config["queue_depth"])
    buffer_mib = int(config["buffer_mib"])
    cpu = float(host["cpu"]["avg_utilization_pct"])
    idle = float(host["gpu"]["idle_ratio"])
    avail = float(host["memory"]["available_ram_min_bytes"]) / (1024**3)
    hdd_q = float((host["io"].get("hdd") or {}).get("avg_queue_depth", 0.0))
    swap_out = int(host["memory"]["swap_out_pages"])

    reasons: list[str] = []
    if avail < 2.0 or swap_out > 0 or cpu > 94.0:
        teacher = max(4, teacher - 1)
        trace = max(4, trace - 1)
        reasons.append("reduce CPU worker pressure")
    elif idle > 0.35 and cpu < 88.0 and avail >= 3.0:
        if teacher <= trace:
            teacher = min(12, teacher + 1)
        else:
            trace = min(12, trace + 1)
        reasons.append("use spare CPU headroom to reduce GPU starvation")
    if hdd_q > 3.0 or float(runtime["fsync_stall_ratio"]) > 0.05:
        queue = max(2, queue - 1)
        buffer_mib = max(16, buffer_mib // 2)
        reasons.append("reduce storage queueing and fsync burst size")
    elif hdd_q < 1.0 and float(runtime["barrier_block_ratio"]) < 0.10:
        queue = min(16, queue + 1)
        reasons.append("increase bounded in-flight depth")
    return {
        "scientific_identity": False,
        "teacher_workers": teacher,
        "trace_workers": trace,
        "queue_depth": queue,
        "buffer_mib": buffer_mib,
        "threads_per_worker": 1,
        "training_dtype": "FP32",
        "amp": False,
        "reasons": reasons or ["retain observed configuration"],
    }


def _expand_argv(argv: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        value = arg
        for key, replacement in replacements.items():
            value = value.replace("{" + key + "}", replacement)
        out.append(value)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="CB16 R11 Stage-2 10-minute high-throughput qualification")
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--legacy-r104-root", required=True)
    ap.add_argument("--work-root", required=True)
    ap.add_argument("--ssd-work-root", required=True)
    ap.add_argument("--hdd-work-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ssd-path", default="/data/cb16_ci")
    ap.add_argument("--hdd-path", default="/data/cb16_hdd")
    ap.add_argument("--sample-interval", type=float, default=1.0)
    args = ap.parse_args()

    trigger = load_trigger(Path(args.trigger).resolve())
    branch = os.environ.get("GITHUB_REF_NAME", "")
    if branch and branch != EXACT_INTEGRATION_BRANCH:
        raise RuntimeError("R11_STAGE2_RUNTIME_BRANCH_GUARD_FAIL")

    package_root = Path(args.package_root).resolve()
    legacy_root = Path(args.legacy_r104_root).resolve()
    work = Path(args.work_root).resolve()
    ssd_work = Path(args.ssd_work_root).resolve()
    hdd_work = Path(args.hdd_work_root).resolve()
    out = Path(args.out).resolve()
    work.mkdir(parents=True, exist_ok=True)
    ssd_work.mkdir(parents=True, exist_ok=True)
    hdd_work.mkdir(parents=True, exist_ok=True)
    telemetry_path = work / "runtime_telemetry.jsonl"
    correctness_path = work / "runtime_correctness.json"
    stdout_path = work / "workload.stdout.log"
    stderr_path = work / "workload.stderr.log"

    hardware = verify_hardware()
    static_guard = verify_static_semantic_contracts(ROOT)
    frozen_before = frozen_authority_hashes(package_root)

    manifest_path = legacy_root / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    manifest_before = _read_json(manifest_path)
    if manifest_before.get("final_holdout_2025_09_accessed", False):
        raise RuntimeError("R11_STAGE2_REFUSES_OPENED_FINAL_HOLDOUT")

    warmup = int(trigger["warmup_seconds"])
    measured_target = int(trigger["measured_seconds"])
    config = {
        "teacher_workers": int(trigger["teacher_workers"]),
        "trace_workers": int(trigger["trace_workers"]),
        "queue_depth": int(trigger["queue_depth"]),
        "buffer_mib": int(trigger["buffer_mib"]),
    }
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update({k: float(v) for k, v in trigger.get("thresholds", {}).items()})
    baseline = trigger.get("baseline_throughput")
    if baseline is not None and not isinstance(baseline, dict):
        raise RuntimeError("R11_STAGE2_BASELINE_THROUGHPUT_MUST_BE_OBJECT")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "CB16_R11_BURST_TELEMETRY_JSONL": str(telemetry_path),
            "CB16_R11_BURST_CORRECTNESS_JSON": str(correctness_path),
            "CB16_R11_BURST_WORK_ROOT": str(work / "runtime"),
            "CB16_R11_BURST_SSD_ROOT": str(ssd_work),
            "CB16_R11_BURST_HDD_ROOT": str(hdd_work),
            "CB16_R11_BURST_PACKAGE_ROOT": str(package_root),
            "CB16_R11_BURST_LEGACY_R104_ROOT": str(legacy_root),
            "CB16_R11_BURST_WARMUP_SECONDS": str(warmup),
            "CB16_R11_BURST_MEASURED_SECONDS": str(measured_target),
            "CB16_R11_BURST_TEACHER_WORKERS": str(config["teacher_workers"]),
            "CB16_R11_BURST_TRACE_WORKERS": str(config["trace_workers"]),
            "CB16_R11_BURST_QUEUE_DEPTH": str(config["queue_depth"]),
            "CB16_R11_BURST_BUFFER_MIB": str(config["buffer_mib"]),
            "CB16_R11_CANONICAL_DTYPE": "FP32",
            "CB16_R11_AMP": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    replacements = {
        "python": sys.executable,
        "repo": str(ROOT),
        "work": str(work / "runtime"),
        "ssd_work": str(ssd_work),
        "hdd_work": str(hdd_work),
        "telemetry": str(telemetry_path),
        "correctness": str(correctness_path),
    }
    argv = _expand_argv(trigger["workload_argv"], replacements)
    ssd_dev = _device_for_path(Path(args.ssd_path))
    hdd_dev = _device_for_path(Path(args.hdd_path))

    samples: list[HostSample] = []
    workload_rc: int | None = None
    start = time.monotonic()
    measured_start = start + warmup
    measured_end = measured_start + measured_target
    drain_deadline = measured_end + 30.0
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            while True:
                now = time.monotonic()
                if proc.poll() is not None:
                    workload_rc = int(proc.returncode)
                    break
                if measured_start <= now <= measured_end:
                    samples.append(sample_host(proc.pid, ssd_dev, hdd_dev))
                if now >= drain_deadline:
                    os.killpg(proc.pid, signal.SIGTERM)
                    workload_rc = proc.wait(timeout=10)
                    break
                time.sleep(max(0.1, float(args.sample_interval)))
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGINT)
                    proc.wait(timeout=20)
                except Exception:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=10)
                    except Exception:
                        os.killpg(proc.pid, signal.SIGKILL)
                workload_rc = proc.returncode
            elif workload_rc is None:
                workload_rc = int(proc.returncode)

    actual_end = min(time.monotonic(), measured_end)
    measured_seconds = max(0.0, actual_end - measured_start)
    if samples and samples[-1].t < measured_end - 1.5 and workload_rc == 0:
        measured_seconds = max(0.0, samples[-1].t - samples[0].t)

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "FAIL",
        "scientific_semantics_changed": False,
        "branch_allowlist": [EXACT_INTEGRATION_BRANCH],
        "duration_policy": {
            "warmup_seconds": warmup,
            "measured_target_seconds": measured_target,
            "measured_actual_seconds": measured_seconds,
            "hard_timeout_minutes": int(trigger["hard_timeout_minutes"]),
            "endurance_test": False,
        },
        "hardware": hardware,
        "config_under_test": {**config, "scientific_identity": False},
        "thresholds": thresholds,
        "workload": {"argv": argv, "returncode": workload_rc},
    }
    failures: list[str] = []
    try:
        if workload_rc != 0:
            failures.append(f"workload_returncode:{workload_rc}")
        if measured_seconds < measured_target * 0.95:
            failures.append(f"measured_duration_short:{measured_seconds:.1f}<{measured_target * 0.95:.1f}")
        if len(samples) < max(30, int(measured_seconds / max(args.sample_interval, 0.1) * 0.80)):
            failures.append("host_sampling_coverage_low")

        host = summarize_host(samples, max(measured_seconds, 1.0))
        rows = load_runtime_telemetry(telemetry_path, measured_start, measured_end)
        runtime = summarize_runtime(rows, max(measured_seconds, 1.0))

        manifest_after = _read_json(manifest_path)
        frozen_after = frozen_authority_hashes(package_root)
        correctness, correctness_failures = validate_correctness(
            correctness_path,
            static_guard=static_guard,
            manifest_before=manifest_before,
            manifest_after=manifest_after,
            frozen_before=frozen_before,
            frozen_after=frozen_after,
        )
        failures.extend("correctness:" + x for x in correctness_failures)
        threshold_failures = evaluate_thresholds(host, runtime, thresholds, baseline)
        failures.extend("threshold:" + x for x in threshold_failures)

        report["correctness"] = correctness
        report["throughput"] = runtime
        report["utilization"] = {
            "cpu": host["cpu"],
            "gpu": host["gpu"],
            "teacher_worker_utilization_avg_pct": runtime["teacher_worker_utilization_avg_pct"],
            "trace_worker_utilization_avg_pct": runtime["trace_worker_utilization_avg_pct"],
            "cpu_gpu_overlap_pct": host["pipeline"]["sampled_cpu_gpu_overlap_pct"],
        }
        report["resource_peaks"] = {
            "rss_peak_bytes": host["memory"]["rss_peak_bytes"],
            "available_ram_min_bytes": host["memory"]["available_ram_min_bytes"],
            "swap_used_peak_bytes": host["memory"]["swap_used_peak_bytes"],
            "thread_count_peak": host["memory"]["thread_count_peak"],
            "vram_peak_mib": host["gpu"]["vram_peak_mib"],
            "load_average_peak_1m": host["cpu"]["load_average_peak_1m"],
            "major_faults_process_tree": host["memory"]["major_faults_process_tree"],
            "major_faults_system": host["memory"]["major_faults_system"],
            "swap_in_pages": host["memory"]["swap_in_pages"],
            "swap_out_pages": host["memory"]["swap_out_pages"],
        }
        report["io"] = {
            **host["io"],
            "ssd_metadata_ops_per_sec_runtime": runtime["ssd_metadata_ops_per_sec"],
            "hdd_read_bytes_per_sec_runtime": runtime["hdd_read_bytes_per_sec_runtime"],
            "hdd_write_bytes_per_sec_runtime": runtime["hdd_write_bytes_per_sec_runtime"],
            "fsync_stall_seconds": runtime["fsync_stall_seconds"],
            "fsync_stall_ratio": runtime["fsync_stall_ratio"],
            "fsync_latency_p95_ms_peak": runtime["fsync_latency_p95_ms_peak"],
        }
        report["bottleneck_attribution"] = bottlenecks(host, runtime)
        report["recommended_runtime_defaults"] = recommend_defaults(config, host, runtime)
        report["static_semantic_guard"] = static_guard
        report["frozen_authority_hashes_before"] = frozen_before
        report["frozen_authority_hashes_after"] = frozen_after
        report["telemetry_samples"] = {"host": len(samples), "runtime": len(rows)}
    except Exception as exc:
        failures.append(f"qualification_exception:{type(exc).__name__}:{exc}")

    report["acceptance"] = {
        "pass": not failures,
        "failures": failures,
        "throughput_regression_policy": "IF_BASELINE_PRESENT_EACH_PRIMARY_RATE_MUST_BE_AT_LEAST_90_PERCENT_OF_BASELINE",
    }
    report["status"] = "PASS" if not failures else "FAIL"
    atomic_json(out, report)
    print(json.dumps({"schema": REPORT_SCHEMA, "status": report["status"], "failures": failures}, sort_keys=True))
    return 0 if not failures else 2


def _write_fatal_report_from_argv(exc: BaseException) -> None:
    try:
        args = sys.argv[1:]
        idx = args.index("--out")
        out = Path(args[idx + 1]).resolve()
    except Exception:
        return
    try:
        atomic_json(
            out,
            {
                "schema": REPORT_SCHEMA,
                "status": "FAIL",
                "scientific_semantics_changed": False,
                "branch_allowlist": [EXACT_INTEGRATION_BRANCH],
                "acceptance": {
                    "pass": False,
                    "failures": [f"fatal:{type(exc).__name__}:{exc}"],
                },
            },
        )
    except Exception:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        _write_fatal_report_from_argv(exc)
        print(json.dumps({"schema": REPORT_SCHEMA, "status": "FAIL", "fatal": f"{type(exc).__name__}:{exc}"}, sort_keys=True), file=sys.stderr)
        raise
