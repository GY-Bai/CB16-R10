from __future__ import annotations

"""Stage-3 E1 persistent data-plane endurance support.

Infrastructure-only.  This module does not own scientific authority transitions.
It supervises long-lived Teacher/trace/training/IO objects and evaluates rolling
resource/throughput telemetry from frozen replay workloads.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "CB16_R11_STAGE3_E1_ENDURANCE_REPORT_V1"
TELEMETRY_SCHEMA = "CB16_R11_STAGE3_E1_TELEMETRY_V1"
CANONICAL_TEACHER_WORKERS = 8
CANONICAL_TRACE_WORKERS = 8
CANONICAL_QUEUE_DEPTH = 8
CANONICAL_BUFFER_MIB = 64


class E1Failure(str, Enum):
    SEMANTIC_IDENTITY_DRIFT = "SEMANTIC_IDENTITY_DRIFT"
    TEACHER_IDENTITY_DRIFT = "TEACHER_IDENTITY_DRIFT"
    H72_IDENTITY_DRIFT = "H72_IDENTITY_DRIFT"
    PRECISION_POLICY_DRIFT = "PRECISION_POLICY_DRIFT"
    FORBIDDEN_GRADIENT = "FORBIDDEN_GRADIENT"
    FROZEN_AUTHORITY_DRIFT = "FROZEN_AUTHORITY_DRIFT"
    FINAL_HOLDOUT_TOUCHED = "FINAL_HOLDOUT_TOUCHED"
    REPLAY_EVIDENCE_VIOLATION = "REPLAY_EVIDENCE_VIOLATION"
    WORKER_DEATH = "WORKER_DEATH"
    PROCESS_DEATH = "PROCESS_DEATH"
    UNEXPECTED_RESTART = "UNEXPECTED_RESTART"
    QUEUE_RUNAWAY = "QUEUE_RUNAWAY"
    RSS_UNBOUNDED_GROWTH = "RSS_UNBOUNDED_GROWTH"
    VRAM_UNBOUNDED_GROWTH = "VRAM_UNBOUNDED_GROWTH"
    THREAD_GROWTH = "THREAD_GROWTH"
    PROCESS_GROWTH = "PROCESS_GROWTH"
    FD_GROWTH = "FD_GROWTH"
    THROUGHPUT_COLLAPSE = "THROUGHPUT_COLLAPSE"
    TELEMETRY_GAP = "TELEMETRY_GAP"
    SWAP_ACTIVITY = "SWAP_ACTIVITY"
    MAJOR_FAULT_SURGE = "MAJOR_FAULT_SURGE"
    HDD_QUEUE_RUNAWAY = "HDD_QUEUE_RUNAWAY"
    FSYNC_LATENCY_REGRESSION = "FSYNC_LATENCY_REGRESSION"
    BARRIER_LATENCY_REGRESSION = "BARRIER_LATENCY_REGRESSION"


@dataclass(frozen=True)
class E1Config:
    duration_seconds: int
    sample_seconds: float = 5.0
    teacher_workers: int = CANONICAL_TEACHER_WORKERS
    trace_workers: int = CANONICAL_TRACE_WORKERS
    queue_depth: int = CANONICAL_QUEUE_DEPTH
    buffer_mib: int = CANONICAL_BUFFER_MIB
    throughput_decay_floor: float = 0.65
    min_windows_for_decay: int = 6
    rss_growth_limit_bytes: int = 512 * 1024 * 1024
    rss_slope_limit_bytes_per_min: float = 64 * 1024 * 1024
    vram_growth_limit_bytes: int = 256 * 1024 * 1024
    vram_slope_limit_bytes_per_min: float = 32 * 1024 * 1024
    fd_growth_limit: int = 32
    thread_growth_limit: int = 4
    process_growth_limit: int = 0
    swap_growth_limit_bytes: int = 64 * 1024 * 1024
    major_faults_per_min_limit: float = 100.0

    def validate(self) -> None:
        if self.duration_seconds <= 0 or self.sample_seconds <= 0:
            raise ValueError("R11_E1_INVALID_DURATION_OR_SAMPLE")
        frozen = {
            "teacher_workers": (self.teacher_workers, CANONICAL_TEACHER_WORKERS),
            "trace_workers": (self.trace_workers, CANONICAL_TRACE_WORKERS),
            "queue_depth": (self.queue_depth, CANONICAL_QUEUE_DEPTH),
            "buffer_mib": (self.buffer_mib, CANONICAL_BUFFER_MIB),
        }
        bad = [f"{k}={a}:EXPECTED={b}" for k, (a, b) in frozen.items() if int(a) != int(b)]
        if bad:
            raise ValueError("R11_E1_STAGE2_DEFAULT_DRIFT:" + ",".join(bad))
        if not 0 < self.throughput_decay_floor <= 1:
            raise ValueError("R11_E1_INVALID_DECAY_FLOOR")


@dataclass(frozen=True)
class CounterSnapshot:
    cycles: int = 0
    traces: int = 0
    optimizer_steps: int = 0
    examples: int = 0
    io_requests: int = 0
    io_bytes_logical: int = 0
    worker_deaths: int = 0
    restart_count: int = 0


@dataclass(frozen=True)
class ResourceSample:
    monotonic_seconds: float
    counters: CounterSnapshot
    rss_bytes: int
    vram_bytes: int | None
    thread_count: int
    process_count: int
    fd_count: int | None
    cpu_utilization_pct: float | None
    gpu_utilization_pct: float | None
    major_faults: int
    swap_in_bytes: int
    swap_out_bytes: int
    queue_depth: int
    queue_capacity: int
    hdd_queue_depth: int | None
    ssd_read_bytes: int | None
    ssd_write_bytes: int | None
    hdd_read_bytes: int | None
    hdd_write_bytes: int | None
    fsync_latency_ms: float | None = None
    barrier_latency_ms: float | None = None
    trace_processes_alive: int | None = None
    storage_writer_alive: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["schema"] = TELEMETRY_SCHEMA
        return out


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _read_status() -> tuple[int, int, int]:
    rss = threads = 0
    major = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    try:
        fields = Path("/proc/self/stat").read_text().split()
        major = int(fields[11])
    except Exception:
        major = 0
    return rss, threads, major


def _children_count() -> int:
    try:
        text = Path(f"/proc/self/task/{os.getpid()}/children").read_text().strip()
        return len(text.split()) if text else 0
    except OSError:
        return 0


def _fd_count() -> int | None:
    try:
        return len(list(Path("/proc/self/fd").iterdir()))
    except OSError:
        return None


def _vmstat() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            k, v = line.split()
            if k in {"pswpin", "pswpout"}:
                values[k] = int(v)
    except OSError:
        pass
    page = os.sysconf("SC_PAGE_SIZE")
    return values.get("pswpin", 0) * page, values.get("pswpout", 0) * page


def _nvidia() -> tuple[int | None, float | None]:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            check=True, capture_output=True, text=True, timeout=3,
        )
        a, b = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
        return int(a) * 1024 * 1024, float(b)
    except Exception:
        return None, None


def block_device_id(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
        return os.major(st.st_dev), os.minor(st.st_dev)
    except OSError:
        return None


def _diskstats(dev: tuple[int, int] | None) -> tuple[int | None, int | None, int | None]:
    if dev is None:
        return None, None, None
    try:
        for line in Path("/proc/diskstats").read_text().splitlines():
            f = line.split()
            if int(f[0]) == dev[0] and int(f[1]) == dev[1]:
                return int(f[5]) * 512, int(f[9]) * 512, int(f[11])
    except (OSError, ValueError, IndexError):
        pass
    return None, None, None


class LinuxResourceSampler:
    def __init__(self, *, ssd_path: Path, hdd_path: Path):
        self.ssd_dev = block_device_id(ssd_path)
        self.hdd_dev = block_device_id(hdd_path)
        self._last_cpu: tuple[int, int] | None = None

    @staticmethod
    def _cpu_ticks() -> tuple[int, int]:
        total = 0
        idle = 0
        try:
            f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            nums = [int(x) for x in f]
            total = sum(nums)
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        except Exception:
            pass
        return total, idle

    def _cpu_pct(self) -> float | None:
        now = self._cpu_ticks()
        old = self._last_cpu
        self._last_cpu = now
        if old is None:
            return None
        dt = now[0] - old[0]
        di = now[1] - old[1]
        if dt <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (dt - di) / dt))

    def sample(
        self,
        *,
        started: float,
        counters: CounterSnapshot,
        queue_depth: int,
        queue_capacity: int,
        fsync_latency_ms: float | None,
        barrier_latency_ms: float | None,
        trace_processes_alive: int | None,
        storage_writer_alive: bool | None,
    ) -> ResourceSample:
        rss, threads, major = _read_status()
        swap_in, swap_out = _vmstat()
        vram, gpu = _nvidia()
        sread, swrite, _sq = _diskstats(self.ssd_dev)
        hread, hwrite, hq = _diskstats(self.hdd_dev)
        return ResourceSample(
            monotonic_seconds=time.monotonic() - started,
            counters=counters,
            rss_bytes=rss,
            vram_bytes=vram,
            thread_count=threads,
            process_count=_children_count(),
            fd_count=_fd_count(),
            cpu_utilization_pct=self._cpu_pct(),
            gpu_utilization_pct=gpu,
            major_faults=major,
            swap_in_bytes=swap_in,
            swap_out_bytes=swap_out,
            queue_depth=int(queue_depth),
            queue_capacity=max(1, int(queue_capacity)),
            hdd_queue_depth=hq,
            ssd_read_bytes=sread,
            ssd_write_bytes=swrite,
            hdd_read_bytes=hread,
            hdd_write_bytes=hwrite,
            fsync_latency_ms=fsync_latency_ms,
            barrier_latency_ms=barrier_latency_ms,
            trace_processes_alive=trace_processes_alive,
            storage_writer_alive=storage_writer_alive,
        )


def percentile(xs: Sequence[float], q: float) -> float | None:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    if len(ys) == 1:
        return ys[0]
    p = (len(ys) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - p) + ys[hi] * (p - lo)


def _rolling_rates(samples: Sequence[ResourceSample], attr: str, width: int = 3) -> list[float]:
    """Counter rates over overlapping rolling windows, not whole-run averages."""
    out: list[float] = []
    if len(samples) < 2:
        return out
    width = max(1, int(width))
    for i in range(1, len(samples)):
        j = max(0, i - width)
        a, b = samples[j], samples[i]
        dt = b.monotonic_seconds - a.monotonic_seconds
        if dt <= 0:
            continue
        av = getattr(a.counters, attr)
        bv = getattr(b.counters, attr)
        out.append(max(0.0, float(bv - av) / dt))
    return out


def _early_late(xs: Sequence[float]) -> tuple[float | None, float | None]:
    if len(xs) < 2:
        return None, None
    n = max(1, len(xs) // 3)
    return statistics.median(xs[:n]), statistics.median(xs[-n:])


def _slope_per_min(samples: Sequence[ResourceSample], attr: str) -> float:
    pts = [(s.monotonic_seconds, getattr(s, attr)) for s in samples if getattr(s, attr) is not None]
    if len(pts) < 2:
        return 0.0
    xbar = sum(x for x, _ in pts) / len(pts)
    ybar = sum(float(y) for _, y in pts) / len(pts)
    denom = sum((x - xbar) ** 2 for x, _ in pts)
    if denom <= 0:
        return 0.0
    return 60.0 * sum((x - xbar) * (float(y) - ybar) for x, y in pts) / denom


def _growth(samples: Sequence[ResourceSample], attr: str) -> float:
    vals = [float(getattr(s, attr)) for s in samples if getattr(s, attr) is not None]
    return vals[-1] - vals[0] if len(vals) >= 2 else 0.0


def _latency_regression(samples: Sequence[ResourceSample], attr: str) -> tuple[bool, dict[str, Any]]:
    vals = [float(getattr(s, attr)) for s in samples if getattr(s, attr) is not None]
    early, late = _early_late(vals)
    fail = bool(early is not None and late is not None and late > max(25.0, early * 3.0))
    return fail, {"early_median_ms": early, "late_median_ms": late, "p90_ms": percentile(vals, .90)}


def analyze_endurance(samples: Sequence[ResourceSample], config: E1Config) -> dict[str, Any]:
    config.validate()
    failures: list[str] = []
    if len(samples) < 2:
        failures.append(E1Failure.TELEMETRY_GAP.value)
        return {"verdict": "FAIL", "failures": failures, "sample_count": len(samples)}

    throughput: dict[str, Any] = {}
    for attr, scale, label in (
        ("cycles", 60.0, "engineering_cycles_per_min"),
        ("traces", 1.0, "traces_per_sec"),
        ("optimizer_steps", 1.0, "fp32_optimizer_steps_per_sec"),
        ("examples", 1.0, "examples_per_sec"),
    ):
        rates = [r * scale for r in _rolling_rates(samples, attr, width=3)]
        early, late = _early_late(rates)
        decay = None if not early or late is None else late / early
        throughput[label] = {
            "median": statistics.median(rates) if rates else 0.0,
            "p10": percentile(rates, .10),
            "p90": percentile(rates, .90),
            "early_median": early,
            "late_median": late,
            "late_over_early": decay,
        }
        if len(rates) >= config.min_windows_for_decay:
            pe, pl = _early_late(rates)
            if pe is not None and pe > 0 and pl is not None and pl / pe < config.throughput_decay_floor:
                failures.append(E1Failure.THROUGHPUT_COLLAPSE.value)

    rss_growth = _growth(samples, "rss_bytes")
    rss_slope = _slope_per_min(samples, "rss_bytes")
    if rss_growth > config.rss_growth_limit_bytes and rss_slope > config.rss_slope_limit_bytes_per_min:
        failures.append(E1Failure.RSS_UNBOUNDED_GROWTH.value)

    vram_growth = _growth(samples, "vram_bytes")
    vram_slope = _slope_per_min(samples, "vram_bytes")
    if vram_growth > config.vram_growth_limit_bytes and vram_slope > config.vram_slope_limit_bytes_per_min:
        failures.append(E1Failure.VRAM_UNBOUNDED_GROWTH.value)

    thread_growth = _growth(samples, "thread_count")
    proc_growth = _growth(samples, "process_count")
    fd_growth = _growth(samples, "fd_count")
    if thread_growth > config.thread_growth_limit:
        failures.append(E1Failure.THREAD_GROWTH.value)
    if proc_growth > config.process_growth_limit:
        failures.append(E1Failure.PROCESS_GROWTH.value)
    if fd_growth > config.fd_growth_limit:
        failures.append(E1Failure.FD_GROWTH.value)

    if any(s.storage_writer_alive is False for s in samples):
        failures.append(E1Failure.WORKER_DEATH.value)
    alive = [s.trace_processes_alive for s in samples if s.trace_processes_alive is not None]
    if alive and min(alive) < config.trace_workers:
        failures.append(E1Failure.PROCESS_DEATH.value)
    if samples[-1].counters.restart_count != samples[0].counters.restart_count:
        failures.append(E1Failure.UNEXPECTED_RESTART.value)
    if samples[-1].counters.worker_deaths != samples[0].counters.worker_deaths:
        failures.append(E1Failure.WORKER_DEATH.value)

    qfrac = [s.queue_depth / max(1, s.queue_capacity) for s in samples]
    if len(qfrac) >= 3 and all(x >= .95 for x in qfrac[-3:]):
        failures.append(E1Failure.QUEUE_RUNAWAY.value)

    hq = [float(s.hdd_queue_depth) for s in samples if s.hdd_queue_depth is not None]
    he, hl = _early_late(hq)
    if he is not None and hl is not None and hl > max(8.0, he * 2.0):
        failures.append(E1Failure.HDD_QUEUE_RUNAWAY.value)

    swap_growth = max(
        samples[-1].swap_in_bytes - samples[0].swap_in_bytes,
        samples[-1].swap_out_bytes - samples[0].swap_out_bytes,
    )
    if swap_growth > config.swap_growth_limit_bytes:
        failures.append(E1Failure.SWAP_ACTIVITY.value)

    elapsed_min = max((samples[-1].monotonic_seconds - samples[0].monotonic_seconds) / 60.0, 1e-9)
    major_rate = (samples[-1].major_faults - samples[0].major_faults) / elapsed_min
    if major_rate > config.major_faults_per_min_limit:
        failures.append(E1Failure.MAJOR_FAULT_SURGE.value)

    fsync_fail, fsync = _latency_regression(samples, "fsync_latency_ms")
    barrier_fail, barrier = _latency_regression(samples, "barrier_latency_ms")
    if fsync_fail:
        failures.append(E1Failure.FSYNC_LATENCY_REGRESSION.value)
    if barrier_fail:
        failures.append(E1Failure.BARRIER_LATENCY_REGRESSION.value)

    failures = sorted(set(failures))
    return {
        "verdict": "PASS" if not failures else "FAIL",
        "failures": failures,
        "sample_count": len(samples),
        "throughput": throughput,
        "rolling_window_samples": 3,
        "resource_trends": {
            "rss_growth_bytes": rss_growth,
            "rss_slope_bytes_per_min": rss_slope,
            "vram_growth_bytes": vram_growth,
            "vram_slope_bytes_per_min": vram_slope,
            "thread_growth": thread_growth,
            "process_growth": proc_growth,
            "fd_growth": fd_growth,
            "swap_growth_bytes": swap_growth,
            "major_faults_per_min": major_rate,
            "queue_fraction_p90": percentile(qfrac, .90),
            "hdd_queue_depth_p90": percentile(hq, .90),
        },
        "latency": {"fsync_commit": fsync, "durability_barrier": barrier},
        "gpu_utilization_role": "STARVATION_CANARY_ONLY__NOT_PRIMARY_ACCEPTANCE",
    }


def make_report(
    *,
    config: E1Config,
    samples: Sequence[ResourceSample],
    correctness_checks: Mapping[str, bool],
    identities: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    correctness_failures: list[str] = []
    mapping = {
        "semantic_freeze_pass": E1Failure.SEMANTIC_IDENTITY_DRIFT,
        "teacher_identity_unchanged": E1Failure.TEACHER_IDENTITY_DRIFT,
        "h72_identity_unchanged": E1Failure.H72_IDENTITY_DRIFT,
        "fp32_amp_false": E1Failure.PRECISION_POLICY_DRIFT,
        "gradient_authority_unchanged": E1Failure.FORBIDDEN_GRADIENT,
        "frozen_authority_unchanged": E1Failure.FROZEN_AUTHORITY_DRIFT,
        "final_holdout_untouched": E1Failure.FINAL_HOLDOUT_TOUCHED,
        "replay_created_zero": E1Failure.REPLAY_EVIDENCE_VIOLATION,
    }
    for key, failure in mapping.items():
        if correctness_checks.get(key) is not True:
            correctness_failures.append(failure.value)
    correctness = {
        "verdict": "PASS" if not correctness_failures else "FAIL",
        "failures": sorted(set(correctness_failures)),
        "checks": dict(correctness_checks),
        "scientific_semantics_changed": False,
        "new_scientific_verdict": False,
        "repeated_replay_role": "ENGINEERING_LOAD_ONLY__NOT_SCIENTIFIC_EVIDENCE",
    }
    endurance = analyze_endurance(samples, config)
    status = "PASS" if correctness["verdict"] == "PASS" and endurance["verdict"] == "PASS" else "FAIL"
    ready_30m = status == "PASS" and config.duration_seconds < 1800
    report = {
        "schema": SCHEMA,
        "status": status,
        "correctness_identity": correctness,
        "runtime_endurance": endurance,
        "identities": dict(identities),
        "accepted_runtime_defaults": {
            "teacher_workers": config.teacher_workers,
            "trace_workers": config.trace_workers,
            "queue_depth": config.queue_depth,
            "buffer_mib": config.buffer_mib,
            "nested_threads": 1,
        },
        "runtime_lifecycle": {
            "teacher_pool_persistent": True,
            "trace_process_pool_persistent": True,
            "training_runtime_persistent": True,
            "io_runtime_persistent": True,
            "authority_model_mutated": False,
            "tournament_or_lineage_exercised": False,
        },
        "failure_taxonomy": [x.value for x in E1Failure],
        "telemetry": [s.as_dict() for s in samples],
        "readiness": {
            "ready_for_30m": ready_30m,
            "ready_for_2h": False,
            "ready_for_6h_plus": False,
            "reason": "NEXT_ALLOWED_RUNG_30M_AFTER_SHORT_SMOKE_PASS"
            if ready_30m else ("30M_RUNG_COMPLETE_OR_NOT_APPLICABLE" if status == "PASS" else "E1_FAILURES_MUST_BE_RESOLVED"),
        },
    }
    if extra:
        report["engineering"] = dict(extra)
    return report


class PersistentTeacherReplayR11:
    """Prepare immutable Teacher geometry once and reuse one worker pool."""

    def __init__(self, *, samples, parents, train_config, val_config, workers: int = 8, block_targets: int = 64):
        from . import teacher_scheduler_r11 as t
        self._t = t
        self.parents = parents
        self.workers = int(workers)
        self.pool = t.TeacherWorkerPoolR11(max_workers=self.workers)
        self.index = t._make_index_immutable_r11(t.build_columnar_teacher_index_r11(samples))
        train_ids, val_ids = t._canonical_target_ids_r11(parents=parents, index=self.index)
        self.train_ids, self.val_ids = train_ids, val_ids
        jobs, regime_count = t._build_jobs_r11(
            train_parent_ids=train_ids, val_parent_ids=val_ids, index=self.index,
            parents=parents, train_config=train_config, val_config=val_config,
            block_targets=int(block_targets),
        )
        self.jobs = tuple(jobs)
        self.regime_count = int(regime_count)
        self._closed = False

    @property
    def pool_identity(self) -> int:
        return self.pool.executor_identity

    def compile(self):
        if self._closed:
            raise RuntimeError("R11_E1_TEACHER_RUNTIME_CLOSED")
        t = self._t
        with t._single_thread_blas_r11(workers=self.workers):
            blocks = t._execute_jobs_r11(
                jobs=self.jobs, index=self.index, workers=self.workers, worker_pool=self.pool
            )
        compiled = {}
        for rows in blocks:
            for e in rows:
                if e.parent_id in compiled:
                    raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{e.parent_id}")
                compiled[e.parent_id] = e
        expected = self.train_ids + self.val_ids
        if set(compiled) != set(expected):
            raise RuntimeError("R11_E1_TEACHER_TARGET_SET_DRIFT")
        train = [compiled[x] for x in self.train_ids]
        val = [compiled[x] for x in self.val_ids]
        return train, val

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.pool.close()
