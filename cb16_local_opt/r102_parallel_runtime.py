from __future__ import annotations

"""Adaptive bounded CPU parallelism for R10 real-historical H72 work.

R8.2 execution policy:
- start with six CPU workers for H72 replay;
- one CPU thread per worker;
- the main process remains the single CUDA owner;
- canonical output ordering is independent of scheduling;
- RAM pressure is monitored from /proc/meminfo and the current cgroup;
- at the high watermark, dispatch pauses and one worker is gracefully retired;
- at the hard watermark, one active worker may be terminated and its pure H72 job requeued;
- at least one worker is always retained.

Worker-count changes are runtime scheduling only and must never invalidate an
already materialized scientific cache.
"""

import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r102_physics import CANDIDATES_R102, FrozenPhysicsRuntimeR102, simulate_h72_branch

_H72_RUNTIME: FrozenPhysicsRuntimeR102 | None = None
_H72_SYMBOL: str | None = None
_H72_TS: np.ndarray | None = None
_H72_BARS: np.ndarray | None = None
_H72_FUNDING: np.ndarray | None = None


def _set_worker_threads(threads: int) -> None:
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


def _h72_worker_init(
    package_root: str,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    threads: int,
) -> None:
    global _H72_RUNTIME, _H72_SYMBOL, _H72_TS, _H72_BARS, _H72_FUNDING
    _set_worker_threads(threads)
    _H72_RUNTIME = FrozenPhysicsRuntimeR102.load(package_root)
    _H72_SYMBOL = str(symbol)
    _H72_TS = np.asarray(hourly_ts)
    _H72_BARS = np.asarray(hourly_ohlcv)
    _H72_FUNDING = np.asarray(funding)


def _require_worker_state() -> tuple[FrozenPhysicsRuntimeR102, str, np.ndarray, np.ndarray, np.ndarray]:
    if _H72_RUNTIME is None or _H72_SYMBOL is None or _H72_TS is None or _H72_BARS is None or _H72_FUNDING is None:
        raise RuntimeError("R102_H72_WORKER_NOT_INITIALIZED")
    return _H72_RUNTIME, _H72_SYMBOL, _H72_TS, _H72_BARS, _H72_FUNDING


@dataclass(frozen=True)
class H72ParentGroupJobR102:
    ordinal: int
    parent_id: str
    parent: Mapping[str, Any]
    decision_time_ms: int


@dataclass(frozen=True)
class H72ActionJobR102:
    ordinal: int
    parent_id: str
    parent: Mapping[str, Any]
    decision_time_ms: int
    candidate_direction_v55: int
    candidate_risk: float


@dataclass(frozen=True)
class RamPressureSampleR82:
    pressure: float
    host_pressure: float | None
    cgroup_pressure: float | None
    host_total_bytes: int | None
    host_available_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_max_bytes: int | None


def _counterfactual_worker(job: H72ParentGroupJobR102) -> tuple[int, str, list[tuple[int, float, dict[str, Any]]]]:
    runtime, symbol, hts, bars, funding = _require_worker_state()
    out: list[tuple[int, float, dict[str, Any]]] = []
    for direction, risk in CANDIDATES_R102:
        result = simulate_h72_branch(
            runtime,
            parent=job.parent,
            symbol=symbol,
            decision_time_ms=int(job.decision_time_ms),
            candidate_direction_v55=int(direction),
            candidate_risk=float(risk),
            hourly_ts=hts,
            hourly_ohlcv=bars,
            funding=funding,
        )
        out.append((int(direction), float(risk), result))
    return job.ordinal, job.parent_id, out


def _single_action_worker(job: H72ActionJobR102) -> tuple[int, str, dict[str, Any]]:
    runtime, symbol, hts, bars, funding = _require_worker_state()
    result = simulate_h72_branch(
        runtime,
        parent=job.parent,
        symbol=symbol,
        decision_time_ms=int(job.decision_time_ms),
        candidate_direction_v55=int(job.candidate_direction_v55),
        candidate_risk=float(job.candidate_risk),
        hourly_ts=hts,
        hourly_ohlcv=bars,
        funding=funding,
    )
    return job.ordinal, job.parent_id, result


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _current_cgroup_dir() -> Path | None:
    try:
        rows = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for row in rows:
        parts = row.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            rel = parts[2].lstrip("/")
            return Path("/sys/fs/cgroup") / rel
    return None


def sample_ram_pressure_r82() -> RamPressureSampleR82:
    host_total = None
    host_avail = None
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            fields = v.strip().split()
            if fields:
                info[k] = int(fields[0]) * 1024
        host_total = info.get("MemTotal")
        host_avail = info.get("MemAvailable")
    except (OSError, ValueError):
        pass
    host_pressure = None
    if host_total and host_avail is not None and host_total > 0:
        host_pressure = max(0.0, min(1.0, 1.0 - host_avail / host_total))

    cg_dir = _current_cgroup_dir()
    cg_current = _read_int(cg_dir / "memory.current") if cg_dir else None
    cg_max = _read_int(cg_dir / "memory.max") if cg_dir else None
    cgroup_pressure = None
    if cg_current is not None and cg_max is not None and cg_max > 0:
        cgroup_pressure = max(0.0, min(1.0, cg_current / cg_max))

    candidates = [x for x in (host_pressure, cgroup_pressure) if x is not None]
    pressure = max(candidates) if candidates else 0.0
    return RamPressureSampleR82(
        pressure=float(pressure),
        host_pressure=host_pressure,
        cgroup_pressure=cgroup_pressure,
        host_total_bytes=host_total,
        host_available_bytes=host_avail,
        cgroup_current_bytes=cg_current,
        cgroup_max_bytes=cg_max,
    )


def ram_action_r82(
    pressure: float,
    current_workers: int,
    *,
    high: float = 0.85,
    hard: float = 0.92,
) -> str:
    if not (0.0 < high < hard < 1.0):
        raise ValueError("R82_INVALID_RAM_THRESHOLDS")
    if int(current_workers) <= 1:
        return "PAUSE" if pressure >= high else "NONE"
    if pressure >= hard:
        return "KILL_ONE"
    if pressure >= high:
        return "RETIRE_ONE"
    return "NONE"


def _pid_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return -1


def _adaptive_worker_loop(
    worker_fn,
    job_q,
    result_q,
    package_root: str,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    threads_per_worker: int,
) -> None:
    _h72_worker_init(
        package_root,
        symbol,
        hourly_ts,
        hourly_ohlcv,
        funding,
        threads_per_worker,
    )
    pid = os.getpid()
    result_q.put(("READY", pid, None, None))
    while True:
        item = job_q.get()
        if item is None:
            result_q.put(("RETIRED", pid, None, None))
            return
        idx, job = item
        result_q.put(("START", pid, idx, None))
        try:
            value = worker_fn(job)
        except BaseException as exc:
            result_q.put(("ERROR", pid, idx, (type(exc).__name__, str(exc), traceback.format_exc())))
            return
        result_q.put(("DONE", pid, idx, value))


def _bounded_process_map(
    worker_fn,
    jobs: Sequence[Any],
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
    ram_backpressure_high: float = 0.85,
    ram_hard_stop: float = 0.92,
    ram_poll_seconds: float = 0.5,
    ram_retire_cooldown_seconds: float = 2.0,
) -> list[Any]:
    if not jobs:
        return []
    workers = int(workers)
    threads_per_worker = int(threads_per_worker)
    max_in_flight = int(max_in_flight)
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("R102_INVALID_H72_WORKER_CONFIG")
    if max_in_flight < workers:
        raise ValueError("R102_H72_MAX_IN_FLIGHT_LT_WORKERS")
    if not (0.0 < float(ram_backpressure_high) < float(ram_hard_stop) < 1.0):
        raise ValueError("R82_INVALID_RAM_THRESHOLDS")

    if workers == 1:
        _h72_worker_init(
            str(Path(package_root).resolve()), symbol, hourly_ts, hourly_ohlcv, funding, threads_per_worker
        )
        return [worker_fn(j) for j in jobs]

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue(maxsize=max_in_flight)
    result_q = ctx.Queue()
    procs: dict[int, mp.Process] = {}
    expected_exit: set[int] = set()
    active_by_pid: dict[int, int] = {}
    queued_indices: set[int] = set()
    retry = deque()
    results: dict[int, Any] = {}
    next_i = 0
    desired_workers = workers
    last_ram_action_at = 0.0
    backpressure = False

    def start_one() -> None:
        p = ctx.Process(
            target=_adaptive_worker_loop,
            args=(
                worker_fn,
                job_q,
                result_q,
                str(Path(package_root).resolve()),
                str(symbol),
                np.asarray(hourly_ts),
                np.asarray(hourly_ohlcv),
                np.asarray(funding),
                threads_per_worker,
            ),
            daemon=False,
        )
        p.start()
        procs[p.pid] = p

    for _ in range(workers):
        start_one()

    def pending_count() -> int:
        return len(queued_indices) + len(active_by_pid)

    def queue_job(idx: int) -> None:
        if idx in results or idx in queued_indices or idx in active_by_pid.values():
            return
        job_q.put((idx, jobs[idx]))
        queued_indices.add(idx)

    def fill_queue() -> None:
        nonlocal next_i
        if backpressure:
            return
        while pending_count() < max_in_flight and (retry or next_i < len(jobs)):
            if retry:
                idx = retry.popleft()
            else:
                idx = next_i
                next_i += 1
            queue_job(idx)

    fill_queue()

    while len(results) < len(jobs):
        drained = False
        while True:
            try:
                kind, pid, idx, payload = result_q.get_nowait()
            except queue.Empty:
                break
            drained = True
            if kind == "READY":
                continue
            if kind == "START":
                queued_indices.discard(int(idx))
                active_by_pid[int(pid)] = int(idx)
                continue
            if kind == "DONE":
                idx = int(idx)
                active_by_pid.pop(int(pid), None)
                queued_indices.discard(idx)
                results.setdefault(idx, payload)
                continue
            if kind == "RETIRED":
                active_by_pid.pop(int(pid), None)
                p = procs.pop(int(pid), None)
                if p is not None:
                    expected_exit.add(int(pid))
                    p.join(timeout=1.0)
                continue
            if kind == "ERROR":
                active_by_pid.pop(int(pid), None)
                name, message, tb = payload
                raise RuntimeError(f"R102_H72_WORKER_ERROR:{name}:{message}\n{tb}")
            raise RuntimeError(f"R102_UNKNOWN_H72_WORKER_MESSAGE:{kind}")

        for pid, p in list(procs.items()):
            if p.is_alive():
                continue
            p.join(timeout=0.1)
            if pid in expected_exit or p.exitcode == 0:
                procs.pop(pid, None)
                active_by_pid.pop(pid, None)
                continue
            idx = active_by_pid.pop(pid, None)
            if idx is not None:
                retry.appendleft(idx)
            procs.pop(pid, None)
            raise RuntimeError(f"R102_H72_WORKER_UNEXPECTED_EXIT:{pid}:{p.exitcode}")

        now = time.monotonic()
        sample = sample_ram_pressure_r82()
        action = ram_action_r82(
            sample.pressure,
            len(procs),
            high=float(ram_backpressure_high),
            hard=float(ram_hard_stop),
        )
        backpressure = sample.pressure >= float(ram_backpressure_high)

        if action in {"RETIRE_ONE", "KILL_ONE"} and now - last_ram_action_at >= float(ram_retire_cooldown_seconds):
            last_ram_action_at = now
            desired_workers = max(1, desired_workers - 1)
            if action == "KILL_ONE" and active_by_pid:
                victim = max(active_by_pid, key=_pid_rss_bytes)
                idx = active_by_pid.pop(victim)
                retry.appendleft(idx)
                p = procs.pop(victim, None)
                if p is not None and p.is_alive():
                    expected_exit.add(victim)
                    p.terminate()
                    p.join(timeout=5.0)
                print(
                    f"[R10_RAM_MONITOR] hard pressure={sample.pressure:.4f}; "
                    f"killed_worker={victim}; requeued_job={idx}; workers={len(procs)}",
                    flush=True,
                )
            else:
                job_q.put(None)
                print(
                    f"[R10_RAM_MONITOR] high pressure={sample.pressure:.4f}; "
                    f"requested_worker_retirement; target_workers={desired_workers}",
                    flush=True,
                )

        fill_queue()

        if not drained:
            time.sleep(float(ram_poll_seconds))

        if not procs and len(results) < len(jobs):
            raise RuntimeError("R102_ALL_H72_WORKERS_EXITED")

    for _ in range(len(procs)):
        job_q.put(None)
    deadline = time.monotonic() + 10.0
    for pid, p in list(procs.items()):
        remaining = max(0.0, deadline - time.monotonic())
        p.join(timeout=remaining)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
    job_q.close()
    result_q.close()
    return [results[i] for i in range(len(jobs))]


def run_counterfactual_h72_farm_r102(
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    jobs: Sequence[H72ParentGroupJobR102],
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
    ram_backpressure_high: float = 0.85,
    ram_hard_stop: float = 0.92,
) -> list[tuple[int, str, list[tuple[int, float, dict[str, Any]]]]]:
    return _bounded_process_map(
        _counterfactual_worker,
        jobs,
        package_root=package_root,
        symbol=symbol,
        hourly_ts=hourly_ts,
        hourly_ohlcv=hourly_ohlcv,
        funding=funding,
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_in_flight=max_in_flight,
        ram_backpressure_high=ram_backpressure_high,
        ram_hard_stop=ram_hard_stop,
    )


def run_single_action_h72_farm_r102(
    *,
    package_root: str | Path,
    symbol: str,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    jobs: Sequence[H72ActionJobR102],
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
    ram_backpressure_high: float = 0.85,
    ram_hard_stop: float = 0.92,
) -> list[tuple[int, str, dict[str, Any]]]:
    return _bounded_process_map(
        _single_action_worker,
        jobs,
        package_root=package_root,
        symbol=symbol,
        hourly_ts=hourly_ts,
        hourly_ohlcv=hourly_ohlcv,
        funding=funding,
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_in_flight=max_in_flight,
        ram_backpressure_high=ram_backpressure_high,
        ram_hard_stop=ram_hard_stop,
    )
