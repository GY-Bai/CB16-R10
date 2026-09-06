from __future__ import annotations

"""Adaptive bounded process farm for R10 probabilistic Teacher compilation.

R8.2 starts six single-threaded Teacher workers. Scheduling is deterministic by
ordinal reordering. RAM pressure may retire workers or terminate one active pure
Teacher job and requeue it; at least one worker is retained.
"""

import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence

from .probabilistic_teacher_r6 import DependenceAwareProbabilisticTeacherR6
from .r102_parallel_runtime import ram_action_r82, sample_ram_pressure_r82

_TRAIN_TEACHER = None
_VAL_TEACHER = None
_TRAIN_INDEX = None
_VAL_INDEX = None
_TRAIN_GROUPS = None


def _set_threads(threads: int) -> None:
    n = str(int(threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n


def _teacher_init(samples, train_groups, train_config, val_config, threads: int) -> None:
    global _TRAIN_TEACHER, _VAL_TEACHER, _TRAIN_INDEX, _VAL_INDEX, _TRAIN_GROUPS
    _set_threads(threads)
    _TRAIN_GROUPS = set(train_groups)
    _TRAIN_TEACHER = DependenceAwareProbabilisticTeacherR6(train_config)
    _VAL_TEACHER = DependenceAwareProbabilisticTeacherR6(val_config)
    _TRAIN_INDEX = _TRAIN_TEACHER.index(samples)
    _VAL_INDEX = _VAL_TEACHER.index(samples)


def _compile_one(job: tuple[int, str, str]):
    ordinal, lane, parent_id = job
    if lane == "TRAIN":
        teacher = _TRAIN_TEACHER
        index = _TRAIN_INDEX
    elif lane == "VALIDATION":
        teacher = _VAL_TEACHER
        index = _VAL_INDEX
    else:
        raise RuntimeError(f"UNKNOWN_TEACHER_LANE:{lane}")
    if teacher is None or index is None or _TRAIN_GROUPS is None:
        raise RuntimeError("R102_TEACHER_WORKER_NOT_INITIALIZED")
    if parent_id not in index.rows_by_parent:
        return ordinal, lane, parent_id, None
    evidence = teacher.compile_one(
        target_parent=parent_id,
        index=index,
        eligible_train_dependence_groups=_TRAIN_GROUPS,
    )
    return ordinal, lane, parent_id, evidence


def _pid_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return -1


def _teacher_worker_loop(
    job_q,
    result_q,
    samples,
    train_groups,
    train_config,
    val_config,
    threads_per_worker: int,
) -> None:
    _teacher_init(samples, train_groups, train_config, val_config, threads_per_worker)
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
            value = _compile_one(job)
        except BaseException as exc:
            result_q.put(("ERROR", pid, idx, (type(exc).__name__, str(exc), traceback.format_exc())))
            return
        result_q.put(("DONE", pid, idx, value))


def compile_teacher_evidence_process_farm_r102(
    *,
    samples,
    train_parent_ids: Sequence[str],
    val_parent_ids: Sequence[str],
    train_groups: Iterable[str],
    train_config,
    val_config,
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
    ram_backpressure_high: float = 0.85,
    ram_hard_stop: float = 0.92,
    ram_poll_seconds: float = 0.5,
    ram_retire_cooldown_seconds: float = 2.0,
):
    workers = int(workers)
    threads_per_worker = int(threads_per_worker)
    max_in_flight = int(max_in_flight)
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("R102_INVALID_TEACHER_WORKER_CONFIG")
    if max_in_flight < workers:
        raise ValueError("R102_TEACHER_MAX_IN_FLIGHT_LT_WORKERS")
    if not (0.0 < float(ram_backpressure_high) < float(ram_hard_stop) < 1.0):
        raise ValueError("R82_INVALID_TEACHER_RAM_THRESHOLDS")

    jobs: list[tuple[int, str, str]] = []
    for p in train_parent_ids:
        jobs.append((len(jobs), "TRAIN", p))
    for p in val_parent_ids:
        jobs.append((len(jobs), "VALIDATION", p))
    if not jobs:
        return [], []

    if workers == 1:
        _teacher_init(samples, set(train_groups), train_config, val_config, threads_per_worker)
        raw = [_compile_one(j) for j in jobs]
    else:
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
        last_ram_action_at = 0.0
        backpressure = False
        retirement_tokens = 0

        def start_one() -> None:
            p = ctx.Process(
                target=_teacher_worker_loop,
                args=(
                    job_q,
                    result_q,
                    samples,
                    set(train_groups),
                    train_config,
                    val_config,
                    threads_per_worker,
                ),
                daemon=False,
            )
            p.start()
            procs[p.pid] = p

        for _ in range(workers):
            start_one()

        def effective_workers() -> int:
            return max(0, len(procs) - retirement_tokens)

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
        try:
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
                        if retirement_tokens > 0:
                            retirement_tokens -= 1
                        if p is not None:
                            expected_exit.add(int(pid))
                            p.join(timeout=1.0)
                        continue
                    if kind == "ERROR":
                        active_by_pid.pop(int(pid), None)
                        name, message, tb = payload
                        raise RuntimeError(f"R102_TEACHER_WORKER_ERROR:{name}:{message}\n{tb}")
                    raise RuntimeError(f"R102_UNKNOWN_TEACHER_WORKER_MESSAGE:{kind}")

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
                    raise RuntimeError(f"R102_TEACHER_WORKER_UNEXPECTED_EXIT:{pid}:{p.exitcode}")

                now = time.monotonic()
                sample = sample_ram_pressure_r82()
                current_effective = effective_workers()
                action = ram_action_r82(
                    sample.pressure,
                    current_effective,
                    high=float(ram_backpressure_high),
                    hard=float(ram_hard_stop),
                )
                backpressure = sample.pressure >= float(ram_backpressure_high)

                if action in {"RETIRE_ONE", "KILL_ONE"} and now - last_ram_action_at >= float(ram_retire_cooldown_seconds):
                    last_ram_action_at = now
                    if action == "KILL_ONE" and active_by_pid and current_effective > 1:
                        victim = max(active_by_pid, key=_pid_rss_bytes)
                        idx = active_by_pid.pop(victim)
                        retry.appendleft(idx)
                        p = procs.pop(victim, None)
                        if p is not None and p.is_alive():
                            expected_exit.add(victim)
                            p.terminate()
                            p.join(timeout=5.0)
                        print(
                            f"[R10_TEACHER_RAM_MONITOR] hard pressure={sample.pressure:.4f}; "
                            f"killed_worker={victim}; requeued_job={idx}; effective_workers={effective_workers()}",
                            flush=True,
                        )
                    elif current_effective > 1:
                        job_q.put(None)
                        retirement_tokens += 1
                        print(
                            f"[R10_TEACHER_RAM_MONITOR] high pressure={sample.pressure:.4f}; "
                            f"requested_worker_retirement; target_effective_workers={effective_workers()}",
                            flush=True,
                        )

                fill_queue()
                if not drained:
                    time.sleep(float(ram_poll_seconds))
                if effective_workers() <= 0 and len(results) < len(jobs):
                    raise RuntimeError("R102_ALL_TEACHER_WORKERS_EXITED")

            raw = [results[i] for i in range(len(jobs))]
        finally:
            for _ in range(max(0, len(procs) - retirement_tokens)):
                try:
                    job_q.put_nowait(None)
                except queue.Full:
                    break
            deadline = time.monotonic() + 10.0
            for p in list(procs.values()):
                remaining = max(0.0, deadline - time.monotonic())
                p.join(timeout=remaining)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=2.0)
            job_q.close()
            result_q.close()

    train_e = []
    val_e = []
    for _, lane, _, evidence in raw:
        if evidence is None:
            continue
        (train_e if lane == "TRAIN" else val_e).append(evidence)
    return train_e, val_e
