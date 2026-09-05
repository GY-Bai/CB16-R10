from __future__ import annotations

"""Bounded R8.1-qualified process farm for R10 probabilistic Teacher compilation."""

import concurrent.futures as cf
import multiprocessing as mp
import os
from typing import Any, Iterable, Sequence

from .probabilistic_teacher_r6 import DependenceAwareProbabilisticTeacherR6

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
):
    workers = int(workers)
    threads_per_worker = int(threads_per_worker)
    max_in_flight = int(max_in_flight)
    if workers <= 0 or threads_per_worker <= 0:
        raise ValueError("R102_INVALID_TEACHER_WORKER_CONFIG")
    if max_in_flight < workers:
        raise ValueError("R102_TEACHER_MAX_IN_FLIGHT_LT_WORKERS")

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
        pending: dict[cf.Future, int] = {}
        results: dict[int, Any] = {}
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_teacher_init,
            initargs=(samples, set(train_groups), train_config, val_config, threads_per_worker),
        ) as ex:
            next_i = 0
            while next_i < len(jobs) or pending:
                while next_i < len(jobs) and len(pending) < max_in_flight:
                    f = ex.submit(_compile_one, jobs[next_i])
                    pending[f] = next_i
                    next_i += 1
                done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    idx = pending.pop(f)
                    results[idx] = f.result()
        raw = [results[i] for i in range(len(jobs))]

    train_e = []
    val_e = []
    for _, lane, _, evidence in raw:
        if evidence is None:
            continue
        (train_e if lane == "TRAIN" else val_e).append(evidence)
    return train_e, val_e
