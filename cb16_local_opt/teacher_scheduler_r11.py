from __future__ import annotations

"""Shared-columnar production scheduler for the frozen R11 probabilistic Teacher.

The scheduler uses threads so every worker shares one immutable
:class:`ColumnarTeacherIndexR11`; only per-target-block scratch is worker-local.
Scientific evidence is independent of worker topology. Stage-2 adds a reusable,
bounded-in-flight thread pool to remove repeated executor/future overhead without
changing target grouping, support geometry, Teacher mathematics, or output identity.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import RLock
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

from .cpu_runtime_r11 import CANONICAL_TEACHER_WORKERS_R11
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6, DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .teacher_vectorized_r11 import (
    ColumnarTeacherIndexR11,
    SupportRegimeR11,
    VectorizedTeacherStatsR11,
    _compile_block_r11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
    prepare_support_regime_r11,
)

R11_THREADED_SCHEDULER = "CB16_R11_SHARED_COLUMNAR_THREAD_SCHEDULER_V3"

_INDEX_ARRAY_FIELDS = (
    "parent_timestamps", "parent_dep_index", "features", "utilities",
    "action_directions", "action_risks", "dep_timestamps", "dep_parent_rows",
)
_REGIME_ARRAY_FIELDS = (
    "dep_rows", "mean", "std", "support_parent_rows", "normalized_support_flat",
    "normalized_support_norm2", "valid_support_flat", "dep_lex_rank",
)


@dataclass(frozen=True)
class ThreadedTeacherStatsR11:
    core: VectorizedTeacherStatsR11
    workers: int
    scheduler: str = R11_THREADED_SCHEDULER
    memory_model: str = "ONE_SHARED_IMMUTABLE_COLUMNAR_INDEX__THREAD_LOCAL_TARGET_BLOCK_SCRATCH"
    nested_blas_threads_required: int = 1
    nested_blas_limit_enforced: bool = True
    completion_collection: str = "BOUNDED_IN_FLIGHT__CANONICAL_SLOT_REASSEMBLY"
    topology_in_scientific_identity: bool = False
    persistent_pool_supplied: bool = False
    max_in_flight: int = 1


@dataclass(frozen=True)
class _TeacherBlockJobR11:
    ordinal: int
    split: str
    target_parent_ids: tuple[str, ...]
    regime: SupportRegimeR11
    config: DependenceAwareTeacherConfigR6


class TeacherWorkerPoolR11:
    """Reusable Teacher threads that own no scientific state."""

    def __init__(self, max_workers: int = CANONICAL_TEACHER_WORKERS_R11) -> None:
        max_workers = int(max_workers)
        if max_workers <= 0:
            raise ValueError("R11_TEACHER_POOL_MAX_WORKERS_MUST_BE_POSITIVE")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cb16-r11-teacher")
        self._lock = RLock()
        self._closed = False
        self._executions = 0

    @property
    def executions(self) -> int:
        with self._lock:
            return int(self._executions)

    @property
    def executor_identity(self) -> int:
        return id(self._executor)

    def execute(self, *, jobs, index: ColumnarTeacherIndexR11, workers: int):
        workers = int(workers)
        if workers <= 0 or workers > self.max_workers:
            raise ValueError(f"R11_TEACHER_POOL_WORKERS_OUT_OF_RANGE:{workers}:MAX={self.max_workers}")
        with self._lock:
            if self._closed:
                raise RuntimeError("R11_TEACHER_POOL_CLOSED")
            self._executions += 1
            return _execute_jobs_with_executor_r11(jobs=jobs, index=index, workers=workers, executor=self._executor)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _set_arrays_readonly_r11(obj: object, fields: Sequence[str]) -> None:
    for name in fields:
        getattr(obj, name).flags.writeable = False


def _make_index_immutable_r11(index: ColumnarTeacherIndexR11) -> ColumnarTeacherIndexR11:
    _set_arrays_readonly_r11(index, _INDEX_ARRAY_FIELDS)
    return replace(index, parent_row_by_id=MappingProxyType(dict(index.parent_row_by_id)), dep_row_by_id=MappingProxyType(dict(index.dep_row_by_id)))


def _make_regime_immutable_r11(regime: SupportRegimeR11) -> SupportRegimeR11:
    _set_arrays_readonly_r11(regime, _REGIME_ARRAY_FIELDS)
    return regime


def _canonical_target_ids_r11(*, parents: Mapping[str, ParentContextR102], index: ColumnarTeacherIndexR11):
    by_split = {"TRAIN": [], "VALIDATION": []}
    for parent in parents.values():
        if parent.split in by_split:
            by_split[parent.split].append(parent.parent_id)
    all_ids = by_split["TRAIN"] + by_split["VALIDATION"]
    if len(all_ids) != len(set(all_ids)):
        seen = set()
        for parent_id in all_ids:
            if parent_id in seen:
                raise RuntimeError(f"R11_DUPLICATE_TARGET:{parent_id}")
            seen.add(parent_id)
    missing = sorted(parent_id for parent_id in all_ids if parent_id not in index.parent_row_by_id)
    if missing:
        raise RuntimeError(f"R11_MISSING_TARGET:{missing[0]}:COUNT={len(missing)}")
    return tuple(sorted(by_split["TRAIN"])), tuple(sorted(by_split["VALIDATION"]))


def _build_jobs_r11(*, train_parent_ids, val_parent_ids, index, parents, train_config, val_config, block_targets):
    eligible_train_groups = {parent.dependence_group_id for parent in parents.values() if parent.split == "TRAIN"}
    jobs = []
    regime_count = 0
    scheduled_ids = []
    for split, target_ids, config in (("TRAIN", train_parent_ids, train_config), ("VALIDATION", val_parent_ids, val_config)):
        support_groups = group_targets_by_support_r11(target_parent_ids=target_ids, index=index, config=config, eligible_train_dependence_groups=eligible_train_groups)
        for dep_rows, regime_targets in support_groups:
            regime = _make_regime_immutable_r11(prepare_support_regime_r11(dep_rows=dep_rows, index=index))
            regime_count += 1
            for start in range(0, len(regime_targets), block_targets):
                target_block = tuple(regime_targets[start:start + block_targets])
                if not target_block:
                    continue
                jobs.append(_TeacherBlockJobR11(len(jobs), split, target_block, regime, config))
                scheduled_ids.extend(target_block)
    expected_ids = tuple(train_parent_ids + val_parent_ids)
    if len(scheduled_ids) != len(expected_ids):
        raise RuntimeError(f"R11_INCOMPLETE_SCHEDULE:EXPECTED={len(expected_ids)}:SCHEDULED={len(scheduled_ids)}")
    if len(scheduled_ids) != len(set(scheduled_ids)):
        seen = set()
        for parent_id in scheduled_ids:
            if parent_id in seen:
                raise RuntimeError(f"R11_DUPLICATE_SCHEDULED_TARGET:{parent_id}")
            seen.add(parent_id)
    if set(scheduled_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(scheduled_ids))
        extra = sorted(set(scheduled_ids) - set(expected_ids))
        raise RuntimeError(f"R11_SCHEDULE_TARGET_SET_DRIFT:MISSING={missing[:1]}:EXTRA={extra[:1]}")
    return jobs, regime_count


def _validate_block_result_r11(*, job, rows):
    rows = tuple(rows)
    expected = job.target_parent_ids
    actual = tuple(row.parent_id for row in rows)
    if len(actual) != len(expected):
        raise RuntimeError(f"R11_INCOMPLETE_BLOCK:{job.ordinal}:EXPECTED={len(expected)}:ACTUAL={len(actual)}")
    if len(actual) != len(set(actual)):
        seen = set()
        for parent_id in actual:
            if parent_id in seen:
                raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{parent_id}")
            seen.add(parent_id)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"R11_INCOMPLETE_BLOCK:{job.ordinal}:MISSING={missing[:1]}:EXTRA={extra[:1]}")
    if actual != expected:
        raise RuntimeError(f"R11_OUTPUT_ORDERING_DRIFT:BLOCK={job.ordinal}")
    return rows


def _run_job_r11(*, job, index):
    try:
        rows = _compile_block_r11(target_parent_ids=job.target_parent_ids, index=index, regime=job.regime, config=job.config)
    except Exception as exc:
        raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
    return _validate_block_result_r11(job=job, rows=rows)


@contextmanager
def _single_thread_blas_r11(*, workers: int) -> Iterator[None]:
    if threadpool_limits is None:
        raise RuntimeError("R11_BLAS_SINGLE_THREAD_CONTROL_UNAVAILABLE")
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def _execute_jobs_with_executor_r11(*, jobs, index, workers: int, executor: ThreadPoolExecutor):
    results = [None] * len(jobs)
    pending: dict[Future, _TeacherBlockJobR11] = {}
    next_index = 0

    def submit_one(job):
        future = executor.submit(_run_job_r11, job=job, index=index)
        pending[future] = job

    while next_index < len(jobs) and len(pending) < workers:
        submit_one(jobs[next_index])
        next_index += 1
    while pending:
        done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in sorted(done, key=lambda f: pending[f].ordinal):
            job = pending.pop(future)
            try:
                results[job.ordinal] = future.result()
            except Exception as exc:
                remainder = tuple(pending)
                for other in remainder:
                    other.cancel()
                if remainder:
                    wait(remainder)
                if isinstance(exc, RuntimeError) and str(exc).startswith("R11_"):
                    raise
                raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
            if next_index < len(jobs):
                submit_one(jobs[next_index])
                next_index += 1
    if any(rows is None for rows in results):
        raise RuntimeError("R11_INCOMPLETE_BLOCK_COLLECTION")
    return [rows for rows in results if rows is not None]


def _execute_jobs_r11(*, jobs, index, workers: int, worker_pool: TeacherWorkerPoolR11 | None = None):
    if workers == 1:
        out = []
        for job in jobs:
            try:
                out.append(_run_job_r11(job=job, index=index))
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith("R11_"):
                    raise
                raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
        return out
    if worker_pool is not None:
        return worker_pool.execute(jobs=jobs, index=index, workers=workers)
    with TeacherWorkerPoolR11(max_workers=workers) as temporary_pool:
        return temporary_pool.execute(jobs=jobs, index=index, workers=workers)


def compile_teacher_evidence_threaded_r11(*, samples: Sequence[CounterfactualBranchSampleR5], parents: Mapping[str, ParentContextR102], train_config: DependenceAwareTeacherConfigR6, val_config: DependenceAwareTeacherConfigR6, workers: int = CANONICAL_TEACHER_WORKERS_R11, block_targets: int = 64, worker_pool: TeacherWorkerPoolR11 | None = None):
    workers = int(workers)
    block_targets = int(block_targets)
    if workers <= 0:
        raise ValueError("workers")
    if block_targets <= 0:
        raise ValueError("block_targets")
    if worker_pool is not None and workers > worker_pool.max_workers:
        raise ValueError(f"R11_TEACHER_POOL_WORKERS_OUT_OF_RANGE:{workers}:MAX={worker_pool.max_workers}")
    train_config.validate()
    val_config.validate()
    index = _make_index_immutable_r11(build_columnar_teacher_index_r11(samples))
    train_parent_ids, val_parent_ids = _canonical_target_ids_r11(parents=parents, index=index)
    jobs, regime_count = _build_jobs_r11(train_parent_ids=train_parent_ids, val_parent_ids=val_parent_ids, index=index, parents=parents, train_config=train_config, val_config=val_config, block_targets=block_targets)
    with _single_thread_blas_r11(workers=workers):
        block_results = _execute_jobs_r11(jobs=jobs, index=index, workers=workers, worker_pool=worker_pool)
    compiled = {}
    for rows in block_results:
        for evidence in rows:
            if evidence.parent_id in compiled:
                raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
            compiled[evidence.parent_id] = evidence
    expected = tuple(train_parent_ids + val_parent_ids)
    missing = [parent_id for parent_id in expected if parent_id not in compiled]
    if missing:
        raise RuntimeError(f"R11_MISSING_COMPILED_TARGET:{missing[0]}:COUNT={len(missing)}")
    extra = sorted(set(compiled) - set(expected))
    if extra:
        raise RuntimeError(f"R11_UNEXPECTED_COMPILED_TARGET:{extra[0]}:COUNT={len(extra)}")
    train = [compiled[parent_id] for parent_id in train_parent_ids]
    val = [compiled[parent_id] for parent_id in val_parent_ids]
    if tuple(x.parent_id for x in train) != train_parent_ids:
        raise RuntimeError("R11_OUTPUT_ORDERING_DRIFT:TRAIN")
    if tuple(x.parent_id for x in val) != val_parent_ids:
        raise RuntimeError("R11_OUTPUT_ORDERING_DRIFT:VALIDATION")
    core_stats = VectorizedTeacherStatsR11(targets=len(train) + len(val), support_regimes=regime_count, geometry_blocks=len(jobs), max_targets_per_block=block_targets, feature_dim=index.feature_dim, actions=index.action_count)
    return train, val, ThreadedTeacherStatsR11(core=core_stats, workers=workers, persistent_pool_supplied=worker_pool is not None, max_in_flight=min(workers, max(1, len(jobs))))
