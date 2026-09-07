from __future__ import annotations

"""Shared-columnar production scheduler for the frozen R11 probabilistic Teacher.

The scheduler deliberately uses threads rather than a Python process farm: every worker
receives the same immutable :class:`ColumnarTeacherIndexR11` object and only owns
per-block scratch allocated by the vectorized Teacher kernel.  Scientific evidence is
therefore independent of worker topology; workers affect execution order only.

Legacy Teacher implementations are qualification oracles, never this hot path.
"""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

try:  # scikit-learn already depends on threadpoolctl in the canonical environment.
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - exercised through explicit fail-closed test/mocking.
    threadpool_limits = None

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


R11_THREADED_SCHEDULER = "CB16_R11_SHARED_COLUMNAR_THREAD_SCHEDULER_V2"

_INDEX_ARRAY_FIELDS = (
    "parent_timestamps",
    "parent_dep_index",
    "features",
    "utilities",
    "action_directions",
    "action_risks",
    "dep_timestamps",
    "dep_parent_rows",
)
_REGIME_ARRAY_FIELDS = (
    "dep_rows",
    "mean",
    "std",
    "support_parent_rows",
    "normalized_support_flat",
    "normalized_support_norm2",
    "valid_support_flat",
    "dep_lex_rank",
)


@dataclass(frozen=True)
class ThreadedTeacherStatsR11:
    core: VectorizedTeacherStatsR11
    workers: int
    scheduler: str = R11_THREADED_SCHEDULER
    memory_model: str = "ONE_SHARED_IMMUTABLE_COLUMNAR_INDEX__THREAD_LOCAL_TARGET_BLOCK_SCRATCH"
    nested_blas_threads_required: int = 1
    nested_blas_limit_enforced: bool = True
    completion_collection: str = "AS_COMPLETED__CANONICAL_SLOT_REASSEMBLY"
    topology_in_scientific_identity: bool = False


@dataclass(frozen=True)
class _TeacherBlockJobR11:
    ordinal: int
    split: str
    target_parent_ids: tuple[str, ...]
    regime: SupportRegimeR11
    config: DependenceAwareTeacherConfigR6


def _set_arrays_readonly_r11(obj: object, fields: Sequence[str]) -> None:
    for name in fields:
        array = getattr(obj, name)
        array.flags.writeable = False


def _make_index_immutable_r11(index: ColumnarTeacherIndexR11) -> ColumnarTeacherIndexR11:
    """Seal the one process-local columnar index before any worker is started."""

    _set_arrays_readonly_r11(index, _INDEX_ARRAY_FIELDS)
    # build_columnar_teacher_index_r11 creates these dictionaries locally, so wrapping
    # copies once here prevents accidental scheduler/worker mutation without per-worker
    # replication of the large columnar arrays.
    return replace(
        index,
        parent_row_by_id=MappingProxyType(dict(index.parent_row_by_id)),
        dep_row_by_id=MappingProxyType(dict(index.dep_row_by_id)),
    )


def _make_regime_immutable_r11(regime: SupportRegimeR11) -> SupportRegimeR11:
    _set_arrays_readonly_r11(regime, _REGIME_ARRAY_FIELDS)
    return regime


def _canonical_target_ids_r11(
    *,
    parents: Mapping[str, ParentContextR102],
    index: ColumnarTeacherIndexR11,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_split: dict[str, list[str]] = {"TRAIN": [], "VALIDATION": []}
    for parent in parents.values():
        if parent.split in by_split:
            by_split[parent.split].append(parent.parent_id)

    all_ids = by_split["TRAIN"] + by_split["VALIDATION"]
    if len(all_ids) != len(set(all_ids)):
        seen: set[str] = set()
        duplicate = ""
        for parent_id in all_ids:
            if parent_id in seen:
                duplicate = parent_id
                break
            seen.add(parent_id)
        raise RuntimeError(f"R11_DUPLICATE_TARGET:{duplicate}")

    missing = sorted(parent_id for parent_id in all_ids if parent_id not in index.parent_row_by_id)
    if missing:
        raise RuntimeError(f"R11_MISSING_TARGET:{missing[0]}:COUNT={len(missing)}")

    # This is the canonical parent ordering already used by the frozen R6 oracle and the
    # R11 vectorized qualification engine.  Runtime topology may never alter it.
    return tuple(sorted(by_split["TRAIN"])), tuple(sorted(by_split["VALIDATION"]))


def _build_jobs_r11(
    *,
    train_parent_ids: tuple[str, ...],
    val_parent_ids: tuple[str, ...],
    index: ColumnarTeacherIndexR11,
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    block_targets: int,
) -> tuple[list[_TeacherBlockJobR11], int]:
    eligible_train_groups = {
        parent.dependence_group_id for parent in parents.values() if parent.split == "TRAIN"
    }
    jobs: list[_TeacherBlockJobR11] = []
    regime_count = 0
    scheduled_ids: list[str] = []

    for split, target_ids, config in (
        ("TRAIN", train_parent_ids, train_config),
        ("VALIDATION", val_parent_ids, val_config),
    ):
        support_groups = group_targets_by_support_r11(
            target_parent_ids=target_ids,
            index=index,
            config=config,
            eligible_train_dependence_groups=eligible_train_groups,
        )
        for dep_rows, regime_targets in support_groups:
            regime = _make_regime_immutable_r11(
                prepare_support_regime_r11(dep_rows=dep_rows, index=index)
            )
            regime_count += 1
            for start in range(0, len(regime_targets), block_targets):
                target_block = tuple(regime_targets[start : start + block_targets])
                if not target_block:
                    continue
                jobs.append(_TeacherBlockJobR11(
                    ordinal=len(jobs),
                    split=split,
                    target_parent_ids=target_block,
                    regime=regime,
                    config=config,
                ))
                scheduled_ids.extend(target_block)

    expected_ids = tuple(train_parent_ids + val_parent_ids)
    if len(scheduled_ids) != len(expected_ids):
        raise RuntimeError(
            f"R11_INCOMPLETE_SCHEDULE:EXPECTED={len(expected_ids)}:SCHEDULED={len(scheduled_ids)}"
        )
    if len(scheduled_ids) != len(set(scheduled_ids)):
        seen: set[str] = set()
        duplicate = ""
        for parent_id in scheduled_ids:
            if parent_id in seen:
                duplicate = parent_id
                break
            seen.add(parent_id)
        raise RuntimeError(f"R11_DUPLICATE_SCHEDULED_TARGET:{duplicate}")
    if set(scheduled_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(scheduled_ids))
        extra = sorted(set(scheduled_ids) - set(expected_ids))
        raise RuntimeError(
            "R11_SCHEDULE_TARGET_SET_DRIFT:"
            f"MISSING={missing[:1]}:EXTRA={extra[:1]}"
        )
    return jobs, regime_count


def _validate_block_result_r11(
    *,
    job: _TeacherBlockJobR11,
    rows: Sequence[DependenceAwareTeacherEvidenceR6],
) -> tuple[DependenceAwareTeacherEvidenceR6, ...]:
    rows = tuple(rows)
    expected = job.target_parent_ids
    actual = tuple(row.parent_id for row in rows)
    if len(actual) != len(expected):
        raise RuntimeError(
            f"R11_INCOMPLETE_BLOCK:{job.ordinal}:EXPECTED={len(expected)}:ACTUAL={len(actual)}"
        )
    if len(actual) != len(set(actual)):
        seen: set[str] = set()
        duplicate = ""
        for parent_id in actual:
            if parent_id in seen:
                duplicate = parent_id
                break
            seen.add(parent_id)
        raise RuntimeError(f"R11_DUPLICATE_COMPILED_TARGET:{duplicate}")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"R11_INCOMPLETE_BLOCK:{job.ordinal}:MISSING={missing[:1]}:EXTRA={extra[:1]}"
        )
    if actual != expected:
        raise RuntimeError(f"R11_OUTPUT_ORDERING_DRIFT:BLOCK={job.ordinal}")
    return rows


def _run_job_r11(
    *,
    job: _TeacherBlockJobR11,
    index: ColumnarTeacherIndexR11,
) -> tuple[DependenceAwareTeacherEvidenceR6, ...]:
    try:
        rows = _compile_block_r11(
            target_parent_ids=job.target_parent_ids,
            index=index,
            regime=job.regime,
            config=job.config,
        )
    except Exception as exc:
        raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
    return _validate_block_result_r11(job=job, rows=rows)


@contextmanager
def _single_thread_blas_r11(*, workers: int) -> Iterator[None]:
    """Enforce one BLAS thread while outer Teacher workers are active.

    Environment variables alone cannot safely reconfigure a BLAS runtime that NumPy has
    already loaded.  threadpoolctl does, so Teacher execution fails closed if that control
    is unavailable instead of risking an unverified BLAS thread count or outer-workers x
    BLAS-threads oversubscription.
    """

    if threadpool_limits is None:
        raise RuntimeError("R11_BLAS_SINGLE_THREAD_CONTROL_UNAVAILABLE")
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def _execute_jobs_r11(
    *,
    jobs: Sequence[_TeacherBlockJobR11],
    index: ColumnarTeacherIndexR11,
    workers: int,
) -> list[tuple[DependenceAwareTeacherEvidenceR6, ...]]:
    if workers == 1:
        out: list[tuple[DependenceAwareTeacherEvidenceR6, ...]] = []
        for job in jobs:
            try:
                out.append(_run_job_r11(job=job, index=index))
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith("R11_"):
                    raise
                raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
        return out

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cb16-r11-teacher")
    future_to_job: dict[Future[tuple[DependenceAwareTeacherEvidenceR6, ...]], _TeacherBlockJobR11] = {}
    results: list[tuple[DependenceAwareTeacherEvidenceR6, ...] | None] = [None] * len(jobs)
    try:
        for job in jobs:
            future = executor.submit(_run_job_r11, job=job, index=index)
            future_to_job[future] = job
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                results[job.ordinal] = future.result()
            except Exception as exc:
                for pending in future_to_job:
                    if pending is not future:
                        pending.cancel()
                if isinstance(exc, RuntimeError) and str(exc).startswith("R11_"):
                    raise
                raise RuntimeError(f"R11_TEACHER_WORKER_EXCEPTION:BLOCK={job.ordinal}") from exc
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if any(rows is None for rows in results):
        raise RuntimeError("R11_INCOMPLETE_BLOCK_COLLECTION")
    return [rows for rows in results if rows is not None]


def compile_teacher_evidence_threaded_r11(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int = 12,
    block_targets: int = 64,
) -> tuple[
    list[DependenceAwareTeacherEvidenceR6],
    list[DependenceAwareTeacherEvidenceR6],
    ThreadedTeacherStatsR11,
]:
    """Compile frozen Teacher evidence with one shared immutable columnar index.

    Worker count and completion order are runtime-only concerns.  The returned evidence
    is reassembled in canonical parent order and contains no topology field.
    """

    workers = int(workers)
    block_targets = int(block_targets)
    if workers <= 0:
        raise ValueError("workers")
    if block_targets <= 0:
        raise ValueError("block_targets")
    train_config.validate()
    val_config.validate()

    index = _make_index_immutable_r11(build_columnar_teacher_index_r11(samples))
    train_parent_ids, val_parent_ids = _canonical_target_ids_r11(parents=parents, index=index)
    jobs, regime_count = _build_jobs_r11(
        train_parent_ids=train_parent_ids,
        val_parent_ids=val_parent_ids,
        index=index,
        parents=parents,
        train_config=train_config,
        val_config=val_config,
        block_targets=block_targets,
    )

    with _single_thread_blas_r11(workers=workers):
        block_results = _execute_jobs_r11(jobs=jobs, index=index, workers=workers)

    compiled: dict[str, DependenceAwareTeacherEvidenceR6] = {}
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

    core_stats = VectorizedTeacherStatsR11(
        targets=len(train) + len(val),
        support_regimes=regime_count,
        geometry_blocks=len(jobs),
        max_targets_per_block=block_targets,
        feature_dim=index.feature_dim,
        actions=index.action_count,
    )
    return train, val, ThreadedTeacherStatsR11(core=core_stats, workers=workers)
