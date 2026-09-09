from __future__ import annotations

"""Dependence-balanced canonical R11 Teacher execution engine.

R2.1/R2.2 qualified one narrow mechanics correction: normalization mass belongs
to independent market-future dependence groups, not to the number of parent rows
materialized inside a group. Distinct AccountState contexts remain distinct;
exact Student-context replicas add zero geometry, support, or target-compute mass.

The downstream R11 vectorized Teacher law is reused unchanged. Legacy R10.2 and
the original R11 vectorized engine remain available for historical reproduction.
"""

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from .cpu_runtime_r11 import CANONICAL_TEACHER_WORKERS_R11
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6, DependenceAwareTeacherEvidenceR6, canonical_hash
from .r102_evidence_cache import ParentContextR102
from .teacher_scheduler_r11 import (
    TeacherWorkerPoolR11,
    ThreadedTeacherStatsR11,
    _TeacherBlockJobR11,
    _canonical_target_ids_r11,
    _execute_jobs_r11,
    _make_index_immutable_r11,
    _make_regime_immutable_r11,
    _single_thread_blas_r11,
)
from .teacher_vectorized_r11 import (
    SupportRegimeR11,
    VectorizedTeacherStatsR11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
)

R11_BALANCED_TEACHER_ENGINE = "CB16_R11_COLUMNAR_DEPENDENCE_BALANCED_TEACHER_V2"
R11_BALANCED_SCHEDULER = "CB16_R11_DEPENDENCE_BALANCED_THREAD_SCHEDULER_V1"
R11_BALANCED_GEOMETRY = "EQUAL_DEPENDENCE_GROUP_MASS__UNIQUE_STUDENT_CONTEXT_WITHIN_GROUP_V1"


def _validate_exact_context_replica_r11(*, prior: int, row: int, dep_id: str, context_id: str, index) -> None:
    if not np.array_equal(index.features[row], index.features[prior]):
        raise RuntimeError(f"R11_BALANCED_CONTEXT_ID_FEATURE_CONFLICT:{dep_id}:{context_id}")
    if not np.array_equal(index.utilities[row], index.utilities[prior]):
        raise RuntimeError(f"R11_BALANCED_CONTEXT_ID_UTILITY_CONFLICT:{dep_id}:{context_id}")
    if int(index.parent_timestamps[row]) != int(index.parent_timestamps[prior]):
        raise RuntimeError(f"R11_BALANCED_CONTEXT_ID_TIMESTAMP_CONFLICT:{dep_id}:{context_id}")


def _unique_parent_rows_by_group_r11(*, dep_rows: np.ndarray, index):
    """Return one canonical support row per unique Student context in each future."""
    groups: list[np.ndarray] = []
    duplicate_count = 0
    for dep_row in np.asarray(dep_rows, dtype=np.int32):
        dep_id = str(index.dep_ids[int(dep_row)])
        raw = np.asarray(index.dep_parent_rows[int(dep_row)], dtype=np.int32)
        raw = raw[raw >= 0]
        by_context: dict[str, int] = {}
        for raw_row in raw:
            row = int(raw_row)
            context_id = str(index.student_context_ids[row])
            prior = by_context.get(context_id)
            if prior is None:
                by_context[context_id] = row
                continue
            duplicate_count += 1
            _validate_exact_context_replica_r11(
                prior=prior, row=row, dep_id=dep_id, context_id=context_id, index=index
            )
            if index.parent_ids[row] < index.parent_ids[prior]:
                by_context[context_id] = row
        # Semantic context order is independent of replica parent naming/volume.
        groups.append(np.asarray([by_context[k] for k in sorted(by_context)], dtype=np.int32))
    return groups, int(duplicate_count)


def _compact_unique_support_matrix_r11(*, unique_groups: Sequence[np.ndarray]) -> np.ndarray:
    if not unique_groups:
        return np.empty((0, 0), dtype=np.int32)
    width = max((len(rows) for rows in unique_groups), default=0)
    matrix = np.full((len(unique_groups), width), -1, dtype=np.int32)
    for i, rows in enumerate(unique_groups):
        if len(rows):
            matrix[i, : len(rows)] = rows
    return matrix


def prepare_support_regime_balanced_r11(*, dep_rows: np.ndarray, index) -> SupportRegimeR11:
    """Build a replica-invariant support regime with equal mass per future group."""
    dep_rows = np.asarray(dep_rows, dtype=np.int32)
    dep_ids = tuple(index.dep_ids[int(i)] for i in dep_rows)
    feature_dim = index.feature_dim
    if len(dep_rows) == 0:
        return SupportRegimeR11(
            dep_rows=dep_rows,
            dep_ids=dep_ids,
            train_dependence_group_hash=canonical_hash(dep_ids),
            mean=np.zeros(feature_dim, dtype=np.float64),
            std=np.ones(feature_dim, dtype=np.float64),
            support_parent_rows=np.empty((0, 0), dtype=np.int32),
            normalized_support_flat=np.empty((0, feature_dim), dtype=np.float64),
            normalized_support_norm2=np.empty(0, dtype=np.float64),
            valid_support_flat=np.empty(0, dtype=np.bool_),
            dep_lex_rank=np.empty(0, dtype=np.int32),
        )

    unique_groups, _ = _unique_parent_rows_by_group_r11(dep_rows=dep_rows, index=index)
    if any(len(rows) == 0 for rows in unique_groups):
        raise RuntimeError("R11_BALANCED_EMPTY_DEPENDENCE_GROUP")

    # Exact replicas are removed not only from normalization mass but from the actual
    # batched kNN support matrix. Base and replica-injected worlds therefore have the
    # same support values, shape, and floating-point reduction order.
    parent_matrix = _compact_unique_support_matrix_r11(unique_groups=unique_groups)
    valid = parent_matrix >= 0
    all_parent_rows = parent_matrix[valid]
    if len(all_parent_rows) == 0:
        raise RuntimeError("R11_BALANCED_EMPTY_SUPPORT")

    # Every independent future contributes equal total normalization mass. Within a
    # future, each distinct AccountState/Student context shares that future's mass.
    group_means = np.stack(
        [np.asarray(index.features[rows], dtype=np.float64).mean(axis=0) for rows in unique_groups],
        axis=0,
    )
    mean = group_means.mean(axis=0)
    group_second_moments = np.stack(
        [
            np.mean((np.asarray(index.features[rows], dtype=np.float64) - mean) ** 2, axis=0)
            for rows in unique_groups
        ],
        axis=0,
    )
    std = np.sqrt(np.maximum(group_second_moments.mean(axis=0), 0.0))
    std = np.where(std < 1e-8, 1.0, std)

    safe_rows = parent_matrix.copy()
    first_valid = int(all_parent_rows[0])
    safe_rows[~valid] = first_valid
    z = (index.features[safe_rows.reshape(-1)] - mean) / std
    z = np.ascontiguousarray(z, dtype=np.float64)
    valid_flat = np.ascontiguousarray(valid.reshape(-1))
    z[~valid_flat] = 0.0
    norm2 = np.einsum("ij,ij->i", z, z, optimize=True)

    lexical = sorted(range(len(dep_ids)), key=lambda i: dep_ids[i])
    dep_lex_rank = np.empty(len(dep_ids), dtype=np.int32)
    for rank, local_idx in enumerate(lexical):
        dep_lex_rank[local_idx] = rank

    return SupportRegimeR11(
        dep_rows=dep_rows,
        dep_ids=dep_ids,
        train_dependence_group_hash=canonical_hash(dep_ids),
        mean=np.ascontiguousarray(mean),
        std=np.ascontiguousarray(std),
        support_parent_rows=parent_matrix,
        normalized_support_flat=z,
        normalized_support_norm2=np.ascontiguousarray(norm2),
        valid_support_flat=valid_flat,
        dep_lex_rank=dep_lex_rank,
    )


def _canonicalize_target_replicas_r11(*, parent_ids: Sequence[str], index):
    """Collapse exact target replicas by semantic identity before vectorized batching."""
    representative_by_key: dict[tuple[str, str], int] = {}
    aliases_by_key: dict[tuple[str, str], list[str]] = {}
    key_by_parent: dict[str, tuple[str, str]] = {}

    for parent_id in parent_ids:
        row = int(index.parent_row_by_id[parent_id])
        dep_id = str(index.dep_ids[int(index.parent_dep_index[row])])
        context_id = str(index.student_context_ids[row])
        key = (dep_id, context_id)
        key_by_parent[parent_id] = key
        aliases_by_key.setdefault(key, []).append(parent_id)
        prior = representative_by_key.get(key)
        if prior is None:
            representative_by_key[key] = row
            continue
        _validate_exact_context_replica_r11(
            prior=prior, row=row, dep_id=dep_id, context_id=context_id, index=index
        )
        if index.parent_ids[row] < index.parent_ids[prior]:
            representative_by_key[key] = row

    # Sort by semantic key, not parent id, so adding a differently named replica cannot
    # alter target batch order or target GEMM shape for the original scientific contexts.
    ordered_keys = sorted(representative_by_key)
    canonical_ids = tuple(index.parent_ids[representative_by_key[key]] for key in ordered_keys)
    canonical_by_key = {
        key: index.parent_ids[representative_by_key[key]] for key in ordered_keys
    }
    parent_to_canonical = {
        parent_id: canonical_by_key[key_by_parent[parent_id]] for parent_id in parent_ids
    }
    return canonical_ids, parent_to_canonical


def _build_balanced_jobs_r11(*, train_parent_ids, val_parent_ids, index, parents, train_config, val_config, block_targets):
    eligible_train_groups = {
        parent.dependence_group_id for parent in parents.values() if parent.split == "TRAIN"
    }
    jobs = []
    regime_count = 0
    scheduled_ids = []
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
        for dep_rows_for_regime, regime_targets in support_groups:
            regime = _make_regime_immutable_r11(
                prepare_support_regime_balanced_r11(dep_rows=dep_rows_for_regime, index=index)
            )
            regime_count += 1
            for start in range(0, len(regime_targets), int(block_targets)):
                target_block = tuple(regime_targets[start : start + int(block_targets)])
                if not target_block:
                    continue
                jobs.append(_TeacherBlockJobR11(len(jobs), split, target_block, regime, config))
                scheduled_ids.extend(target_block)

    expected_ids = tuple(train_parent_ids + val_parent_ids)
    if len(scheduled_ids) != len(expected_ids) or set(scheduled_ids) != set(expected_ids):
        raise RuntimeError("R11_BALANCED_SCHEDULE_TARGET_SET_DRIFT")
    if len(scheduled_ids) != len(set(scheduled_ids)):
        raise RuntimeError("R11_BALANCED_DUPLICATE_SCHEDULED_TARGET")
    return jobs, regime_count


def _rebind_evidence_to_parent_r11(*, evidence: DependenceAwareTeacherEvidenceR6, parent_id: str, index):
    if evidence.parent_id == parent_id:
        return evidence
    row = int(index.parent_row_by_id[parent_id])
    dep_id = str(index.dep_ids[int(index.parent_dep_index[row])])
    context_id = str(index.student_context_ids[row])
    if dep_id != evidence.target_dependence_group_id or context_id != evidence.student_context_object_id:
        raise RuntimeError(f"R11_BALANCED_TARGET_REBIND_IDENTITY_DRIFT:{parent_id}")
    return replace(
        evidence,
        evidence_id=f"R6E:{parent_id}:{evidence.teacher_protocol_hash[:12]}",
        parent_id=parent_id,
        student_context_object_id=context_id,
        target_dependence_group_id=dep_id,
        timestamp=int(index.parent_timestamps[row]),
    )


def compile_teacher_evidence_balanced_r11(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int = CANONICAL_TEACHER_WORKERS_R11,
    block_targets: int = 64,
    worker_pool: TeacherWorkerPoolR11 | None = None,
) -> tuple[list[DependenceAwareTeacherEvidenceR6], list[DependenceAwareTeacherEvidenceR6], ThreadedTeacherStatsR11]:
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
    all_train_ids, all_val_ids = _canonical_target_ids_r11(parents=parents, index=index)
    train_parent_ids, train_parent_to_canonical = _canonicalize_target_replicas_r11(
        parent_ids=all_train_ids, index=index
    )
    val_parent_ids, val_parent_to_canonical = _canonicalize_target_replicas_r11(
        parent_ids=all_val_ids, index=index
    )

    jobs, regime_count = _build_balanced_jobs_r11(
        train_parent_ids=train_parent_ids,
        val_parent_ids=val_parent_ids,
        index=index,
        parents=parents,
        train_config=train_config,
        val_config=val_config,
        block_targets=block_targets,
    )
    with _single_thread_blas_r11(workers=workers):
        block_results = _execute_jobs_r11(
            jobs=jobs, index=index, workers=workers, worker_pool=worker_pool
        )

    compiled_canonical: dict[str, DependenceAwareTeacherEvidenceR6] = {}
    for rows in block_results:
        for evidence in rows:
            if evidence.parent_id in compiled_canonical:
                raise RuntimeError(f"R11_BALANCED_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
            compiled_canonical[evidence.parent_id] = evidence
    expected_canonical = tuple(train_parent_ids + val_parent_ids)
    missing = [parent_id for parent_id in expected_canonical if parent_id not in compiled_canonical]
    if missing:
        raise RuntimeError(f"R11_BALANCED_MISSING_COMPILED_TARGET:{missing[0]}:COUNT={len(missing)}")
    extra = sorted(set(compiled_canonical) - set(expected_canonical))
    if extra:
        raise RuntimeError(f"R11_BALANCED_UNEXPECTED_COMPILED_TARGET:{extra[0]}:COUNT={len(extra)}")

    train = [
        _rebind_evidence_to_parent_r11(
            evidence=compiled_canonical[train_parent_to_canonical[parent_id]],
            parent_id=parent_id,
            index=index,
        )
        for parent_id in all_train_ids
    ]
    val = [
        _rebind_evidence_to_parent_r11(
            evidence=compiled_canonical[val_parent_to_canonical[parent_id]],
            parent_id=parent_id,
            index=index,
        )
        for parent_id in all_val_ids
    ]

    core_stats = VectorizedTeacherStatsR11(
        targets=len(train) + len(val),
        support_regimes=regime_count,
        geometry_blocks=len(jobs),
        max_targets_per_block=block_targets,
        feature_dim=index.feature_dim,
        actions=index.action_count,
        engine=R11_BALANCED_TEACHER_ENGINE,
    )
    return train, val, ThreadedTeacherStatsR11(
        core=core_stats,
        workers=workers,
        scheduler=R11_BALANCED_SCHEDULER,
        persistent_pool_supplied=worker_pool is not None,
        max_in_flight=min(workers, max(1, len(jobs))),
    )


__all__ = [
    "R11_BALANCED_TEACHER_ENGINE",
    "R11_BALANCED_SCHEDULER",
    "R11_BALANCED_GEOMETRY",
    "prepare_support_regime_balanced_r11",
    "compile_teacher_evidence_balanced_r11",
]
