from __future__ import annotations

"""Dependence-balanced canonical R11 Teacher execution engine.

R2.1/R2.2 qualified one narrow mechanics correction: normalization mass belongs
to independent market-future dependence groups, not to the number of parent rows
materialized inside a group. Distinct AccountState contexts remain distinct;
exact Student-context replicas add zero normalization mass.

The downstream R11 vectorized Teacher law is reused unchanged. Legacy R10.2 and
the original R11 vectorized engine remain available for historical reproduction.
"""

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


def _unique_parent_rows_by_group_r11(*, dep_rows: np.ndarray, index):
    groups: list[np.ndarray] = []
    duplicate_count = 0
    for dep_row in np.asarray(dep_rows, dtype=np.int32):
        raw = np.asarray(index.dep_parent_rows[int(dep_row)], dtype=np.int32)
        raw = raw[raw >= 0]
        seen: dict[str, int] = {}
        unique: list[int] = []
        for row in raw:
            row = int(row)
            context_id = str(index.student_context_ids[row])
            prior = seen.get(context_id)
            if prior is None:
                seen[context_id] = row
                unique.append(row)
                continue
            duplicate_count += 1
            # Same Student identity is allowed to collapse only when it is truly the
            # same observable context and the same realized counterfactual utility law.
            if not np.array_equal(index.features[row], index.features[prior]):
                raise RuntimeError(
                    f"R11_BALANCED_CONTEXT_ID_FEATURE_CONFLICT:{index.dep_ids[int(dep_row)]}:{context_id}"
                )
            if not np.array_equal(index.utilities[row], index.utilities[prior]):
                raise RuntimeError(
                    f"R11_BALANCED_CONTEXT_ID_UTILITY_CONFLICT:{index.dep_ids[int(dep_row)]}:{context_id}"
                )
        groups.append(np.asarray(unique, dtype=np.int32))
    return groups, int(duplicate_count)


def prepare_support_regime_balanced_r11(*, dep_rows: np.ndarray, index) -> SupportRegimeR11:
    """Build normalization with equal total mass per independent future group."""
    dep_rows = np.asarray(dep_rows, dtype=np.int32)
    dep_ids = tuple(index.dep_ids[int(i)] for i in dep_rows)
    parent_matrix = np.asarray(index.dep_parent_rows[dep_rows], dtype=np.int32)
    valid = parent_matrix >= 0
    all_parent_rows = parent_matrix[valid]
    if len(all_parent_rows) == 0:
        feature_dim = index.feature_dim
        return SupportRegimeR11(
            dep_rows=dep_rows,
            dep_ids=dep_ids,
            train_dependence_group_hash=canonical_hash(dep_ids),
            mean=np.zeros(feature_dim, dtype=np.float64),
            std=np.ones(feature_dim, dtype=np.float64),
            support_parent_rows=parent_matrix,
            normalized_support_flat=np.empty((0, feature_dim), dtype=np.float64),
            normalized_support_norm2=np.empty(0, dtype=np.float64),
            valid_support_flat=np.empty(0, dtype=np.bool_),
            dep_lex_rank=np.empty(0, dtype=np.int32),
        )

    unique_groups, _ = _unique_parent_rows_by_group_r11(dep_rows=dep_rows, index=index)
    if any(len(rows) == 0 for rows in unique_groups):
        raise RuntimeError("R11_BALANCED_EMPTY_DEPENDENCE_GROUP")

    # Always use the same dependence-balanced arithmetic, even when the input happens
    # to be perfectly balanced. Otherwise base and replica-injected worlds could take
    # different floating-point reduction orders and violate strict replica invariance.
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
        for dep_rows, regime_targets in support_groups:
            regime = _make_regime_immutable_r11(
                prepare_support_regime_balanced_r11(dep_rows=dep_rows, index=index)
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
    train_parent_ids, val_parent_ids = _canonical_target_ids_r11(parents=parents, index=index)
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

    compiled = {}
    for rows in block_results:
        for evidence in rows:
            if evidence.parent_id in compiled:
                raise RuntimeError(f"R11_BALANCED_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
            compiled[evidence.parent_id] = evidence
    expected = tuple(train_parent_ids + val_parent_ids)
    missing = [parent_id for parent_id in expected if parent_id not in compiled]
    if missing:
        raise RuntimeError(f"R11_BALANCED_MISSING_COMPILED_TARGET:{missing[0]}:COUNT={len(missing)}")
    extra = sorted(set(compiled) - set(expected))
    if extra:
        raise RuntimeError(f"R11_BALANCED_UNEXPECTED_COMPILED_TARGET:{extra[0]}:COUNT={len(extra)}")
    train = [compiled[parent_id] for parent_id in train_parent_ids]
    val = [compiled[parent_id] for parent_id in val_parent_ids]

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
