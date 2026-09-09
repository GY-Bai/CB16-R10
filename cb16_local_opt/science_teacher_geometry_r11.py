from __future__ import annotations

"""Shadow-only geometry diagnostics for the R11 probabilistic Teacher.

R2 found that the frozen R11/R6 Teacher counts independent market-future
support correctly, but its feature normalization is parent-row weighted.  Thus
exact AccountState replicas inside one dependence group can alter the metric
without adding independent future support.

This module does NOT replace Teacher authority.  It constructs one candidate
shadow metric whose normalization gives every dependence group equal total
weight and deduplicates exact Student context identities within each group.
The existing nearest-context-per-group law, cross-fit support, utility law,
quantiles, admission thresholds, and decision projection are otherwise reused
byte-for-byte through the qualified vectorized kernel.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import (
    DependenceAwareTeacherConfigR6,
    DependenceAwareTeacherEvidenceR6,
    canonical_hash,
)
from .r102_evidence_cache import ParentContextR102
from .teacher_vectorized_r11 import (
    ColumnarTeacherIndexR11,
    SupportRegimeR11,
    VectorizedTeacherStatsR11,
    _compile_block_r11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
)

SHADOW_ENGINE = "CB16_R11_DEPENDENCE_BALANCED_UNIQUE_CONTEXT_GEOMETRY_SHADOW_V1"


@dataclass(frozen=True)
class GeometryShadowStatsR11:
    targets: int
    support_regimes: int
    geometry_blocks: int
    max_targets_per_block: int
    feature_dim: int
    actions: int
    engine: str = SHADOW_ENGINE


def _unique_rows_by_context_within_group(
    index: ColumnarTeacherIndexR11,
    dep_row: int,
) -> np.ndarray:
    rows = np.asarray(index.dep_parent_rows[int(dep_row)], dtype=np.int32)
    rows = rows[rows >= 0]
    if len(rows) == 0:
        return rows
    seen: set[str] = set()
    unique: list[int] = []
    for row in rows:
        context_id = str(index.student_context_ids[int(row)])
        if context_id in seen:
            continue
        seen.add(context_id)
        unique.append(int(row))
    return np.asarray(unique, dtype=np.int32)


def dependence_balanced_unique_context_moments_r11(
    *,
    dep_rows: np.ndarray,
    index: ColumnarTeacherIndexR11,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Equal mass per future group, then equal mass per unique context in group."""

    dep_rows = np.asarray(dep_rows, dtype=np.int32)
    feature_dim = index.feature_dim
    if len(dep_rows) == 0:
        return (
            np.zeros(feature_dim, dtype=np.float64),
            np.ones(feature_dim, dtype=np.float64),
            {"dependence_groups": 0, "physical_parent_rows": 0, "unique_context_rows": 0},
        )

    group_features: list[np.ndarray] = []
    physical = 0
    unique_total = 0
    for dep_row in dep_rows:
        all_rows = np.asarray(index.dep_parent_rows[int(dep_row)], dtype=np.int32)
        physical += int(np.count_nonzero(all_rows >= 0))
        rows = _unique_rows_by_context_within_group(index, int(dep_row))
        if len(rows) == 0:
            raise RuntimeError(f"R11_R2_1_EMPTY_DEPENDENCE_GROUP:{int(dep_row)}")
        unique_total += int(len(rows))
        group_features.append(np.asarray(index.features[rows], dtype=np.float64))

    # Equal total mass per dependence group.  Within a group, exact Student
    # context identities are deduplicated and the remaining contexts share the
    # group's mass equally.
    group_means = np.stack([x.mean(axis=0) for x in group_features], axis=0)
    mean = group_means.mean(axis=0)
    group_second = np.stack(
        [np.mean((x - mean) ** 2, axis=0) for x in group_features],
        axis=0,
    )
    variance = group_second.mean(axis=0)
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(std < 1e-8, 1.0, std)
    return (
        np.ascontiguousarray(mean, dtype=np.float64),
        np.ascontiguousarray(std, dtype=np.float64),
        {
            "dependence_groups": int(len(dep_rows)),
            "physical_parent_rows": int(physical),
            "unique_context_rows": int(unique_total),
        },
    )


def prepare_dependence_balanced_support_regime_r11(
    *,
    dep_rows: np.ndarray,
    index: ColumnarTeacherIndexR11,
) -> tuple[SupportRegimeR11, dict[str, int]]:
    dep_rows = np.asarray(dep_rows, dtype=np.int32)
    dep_ids = tuple(index.dep_ids[int(i)] for i in dep_rows)
    parent_matrix = np.asarray(index.dep_parent_rows[dep_rows], dtype=np.int32)
    valid = parent_matrix >= 0
    parent_rows = parent_matrix[valid]
    mean, std, counts = dependence_balanced_unique_context_moments_r11(
        dep_rows=dep_rows,
        index=index,
    )
    if len(parent_rows) == 0:
        return (
            SupportRegimeR11(
                dep_rows=dep_rows,
                dep_ids=dep_ids,
                train_dependence_group_hash=canonical_hash(dep_ids),
                mean=mean,
                std=std,
                support_parent_rows=parent_matrix,
                normalized_support_flat=np.empty((0, index.feature_dim), dtype=np.float64),
                normalized_support_norm2=np.empty(0, dtype=np.float64),
                valid_support_flat=np.empty(0, dtype=np.bool_),
                dep_lex_rank=np.empty(0, dtype=np.int32),
            ),
            counts,
        )

    safe_rows = parent_matrix.copy()
    first_valid = int(safe_rows[safe_rows >= 0][0])
    safe_rows[~valid] = first_valid
    z = (index.features[safe_rows.reshape(-1)] - mean) / std
    z = np.ascontiguousarray(z, dtype=np.float64)
    valid_flat = valid.reshape(-1)
    z[~valid_flat] = 0.0
    norm2 = np.einsum("ij,ij->i", z, z, optimize=True)

    lexical = sorted(range(len(dep_ids)), key=lambda i: dep_ids[i])
    dep_lex_rank = np.empty(len(dep_ids), dtype=np.int32)
    for rank, local_idx in enumerate(lexical):
        dep_lex_rank[local_idx] = rank

    return (
        SupportRegimeR11(
            dep_rows=dep_rows,
            dep_ids=dep_ids,
            train_dependence_group_hash=canonical_hash(dep_ids),
            mean=mean,
            std=std,
            support_parent_rows=parent_matrix,
            normalized_support_flat=z,
            normalized_support_norm2=np.ascontiguousarray(norm2),
            valid_support_flat=np.ascontiguousarray(valid_flat),
            dep_lex_rank=dep_lex_rank,
        ),
        counts,
    )


def compile_teacher_evidence_dependence_balanced_shadow_r11(
    *,
    samples,
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    block_targets: int = 64,
) -> tuple[
    list[DependenceAwareTeacherEvidenceR6],
    list[DependenceAwareTeacherEvidenceR6],
    GeometryShadowStatsR11,
    dict[str, int],
]:
    if int(block_targets) <= 0:
        raise ValueError("block_targets")
    train_config.validate(); val_config.validate()
    index = build_columnar_teacher_index_r11(samples)
    train_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "TRAIN" and p.parent_id in index.parent_row_by_id
    )
    val_parent_ids = sorted(
        p.parent_id for p in parents.values()
        if p.split == "VALIDATION" and p.parent_id in index.parent_row_by_id
    )
    eligible_train_groups = {
        p.dependence_group_id for p in parents.values() if p.split == "TRAIN"
    }

    compiled: dict[str, DependenceAwareTeacherEvidenceR6] = {}
    regimes = 0; blocks = 0
    aggregate = {
        "support_regime_physical_parent_rows": 0,
        "support_regime_unique_context_rows": 0,
        "support_regime_dependence_groups": 0,
    }
    for target_ids, config in ((train_parent_ids, train_config), (val_parent_ids, val_config)):
        grouped = group_targets_by_support_r11(
            target_parent_ids=target_ids,
            index=index,
            config=config,
            eligible_train_dependence_groups=eligible_train_groups,
        )
        for dep_rows, regime_targets in grouped:
            regime, counts = prepare_dependence_balanced_support_regime_r11(
                dep_rows=dep_rows,
                index=index,
            )
            regimes += 1
            aggregate["support_regime_physical_parent_rows"] += counts["physical_parent_rows"]
            aggregate["support_regime_unique_context_rows"] += counts["unique_context_rows"]
            aggregate["support_regime_dependence_groups"] += counts["dependence_groups"]
            for start in range(0, len(regime_targets), int(block_targets)):
                chunk = regime_targets[start:start + int(block_targets)]
                for evidence in _compile_block_r11(
                    target_parent_ids=chunk,
                    index=index,
                    regime=regime,
                    config=config,
                ):
                    if evidence.parent_id in compiled:
                        raise RuntimeError(f"R11_R2_1_DUPLICATE_SHADOW_TARGET:{evidence.parent_id}")
                    compiled[evidence.parent_id] = evidence
                blocks += 1

    train = [compiled[p] for p in train_parent_ids]
    val = [compiled[p] for p in val_parent_ids]
    stats = GeometryShadowStatsR11(
        targets=len(train) + len(val),
        support_regimes=regimes,
        geometry_blocks=blocks,
        max_targets_per_block=int(block_targets),
        feature_dim=index.feature_dim,
        actions=index.action_count,
    )
    return train, val, stats, aggregate


def compare_teacher_evidence_sets_r11(
    a: Sequence[DependenceAwareTeacherEvidenceR6],
    b: Sequence[DependenceAwareTeacherEvidenceR6],
) -> dict[str, object]:
    aa = {x.parent_id: x for x in a}; bb = {x.parent_id: x for x in b}
    if aa.keys() != bb.keys():
        raise RuntimeError("R11_R2_1_TEACHER_TARGET_SET_DRIFT")
    changed = 0
    admission_changes = 0
    max_prob = 0.0; max_risk = 0.0; max_mean = 0.0; max_quantile = 0.0
    for parent_id in sorted(aa):
        x = aa[parent_id]; y = bb[parent_id]
        if x.content_hash != y.content_hash:
            changed += 1
        if x.admission.status != y.admission.status:
            admission_changes += 1
        max_prob = max(max_prob, max(abs(float(u)-float(v)) for u,v in zip(x.direction_target_probs,y.direction_target_probs)))
        max_risk = max(max_risk, abs(float(x.requested_risk_target)-float(y.requested_risk_target)))
        if len(x.action_laws) != len(y.action_laws):
            raise RuntimeError("R11_R2_1_ACTION_LAW_COUNT_DRIFT")
        for lx, ly in zip(x.action_laws, y.action_laws):
            max_mean = max(max_mean, abs(float(lx.mean_utility)-float(ly.mean_utility)))
            if len(lx.quantiles) != len(ly.quantiles):
                raise RuntimeError("R11_R2_1_QUANTILE_COUNT_DRIFT")
            max_quantile = max(max_quantile, max(abs(float(u)-float(v)) for u,v in zip(lx.quantiles,ly.quantiles)))
    return {
        "targets": len(aa),
        "content_hash_changed_targets": int(changed),
        "content_hash_change_rate": float(changed / max(len(aa), 1)),
        "admission_status_changes": int(admission_changes),
        "maximum_direction_probability_abs_delta": float(max_prob),
        "maximum_requested_risk_abs_delta": float(max_risk),
        "maximum_action_mean_utility_abs_delta": float(max_mean),
        "maximum_action_quantile_abs_delta": float(max_quantile),
    }


__all__ = [
    "SHADOW_ENGINE",
    "GeometryShadowStatsR11",
    "dependence_balanced_unique_context_moments_r11",
    "prepare_dependence_balanced_support_regime_r11",
    "compile_teacher_evidence_dependence_balanced_shadow_r11",
    "compare_teacher_evidence_sets_r11",
]
