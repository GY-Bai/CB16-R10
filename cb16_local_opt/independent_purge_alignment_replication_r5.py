from __future__ import annotations

"""R11 Science G0 R5 helpers for independent purge-gap alignment replication.

R5 does not change Teacher or Student authority. It provides two narrow pieces:
(1) candidate-only execution of the already-authoritative R11 R2.4 validation
Teacher against a fixed legacy TRAIN support surface, and (2) the pre-registered
ranking adjudication for ALIGNED versus the five frozen R4.1 shuffle controls.
"""

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .probabilistic_teacher_r6 import DependenceAwareTeacherEvidenceR6
from .r102_evidence_cache import ParentContextR102
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_balanced_runtime_r11 import (
    R11_BALANCED_GEOMETRY,
    R11_BALANCED_TEACHER_ENGINE,
    prepare_support_regime_balanced_r11,
)
from .teacher_vectorized_r11 import (
    _compile_block_r11,
    build_columnar_teacher_index_r11,
    group_targets_by_support_r11,
)

R5_RUNTIME = "CB16_R11_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_V1"
R5_SHIFTS = (1, 7, 13, 23, 31)
R5_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
)
R5_CANDIDATE_TIMESTAMP_MS = 1_735_372_800_000  # 2024-12-28T08:00:00Z


def _require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def compile_validation_targets_only_r5(
    *,
    samples: Sequence[CounterfactualBranchSampleR5],
    parents: Mapping[str, ParentContextR102],
    target_parent_ids: Sequence[str],
    block_targets: int = 32,
) -> tuple[list[DependenceAwareTeacherEvidenceR6], dict[str, Any]]:
    """Compile only requested VALIDATION targets with exact R11 R2.4 mechanics.

    TRAIN parents remain support authority but are deliberately not scheduled as
    output targets. This is execution pruning only. The target support set,
    dependence-balanced normalization, kNN law, admission law, protocol hash and
    floating-point target law are the production R11 R2.4 definitions.
    """
    block_targets = int(block_targets)
    _require(block_targets > 0, "R11_R5_INVALID_TEACHER_BLOCK_TARGETS")
    target_ids = tuple(sorted(str(x) for x in target_parent_ids))
    _require(bool(target_ids), "R11_R5_NO_CANDIDATE_TARGETS")
    _require(len(target_ids) == len(set(target_ids)), "R11_R5_DUPLICATE_CANDIDATE_TARGET")

    index = build_columnar_teacher_index_r11(samples)
    for parent_id in target_ids:
        _require(parent_id in parents, f"R11_R5_TARGET_PARENT_MISSING:{parent_id}")
        _require(parent_id in index.parent_row_by_id, f"R11_R5_TARGET_SAMPLE_MISSING:{parent_id}")
        _require(parents[parent_id].split == "VALIDATION", f"R11_R5_TARGET_NOT_VALIDATION:{parent_id}")

    eligible_train_groups = {
        p.dependence_group_id for p in parents.values() if p.split == "TRAIN"
    }
    _require(
        len(eligible_train_groups) >= int(R11_VALIDATION_TEACHER_CONFIG.min_train_dependence_groups),
        f"R11_R5_INSUFFICIENT_LEGACY_TRAIN_GROUPS:{len(eligible_train_groups)}",
    )

    support_groups = group_targets_by_support_r11(
        target_parent_ids=target_ids,
        index=index,
        config=R11_VALIDATION_TEACHER_CONFIG,
        eligible_train_dependence_groups=eligible_train_groups,
    )
    _require(bool(support_groups), "R11_R5_EMPTY_CANDIDATE_SUPPORT_GROUPING")

    compiled: dict[str, DependenceAwareTeacherEvidenceR6] = {}
    regime_receipts: list[dict[str, Any]] = []
    blocks = 0
    for dep_rows, regime_targets in support_groups:
        regime = prepare_support_regime_balanced_r11(dep_rows=np.asarray(dep_rows, dtype=np.int32), index=index)
        regime_receipts.append({
            "target_count": int(len(regime_targets)),
            "eligible_support_dependence_groups": int(regime.group_count),
            "train_dependence_group_hash": str(regime.train_dependence_group_hash),
            "max_unique_parent_contexts_per_group": int(regime.max_parents_per_group),
        })
        for start in range(0, len(regime_targets), block_targets):
            chunk = tuple(regime_targets[start:start + block_targets])
            rows = _compile_block_r11(
                target_parent_ids=chunk,
                index=index,
                regime=regime,
                config=R11_VALIDATION_TEACHER_CONFIG,
            )
            for evidence in rows:
                _require(evidence.parent_id not in compiled, f"R11_R5_DUPLICATE_COMPILED_TARGET:{evidence.parent_id}")
                compiled[evidence.parent_id] = evidence
            blocks += 1

    _require(set(compiled) == set(target_ids), "R11_R5_CANDIDATE_TARGET_SET_DRIFT")
    ordered = [compiled[parent_id] for parent_id in target_ids]
    protocol_hashes = {x.teacher_protocol_hash for x in ordered}
    _require(protocol_hashes == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "R11_R5_TEACHER_PROTOCOL_DRIFT")
    return ordered, {
        "schema": "CB16_R11_R5_TARGET_ONLY_TEACHER_EXECUTION_V1",
        "runtime": R5_RUNTIME,
        "execution_pruning_only": True,
        "teacher_semantics_changed": False,
        "teacher_engine": R11_BALANCED_TEACHER_ENGINE,
        "teacher_geometry": R11_BALANCED_GEOMETRY,
        "teacher_mode": R11_VALIDATION_TEACHER_CONFIG.mode,
        "teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "target_count": len(ordered),
        "legacy_train_dependence_groups": len(eligible_train_groups),
        "support_regime_count": len(regime_receipts),
        "geometry_blocks": blocks,
        "block_targets": block_targets,
        "regimes": regime_receipts,
    }


def _finite_metrics(row: Mapping[str, Any], label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("loss", "direction_loss", "sizing_loss"):
        _require(key in row, f"R11_R5_METRIC_MISSING:{label}:{key}")
        value = float(row[key])
        _require(math.isfinite(value), f"R11_R5_NONFINITE_METRIC:{label}:{key}")
        out[key] = value
    return out


def adjudicate_alignment_replication_r5(
    *,
    champion: Mapping[str, Any],
    aligned: Mapping[str, Any],
    shuffled: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the gate frozen before candidate outcomes were opened."""
    c = _finite_metrics(champion, "CHAMPION")
    a = _finite_metrics(aligned, "ALIGNED")
    _require(tuple(sorted(int(x) for x in shuffled)) == R5_SHIFTS, "R11_R5_SHUFFLE_ARM_SET_DRIFT")
    s = {int(k): _finite_metrics(v, f"SHUFFLE_{int(k)}") for k, v in shuffled.items()}

    total_wins = sum(a["loss"] < s[k]["loss"] for k in R5_SHIFTS)
    direction_wins = sum(a["direction_loss"] < s[k]["direction_loss"] for k in R5_SHIFTS)
    sizing_wins = sum(a["sizing_loss"] < s[k]["sizing_loss"] for k in R5_SHIFTS)
    aligned_total_better_than_champion = a["loss"] < c["loss"]
    aligned_direction_better_than_champion = a["direction_loss"] < c["direction_loss"]
    supported = bool(
        aligned_total_better_than_champion
        and aligned_direction_better_than_champion
        and total_wins == len(R5_SHIFTS)
        and direction_wins == len(R5_SHIFTS)
    )

    total_rank_rows = [("G0_CHAMPION", c["loss"]), ("ALIGNED", a["loss"])] + [
        (f"SHUFFLE_{k}", s[k]["loss"]) for k in R5_SHIFTS
    ]
    direction_rank_rows = [("G0_CHAMPION", c["direction_loss"]), ("ALIGNED", a["direction_loss"])] + [
        (f"SHUFFLE_{k}", s[k]["direction_loss"]) for k in R5_SHIFTS
    ]
    total_ranking = [name for name, _ in sorted(total_rank_rows, key=lambda x: (x[1], x[0]))]
    direction_ranking = [name for name, _ in sorted(direction_rank_rows, key=lambda x: (x[1], x[0]))]

    return {
        "schema": "CB16_R11_R5_ALIGNMENT_REPLICATION_ADJUDICATION_V1",
        "pre_registered_rule": "ALIGNED_TOTAL_AND_DIRECTION_STRICTLY_LOWER_THAN_G0_AND_ALL_FIVE_SHUFFLES",
        "aligned_lower_total_count_vs_shuffles": int(total_wins),
        "aligned_lower_direction_count_vs_shuffles": int(direction_wins),
        "aligned_lower_sizing_count_vs_shuffles": int(sizing_wins),
        "aligned_total_better_than_champion": bool(aligned_total_better_than_champion),
        "aligned_direction_better_than_champion": bool(aligned_direction_better_than_champion),
        "total_ranking_best_to_worst": total_ranking,
        "direction_ranking_best_to_worst": direction_ranking,
        "mechanistic_alignment_replication_supported": supported,
        "conclusion": (
            "STATE_ALIGNMENT_MECHANISM_REPLICATED_ON_ONE_INDEPENDENT_PURGE_CLOCK_BLOCK__SUPPORT_LIMITED"
            if supported
            else "STATE_ALIGNMENT_NOT_REPLICATED_ON_INDEPENDENT_PURGE_SUPPORT"
        ),
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "market_information_verdict_reopened": False,
    }


__all__ = [
    "R5_RUNTIME", "R5_SHIFTS", "R5_SYMBOLS", "R5_CANDIDATE_TIMESTAMP_MS",
    "compile_validation_targets_only_r5", "adjudicate_alignment_replication_r5",
]
