from __future__ import annotations

"""H5.10 temporal half-block stability audit.

This module reuses the exact H5.8 per-future-group Operator-conditional Medium48
statistics and changes only final target aggregation.  Each already-frozen R6 outer
evaluation block is split into two contiguous near-equal-count halves in the existing
chronological future-group order.  No support, normalization, distance, null mapping,
representation, Teacher, Student, or fusion weight is changed.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .operator_conditional_medium_geometry_h58 import H58_SHIFTS, run_fold_h58
from .teacher_temporal_transport_audit_h5 import require

H510_RUNTIME = "CB16_R11_H5_10_MEDIUM48_TEMPORAL_HALFBLOCK_STABILITY_R0_V1"
H510_FOLDS = (1, 2, 3, 4, 5)
H510_BASELINE_TOL = 1e-12
H510_EXPECTED = {
    1: {"aligned_partial_rho": -0.012411348281499165, "null_median": 0.027034529350260694, "orientation": "ANTI_ALIGNMENT"},
    2: {"aligned_partial_rho": 0.07585785741564523, "null_median": 0.02033114021581147, "orientation": "POSITIVE_ALIGNMENT"},
    3: {"aligned_partial_rho": -0.06450510517280354, "null_median": -0.010316589086094265, "orientation": "ANTI_ALIGNMENT"},
    4: {"aligned_partial_rho": 0.027351639484904336, "null_median": 0.006963813629549571, "orientation": "POSITIVE_ALIGNMENT"},
    5: {"aligned_partial_rho": 0.024765647270181372, "null_median": 0.01168198726845212, "orientation": "POSITIVE_ALIGNMENT"},
}


def orientation_h510(aligned: float, null_median: float) -> str:
    a = float(aligned)
    n = float(null_median)
    if a > 0.0 and a > n:
        return "POSITIVE_ALIGNMENT"
    if a < 0.0 and a < n:
        return "ANTI_ALIGNMENT"
    return "MIXED"


def _future_group_timestamp_h510(gid: str) -> int:
    parts = str(gid).rsplit(":", 1)
    require(len(parts) == 2 and parts[1].isdigit(), f"H510_GROUP_ID_TIMESTAMP_PARSE:{gid}")
    return int(parts[1])


def _aggregate_group_rows_h510(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(rows) >= 1, "H510_EMPTY_HALF")
    aligned = float(np.mean([float(x["aligned_partial_rho"]) for x in rows]))
    nulls = {
        int(s): float(np.mean([float(x["null_partial_rho_by_shift"][int(s)]) for x in rows]))
        for s in H58_SHIFTS
    }
    med = float(statistics.median(nulls.values()))
    ts = [_future_group_timestamp_h510(str(x["future_group_id"])) for x in rows]
    return {
        "future_group_count": int(len(rows)),
        "first_timestamp_ms": int(min(ts)),
        "last_timestamp_ms": int(max(ts)),
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "null_median_partial_rho": med,
        "aligned_minus_null_median": float(aligned - med),
        "orientation": orientation_h510(aligned, med),
    }


def run_fold_h510(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    require(fold in H510_FOLDS, f"H510_UNKNOWN_FOLD:{fold}")
    base = run_fold_h58(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    group_rows = list(base["group_rows"])
    require(len(group_rows) >= 64, f"H510_TOO_FEW_FUTURE_GROUPS:{fold}:{len(group_rows)}")
    timestamps = [_future_group_timestamp_h510(str(x["future_group_id"])) for x in group_rows]
    require(timestamps == sorted(timestamps), f"H510_GROUP_ORDER_NOT_CHRONOLOGICAL:{fold}")

    indices = np.array_split(np.arange(len(group_rows), dtype=np.int32), 2)
    require(len(indices) == 2 and all(len(x) >= 32 for x in indices), f"H510_HALF_SIZE_DRIFT:{fold}")
    halves = []
    for half_idx, idx in enumerate(indices, start=1):
        rows = [group_rows[int(i)] for i in idx]
        receipt = _aggregate_group_rows_h510(rows)
        receipt["half"] = int(half_idx)
        halves.append(receipt)

    require(int(halves[0]["last_timestamp_ms"]) <= int(halves[1]["first_timestamp_ms"]), f"H510_HALF_CHRONOLOGY_DRIFT:{fold}")
    expected = H510_EXPECTED[fold]
    aerr = abs(float(base["aligned_partial_rho"]) - float(expected["aligned_partial_rho"]))
    nerr = abs(float(base["median_null_partial_rho"]) - float(expected["null_median"]))
    reproduced = bool(aerr <= H510_BASELINE_TOL and nerr <= H510_BASELINE_TOL)

    return {
        "fold": fold,
        "full_fold": {
            "aligned_partial_rho": float(base["aligned_partial_rho"]),
            "null_median_partial_rho": float(base["median_null_partial_rho"]),
            "aligned_minus_null_median": float(base["aligned_minus_null_median"]),
            "orientation": orientation_h510(float(base["aligned_partial_rho"]), float(base["median_null_partial_rho"])),
            "expected_orientation": str(expected["orientation"]),
            "aligned_abs_error_vs_h5_8": float(aerr),
            "null_median_abs_error_vs_h5_8": float(nerr),
            "reproduced": reproduced,
        },
        "halves": halves,
        "halves_match_parent_orientation": bool(all(str(x["orientation"]) == str(expected["orientation"]) for x in halves)),
        "eval_dependence_groups": int(base["eval_dependence_groups"]),
        "local_support_future_groups_min": int(base["local_support_future_groups_min"]),
        "local_support_future_groups_max": int(base["local_support_future_groups_max"]),
        "zero_information_target_count": int(base["zero_information_target_count"]),
        "all_null_medium_distance_multisets_preserved": bool(base["all_null_medium_distance_multisets_preserved"]),
        "partition_boundary_uses_utility": False,
        "partition_boundary_uses_features": False,
        "partition_changes_support_normalization_distance_or_null": False,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
        "student_training_used": False,
        "student_inference_used": False,
        "new_representation_used": False,
        "fusion_weights_searched": False,
        "distance_metric_tuned": False,
    }


def adjudicate_h510(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == H510_FOLDS, "H510_FOLD_SET_DRIFT")
    baseline_ok = bool(all(bool(x["full_fold"]["reproduced"]) for x in rows))

    by_fold = {int(x["fold"]): x for x in rows}
    failure_persist = bool(
        all(
            all(str(h["orientation"]) == "ANTI_ALIGNMENT" for h in by_fold[f]["halves"])
            for f in (1, 3)
        )
    )
    positive_persist = bool(
        all(
            all(str(h["orientation"]) == "POSITIVE_ALIGNMENT" for h in by_fold[f]["halves"])
            for f in (2, 4, 5)
        )
    )

    if not baseline_ok:
        classification = "H5_10_BASELINE_REPRODUCTION_FAILED"
    elif failure_persist and positive_persist:
        classification = "WITHIN_FOLD_RELATION_STABLE__CROSS_FOLD_SIGN_REVERSAL"
    elif failure_persist and not positive_persist:
        classification = "FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED"
    elif (not failure_persist) and positive_persist:
        classification = "FAILURE_FOLDS_INTERNALLY_MIXED__POSITIVE_FOLDS_STABLE"
    else:
        classification = "TEMPORAL_RELATION_MIXED_WITHIN_AND_ACROSS_FOLDS"

    sequence = []
    for row in rows:
        for half in row["halves"]:
            sequence.append({
                "fold": int(row["fold"]),
                "half": int(half["half"]),
                "orientation": str(half["orientation"]),
                "aligned_partial_rho": float(half["aligned_partial_rho"]),
                "null_median_partial_rho": float(half["null_median_partial_rho"]),
                "aligned_minus_null_median": float(half["aligned_minus_null_median"]),
                "future_group_count": int(half["future_group_count"]),
                "first_timestamp_ms": int(half["first_timestamp_ms"]),
                "last_timestamp_ms": int(half["last_timestamp_ms"]),
            })

    return {
        "classification": classification,
        "conclusion": f"H5_10_{classification}",
        "baseline_reproduction_passed": baseline_ok,
        "failure_fold_persistence_4_of_4": failure_persist,
        "positive_fold_persistence_6_of_6": positive_persist,
        "all_fold_orientation_stability": bool(failure_persist and positive_persist),
        "chronological_half_sequence": sequence,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "time_gating_authorized": False,
        "handcrafted_regime_activation_authorized": False,
    }
