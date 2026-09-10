from __future__ import annotations

"""H5.10.1 strict-clock replication of H5.10.

The H5.8 per-future-group Operator-conditional Medium48 statistic is unchanged.
Only the final half partition changes: unique decision clocks, rather than future-group
rows, are split into two contiguous chronological halves.  Every future group sharing
a decision clock is therefore assigned to the same half.  No support, normalization,
distance, null mapping, representation, Teacher, Student, or fusion weight is changed.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .medium48_temporal_halfblock_stability_h510 import (
    H510_BASELINE_TOL,
    H510_EXPECTED,
    _aggregate_group_rows_h510,
    _future_group_timestamp_h510,
    orientation_h510,
)
from .operator_conditional_medium_geometry_h58 import run_fold_h58
from .teacher_temporal_transport_audit_h5 import require

H5101_RUNTIME = "CB16_R11_H5_10_1_STRICT_CLOCK_HALFBLOCK_REPLICATION_R0_V1"
H5101_FOLDS = (1, 2, 3, 4, 5)
H5101_MIN_HALF_GROUPS = 32


def strict_clock_partition_h5101(
    group_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[list[Mapping[str, Any]]], list[list[int]]]:
    """Split by unique timestamp and never divide one timestamp across halves."""
    rows = list(group_rows)
    require(len(rows) >= 2 * H5101_MIN_HALF_GROUPS, f"H5101_TOO_FEW_FUTURE_GROUPS:{len(rows)}")
    timestamps = [_future_group_timestamp_h510(str(x["future_group_id"])) for x in rows]
    require(timestamps == sorted(timestamps), "H5101_GROUP_ORDER_NOT_CHRONOLOGICAL")

    unique_clocks = np.asarray(sorted(set(timestamps)), dtype=np.int64)
    require(len(unique_clocks) >= 2, f"H5101_TOO_FEW_UNIQUE_CLOCKS:{len(unique_clocks)}")
    clock_parts = [np.asarray(x, dtype=np.int64) for x in np.array_split(unique_clocks, 2)]
    require(len(clock_parts) == 2 and all(len(x) > 0 for x in clock_parts), "H5101_EMPTY_CLOCK_HALF")
    clock_sets = [set(int(x) for x in part.tolist()) for part in clock_parts]
    require(clock_sets[0].isdisjoint(clock_sets[1]), "H5101_CLOCK_SET_OVERLAP")
    require(max(clock_sets[0]) < min(clock_sets[1]), "H5101_CLOCK_HALVES_NOT_STRICTLY_ORDERED")

    row_parts: list[list[Mapping[str, Any]]] = []
    for clocks in clock_sets:
        part = [row for row, ts in zip(rows, timestamps) if int(ts) in clocks]
        require(len(part) >= H5101_MIN_HALF_GROUPS, f"H5101_HALF_GROUP_SHORTFALL:{len(part)}")
        row_parts.append(part)

    require(sum(len(x) for x in row_parts) == len(rows), "H5101_ROW_ASSIGNMENT_COUNT_DRIFT")
    assigned_ids = [str(x["future_group_id"]) for part in row_parts for x in part]
    require(len(assigned_ids) == len(set(assigned_ids)), "H5101_DUPLICATE_FUTURE_GROUP_ASSIGNMENT")
    original_ids = [str(x["future_group_id"]) for x in rows]
    require(set(assigned_ids) == set(original_ids), "H5101_FUTURE_GROUP_ASSIGNMENT_DRIFT")

    # Strong invariant: a decision clock occurs in exactly one output half.
    observed_by_half = [
        {_future_group_timestamp_h510(str(x["future_group_id"])) for x in part}
        for part in row_parts
    ]
    require(observed_by_half[0].isdisjoint(observed_by_half[1]), "H5101_OUTPUT_CLOCK_OVERLAP")
    require(observed_by_half == clock_sets, "H5101_OUTPUT_CLOCK_SET_DRIFT")
    return row_parts, [sorted(x) for x in clock_sets]


def run_fold_h5101(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    require(fold in H5101_FOLDS, f"H5101_UNKNOWN_FOLD:{fold}")
    base = run_fold_h58(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    group_rows = list(base["group_rows"])
    row_parts, clock_parts = strict_clock_partition_h5101(group_rows)

    halves: list[dict[str, Any]] = []
    for half_idx, (rows, clocks) in enumerate(zip(row_parts, clock_parts), start=1):
        receipt = _aggregate_group_rows_h510(rows)
        receipt["half"] = int(half_idx)
        receipt["decision_clock_count"] = int(len(clocks))
        receipt["first_decision_clock_ms"] = int(clocks[0])
        receipt["last_decision_clock_ms"] = int(clocks[-1])
        halves.append(receipt)

    half1_clocks = set(clock_parts[0]); half2_clocks = set(clock_parts[1])
    require(half1_clocks.isdisjoint(half2_clocks), f"H5101_HALF_CLOCK_OVERLAP:{fold}")
    require(max(half1_clocks) < min(half2_clocks), f"H5101_HALF_CLOCK_CHRONOLOGY_DRIFT:{fold}")
    require(int(halves[0]["last_timestamp_ms"]) < int(halves[1]["first_timestamp_ms"]), f"H5101_HALF_TIMESTAMP_OVERLAP:{fold}")

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
        "strict_clock_sets_disjoint": True,
        "strict_clock_order_passed": True,
        "same_decision_clock_never_split": True,
        "eval_dependence_groups": int(base["eval_dependence_groups"]),
        "local_support_future_groups_min": int(base["local_support_future_groups_min"]),
        "local_support_future_groups_max": int(base["local_support_future_groups_max"]),
        "zero_information_target_count": int(base["zero_information_target_count"]),
        "all_null_medium_distance_multisets_preserved": bool(base["all_null_medium_distance_multisets_preserved"]),
        "partition_boundary_uses_utility": False,
        "partition_boundary_uses_features": False,
        "change_point_search_used": False,
        "calendar_or_market_regime_labels_used": False,
        "partition_changes_support_normalization_distance_or_null": False,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
        "student_training_used": False,
        "student_inference_used": False,
        "new_representation_used": False,
        "medium_dimensions_subselected": False,
        "fusion_weights_searched": False,
        "distance_metric_tuned": False,
    }


def adjudicate_h5101(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == H5101_FOLDS, "H5101_FOLD_SET_DRIFT")
    baseline_ok = bool(all(bool(x["full_fold"]["reproduced"]) for x in rows))
    by_fold = {int(x["fold"]): x for x in rows}

    failure_persist = bool(all(
        all(str(h["orientation"]) == "ANTI_ALIGNMENT" for h in by_fold[f]["halves"])
        for f in (1, 3)
    ))
    positive_persist = bool(all(
        all(str(h["orientation"]) == "POSITIVE_ALIGNMENT" for h in by_fold[f]["halves"])
        for f in (2, 4, 5)
    ))

    if not baseline_ok:
        classification = "H5_10_1_BASELINE_REPRODUCTION_FAILED"
    elif failure_persist and positive_persist:
        classification = "STRICT_CLOCK_WITHIN_FOLD_RELATION_STABLE__CROSS_FOLD_SIGN_REVERSAL"
    elif failure_persist and not positive_persist:
        classification = "STRICT_CLOCK_FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED"
    elif (not failure_persist) and positive_persist:
        classification = "STRICT_CLOCK_FAILURE_FOLDS_INTERNALLY_MIXED__POSITIVE_FOLDS_STABLE"
    else:
        classification = "STRICT_CLOCK_TEMPORAL_RELATION_MIXED_WITHIN_AND_ACROSS_FOLDS"

    sequence: list[dict[str, Any]] = []
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
                "decision_clock_count": int(half["decision_clock_count"]),
                "first_timestamp_ms": int(half["first_timestamp_ms"]),
                "last_timestamp_ms": int(half["last_timestamp_ms"]),
            })

    return {
        "classification": classification,
        "conclusion": f"H5_10_1_{classification}",
        "baseline_reproduction_passed": baseline_ok,
        "failure_fold_persistence_4_of_4": failure_persist,
        "positive_fold_persistence_6_of_6": positive_persist,
        "strict_clock_half_sequence": sequence,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "time_gating_authorized": False,
        "handcrafted_regime_activation_authorized": False,
    }
