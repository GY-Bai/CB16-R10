from __future__ import annotations

import inspect

import numpy as np

from cb16_local_opt.medium48_strict_clock_halfblock_replication_h5101 import (
    H5101_FOLDS,
    adjudicate_h5101,
    run_fold_h5101,
    strict_clock_partition_h5101,
)
from cb16_local_opt.medium48_temporal_halfblock_stability_h510 import H510_EXPECTED


def _row(symbol: str, ts: int, aligned: float = 0.1, null: float = 0.0) -> dict:
    return {
        "future_group_id": f"FUT:{symbol}:{ts}",
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": {1: null, 7: null, 13: null, 23: null, 31: null},
    }


def test_h5101_frozen_folds_match_parent_fold_set():
    assert H5101_FOLDS == (1, 2, 3, 4, 5)
    assert tuple(H510_EXPECTED) == H5101_FOLDS


def test_run_fold_signature_has_no_teacher_student_weight_or_boundary_inputs():
    params = set(inspect.signature(run_fold_h5101).parameters)
    forbidden = {
        "teacher",
        "teacher_config",
        "kernel",
        "student",
        "organ_weight",
        "fusion_weight",
        "distance_weight",
        "change_point",
        "boundary",
        "regime_selector",
        "clock_boundary",
    }
    assert not (params & forbidden)


def test_strict_clock_partition_never_splits_a_shared_clock_even_when_group_count_split_would():
    # 101 ordered future groups across four clocks.  A direct np.array_split over
    # rows creates 51/50 groups and necessarily puts clock=3000 in both halves.
    counts = [(1000, 30), (2000, 20), (3000, 30), (4000, 21)]
    rows = []
    for ts, n in counts:
        rows.extend(_row(f"S{i:02d}", ts) for i in range(n))
    assert len(rows) == 101

    raw_indices = np.array_split(np.arange(len(rows)), 2)
    raw_clock_sets = [
        {int(rows[int(i)]["future_group_id"].rsplit(":", 1)[1]) for i in idx}
        for idx in raw_indices
    ]
    assert raw_clock_sets[0] & raw_clock_sets[1] == {3000}

    parts, clocks = strict_clock_partition_h5101(rows)
    assert [len(x) for x in parts] == [50, 51]
    assert clocks == [[1000, 2000], [3000, 4000]]
    assert set(clocks[0]).isdisjoint(clocks[1])
    assert max(clocks[0]) < min(clocks[1])
    part_clock_sets = [
        {int(x["future_group_id"].rsplit(":", 1)[1]) for x in part}
        for part in parts
    ]
    assert part_clock_sets == [set(clocks[0]), set(clocks[1])]


def test_strict_clock_partition_preserves_all_rows_exactly_once():
    rows = []
    for ts in range(1000, 9000, 1000):
        rows.extend(_row(f"S{i:02d}", ts) for i in range(10))
    parts, clocks = strict_clock_partition_h5101(rows)
    original = [x["future_group_id"] for x in rows]
    observed = [x["future_group_id"] for part in parts for x in part]
    assert len(observed) == len(original) == 80
    assert set(observed) == set(original)
    assert len(observed) == len(set(observed))
    assert set(clocks[0]).isdisjoint(clocks[1])


def _fold_result(fold: int, half_orientations: tuple[str, str], *, reproduce: bool = True) -> dict:
    halves = []
    for i, orient in enumerate(half_orientations, start=1):
        if orient == "ANTI_ALIGNMENT":
            aligned, null = -0.2, -0.1
        elif orient == "POSITIVE_ALIGNMENT":
            aligned, null = 0.2, 0.1
        else:
            aligned, null = 0.05, 0.1
        halves.append({
            "half": i,
            "orientation": orient,
            "aligned_partial_rho": aligned,
            "null_median_partial_rho": null,
            "aligned_minus_null_median": aligned - null,
            "future_group_count": 40,
            "decision_clock_count": 4,
            "first_timestamp_ms": fold * 100000 + i * 1000,
            "last_timestamp_ms": fold * 100000 + i * 1000 + 500,
        })
    return {
        "fold": fold,
        "full_fold": {"reproduced": reproduce},
        "halves": halves,
    }


def _stable_rows() -> list[dict]:
    return [
        _fold_result(1, ("ANTI_ALIGNMENT", "ANTI_ALIGNMENT")),
        _fold_result(2, ("POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT")),
        _fold_result(3, ("ANTI_ALIGNMENT", "ANTI_ALIGNMENT")),
        _fold_result(4, ("POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT")),
        _fold_result(5, ("POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT")),
    ]


def test_adjudication_strict_clock_full_stability_classification():
    out = adjudicate_h5101(_stable_rows())
    assert out["baseline_reproduction_passed"] is True
    assert out["failure_fold_persistence_4_of_4"] is True
    assert out["positive_fold_persistence_6_of_6"] is True
    assert out["classification"] == "STRICT_CLOCK_WITHIN_FOLD_RELATION_STABLE__CROSS_FOLD_SIGN_REVERSAL"


def test_adjudication_strict_clock_failure_stable_positive_mixed_classification():
    rows = _stable_rows()
    rows[4] = _fold_result(5, ("MIXED", "POSITIVE_ALIGNMENT"))
    out = adjudicate_h5101(rows)
    assert out["failure_fold_persistence_4_of_4"] is True
    assert out["positive_fold_persistence_6_of_6"] is False
    assert out["classification"] == "STRICT_CLOCK_FAILURE_FOLDS_STABLE__POSITIVE_FOLDS_INTERNALLY_MIXED"


def test_adjudication_strict_clock_failure_mixed_positive_stable_classification():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "MIXED"))
    out = adjudicate_h5101(rows)
    assert out["failure_fold_persistence_4_of_4"] is False
    assert out["positive_fold_persistence_6_of_6"] is True
    assert out["classification"] == "STRICT_CLOCK_FAILURE_FOLDS_INTERNALLY_MIXED__POSITIVE_FOLDS_STABLE"


def test_adjudication_strict_clock_mixed_both_sides_classification():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "MIXED"))
    rows[4] = _fold_result(5, ("MIXED", "POSITIVE_ALIGNMENT"))
    out = adjudicate_h5101(rows)
    assert out["classification"] == "STRICT_CLOCK_TEMPORAL_RELATION_MIXED_WITHIN_AND_ACROSS_FOLDS"


def test_adjudication_fails_closed_when_h58_baseline_does_not_reproduce():
    rows = _stable_rows()
    rows[0] = _fold_result(1, ("ANTI_ALIGNMENT", "ANTI_ALIGNMENT"), reproduce=False)
    out = adjudicate_h5101(rows)
    assert out["baseline_reproduction_passed"] is False
    assert out["classification"] == "H5_10_1_BASELINE_REPRODUCTION_FAILED"
