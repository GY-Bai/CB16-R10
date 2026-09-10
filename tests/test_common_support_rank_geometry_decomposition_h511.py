from __future__ import annotations

import numpy as np

from cb16_local_opt.common_support_rank_geometry_decomposition_h511 import (
    H511_NATIVE_ORIENTATION,
    H511_PAIRS,
    _post_weight_cell_mass_h511,
    _raw_rank_h511,
    adjudicate_h511,
    build_pair_overlap_design_h511,
    cell_ids_h511,
    rank_percentile_h511,
    weighted_partial_rank_h511,
)
from cb16_local_opt.operator_conditional_medium_geometry_h58 import partial_spearman_rank_h58


def test_h511_cell_boundaries_are_frozen_right_open_except_final():
    p = np.asarray([0.0, 1.0 / 3.0 - 1e-10, 1.0 / 3.0, 2.0 / 3.0 - 1e-10, 2.0 / 3.0, 1.0])
    cells = cell_ids_h511(p, p)
    bins = [int(x) // 3 for x in cells]
    assert bins == [0, 0, 1, 1, 2, 2]
    assert [int(x) % 3 for x in cells] == [0, 0, 1, 1, 2, 2]


def test_h511_rank_percentile_uses_average_ties_and_endpoints():
    x = np.asarray([10.0, 20.0, 20.0, 40.0])
    rank, pct = rank_percentile_h511(x)
    assert np.allclose(rank, [1.0, 2.5, 2.5, 4.0])
    assert np.allclose(pct, [0.0, 0.5, 0.5, 1.0])


def _design(fold: int, p, clocks=5):
    return {
        "fold": fold,
        "cell_mass": list(map(float, p)),
        "unique_target_decision_clocks_by_cell": [int(clocks) if float(x) > 0 else 0 for x in p],
        "utility_accessed": False,
    }


def test_h511_overlap_q_is_discrete_symmetric_overlap_target():
    pa = np.asarray([0.2, 0.3, 0.0, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1])
    pb = np.asarray([0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1])
    pa = pa / pa.sum(); pb = pb / pb.sum()
    d = build_pair_overlap_design_h511(_design(1, pa), _design(2, pb))
    raw = np.zeros(9)
    mask = (pa > 0) & (pb > 0)
    raw[mask] = pa[mask] * pb[mask] / (pa[mask] + pb[mask])
    expected = raw / raw.sum()
    assert np.allclose(d["q"], expected)
    assert d["q"][2] == 0.0
    assert d["common_support_valid"] is True
    assert d["utility_used_to_construct_q"] is False
    assert d["propensity_model_fit"] is False


def test_h511_overlap_multiplier_exactly_maps_each_environment_to_q():
    pa = np.asarray([0.10, 0.15, 0.05, 0.10, 0.15, 0.10, 0.10, 0.10, 0.05])
    pb = np.asarray([0.05, 0.10, 0.10, 0.15, 0.10, 0.10, 0.15, 0.05, 0.10])
    d = build_pair_overlap_design_h511(_design(1, pa), _design(2, pb))
    q = np.asarray(d["q"])
    qa = _post_weight_cell_mass_h511(pa, np.asarray(d["multiplier_a"]))
    qb = _post_weight_cell_mass_h511(pb, np.asarray(d["multiplier_b"]))
    assert np.max(np.abs(qa - q)) < 1e-14
    assert np.max(np.abs(qb - q)) < 1e-14


def test_h511_common_support_counts_unique_target_clocks_not_pair_rows():
    p = np.asarray([0.2, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.0, 0.4])
    a = _design(1, p, clocks=5)
    b = _design(2, p, clocks=5)
    # One positive-overlap cell has only two independent target decision clocks.
    a["unique_target_decision_clocks_by_cell"][8] = 2
    d = build_pair_overlap_design_h511(a, b)
    assert d["clock_support_gate_passed"] is False
    assert d["common_support_valid"] is False


def test_h511_common_support_requires_both_rank_axes_to_span():
    p = np.asarray([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = build_pair_overlap_design_h511(_design(1, p), _design(2, p))
    assert d["operator_bins_spanned"] == [0]
    assert d["axis_span_gate_passed"] is False
    assert d["common_support_valid"] is False


def test_h511_uniform_weight_partial_rank_reproduces_h58():
    operator = np.asarray([0.8, 0.1, 0.4, 0.7, 0.2, 0.5, 0.9, 0.3, 0.6])
    medium = np.asarray([0.4, 0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.5])
    utility = np.asarray([0.7, 0.2, 0.8, 0.1, 0.5, 0.9, 0.3, 0.6, 0.4])
    expected, _ = partial_spearman_rank_h58(operator, medium, utility)
    got, receipt = weighted_partial_rank_h511(
        _raw_rank_h511(operator),
        _raw_rank_h511(medium),
        _raw_rank_h511(utility),
        np.ones(len(operator)),
    )
    assert abs(got - expected) <= 1e-12
    assert receipt["zero_information"] is False


def _native_rows():
    return [
        {"fold": f, "reproduced": True, "orientation": H511_NATIVE_ORIENTATION[f]}
        for f in range(1, 6)
    ]


def _pair_result(a, b, oa, ob, valid=True):
    return {"pair": [a, b], "common_support_valid": valid, "side_a": {"orientation": oa}, "side_b": {"orientation": ob}}


def test_h511_adjudication_strong_conditional_mapping_class_requires_3_of_3_plus_control():
    pairs = [
        _pair_result(1, 2, "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair_result(2, 3, "POSITIVE_ALIGNMENT", "ANTI_ALIGNMENT"),
        _pair_result(3, 4, "ANTI_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair_result(4, 5, "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
    ]
    s = adjudicate_h511(_native_rows(), pairs)
    assert s["classification"] == "COARSENED_COMMON_SUPPORT_CONDITIONAL_MAPPING_INSTABILITY_SUPPORTED"
    assert s["native_flip_persistence_count_of_3"] == 3
    assert s["positive_control_pair_stable"] is True


def test_h511_adjudication_strong_occupancy_class_requires_clean_removal_plus_control():
    pairs = [
        _pair_result(1, 2, "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair_result(2, 3, "ANTI_ALIGNMENT", "ANTI_ALIGNMENT"),
        _pair_result(3, 4, "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
        _pair_result(4, 5, "POSITIVE_ALIGNMENT", "POSITIVE_ALIGNMENT"),
    ]
    s = adjudicate_h511(_native_rows(), pairs)
    assert s["classification"] == "COARSENED_OCCUPANCY_STANDARDIZATION_REMOVES_NATIVE_ORIENTATION_TRANSITIONS"
    assert s["native_flip_clean_removal_count_of_3"] == 3


def test_h511_adjudication_fails_closed_on_baseline_or_support():
    native = _native_rows(); native[0] = dict(native[0], reproduced=False)
    pairs = [_pair_result(*p, H511_NATIVE_ORIENTATION[p[0]], H511_NATIVE_ORIENTATION[p[1]]) for p in H511_PAIRS]
    assert adjudicate_h511(native, pairs)["classification"] == "EXECUTION_BLOCKED__H5_8_BASELINE_REPRODUCTION_FAILED"
    native = _native_rows(); pairs[0] = dict(pairs[0], common_support_valid=False)
    assert adjudicate_h511(native, pairs)["classification"] == "EXECUTION_BLOCKED__INSUFFICIENT_COMMON_SUPPORT"
