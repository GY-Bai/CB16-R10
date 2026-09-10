from __future__ import annotations

"""H5.9 breadth audit for the frozen H5.8 Medium48 conditional anti-alignment.

This module does not invent a new representation or split the dense Medium48 adapter.
It recomputes the exact H5.8 per-anchor statistic on the two preregistered failure
folds, then changes only the final aggregation by leaving out one frozen symbol or
one frozen account scenario at a time. Support, normalization, distances, utility
truth, and structured Medium null mappings are all computed before any omission.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .operator_conditional_medium_geometry_h58 import H58_SHIFTS, _scenario_target_h58
from .teacher_temporal_transport_audit_h5 import (
    H5_SCENARIOS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    require,
)
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11
from .time_local_vs_forward_geometry_contrast_h56 import _same_scenario_leave_group_out_normalization_h56

H59_RUNTIME = "CB16_R11_H5_9_MEDIUM48_ANTIALIGNMENT_BREADTH_R0_V1"
H59_FOLDS = (1, 3)
H59_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "LINKUSDT",
    "LTCUSDT",
    "SOLUSDT",
)
H59_BASELINE = {
    1: {"aligned_partial_rho": -0.012411348281499165, "null_median": 0.027034529350260694},
    3: {"aligned_partial_rho": -0.06450510517280354, "null_median": -0.010316589086094265},
}
H59_BASELINE_TOL = 1e-12


def _anti_alignment_h59(aligned: float, null_median: float) -> bool:
    return bool(float(aligned) < 0.0 and float(aligned) < float(null_median))


def _aggregate_records_h59(
    records: Sequence[Mapping[str, Any]],
    *,
    omit_symbol: str | None = None,
    omit_scenario: str | None = None,
) -> dict[str, Any]:
    require(not (omit_symbol is not None and omit_scenario is not None), "H59_DOUBLE_OMISSION_FORBIDDEN")
    selected = [
        r
        for r in records
        if (omit_symbol is None or str(r["symbol"]) != str(omit_symbol))
        and (omit_scenario is None or str(r["scenario"]) != str(omit_scenario))
    ]
    require(selected, "H59_EMPTY_AGGREGATION")

    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for r in selected:
        by_group.setdefault(str(r["future_group_id"]), []).append(r)

    expected_scenarios = len(H5_SCENARIOS) - (1 if omit_scenario is not None else 0)
    require(expected_scenarios >= 1, "H59_BAD_EXPECTED_SCENARIOS")
    group_aligned: list[float] = []
    group_nulls: dict[int, list[float]] = {int(s): [] for s in H58_SHIFTS}
    for gid in sorted(by_group):
        rows = by_group[gid]
        require(len(rows) == expected_scenarios, f"H59_GROUP_SCENARIO_COUNT_DRIFT:{gid}:{len(rows)}")
        require(len({str(r['scenario']) for r in rows}) == expected_scenarios, f"H59_GROUP_SCENARIO_DUPLICATE:{gid}")
        group_aligned.append(float(np.mean([float(r["aligned_partial_rho"]) for r in rows])))
        for shift in H58_SHIFTS:
            group_nulls[int(shift)].append(
                float(np.mean([float(r["null_partial_rho_by_shift"][int(shift)]) for r in rows]))
            )

    aligned = float(np.mean(group_aligned))
    nulls = {int(s): float(np.mean(group_nulls[int(s)])) for s in H58_SHIFTS}
    null_median = float(statistics.median(nulls.values()))
    return {
        "aligned_partial_rho": aligned,
        "null_partial_rho_by_shift": nulls,
        "null_median_partial_rho": null_median,
        "aligned_minus_null_median": float(aligned - null_median),
        "anti_alignment": _anti_alignment_h59(aligned, null_median),
        "included_future_groups": int(len(by_group)),
        "included_target_rows": int(len(selected)),
        "included_scenarios_per_future_group": int(expected_scenarios),
    }


def _group_symbol_map_h59(eval_parents: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in eval_parents.values():
        gid = str(p.dependence_group_id)
        symbol = str(p.symbol)
        if gid in out:
            require(out[gid] == symbol, f"H59_GROUP_SYMBOL_DRIFT:{gid}")
        else:
            out[gid] = symbol
    return out


def run_fold_h59(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    require(fold in H59_FOLDS, f"H59_NONSCIENTIFIC_FOLD_REQUESTED:{fold}")
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    index = build_columnar_teacher_index_r11(samples)
    eval_parent_ids = tuple(sorted(eval_parents))
    eval_order, eval_rows_map = _eval_group_scenario_rows_h5(
        index=index, eval_parents=eval_parents, eval_parent_ids=eval_parent_ids
    )
    symbol_by_group = _group_symbol_map_h59(eval_parents)
    require(set(symbol_by_group) == set(eval_order), "H59_GROUP_SYMBOL_SET_DRIFT")
    observed_symbols = {symbol_by_group[g] for g in eval_order}
    require(observed_symbols == set(H59_SYMBOLS), f"H59_SYMBOL_SET_DRIFT:{sorted(observed_symbols)}")
    require(len(H5_SCENARIOS) == 6, f"H59_SCENARIO_COUNT_DRIFT:{len(H5_SCENARIOS)}")

    normalizers = _same_scenario_leave_group_out_normalization_h56(
        index=index, eval_order=eval_order, eval_rows_map=eval_rows_map
    )

    records: list[dict[str, Any]] = []
    all_multisets = True
    min_support = 10**9
    max_support = 0
    zero_information_count = 0
    for gid in eval_order:
        for scenario in H5_SCENARIOS:
            mean, std = normalizers[(str(gid), str(scenario))]
            row = _scenario_target_h58(
                index=index,
                gid=str(gid),
                scenario=str(scenario),
                eval_order=eval_order,
                eval_rows_map=eval_rows_map,
                mean=mean,
                std=std,
            )
            all_multisets = bool(
                all_multisets and all(bool(x) for x in row["null_medium_distance_multiset_identity"].values())
            )
            min_support = min(min_support, int(row["support_future_group_count"]))
            max_support = max(max_support, int(row["support_future_group_count"]))
            zero_information_count += int(bool(row["zero_information"]))
            records.append(
                {
                    "future_group_id": str(gid),
                    "symbol": str(symbol_by_group[str(gid)]),
                    "scenario": str(scenario),
                    "aligned_partial_rho": float(row["aligned_partial_rho"]),
                    "null_partial_rho_by_shift": {
                        int(s): float(row["null_partial_rho_by_shift"][int(s)]) for s in H58_SHIFTS
                    },
                }
            )

    baseline = _aggregate_records_h59(records)
    symbol_loo = {
        symbol: _aggregate_records_h59(records, omit_symbol=symbol) for symbol in H59_SYMBOLS
    }
    scenario_loo = {
        scenario: _aggregate_records_h59(records, omit_scenario=scenario) for scenario in H5_SCENARIOS
    }
    symbol_robust = bool(all(x["anti_alignment"] for x in symbol_loo.values()))
    scenario_robust = bool(all(x["anti_alignment"] for x in scenario_loo.values()))

    return {
        "fold": fold,
        "baseline": baseline,
        "symbol_leave_one_out": symbol_loo,
        "scenario_leave_one_out": scenario_loo,
        "symbol_loo_robust": symbol_robust,
        "scenario_loo_robust": scenario_robust,
        "symbol_sensitive_omissions": [k for k, v in symbol_loo.items() if not bool(v["anti_alignment"])],
        "scenario_sensitive_omissions": [k for k, v in scenario_loo.items() if not bool(v["anti_alignment"])],
        "symbol_loo_min_null_minus_aligned_margin": float(
            min(float(v["null_median_partial_rho"]) - float(v["aligned_partial_rho"]) for v in symbol_loo.values())
        ),
        "scenario_loo_min_null_minus_aligned_margin": float(
            min(float(v["null_median_partial_rho"]) - float(v["aligned_partial_rho"]) for v in scenario_loo.values())
        ),
        "symbol_loo_max_aligned_partial_rho": float(max(float(v["aligned_partial_rho"]) for v in symbol_loo.values())),
        "scenario_loo_max_aligned_partial_rho": float(max(float(v["aligned_partial_rho"]) for v in scenario_loo.values())),
        "eval_dependence_groups": int(len(eval_order)),
        "target_row_count": int(len(records)),
        "local_support_future_groups_min": int(min_support),
        "local_support_future_groups_max": int(max_support),
        "zero_information_target_count": int(zero_information_count),
        "all_null_medium_distance_multisets_preserved": bool(all_multisets),
        "omission_changes_support_or_normalization": False,
        "omission_changes_null_mapping": False,
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


def adjudicate_h59(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == H59_FOLDS, "H59_FOLD_SET_DRIFT")

    baseline_ok = True
    baseline_receipt: dict[str, Any] = {}
    for row in rows:
        fold = int(row["fold"])
        expected = H59_BASELINE[fold]
        aligned = float(row["baseline"]["aligned_partial_rho"])
        null_median = float(row["baseline"]["null_median_partial_rho"])
        aerr = abs(aligned - float(expected["aligned_partial_rho"]))
        nerr = abs(null_median - float(expected["null_median"]))
        anti = bool(row["baseline"]["anti_alignment"])
        ok = bool(aerr <= H59_BASELINE_TOL and nerr <= H59_BASELINE_TOL and anti)
        baseline_ok = bool(baseline_ok and ok)
        baseline_receipt[str(fold)] = {
            "aligned_partial_rho": aligned,
            "expected_aligned_partial_rho": float(expected["aligned_partial_rho"]),
            "aligned_abs_error": float(aerr),
            "null_median_partial_rho": null_median,
            "expected_null_median_partial_rho": float(expected["null_median"]),
            "null_median_abs_error": float(nerr),
            "anti_alignment": anti,
            "reproduced": ok,
        }

    symbol_both = bool(all(bool(x["symbol_loo_robust"]) for x in rows))
    scenario_both = bool(all(bool(x["scenario_loo_robust"]) for x in rows))

    if not baseline_ok:
        classification = "H5_9_BASELINE_REPRODUCTION_FAILED"
    elif symbol_both and scenario_both:
        classification = "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_BROAD_ACROSS_SYMBOLS_AND_SCENARIOS"
    elif (not symbol_both) and scenario_both:
        classification = "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_SENSITIVE__SCENARIO_ROBUST"
    elif symbol_both and (not scenario_both):
        classification = "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_ROBUST__SCENARIO_SENSITIVE"
    else:
        classification = "MEDIUM48_CONDITIONAL_ANTI_ALIGNMENT_SYMBOL_AND_SCENARIO_SENSITIVE_OR_MIXED"

    return {
        "classification": classification,
        "conclusion": f"H5_9_{classification}",
        "baseline_reproduction_passed": bool(baseline_ok),
        "baseline_receipt": baseline_receipt,
        "symbol_loo_robust_both_failure_folds": symbol_both,
        "scenario_loo_robust_both_failure_folds": scenario_both,
        "per_fold": {
            str(int(x["fold"])): {
                "symbol_loo_robust": bool(x["symbol_loo_robust"]),
                "scenario_loo_robust": bool(x["scenario_loo_robust"]),
                "symbol_sensitive_omissions": list(x["symbol_sensitive_omissions"]),
                "scenario_sensitive_omissions": list(x["scenario_sensitive_omissions"]),
                "symbol_loo_min_null_minus_aligned_margin": float(x["symbol_loo_min_null_minus_aligned_margin"]),
                "scenario_loo_min_null_minus_aligned_margin": float(x["scenario_loo_min_null_minus_aligned_margin"]),
                "symbol_loo_max_aligned_partial_rho": float(x["symbol_loo_max_aligned_partial_rho"]),
                "scenario_loo_max_aligned_partial_rho": float(x["scenario_loo_max_aligned_partial_rho"]),
            }
            for x in rows
        },
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "organ_change_authorized": False,
        "handcrafted_regime_activation_authorized": False,
    }
