from __future__ import annotations

"""M-series M1: read-only sufficiency audit for reduced Teacher Direction targets."""

from typing import Any, Mapping, Sequence

import numpy as np

M1_RUNTIME = "CB16_R11_M_SERIES_M1_DIRECTION_RELATIVE_SUFFICIENCY_V1"
TAU = 0.002
LOGIT_CLIP = 60.0
GAP_SATURATION = TAU * LOGIT_CLIP
DIRECTION_VALUES = (-1, 0, 1)
PAIR_INDEX = ((0, 1), (0, 2), (1, 2))


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def _prob_matrix(x: Any) -> np.ndarray:
    p = np.asarray(x, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3, "M1_PROB_SHAPE")
    require(np.isfinite(p).all(), "M1_PROB_NONFINITE")
    require(np.all(p > 0.0), "M1_PROB_NONPOSITIVE")
    # Reduced Teacher probabilities originate in float64 while G0 probabilities originate
    # in the canonical FP32 Student. M1 tests information sufficiency, not serialization-level
    # probability normalization, so accept the same numerical scale as the frozen runtime.
    require(np.all(np.abs(p.sum(axis=1) - 1.0) <= 1e-5), "M1_PROB_NOT_DISTRIBUTION")
    return p


def best_direction_geometry_from_action_laws_m1(evidence: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    means = np.empty((len(evidence), 3), dtype=np.float64)
    risks = np.empty((len(evidence), 3), dtype=np.float64)
    for row, e in enumerate(evidence):
        laws = tuple(getattr(e, "action_laws", ()))
        require(bool(laws), f"M1_ACTION_LAWS_MISSING:{row}")
        for cls, direction in enumerate(DIRECTION_VALUES):
            candidates = [law for law in laws if int(law.direction) == direction]
            require(bool(candidates), f"M1_DIRECTION_LAW_MISSING:{row}:{direction}")
            best = max(
                candidates,
                key=lambda law: (float(law.mean_utility), -float(law.requested_risk)),
            )
            means[row, cls] = float(best.mean_utility)
            risks[row, cls] = float(best.requested_risk)
    require(np.isfinite(means).all(), "M1_MEAN_NONFINITE")
    require(np.isfinite(risks).all(), "M1_RISK_NONFINITE")
    return means, risks


def audit_direction_sufficiency_m1(
    *,
    evidence: Sequence[Any],
    g0_direction_probs: Any,
) -> dict[str, Any]:
    p = _prob_matrix([e.direction_target_probs for e in evidence])
    g0p = _prob_matrix(g0_direction_probs)
    require(len(p) == len(g0p), "M1_G0_ROW_COUNT")
    n = len(p)
    means, risks = best_direction_geometry_from_action_laws_m1(evidence)
    rich_best = np.argmax(means, axis=1).astype(np.int64)
    reduced_best = np.argmax(p, axis=1).astype(np.int64)
    g0_dir = np.argmax(g0p, axis=1).astype(np.int64)
    rows = np.arange(n)

    actual_adv = means[rows, rich_best] - means[rows, g0_dir]
    actual_adv[np.abs(actual_adv) < 1e-15] = 0.0
    require(np.all(actual_adv >= -1e-12), "M1_NEGATIVE_BEST_ADVANTAGE")
    actual_adv = np.maximum(actual_adv, 0.0)
    implied_adv = TAU * (np.log(p[rows, reduced_best]) - np.log(p[rows, g0_dir]))
    implied_adv[np.abs(implied_adv) < 1e-15] = 0.0
    saturation = actual_adv > GAP_SATURATION
    unsat = ~saturation
    adv_err = implied_adv - actual_adv

    max_mu = np.max(means, axis=1, keepdims=True)
    direction_saturated = (max_mu - means) > GAP_SATURATION
    pair_actual: list[np.ndarray] = []
    pair_implied: list[np.ndarray] = []
    pair_saturated: list[np.ndarray] = []
    for i, j in PAIR_INDEX:
        pair_actual.append(means[:, i] - means[:, j])
        pair_implied.append(TAU * (np.log(p[:, i]) - np.log(p[:, j])))
        pair_saturated.append(direction_saturated[:, i] | direction_saturated[:, j])
    pa = np.stack(pair_actual, axis=1)
    pi = np.stack(pair_implied, axis=1)
    ps = np.stack(pair_saturated, axis=1)
    pair_unsat = ~ps
    pair_err = pi - pa

    entropy = -np.sum(p * np.log(p), axis=1)
    top_probability = np.max(p, axis=1)
    sorted_means = np.sort(means, axis=1)
    best_second_gap = sorted_means[:, 2] - sorted_means[:, 1]
    common_level = np.mean(means, axis=1)

    mismatch_count = int(np.sum(rich_best != reduced_best))
    sat_count = int(np.sum(saturation))
    unsat_abs = np.abs(adv_err[unsat])
    pair_unsat_abs = np.abs(pair_err[pair_unsat])
    return {
        "schema": "CB16_R11_M_SERIES_M1_DIRECTION_SUFFICIENCY_FOLD_V1",
        "rows": n,
        "rich_vs_reduced_best_direction_argmax_mismatch_count": mismatch_count,
        "best_vs_g0_advantage_saturation_row_count": sat_count,
        "best_vs_g0_advantage_saturation_rate": float(sat_count / n),
        "best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows": (
            0.0 if unsat_abs.size == 0 else float(np.max(unsat_abs))
        ),
        "best_vs_g0_advantage_reconstruction_mae_on_unsaturated_rows": (
            0.0 if unsat_abs.size == 0 else float(np.mean(unsat_abs))
        ),
        "actual_best_vs_g0_advantage_max": float(np.max(actual_adv)),
        "actual_best_vs_g0_advantage_mean": float(np.mean(actual_adv)),
        "all_pairwise_gap_saturation_count": int(np.sum(ps)),
        "all_pairwise_gap_count": int(ps.size),
        "all_pairwise_gap_saturation_rate": float(np.mean(ps)),
        "all_pairwise_gap_reconstruction_max_abs_error_on_unsaturated_pairs": (
            0.0 if pair_unsat_abs.size == 0 else float(np.max(pair_unsat_abs))
        ),
        "all_pairwise_gap_reconstruction_mae_on_unsaturated_pairs": (
            0.0 if pair_unsat_abs.size == 0 else float(np.mean(pair_unsat_abs))
        ),
        "teacher_target_entropy_mean": float(np.mean(entropy)),
        "teacher_target_entropy_min": float(np.min(entropy)),
        "teacher_target_entropy_max": float(np.max(entropy)),
        "teacher_top_probability_mean": float(np.mean(top_probability)),
        "teacher_top_probability_min": float(np.min(top_probability)),
        "teacher_top_probability_max": float(np.max(top_probability)),
        "actual_best_vs_second_best_mean_gap_mean": float(np.mean(best_second_gap)),
        "actual_best_vs_second_best_mean_gap_max": float(np.max(best_second_gap)),
        "common_best_direction_utility_level_mean": float(np.mean(common_level)),
        "common_best_direction_utility_level_std": float(np.std(common_level)),
        "common_utility_offset_identifiable_from_direction_probs": False,
        "within_direction_risk_curve_retained_by_three_direction_probs": False,
        "tails_std_quantiles_support_strength_retained_by_three_direction_probs": False,
        "continuous_g0_requested_risk_used": False,
        "optimizer_steps": 0,
    }


def adjudicate_m1(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(folds) == 5, f"M1_FOLD_COUNT:{len(folds)}")
    ordered = sorted(folds, key=lambda x: int(x["fold"]))
    require([int(x["fold"]) for x in ordered] == [1, 2, 3, 4, 5], "M1_FOLD_IDS")
    retained = all(
        int(x["rich_vs_reduced_best_direction_argmax_mismatch_count"]) == 0
        and int(x["best_vs_g0_advantage_saturation_row_count"]) == 0
        and float(x["best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows"]) <= 1e-12
        for x in ordered
    )
    conclusion = (
        "DIRECTION_BEST_MEAN_GEOMETRY_RETAINED_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
        if retained
        else "DIRECTION_BEST_MEAN_GEOMETRY_PARTIALLY_COMPRESSED_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
    )
    return {
        "schema": "CB16_R11_M_SERIES_M1_DIRECTION_RELATIVE_SUFFICIENCY_SUMMARY_V1",
        "conclusion": conclusion,
        "direction_mean_geometry_retained": bool(retained),
        "total_rows": int(sum(int(x["rows"]) for x in ordered)),
        "total_argmax_mismatch_count": int(
            sum(int(x["rich_vs_reduced_best_direction_argmax_mismatch_count"]) for x in ordered)
        ),
        "total_best_vs_g0_saturation_rows": int(
            sum(int(x["best_vs_g0_advantage_saturation_row_count"]) for x in ordered)
        ),
        "global_max_reconstruction_error_unsaturated": float(
            max(float(x["best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows"]) for x in ordered)
        ),
        "per_fold": [
            {
                "fold": int(x["fold"]),
                "rows": int(x["rows"]),
                "argmax_mismatch_count": int(x["rich_vs_reduced_best_direction_argmax_mismatch_count"]),
                "best_vs_g0_saturation_rows": int(x["best_vs_g0_advantage_saturation_row_count"]),
                "max_reconstruction_error": float(
                    x["best_vs_g0_advantage_reconstruction_max_abs_error_on_unsaturated_rows"]
                ),
                "actual_advantage_max": float(x["actual_best_vs_g0_advantage_max"]),
                "pairwise_saturation_rate": float(x["all_pairwise_gap_saturation_rate"]),
            }
            for x in ordered
        ],
        "r7_candidate_evaluated": False,
        "market_information_verdict": False,
        "canonical_promotion_authorized": False,
    }
