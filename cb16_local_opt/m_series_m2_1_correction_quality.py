from __future__ import annotations

"""M-series M2.1 correction-quality attribution helpers.

This module does not define a new learner. It decomposes behavior produced by
exact M2 residual training using the aligned evaluation Teacher mean-utility
surface. M2 remains failed regardless of any M2.1 result.
"""

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .m_series_m2_frozen_g0_direction_residual import (
    DirectionResidualM2,
    FrozenDirectionCacheM2,
    M2_SHIFTS,
    require,
)

M21_RUNTIME = "CB16_R11_M_SERIES_M2_1_CORRECTION_QUALITY_ATTRIBUTION_V1"


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    require(x.shape == w.shape, "M21_WEIGHTED_MEAN_SHAPE")
    d = float(np.sum(w))
    require(math.isfinite(d) and d > 0.0, "M21_WEIGHTED_MEAN_DENOM")
    return float(np.sum(x * w) / d)


def _conditional_weighted_mean(x: np.ndarray, mask: np.ndarray, w: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    w = np.asarray(w, dtype=np.float64)
    require(x.shape == mask.shape == w.shape, "M21_CONDITIONAL_SHAPE")
    d = float(np.sum(w[mask]))
    if d <= 0.0:
        return None
    return float(np.sum(x[mask] * w[mask]) / d)


def correction_quality_m21(
    *,
    residual: DirectionResidualM2,
    cache: FrozenDirectionCacheM2,
    aligned_best_direction_means: np.ndarray,
    group_weight: torch.Tensor,
) -> dict[str, Any]:
    residual.eval()
    with torch.inference_mode():
        delta = residual(cache.shared256)
        new_probs_t = torch.softmax(cache.base_logits + delta, dim=-1)
    new_probs = new_probs_t.detach().cpu().numpy().astype(np.float64, copy=False)
    g0_probs = cache.base_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    means = np.asarray(aligned_best_direction_means, dtype=np.float64)
    w = group_weight.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    require(new_probs.shape == g0_probs.shape == means.shape, "M21_EVAL_SHAPE")
    n = len(new_probs)
    require(len(w) == n, "M21_WEIGHT_ROWS")
    row = np.arange(n)
    g0_dir = np.argmax(g0_probs, axis=1).astype(np.int64)
    new_dir = np.argmax(new_probs, axis=1).astype(np.int64)
    teacher_best = np.argmax(means, axis=1).astype(np.int64)

    mu_g0 = means[row, g0_dir]
    mu_new = means[row, new_dir]
    mu_best = means[row, teacher_best]
    available = np.maximum(mu_best - mu_g0, 0.0)
    captured = np.maximum(mu_new - mu_g0, 0.0)
    harmful = np.maximum(mu_g0 - mu_new, 0.0)
    require(np.all(captured <= available + 1e-12), "M21_CAPTURE_EXCEEDS_AVAILABLE")

    disagreement = teacher_best != g0_dir
    agreement = ~disagreement
    changed = new_dir != g0_dir
    direct = disagreement & (new_dir == teacher_best)
    stayed = disagreement & (new_dir == g0_dir)
    third = disagreement & changed & (new_dir != teacher_best)
    agreement_harm = agreement & changed
    require(np.all(direct | stayed | third | agreement), "M21_ROW_CLASSIFICATION_INCOMPLETE")

    available_mass = float(np.sum(w * available))
    require(math.isfinite(available_mass) and available_mass > 0.0, "M21_NO_AVAILABLE_ADVANTAGE")
    beneficial_mean = _weighted_mean(captured, w)
    harmful_mean = _weighted_mean(harmful, w)
    net_mean = beneficial_mean - harmful_mean
    total_motion_value = beneficial_mean + harmful_mean
    efficiency = None if total_motion_value <= 0.0 else float(beneficial_mean / total_motion_value)
    correction_coverage = float(np.sum(w * available * direct.astype(np.float64)) / available_mass)
    capture_fraction = float(np.sum(w * captured) / available_mass)
    raw_move = _conditional_weighted_mean(direct.astype(np.float64), disagreement, w)

    category = {
        "direct_teacher_correction": {
            "rows": int(np.sum(direct)),
            "weight_fraction": _weighted_mean(direct.astype(np.float64), w),
            "beneficial_gain_mean_all_rows": _weighted_mean(captured * direct, w),
            "harmful_cost_mean_all_rows": _weighted_mean(harmful * direct, w),
        },
        "stayed_g0_on_disagreement": {
            "rows": int(np.sum(stayed)),
            "weight_fraction": _weighted_mean(stayed.astype(np.float64), w),
            "beneficial_gain_mean_all_rows": _weighted_mean(captured * stayed, w),
            "harmful_cost_mean_all_rows": _weighted_mean(harmful * stayed, w),
        },
        "third_direction_change": {
            "rows": int(np.sum(third)),
            "weight_fraction": _weighted_mean(third.astype(np.float64), w),
            "beneficial_gain_mean_all_rows": _weighted_mean(captured * third, w),
            "harmful_cost_mean_all_rows": _weighted_mean(harmful * third, w),
        },
        "agreement_harmful_drift": {
            "rows": int(np.sum(agreement_harm)),
            "weight_fraction": _weighted_mean(agreement_harm.astype(np.float64), w),
            "beneficial_gain_mean_all_rows": _weighted_mean(captured * agreement_harm, w),
            "harmful_cost_mean_all_rows": _weighted_mean(harmful * agreement_harm, w),
        },
    }
    return {
        "schema": "CB16_R11_M2_1_CORRECTION_QUALITY_FOLD_ARM_V1",
        "rows": int(n),
        "available_advantage_mean": _weighted_mean(available, w),
        "available_advantage_mass": available_mass,
        "advantage_weighted_teacher_correction_coverage": correction_coverage,
        "available_advantage_capture_fraction": capture_fraction,
        "beneficial_gain_mean": beneficial_mean,
        "harmful_cost_mean": harmful_mean,
        "net_gain_mean": net_mean,
        "gain_efficiency": efficiency,
        "agreement_harmful_cost_mean": _conditional_weighted_mean(harmful, agreement, w),
        "disagreement_harmful_cost_mean": _conditional_weighted_mean(harmful, disagreement, w),
        "raw_move_to_teacher_rate": raw_move,
        "negative_gain_rate": _weighted_mean((mu_new < mu_g0).astype(np.float64), w),
        "category_decomposition": category,
    }


def adjudicate_m21(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(fold_results) == 5, f"M21_FOLD_COUNT:{len(fold_results)}")
    by_fold = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in by_fold) == (1, 2, 3, 4, 5), "M21_FOLD_IDS")
    correction_wins = 0
    capture_wins = 0
    harm_wins = 0
    selective_vs_abs_harm_wins = 0
    rows = []
    for f in by_fold:
        arms = f["arms"]
        sa = arms["SELECTIVE_ALIGNED"]["quality"]
        aa = arms["ABS_CE_ALIGNED"]["quality"]
        sh = [arms[f"SELECTIVE_SHUFFLE_{s}"]["quality"] for s in M2_SHIFTS]
        med_corr = float(statistics.median(float(x["advantage_weighted_teacher_correction_coverage"]) for x in sh))
        med_capture = float(statistics.median(float(x["available_advantage_capture_fraction"]) for x in sh))
        med_harm = float(statistics.median(float(x["harmful_cost_mean"]) for x in sh))
        corr = float(sa["advantage_weighted_teacher_correction_coverage"])
        capture = float(sa["available_advantage_capture_fraction"])
        harm = float(sa["harmful_cost_mean"])
        abs_harm = float(aa["harmful_cost_mean"])
        c_corr = corr > med_corr
        c_capture = capture > med_capture
        c_harm = harm < med_harm
        c_abs_harm = harm < abs_harm
        correction_wins += int(c_corr)
        capture_wins += int(c_capture)
        harm_wins += int(c_harm)
        selective_vs_abs_harm_wins += int(c_abs_harm)
        rows.append({
            "fold": int(f["fold"]),
            "selective_aligned_advantage_weighted_teacher_correction_coverage": corr,
            "median_selective_shuffle_advantage_weighted_teacher_correction_coverage": med_corr,
            "selective_aligned_available_advantage_capture_fraction": capture,
            "median_selective_shuffle_available_advantage_capture_fraction": med_capture,
            "selective_aligned_harmful_cost_mean": harm,
            "median_selective_shuffle_harmful_cost_mean": med_harm,
            "abs_ce_aligned_harmful_cost_mean": abs_harm,
            "selective_aligned_net_gain_mean": float(sa["net_gain_mean"]),
            "selective_aligned_raw_move_to_teacher_rate": float(sa["raw_move_to_teacher_rate"]),
            "correction_quality_gt_shuffle_median": c_corr,
            "capture_fraction_gt_shuffle_median": c_capture,
            "harmful_cost_lt_shuffle_median": c_harm,
            "harmful_cost_lt_abs_ce_aligned": c_abs_harm,
        })
    correction_pass = correction_wins >= 4
    capture_pass = capture_wins >= 4
    harm_pass = harm_wins >= 4
    if correction_pass and harm_pass:
        joint = "M2_RAW_MOVE_RATE_FAILURE_LOCALIZED_TO_READOUT_INSUFFICIENCY_ON_CONSUMED_SUPPORT__M2_REMAINS_FAILED"
    elif harm_pass:
        joint = "M2_SELECTIVE_POSITIVE_GAIN_IS_PRIMARILY_PRESERVATION_HARM_SUPPRESSION_NOT_SUPERIOR_DISAGREEMENT_CORRECTION_SELECTION"
    elif correction_pass:
        joint = "M2_SELECTIVE_CORRECTION_QUALITY_EXISTS_WITHOUT_RELIABLE_HARM_SUPPRESSION"
    else:
        joint = "M2_SELECTIVE_MECHANISM_NOT_SUPPORTED_BY_CORRECTION_QUALITY_DECOMPOSITION"
    return {
        "schema": "CB16_R11_M_SERIES_M2_1_ADJUDICATION_SUMMARY_V1",
        "correction_quality_gt_shuffle_median_fold_count": int(correction_wins),
        "capture_fraction_gt_shuffle_median_fold_count": int(capture_wins),
        "harmful_cost_lt_shuffle_median_fold_count": int(harm_wins),
        "selective_harmful_cost_lt_abs_ce_aligned_fold_count": int(selective_vs_abs_harm_wins),
        "correction_quality_pass": bool(correction_pass),
        "capture_pass": bool(capture_pass),
        "harm_suppression_pass": bool(harm_pass),
        "correction_quality_conclusion": (
            "SELECTIVE_ALIGNMENT_CONTAINS_ADVANTAGE_WEIGHTED_CORRECTION_QUALITY_NOT_VISIBLE_IN_RAW_FREQUENCY"
            if correction_pass else
            "SELECTIVE_ALIGNMENT_DOES_NOT_SHOW_ADVANTAGE_WEIGHTED_CORRECTION_SELECTION_OVER_SHUFFLE"
        ),
        "harm_suppression_conclusion": (
            "SELECTIVE_ALIGNMENT_SUPPRESSES_TEACHER_IMPLIED_HARM_RELATIVE_TO_SHUFFLE"
            if harm_pass else
            "SELECTIVE_ALIGNMENT_HARM_SUPPRESSION_OVER_SHUFFLE_NOT_ESTABLISHED"
        ),
        "joint_interpretation": joint,
        "per_fold": rows,
        "m2_retroactive_status_change": False,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "m3_automatic_authorization": False,
    }
