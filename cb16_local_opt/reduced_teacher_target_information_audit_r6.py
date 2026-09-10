from __future__ import annotations

"""R11 Science G0 R6 train-only reduced Teacher target information audit helpers."""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical_state_alignment_falsification_r41 import R41_SHIFTS
from .training_runtime_r11 import PreparedEvidenceR11, SMOOTH_L1_BETA_R11

R6_RUNTIME = "CB16_R11_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_TRAIN_ONLY_V1"
R6_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
)
R6_FOLDS = (1, 2, 3, 4, 5)
R6_SHIFTS = tuple(R41_SHIFTS)
R6_CLOCK_BLOCKS = 6
R6_MIN_TRAIN_GROUPS = 32
R6_MIN_EVAL_GROUPS = 8
R6_MIN_GAP_HOURS = 72
HOUR_MS = 3_600_000


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def build_outer_folds_r6(parents: Mapping[str, Any]) -> list[dict[str, Any]]:
    eligible = [
        p for p in parents.values()
        if str(getattr(p, "split", "")) == "TRAIN"
        and str(getattr(p, "symbol", "")) in R6_SYMBOLS
    ]
    require(bool(eligible), "R11_R6_NO_TRAIN_PARENTS")
    observed_symbols = {str(p.symbol) for p in eligible}
    require(observed_symbols == set(R6_SYMBOLS), f"R11_R6_SYMBOL_SET_DRIFT:{sorted(observed_symbols)}")
    clocks = np.asarray(sorted({int(p.decision_time_ms) for p in eligible}), dtype=np.int64)
    require(len(clocks) >= R6_CLOCK_BLOCKS, f"R11_R6_TOO_FEW_CLOCKS:{len(clocks)}")
    raw_blocks = [np.asarray(x, dtype=np.int64) for x in np.array_split(clocks, R6_CLOCK_BLOCKS)]
    require(all(len(x) > 0 for x in raw_blocks), "R11_R6_EMPTY_CLOCK_BLOCK")
    group_by_clock: dict[int, set[str]] = {}
    symbol_by_clock: dict[int, set[str]] = {}
    for p in eligible:
        t = int(p.decision_time_ms)
        group_by_clock.setdefault(t, set()).add(str(p.dependence_group_id))
        symbol_by_clock.setdefault(t, set()).add(str(p.symbol))
    out: list[dict[str, Any]] = []
    for fold in R6_FOLDS:
        train_clock_arr = np.concatenate(raw_blocks[:fold])
        eval_clock_arr = raw_blocks[fold]
        train_clocks = tuple(int(x) for x in train_clock_arr.tolist())
        eval_clocks = tuple(int(x) for x in eval_clock_arr.tolist())
        require(max(train_clocks) < min(eval_clocks), f"R11_R6_NONCHRONOLOGICAL_FOLD:{fold}")
        gap_hours = (min(eval_clocks) - max(train_clocks)) / HOUR_MS
        require(gap_hours >= R6_MIN_GAP_HOURS, f"R11_R6_OUTER_GAP_TOO_SMALL:{fold}:{gap_hours}")
        train_groups = sorted({g for t in train_clocks for g in group_by_clock.get(t, ())})
        eval_groups = sorted({g for t in eval_clocks for g in group_by_clock.get(t, ())})
        require(len(train_groups) >= R6_MIN_TRAIN_GROUPS, f"R11_R6_TRAIN_GROUP_SHORTFALL:{fold}:{len(train_groups)}")
        require(len(eval_groups) >= R6_MIN_EVAL_GROUPS, f"R11_R6_EVAL_GROUP_SHORTFALL:{fold}:{len(eval_groups)}")
        train_symbols = sorted({s for t in train_clocks for s in symbol_by_clock.get(t, ())})
        eval_symbols = sorted({s for t in eval_clocks for s in symbol_by_clock.get(t, ())})
        require(set(train_symbols) == set(R6_SYMBOLS), f"R11_R6_TRAIN_SYMBOL_COVERAGE:{fold}:{train_symbols}")
        require(set(eval_symbols) == set(R6_SYMBOLS), f"R11_R6_EVAL_SYMBOL_COVERAGE:{fold}:{eval_symbols}")
        out.append({
            "fold": int(fold),
            "train_clocks": train_clocks,
            "eval_clocks": eval_clocks,
            "train_group_count": len(train_groups),
            "eval_group_count": len(eval_groups),
            "train_clock_count": len(train_clocks),
            "eval_clock_count": len(eval_clocks),
            "train_symbols": train_symbols,
            "eval_symbols": eval_symbols,
            "train_last_timestamp_ms": max(train_clocks),
            "eval_first_timestamp_ms": min(eval_clocks),
            "train_to_eval_gap_hours": float(gap_hours),
        })
    return out


def _smooth_l1_numpy(pred: float, target: np.ndarray, beta: float) -> np.ndarray:
    d = np.abs(np.asarray(target, dtype=np.float64) - float(pred))
    return np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)


def analytic_global_marginal_baseline_r6(
    train: PreparedEvidenceR11,
    evaluation: PreparedEvidenceR11,
) -> dict[str, Any]:
    """Gate-authorized state-free marginal: mean direction distribution + mean risk."""
    train.validate()
    evaluation.validate()
    tw = train.group_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    tp = train.direction_target_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    tr = train.requested_risk_target.detach().cpu().numpy().astype(np.float64, copy=False)
    direction_mean = np.sum(tp * tw[:, None], axis=0) / np.sum(tw)
    direction_mean = np.clip(direction_mean, 1e-12, 1.0)
    direction_mean = direction_mean / direction_mean.sum()
    risk_mean = float(np.sum(tr * tw) / np.sum(tw))

    ew = evaluation.group_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    ep = evaluation.direction_target_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    er = evaluation.requested_risk_target.detach().cpu().numpy().astype(np.float64, copy=False)
    row_direction = -np.sum(ep * np.log(direction_mean[None, :]), axis=1)
    row_sizing = _smooth_l1_numpy(risk_mean, er, float(SMOOTH_L1_BETA_R11))
    denom = float(np.sum(ew))
    direction_loss = float(np.sum(row_direction * ew) / denom)
    sizing_loss = float(np.sum(row_sizing * ew) / denom)
    return {
        "schema": "CB16_R11_R6_ANALYTIC_GLOBAL_MARGINAL_BASELINE_V1",
        "no_state_features": True,
        "direction_distribution": [float(x) for x in direction_mean],
        "requested_risk_mean": risk_mean,
        "loss": direction_loss + sizing_loss,
        "direction_loss": direction_loss,
        "sizing_loss": sizing_loss,
        "train_rows": int(train.rows),
        "eval_rows": int(evaluation.rows),
    }


def target_dispersion_r6(prepared: PreparedEvidenceR11) -> dict[str, Any]:
    prepared.validate()
    p = prepared.direction_target_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    r = prepared.requested_risk_target.detach().cpu().numpy().astype(np.float64, copy=False)
    return {
        "rows": int(prepared.rows),
        "independent_groups": int(len(set(prepared.dependence_group_ids))),
        "direction_component_mean": [float(x) for x in np.mean(p, axis=0)],
        "direction_component_std": [float(x) for x in np.std(p, axis=0)],
        "direction_component_min": [float(x) for x in np.min(p, axis=0)],
        "direction_component_max": [float(x) for x in np.max(p, axis=0)],
        "requested_risk_mean": float(np.mean(r)),
        "requested_risk_std": float(np.std(r)),
        "requested_risk_min": float(np.min(r)),
        "requested_risk_max": float(np.max(r)),
    }


def summarize_r6(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(fold_results) == 5, f"R11_R6_FOLD_RESULT_COUNT:{len(fold_results)}")
    by_fold = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in by_fold) == R6_FOLDS, "R11_R6_FOLD_ID_SET_DRIFT")
    per_fold: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    total_vs_median_count = 0
    direction_vs_median_count = 0
    direction_vs_marginal_count = 0
    sizing_vs_median_count = 0
    sizing_vs_marginal_count = 0
    for row in by_fold:
        aligned = row["arms"]["ALIGNED"]["validation_after"]
        marginal = row["marginal_baseline"]
        shuffled = [row["arms"][f"SHUFFLE_{s}"]["validation_after"] for s in R6_SHIFTS]
        med_total = float(statistics.median(float(x["loss"]) for x in shuffled))
        med_dir = float(statistics.median(float(x["direction_loss"]) for x in shuffled))
        med_size = float(statistics.median(float(x["sizing_loss"]) for x in shuffled))
        a_total = float(aligned["loss"])
        a_dir = float(aligned["direction_loss"])
        a_size = float(aligned["sizing_loss"])
        total_win = a_total < med_total
        dir_win = a_dir < med_dir
        dir_marginal_win = a_dir < float(marginal["direction_loss"])
        size_win = a_size < med_size
        size_marginal_win = a_size < float(marginal["sizing_loss"])
        total_vs_median_count += int(total_win)
        direction_vs_median_count += int(dir_win)
        direction_vs_marginal_count += int(dir_marginal_win)
        sizing_vs_median_count += int(size_win)
        sizing_vs_marginal_count += int(size_marginal_win)
        fold_pair_total = 0
        fold_pair_dir = 0
        fold_pair_size = 0
        for shift, src in zip(R6_SHIFTS, shuffled):
            comp = {
                "fold": int(row["fold"]),
                "shift": int(shift),
                "aligned_minus_shuffle_total": a_total - float(src["loss"]),
                "aligned_minus_shuffle_direction": a_dir - float(src["direction_loss"]),
                "aligned_minus_shuffle_sizing": a_size - float(src["sizing_loss"]),
                "aligned_lower_total": a_total < float(src["loss"]),
                "aligned_lower_direction": a_dir < float(src["direction_loss"]),
                "aligned_lower_sizing": a_size < float(src["sizing_loss"]),
            }
            fold_pair_total += int(comp["aligned_lower_total"])
            fold_pair_dir += int(comp["aligned_lower_direction"])
            fold_pair_size += int(comp["aligned_lower_sizing"])
            pairwise.append(comp)
        per_fold.append({
            "fold": int(row["fold"]),
            "aligned_total": a_total,
            "aligned_direction": a_dir,
            "aligned_sizing": a_size,
            "median_shuffle_total": med_total,
            "median_shuffle_direction": med_dir,
            "median_shuffle_sizing": med_size,
            "marginal_total": float(marginal["loss"]),
            "marginal_direction": float(marginal["direction_loss"]),
            "marginal_sizing": float(marginal["sizing_loss"]),
            "aligned_total_lower_than_shuffle_median": total_win,
            "aligned_direction_lower_than_shuffle_median": dir_win,
            "aligned_direction_lower_than_marginal": dir_marginal_win,
            "aligned_sizing_lower_than_shuffle_median": size_win,
            "aligned_sizing_lower_than_marginal": size_marginal_win,
            "aligned_lower_total_pair_count": fold_pair_total,
            "aligned_lower_direction_pair_count": fold_pair_dir,
            "aligned_lower_sizing_pair_count": fold_pair_size,
        })
    supported = (
        total_vs_median_count >= 4
        and direction_vs_median_count >= 4
        and direction_vs_marginal_count >= 4
    )
    return {
        "schema": "CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_SUMMARY_V1",
        "primary_rule": "ALIGNED_TOTAL_LOWER_THAN_MEDIAN_OF_FIVE_SHUFFLES_IN_AT_LEAST_4_OF_5_FOLDS__AND_ALIGNED_DIRECTION_LOWER_THAN_MEDIAN_OF_FIVE_SHUFFLES_IN_AT_LEAST_4_OF_5_FOLDS__AND_ALIGNED_DIRECTION_LOWER_THAN_ANALYTIC_MARGINAL_BASELINE_IN_AT_LEAST_4_OF_5_FOLDS",
        "conditional_information_supported": bool(supported),
        "conclusion": (
            "TRAIN_ONLY_CONDITIONAL_INFORMATION_SUPPORTED"
            if supported else "TRAIN_ONLY_CONDITIONAL_INFORMATION_NOT_SUPPORTED"
        ),
        "total_lower_than_shuffle_median_fold_count": int(total_vs_median_count),
        "direction_lower_than_shuffle_median_fold_count": int(direction_vs_median_count),
        "direction_lower_than_marginal_fold_count": int(direction_vs_marginal_count),
        "sizing_lower_than_shuffle_median_fold_count": int(sizing_vs_median_count),
        "sizing_lower_than_marginal_fold_count": int(sizing_vs_marginal_count),
        "aligned_lower_total_pair_count_of_25": int(sum(x["aligned_lower_total"] for x in pairwise)),
        "aligned_lower_direction_pair_count_of_25": int(sum(x["aligned_lower_direction"] for x in pairwise)),
        "aligned_lower_sizing_pair_count_of_25": int(sum(x["aligned_lower_sizing"] for x in pairwise)),
        "per_fold": per_fold,
        "all_25_pairwise": pairwise,
        "market_information_verdict": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
    }
