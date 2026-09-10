from __future__ import annotations

"""M-series M0: read-only attribution metrics for canonical R6 policy migration.

M0 never changes Teacher, Student, Physics, Supervisor, training rules, or support.
It receives before/after Student outputs plus the unchanged R6 eval Teacher evidence
and attributes discrete Direction migration relative to the immutable G0.
"""

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

M0_RUNTIME = "CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_V1"
DIRECTION_NAMES = ("SHORT", "FLAT", "LONG")
DIRECTION_VALUES = (-1, 0, 1)
ROW_CLASSES = (
    "PRESERVED_TEACHER_AGREEMENT",
    "DRIFT_ON_TEACHER_AGREEMENT",
    "DIRECT_TEACHER_CORRECTION",
    "STAYED_G0_DESPITE_TEACHER_DISAGREEMENT",
    "THIRD_DIRECTION_DRIFT",
)


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def _prob_matrix(x: Any, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 2 and a.shape[1] == 3, f"M0_{name}_SHAPE")
    require(np.isfinite(a).all(), f"M0_{name}_NONFINITE")
    require(np.all(a >= 0.0), f"M0_{name}_NEGATIVE")
    sums = a.sum(axis=1)
    require(np.all(np.abs(sums - 1.0) <= 1e-5), f"M0_{name}_NOT_DISTRIBUTION")
    return a


def _risk_vector(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    require(len(a) == n, f"M0_{name}_LENGTH")
    require(np.isfinite(a).all(), f"M0_{name}_NONFINITE")
    require(np.all((a >= 0.0) & (a <= 1.0)), f"M0_{name}_RANGE")
    return a


def teacher_direction_geometry_from_action_laws_m0(
    *,
    evidence: Sequence[Any],
    g0_direction: Any,
    reduced_teacher_probs: Any,
) -> dict[str, np.ndarray]:
    """Recover exact best-by-Direction Teacher mean utilities from rich action_laws.

    This intentionally does not reconstruct utility gaps from reduced softmax probabilities.
    The Teacher implementation clips normalized logits before exponentiation, so log-probability
    ratios can saturate. M0 V2 instead reads the already-existing eval Teacher action_laws and
    reproduces the compiler's within-Direction max(mean_utility, -requested_risk) choice.
    """
    probs = _prob_matrix(reduced_teacher_probs, "TEACHER_PROBS")
    n = len(probs)
    require(len(evidence) == n, "M0_TEACHER_EVIDENCE_LENGTH")
    g0 = np.asarray(g0_direction, dtype=np.int64).reshape(-1)
    require(len(g0) == n, "M0_G0_DIRECTION_LENGTH")
    require(np.all((g0 >= 0) & (g0 <= 2)), "M0_G0_DIRECTION_RANGE")

    means = np.empty((n, 3), dtype=np.float64)
    risks = np.empty((n, 3), dtype=np.float64)
    for i, e in enumerate(evidence):
        laws = tuple(getattr(e, "action_laws", ()))
        require(bool(laws), f"M0_ACTION_LAWS_MISSING:{i}")
        for cls, direction in enumerate(DIRECTION_VALUES):
            candidates = [law for law in laws if int(law.direction) == int(direction)]
            require(bool(candidates), f"M0_DIRECTION_LAW_MISSING:{i}:{direction}")
            best = max(
                candidates,
                key=lambda law: (float(law.mean_utility), -float(law.requested_risk)),
            )
            means[i, cls] = float(best.mean_utility)
            risks[i, cls] = float(best.requested_risk)

    require(np.isfinite(means).all(), "M0_BEST_DIRECTION_MEANS_NONFINITE")
    require(np.isfinite(risks).all(), "M0_BEST_DIRECTION_RISKS_NONFINITE")
    best_from_laws = np.argmax(means, axis=1).astype(np.int64)
    best_from_reduced = np.argmax(probs, axis=1).astype(np.int64)
    require(
        np.array_equal(best_from_laws, best_from_reduced),
        "M0_TEACHER_BEST_DIRECTION_REDUCED_RICH_MISMATCH",
    )
    row = np.arange(n)
    advantage = means[row, best_from_laws] - means[row, g0]
    advantage[np.abs(advantage) < 1e-15] = 0.0
    require(np.all(advantage >= -1e-12), "M0_NEGATIVE_BEST_DIRECTION_ADVANTAGE")
    return {
        "best_mean_by_direction": means,
        "best_risk_by_direction": risks,
        "teacher_best_direction": best_from_laws,
        "direction_advantage_over_g0": np.maximum(advantage, 0.0),
    }


def jensen_shannon_rows(p: Any, q: Any) -> np.ndarray:
    a = _prob_matrix(p, "JS_P")
    b = _prob_matrix(q, "JS_Q")
    require(a.shape == b.shape, "M0_JS_SHAPE_MISMATCH")
    tiny = np.finfo(np.float64).tiny
    a = np.clip(a, tiny, 1.0)
    b = np.clip(b, tiny, 1.0)
    a = a / a.sum(axis=1, keepdims=True)
    b = b / b.sum(axis=1, keepdims=True)
    m = 0.5 * (a + b)
    return 0.5 * np.sum(a * (np.log(a) - np.log(m)), axis=1) + 0.5 * np.sum(
        b * (np.log(b) - np.log(m)), axis=1
    )


def _safe_rate(num: int, den: int) -> float | None:
    return None if den == 0 else float(num / den)


def _mean_or_none(x: np.ndarray) -> float | None:
    return None if x.size == 0 else float(np.mean(x))


def _median_or_none(x: np.ndarray) -> float | None:
    return None if x.size == 0 else float(np.median(x))


def attribute_direction_migration_m0(
    *,
    parent_ids: Sequence[str],
    dependence_group_ids: Sequence[str],
    teacher_probs: Any,
    teacher_best_mean_by_direction: Any,
    teacher_best_risk_by_direction: Any,
    teacher_direction_advantage_over_g0: Any,
    g0_probs: Any,
    challenger_probs: Any,
    g0_requested_risk_raw: Any,
    challenger_requested_risk_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher = _prob_matrix(teacher_probs, "TEACHER_PROBS")
    g0 = _prob_matrix(g0_probs, "G0_PROBS")
    challenger = _prob_matrix(challenger_probs, "CHALLENGER_PROBS")
    require(teacher.shape == g0.shape == challenger.shape, "M0_PROB_SHAPE_MISMATCH")
    n = len(teacher)
    require(len(parent_ids) == n, "M0_PARENT_ID_LENGTH")
    require(len(dependence_group_ids) == n, "M0_GROUP_ID_LENGTH")
    require(len(set(parent_ids)) == n, "M0_DUPLICATE_PARENT_IDS")

    best_means = np.asarray(teacher_best_mean_by_direction, dtype=np.float64)
    best_risks = np.asarray(teacher_best_risk_by_direction, dtype=np.float64)
    require(best_means.shape == (n, 3), "M0_BEST_MEAN_SHAPE")
    require(best_risks.shape == (n, 3), "M0_BEST_RISK_SHAPE")
    require(np.isfinite(best_means).all(), "M0_BEST_MEAN_NONFINITE")
    require(np.isfinite(best_risks).all(), "M0_BEST_RISK_NONFINITE")
    advantage = np.asarray(teacher_direction_advantage_over_g0, dtype=np.float64).reshape(-1)
    require(len(advantage) == n, "M0_ADVANTAGE_LENGTH")
    require(np.isfinite(advantage).all(), "M0_ADVANTAGE_NONFINITE")
    require(np.all(advantage >= -1e-12), "M0_ADVANTAGE_NEGATIVE")
    advantage = np.maximum(advantage, 0.0)

    g0_risk = _risk_vector(g0_requested_risk_raw, n, "G0_RISK")
    challenger_risk = _risk_vector(challenger_requested_risk_raw, n, "CHALLENGER_RISK")

    teacher_dir = np.argmax(teacher, axis=1).astype(np.int64)
    rich_teacher_dir = np.argmax(best_means, axis=1).astype(np.int64)
    require(np.array_equal(teacher_dir, rich_teacher_dir), "M0_RICH_REDUCED_TEACHER_DIR_MISMATCH")
    g0_dir = np.argmax(g0, axis=1).astype(np.int64)
    challenger_dir = np.argmax(challenger, axis=1).astype(np.int64)
    expected_advantage = best_means[np.arange(n), teacher_dir] - best_means[np.arange(n), g0_dir]
    expected_advantage[np.abs(expected_advantage) < 1e-15] = 0.0
    require(
        np.allclose(advantage, np.maximum(expected_advantage, 0.0), rtol=0.0, atol=1e-12),
        "M0_ADVANTAGE_SOURCE_DRIFT",
    )

    js = jensen_shannon_rows(g0, challenger)
    raw_risk_delta = np.abs(challenger_risk - g0_risk)
    g0_composed_risk = np.where(g0_dir == 1, 0.0, g0_risk)
    challenger_composed_risk = np.where(challenger_dir == 1, 0.0, challenger_risk)
    composed_risk_delta = np.abs(challenger_composed_risk - g0_composed_risk)

    teacher_agrees = teacher_dir == g0_dir
    direction_changed = challenger_dir != g0_dir
    direct_correction = (~teacher_agrees) & (challenger_dir == teacher_dir)
    third_direction = (~teacher_agrees) & direction_changed & (challenger_dir != teacher_dir)

    classes = np.empty(n, dtype=object)
    classes[teacher_agrees & (~direction_changed)] = "PRESERVED_TEACHER_AGREEMENT"
    classes[teacher_agrees & direction_changed] = "DRIFT_ON_TEACHER_AGREEMENT"
    classes[(~teacher_agrees) & direct_correction] = "DIRECT_TEACHER_CORRECTION"
    classes[(~teacher_agrees) & (~direction_changed)] = "STAYED_G0_DESPITE_TEACHER_DISAGREEMENT"
    classes[third_direction] = "THIRD_DIRECTION_DRIFT"
    require(all(str(x) in ROW_CLASSES for x in classes), "M0_UNCLASSIFIED_ROW")

    rows: list[dict[str, Any]] = []
    for i in range(n):
        rows.append(
            {
                "parent_id": str(parent_ids[i]),
                "dependence_group_id": str(dependence_group_ids[i]),
                "class": str(classes[i]),
                "teacher_direction_class": int(teacher_dir[i]),
                "teacher_direction": DIRECTION_NAMES[int(teacher_dir[i])],
                "g0_direction_class": int(g0_dir[i]),
                "g0_direction": DIRECTION_NAMES[int(g0_dir[i])],
                "challenger_direction_class": int(challenger_dir[i]),
                "challenger_direction": DIRECTION_NAMES[int(challenger_dir[i])],
                "teacher_best_mean_by_direction": [float(x) for x in best_means[i]],
                "teacher_best_risk_by_direction": [float(x) for x in best_risks[i]],
                "teacher_direction_advantage_over_g0": float(advantage[i]),
                "direction_changed": bool(direction_changed[i]),
                "direct_teacher_correction": bool(direct_correction[i]),
                "direction_js_g0_vs_challenger": float(js[i]),
                "g0_requested_risk_raw": float(g0_risk[i]),
                "challenger_requested_risk_raw": float(challenger_risk[i]),
                "abs_requested_risk_raw_delta": float(raw_risk_delta[i]),
                "g0_requested_risk_composed": float(g0_composed_risk[i]),
                "challenger_requested_risk_composed": float(challenger_composed_risk[i]),
                "abs_requested_risk_delta": float(composed_risk_delta[i]),
                "teacher_direction_probs": [float(x) for x in teacher[i]],
                "g0_direction_probs": [float(x) for x in g0[i]],
                "challenger_direction_probs": [float(x) for x in challenger[i]],
            }
        )

    counts = Counter(str(x) for x in classes)
    change_count = int(np.sum(direction_changed))
    direct_count = int(np.sum(direct_correction))
    non_teacher_count = change_count - direct_count
    agreement_count = int(np.sum(teacher_agrees))
    disagreement_count = n - agreement_count
    agreement_change_count = int(np.sum(teacher_agrees & direction_changed))

    class_metrics: dict[str, Any] = {}
    for label in ROW_CLASSES:
        mask = classes == label
        class_metrics[label] = {
            "rows": int(np.sum(mask)),
            "mean_direction_advantage": _mean_or_none(advantage[mask]),
            "mean_direction_js": _mean_or_none(js[mask]),
            "mean_abs_requested_risk_delta": _mean_or_none(composed_risk_delta[mask]),
        }

    disagreement_mask = ~teacher_agrees
    quartiles: list[dict[str, Any]] = []
    if int(np.sum(disagreement_mask)) > 0:
        a = advantage[disagreement_mask]
        edges = np.quantile(a, [0.25, 0.50, 0.75])
        bin_ids = np.searchsorted(edges, a, side="right")
        direct_d = direct_correction[disagreement_mask]
        js_d = js[disagreement_mask]
        risk_d = composed_risk_delta[disagreement_mask]
        for b in range(4):
            mask = bin_ids == b
            quartiles.append(
                {
                    "quartile": b + 1,
                    "rows": int(np.sum(mask)),
                    "advantage_min": None if not np.any(mask) else float(np.min(a[mask])),
                    "advantage_max": None if not np.any(mask) else float(np.max(a[mask])),
                    "move_to_teacher_rate": None if not np.any(mask) else float(np.mean(direct_d[mask])),
                    "mean_direction_js": _mean_or_none(js_d[mask]),
                    "mean_abs_requested_risk_delta": _mean_or_none(risk_d[mask]),
                }
            )

    summary = {
        "schema": "CB16_R11_M_SERIES_M0_FOLD_POLICY_DRIFT_ATTRIBUTION_V1",
        "rows": int(n),
        "independent_dependence_groups": int(len(set(str(x) for x in dependence_group_ids))),
        "class_counts": {label: int(counts.get(label, 0)) for label in ROW_CLASSES},
        "direction_change_count": change_count,
        "direction_change_rate": float(change_count / n),
        "direct_teacher_correction_count": direct_count,
        "non_teacher_attributable_change_count": non_teacher_count,
        "direct_teacher_correction_fraction_of_all_direction_changes": _safe_rate(direct_count, change_count),
        "non_teacher_attributable_fraction_of_all_direction_changes": _safe_rate(non_teacher_count, change_count),
        "teacher_g0_agreement_rows": agreement_count,
        "teacher_g0_disagreement_rows": disagreement_count,
        "teacher_g0_agreement_direction_change_rate": _safe_rate(agreement_change_count, agreement_count),
        "teacher_g0_disagreement_move_to_teacher_rate": _safe_rate(direct_count, disagreement_count),
        "mean_direction_js_g0_vs_challenger": float(np.mean(js)),
        "median_direction_js_g0_vs_challenger": float(np.median(js)),
        "mean_abs_requested_risk_delta": float(np.mean(composed_risk_delta)),
        "median_abs_requested_risk_delta": float(np.median(composed_risk_delta)),
        "mean_abs_requested_risk_raw_delta": float(np.mean(raw_risk_delta)),
        "teacher_direction_advantage_mean": float(np.mean(advantage)),
        "teacher_direction_advantage_mean_changed": _mean_or_none(advantage[direction_changed]),
        "teacher_direction_advantage_mean_unchanged": _mean_or_none(advantage[~direction_changed]),
        "teacher_direction_advantage_median_changed": _median_or_none(advantage[direction_changed]),
        "teacher_direction_advantage_median_unchanged": _median_or_none(advantage[~direction_changed]),
        "class_metrics": class_metrics,
        "advantage_quartiles_on_teacher_g0_disagreement": quartiles,
        "direction_advantage_source": "EXISTING_R6_EVAL_TEACHER_ACTION_LAWS_BEST_MEAN_BY_DIRECTION",
        "softmax_log_ratio_advantage_used": False,
        "continuous_g0_requested_risk_projection_used": False,
        "independent_quantile_subtraction_used": False,
    }
    return rows, summary


def adjudicate_m0(fold_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(fold_summaries) == 5, f"M0_FOLD_COUNT:{len(fold_summaries)}")
    ordered = sorted(fold_summaries, key=lambda x: int(x["fold"]))
    require([int(x["fold"]) for x in ordered] == [1, 2, 3, 4, 5], "M0_FOLD_IDS")

    direct_majority = 0
    non_teacher_majority = 0
    zero_change = 0
    per_fold: list[dict[str, Any]] = []
    for x in ordered:
        changes = int(x["direction_change_count"])
        direct = x["direct_teacher_correction_fraction_of_all_direction_changes"]
        non_teacher = x["non_teacher_attributable_fraction_of_all_direction_changes"]
        if changes == 0:
            zero_change += 1
        else:
            direct_majority += int(float(direct) > 0.5)
            non_teacher_majority += int(float(non_teacher) > 0.5)
        per_fold.append(
            {
                "fold": int(x["fold"]),
                "direction_change_count": changes,
                "direction_change_rate": float(x["direction_change_rate"]),
                "direct_teacher_correction_fraction_of_all_direction_changes": direct,
                "non_teacher_attributable_fraction_of_all_direction_changes": non_teacher,
                "teacher_g0_agreement_direction_change_rate": x["teacher_g0_agreement_direction_change_rate"],
                "teacher_g0_disagreement_move_to_teacher_rate": x["teacher_g0_disagreement_move_to_teacher_rate"],
            }
        )

    if direct_majority >= 4:
        conclusion = "DIRECTION_UPDATE_MAJORITY_TEACHER_ATTRIBUTABLE_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
    elif non_teacher_majority >= 4:
        conclusion = "DIRECTION_UPDATE_MAJORITY_NOT_TEACHER_ATTRIBUTABLE_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
    else:
        conclusion = "MIXED_POLICY_DRIFT_ATTRIBUTION_ON_CONSUMED_TRAIN_ONLY_SUPPORT"

    return {
        "schema": "CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_SUMMARY_V1",
        "conclusion": conclusion,
        "direct_teacher_correction_majority_fold_count": int(direct_majority),
        "non_teacher_attributable_majority_fold_count": int(non_teacher_majority),
        "zero_direction_change_fold_count": int(zero_change),
        "per_fold": per_fold,
        "market_information_verdict": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "r7_candidate_evaluated": False,
    }
