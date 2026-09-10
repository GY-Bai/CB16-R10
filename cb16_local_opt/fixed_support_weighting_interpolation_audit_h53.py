from __future__ import annotations

"""H5.3 fixed-support weighting/interpolation audit.

Canonical FULL102 support dependence groups and nearest parent rows are selected once
with the frozen Teacher geometry. Every shadow arm reuses those exact parent rows and
changes only the post-selection weights. No Student is loaded and no production Teacher
configuration is changed.
"""

from typing import Any, Mapping, Sequence

import numpy as np

from .medium48_support_selection_causal_audit_h52 import active_dimensions_h52
from .probabilistic_teacher_r6 import canonical_hash
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_failure_localization_h51 import _fold_material_h5, _group_equal_mean, _support_and_aligned_h5, require
from .teacher_temporal_transport_audit_h5 import H5_QUANTILES, pinball_score_h5, score_teacher_evidence_h5
from .teacher_vectorized_r11 import _weighted_quantiles_batch_r11

H53_RUNTIME = "CB16_R11_H5_3_FIXED_SUPPORT_WEIGHTING_INTERPOLATION_AUDIT_R0_V1"
H53_ARMS = (
    "FULL102_CANONICAL",
    "DROP_MEDIUM48_WEIGHT_ONLY",
    "DROP_OPERATOR48_WEIGHT_ONLY",
    "UNIFORM_WEIGHT_FIXED_SUPPORT",
)


def _canonical_fixed_support_block_h53(
    *, index, regime, parent_ids: Sequence[str], evidence_by_parent: Mapping[str, Any]
) -> dict[str, Any]:
    """Reproduce frozen Teacher selection and retain exact top-64 distances."""
    ids = tuple(str(x) for x in parent_ids)
    require(bool(ids), "H53_EMPTY_TARGET_BLOCK")
    target_rows = np.asarray([index.parent_row_by_id[p] for p in ids], dtype=np.int32)
    target = (index.features[target_rows] - regime.mean) / regime.std
    b = len(ids)
    g = int(regime.group_count)
    rmax = int(regime.max_parents_per_group)
    require(g > 0 and rmax > 0, "H53_EMPTY_SUPPORT_REGIME")

    target_norm2 = np.einsum("ij,ij->i", target, target, optimize=True)
    dot = target @ regime.normalized_support_flat.T
    sq = (target_norm2[:, None] + regime.normalized_support_norm2[None, :] - 2.0 * dot) / float(index.feature_dim)
    np.maximum(sq, 0.0, out=sq)
    flat_dist = np.sqrt(sq, out=sq)
    flat_dist[:, ~regime.valid_support_flat] = np.inf
    parent_dist = flat_dist.reshape(b, g, rmax)

    nearest_slot = np.argmin(parent_dist, axis=2)
    nearest_distance = np.take_along_axis(parent_dist, nearest_slot[..., None], axis=2)[..., 0]
    safe_parent_matrix = regime.support_parent_rows.copy()
    first_valid = int(safe_parent_matrix[safe_parent_matrix >= 0][0])
    safe_parent_matrix[safe_parent_matrix < 0] = first_valid
    nearest_parent_row = np.take_along_axis(
        np.broadcast_to(safe_parent_matrix[None, :, :], parent_dist.shape),
        nearest_slot[..., None], axis=2,
    )[..., 0]
    rank = np.broadcast_to(regime.dep_lex_rank[None, :], nearest_distance.shape)
    order = np.lexsort((rank, nearest_distance), axis=1)
    k = min(int(R11_VALIDATION_TEACHER_CONFIG.k_dependence_groups), g)
    top_dep = order[:, :k]
    top_distance = np.take_along_axis(nearest_distance, top_dep, axis=1)
    selected_rows = np.take_along_axis(nearest_parent_row, top_dep, axis=1)

    for i, pid in enumerate(ids):
        law = evidence_by_parent[pid].action_laws[0]
        dep_ids = tuple(str(regime.dep_ids[int(x)]) for x in top_dep[i])
        require(canonical_hash(dep_ids) == str(law.support_dependence_group_hash), f"H53_SUPPORT_HASH_DRIFT:{pid}")
        require(abs(float(top_distance[i, 0]) - float(law.nearest_distance)) <= 1e-12, f"H53_NEAREST_DRIFT:{pid}")
        require(abs(float(top_distance[i, -1]) - float(law.max_distance_used)) <= 1e-12, f"H53_MAX_DISTANCE_DRIFT:{pid}")
    return {
        "parent_ids": ids,
        "target_rows": target_rows,
        "target_z": np.ascontiguousarray(target, dtype=np.float64),
        "selected_parent_rows": np.ascontiguousarray(selected_rows, dtype=np.int32),
        "canonical_top_distances": np.ascontiguousarray(top_distance, dtype=np.float64),
        "k": int(k),
    }


def _weights_from_distance_h53(distance: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    require(d.ndim == 2 and d.shape[1] > 0 and np.isfinite(d).all(), "H53_BAD_DISTANCE_MATRIX")
    t = float(R11_VALIDATION_TEACHER_CONFIG.distance_temperature)
    w = np.exp(-0.5 * (d / t) ** 2) + 1e-12
    w /= np.sum(w, axis=1, keepdims=True)
    require(np.isfinite(w).all() and np.all(w > 0.0), "H53_BAD_WEIGHT_MATRIX")
    require(np.allclose(np.sum(w, axis=1), 1.0, rtol=0.0, atol=1e-14), "H53_WEIGHT_SUM_DRIFT")
    return np.ascontiguousarray(w, dtype=np.float64)


def _shadow_distance_h53(*, index, regime, target_rows: np.ndarray, selected_rows: np.ndarray, metric_name: str) -> np.ndarray:
    active = active_dimensions_h52(metric_name, int(index.feature_dim))
    target_z = (index.features[target_rows] - regime.mean) / regime.std
    support_z = (index.features[selected_rows] - regime.mean) / regime.std
    delta = support_z[:, :, active] - target_z[:, None, active]
    sq = np.mean(delta * delta, axis=2)
    np.maximum(sq, 0.0, out=sq)
    return np.sqrt(sq, out=sq)


def _laws_from_weights_h53(*, selected_utility: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    u = np.asarray(selected_utility, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    require(u.ndim == 3 and w.shape == u.shape[:2], "H53_UTILITY_WEIGHT_SHAPE")
    means = np.einsum("bk,bka->ba", w, u, optimize=True)
    centered = u - means[:, None, :]
    variances = np.einsum("bk,bka->ba", w, centered * centered, optimize=True)
    stds = np.sqrt(np.maximum(variances, 0.0))
    quantiles = _weighted_quantiles_batch_r11(u, w, H5_QUANTILES)
    effective_n = 1.0 / np.sum(w * w, axis=1)
    return {
        "means": np.ascontiguousarray(means),
        "stds": np.ascontiguousarray(stds),
        "quantiles": np.ascontiguousarray(quantiles),
        "effective_n": np.ascontiguousarray(effective_n),
    }


def compile_fixed_support_weight_arms_h53(
    *, index, regime, eval_parent_ids: Sequence[str], aligned_evidence: Sequence[Any], block_targets: int = 32
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    ids = tuple(str(x) for x in eval_parent_ids)
    by_ev = {x.parent_id: x for x in aligned_evidence}
    require(set(by_ev) == set(ids), "H53_ALIGNED_TARGET_SET_DRIFT")
    out = {arm: {} for arm in H53_ARMS}
    tv_rows = {arm: [] for arm in H53_ARMS if arm != "FULL102_CANONICAL"}

    for start in range(0, len(ids), int(block_targets)):
        block_ids = ids[start:start + int(block_targets)]
        fixed = _canonical_fixed_support_block_h53(
            index=index, regime=regime, parent_ids=block_ids, evidence_by_parent=by_ev
        )
        selected_rows = fixed["selected_parent_rows"]
        target_rows = fixed["target_rows"]
        selected_utility = np.asarray(index.utilities[selected_rows], dtype=np.float64)
        canonical_w = _weights_from_distance_h53(fixed["canonical_top_distances"])
        drop_m_w = _weights_from_distance_h53(_shadow_distance_h53(
            index=index, regime=regime, target_rows=target_rows, selected_rows=selected_rows, metric_name="DROP_MEDIUM48"
        ))
        drop_o_w = _weights_from_distance_h53(_shadow_distance_h53(
            index=index, regime=regime, target_rows=target_rows, selected_rows=selected_rows, metric_name="DROP_OPERATOR48"
        ))
        uniform_w = np.full_like(canonical_w, 1.0 / float(canonical_w.shape[1]))
        weights_by_arm = {
            "FULL102_CANONICAL": canonical_w,
            "DROP_MEDIUM48_WEIGHT_ONLY": drop_m_w,
            "DROP_OPERATOR48_WEIGHT_ONLY": drop_o_w,
            "UNIFORM_WEIGHT_FIXED_SUPPORT": uniform_w,
        }
        laws_by_arm = {arm: _laws_from_weights_h53(selected_utility=selected_utility, weights=w) for arm, w in weights_by_arm.items()}

        for bi, pid in enumerate(block_ids):
            canonical_law0 = by_ev[pid].action_laws[0]
            full = laws_by_arm["FULL102_CANONICAL"]
            require(abs(float(full["effective_n"][bi]) - float(canonical_law0.effective_dependence_n)) <= 1e-10,
                    f"H53_FULL_EFFECTIVE_N_DRIFT:{pid}")
            for a, law in enumerate(by_ev[pid].action_laws):
                require(abs(float(full["means"][bi, a]) - float(law.mean_utility)) <= 1e-12, f"H53_FULL_MEAN_DRIFT:{pid}:{a}")
                require(abs(float(full["stds"][bi, a]) - float(law.std_utility)) <= 1e-12, f"H53_FULL_STD_DRIFT:{pid}:{a}")
                require(np.allclose(full["quantiles"][bi, a], np.asarray(law.quantiles, dtype=np.float64), rtol=0.0, atol=1e-12),
                        f"H53_FULL_QUANTILE_DRIFT:{pid}:{a}")
            selected_tuple = tuple(int(x) for x in selected_rows[bi])
            for arm in H53_ARMS:
                lawset = laws_by_arm[arm]
                out[arm][pid] = {
                    "selected_parent_rows": selected_tuple,
                    "means": tuple(float(x) for x in lawset["means"][bi]),
                    "stds": tuple(float(x) for x in lawset["stds"][bi]),
                    "quantiles": tuple(tuple(float(y) for y in lawset["quantiles"][bi, a]) for a in range(index.action_count)),
                    "effective_n": float(lawset["effective_n"][bi]),
                }
                if arm != "FULL102_CANONICAL":
                    tv_rows[arm].append((pid, float(0.5 * np.sum(np.abs(weights_by_arm[arm][bi] - canonical_w[bi])))))

    for pid in ids:
        ref = out["FULL102_CANONICAL"][pid]["selected_parent_rows"]
        for arm in H53_ARMS[1:]:
            require(out[arm][pid]["selected_parent_rows"] == ref, f"H53_SELECTED_PARENT_IDENTITY_DRIFT:{arm}:{pid}")
    return out, {
        "target_count": len(ids),
        "all_shadow_arms_selected_parent_rows_identical": True,
        "full102_reference_law_identity_guard": "PASS",
        "weight_total_variation_rows": tv_rows,
    }


def score_weight_arm_h53(*, arm_rows: Mapping[str, Mapping[str, Any]], index, eval_parents: Mapping[str, Any]) -> dict[str, Any]:
    overall = []
    mean_mse = []
    c1090 = []
    c0595 = []
    for pid in sorted(arm_rows):
        row = arm_rows[pid]
        gid = str(eval_parents[pid].dependence_group_id)
        target_row = int(index.parent_row_by_id[pid])
        y = np.asarray(index.utilities[target_row], dtype=np.float64)
        means = np.asarray(row["means"], dtype=np.float64)
        q = np.asarray(row["quantiles"], dtype=np.float64)
        require(q.shape == (int(index.action_count), len(H5_QUANTILES)), f"H53_Q_SHAPE:{pid}")
        losses = [pinball_score_h5(float(y[a]), q[a], H5_QUANTILES) for a in range(index.action_count)]
        overall.append((gid, float(np.mean(losses))))
        mean_mse.append((gid, float(np.mean((y - means) ** 2))))
        c1090.append((gid, float(np.mean((q[:, 1] <= y) & (y <= q[:, 5])))))
        c0595.append((gid, float(np.mean((q[:, 0] <= y) & (y <= q[:, 6])))))
    return {
        "qscore": _group_equal_mean(overall),
        "mean_utility_mse": _group_equal_mean(mean_mse),
        "coverage_10_90": _group_equal_mean(c1090),
        "coverage_05_95": _group_equal_mean(c0595),
        "future_dependence_groups": len({gid for gid, _ in overall}),
        "parent_contexts": len(overall),
    }


def _group_equal_tv_h53(rows: Sequence[tuple[str, float]], eval_parents: Mapping[str, Any]) -> float:
    grouped = []
    for pid, value in rows:
        grouped.append((str(eval_parents[pid].dependence_group_id), float(value)))
    return _group_equal_mean(grouped)


def run_fold_h53(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any], block_targets: int = 32
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    eval_parent_ids = tuple(sorted(eval_parents))
    index, regime, aligned, execution = _support_and_aligned_h5(
        parents=parents, samples=samples, eval_parent_ids=eval_parent_ids, block_targets=block_targets
    )
    arms, receipt = compile_fixed_support_weight_arms_h53(
        index=index, regime=regime, eval_parent_ids=eval_parent_ids, aligned_evidence=aligned, block_targets=block_targets
    )
    scores = {arm: score_weight_arm_h53(arm_rows=arms[arm], index=index, eval_parents=eval_parents) for arm in H53_ARMS}
    frozen = score_teacher_evidence_h5(evidence=aligned, index=index, eval_parents=eval_parents)
    require(abs(float(scores["FULL102_CANONICAL"]["qscore"]) - float(frozen["qscore"])) <= 1e-15,
            f"H53_FULL_QSCORE_IDENTITY_DRIFT:{fold}")
    tv = {
        arm: _group_equal_tv_h53(receipt["weight_total_variation_rows"][arm], eval_parents)
        for arm in H53_ARMS[1:]
    }
    return {
        "fold": fold,
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": int(regime.group_count),
        "eval_dependence_groups": int(scores["FULL102_CANONICAL"]["future_dependence_groups"]),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "arms": scores,
        "weight_total_variation_vs_full102": tv,
        "full102_frozen_teacher_qscore_identity_guard": "PASS",
        "full102_support_hash_identity_guard": "PASS",
        "selected_parent_identity_across_all_arms_guard": "PASS",
        "teacher_execution_identity": execution,
    }


def adjudicate_h53(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H53_FOLD_SET_DRIFT")
    late = [rows[3], rows[4]]
    def beats_both(arm: str) -> bool:
        return all(float(x["arms"][arm]["qscore"]) < float(x["arms"]["FULL102_CANONICAL"]["qscore"]) for x in late)
    dm = beats_both("DROP_MEDIUM48_WEIGHT_ONLY")
    do = beats_both("DROP_OPERATOR48_WEIGHT_ONLY")
    uni = beats_both("UNIFORM_WEIGHT_FIXED_SUPPORT")
    identities = all(
        x["full102_frozen_teacher_qscore_identity_guard"] == "PASS"
        and x["full102_support_hash_identity_guard"] == "PASS"
        and x["selected_parent_identity_across_all_arms_guard"] == "PASS"
        for x in rows
    )
    if dm and (not do) and (not uni):
        classification = "MEDIUM48_WEIGHTING_PRIMARY"
    elif uni:
        classification = "BROAD_KERNEL_WEIGHTING_FAILURE"
    elif do and (not dm):
        classification = "OPERATOR48_WEIGHTING_PRIMARY"
    else:
        classification = "H5_3_WEIGHTING_INTERPOLATION_UNRESOLVED"
    return {
        "schema": "CB16_R11_M_SERIES_H5_3_FIXED_SUPPORT_WEIGHTING_INTERPOLATION_AUDIT_ADJUDICATION_R0_V1",
        "classification": classification,
        "drop_medium48_weight_beats_full102_both_late_folds": bool(dm),
        "drop_operator48_weight_beats_full102_both_late_folds": bool(do),
        "uniform_weight_beats_full102_both_late_folds": bool(uni),
        "all_identity_guards_pass": bool(identities),
        "conclusion": f"H5_3_{classification}",
        "per_fold": [{
            "fold": int(x["fold"]),
            "full102_qscore": float(x["arms"]["FULL102_CANONICAL"]["qscore"]),
            "drop_medium48_weight_qscore": float(x["arms"]["DROP_MEDIUM48_WEIGHT_ONLY"]["qscore"]),
            "drop_operator48_weight_qscore": float(x["arms"]["DROP_OPERATOR48_WEIGHT_ONLY"]["qscore"]),
            "uniform_weight_qscore": float(x["arms"]["UNIFORM_WEIGHT_FIXED_SUPPORT"]["qscore"]),
            "weight_total_variation_vs_full102": x["weight_total_variation_vs_full102"],
        } for x in rows],
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
    }


__all__ = [
    "H53_RUNTIME", "H53_ARMS", "compile_fixed_support_weight_arms_h53", "score_weight_arm_h53",
    "run_fold_h53", "adjudicate_h53",
]
