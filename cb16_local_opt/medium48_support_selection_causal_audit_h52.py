from __future__ import annotations

"""H5.2 Student-free, Teacher-output-free causal audit of support selection.

Selection functions consume state geometry only. Realized 9-action utilities are read
only after support identities are frozen for a target. This file never compiles a new
Teacher law and never changes the canonical Teacher configuration.
"""

import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import canonical_hash
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_failure_localization_h51 import HOUR_MS, _fold_material_h5, _group_equal_mean, _support_and_aligned_h5, require

H52_RUNTIME = "CB16_R11_H5_2_MEDIUM48_SUPPORT_SELECTION_CAUSAL_AUDIT_R0_V1"
METRIC_NAMES = ("FULL102", "DROP_MEDIUM48", "DROP_OPERATOR48")


def active_dimensions_h52(name: str, feature_dim: int) -> np.ndarray:
    require(int(feature_dim) == 102, f"H52_FEATURE_DIM_DRIFT:{feature_dim}")
    name = str(name)
    if name == "FULL102":
        idx = np.arange(102, dtype=np.int32)
    elif name == "DROP_MEDIUM48":
        idx = np.asarray([*range(0, 48), *range(96, 102)], dtype=np.int32)
    elif name == "DROP_OPERATOR48":
        idx = np.asarray(range(48, 102), dtype=np.int32)
    else:
        raise RuntimeError(f"H52_UNKNOWN_METRIC:{name}")
    require(len(idx) > 0 and len(set(idx.tolist())) == len(idx), f"H52_BAD_ACTIVE_DIMS:{name}")
    return idx


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    require(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all(), "H52_BAD_RANK_INPUT")
    order = np.argsort(a, kind="mergesort")
    out = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * ((i + 1) + j)
        out[order[i:j]] = rank
        i = j
    return out


def _spearman9(a: np.ndarray, b: np.ndarray) -> float:
    ra = _rankdata_average(a)
    rb = _rankdata_average(b)
    da = ra - float(np.mean(ra))
    db = rb - float(np.mean(rb))
    den = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    if den <= 0.0:
        return 0.0
    return float(np.sum(da * db) / den)


def select_support_block_h52(
    *,
    index,
    regime,
    parent_ids: Sequence[str],
    metric_name: str,
    full_evidence_by_parent: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Select support using state features only; utility tensors are intentionally absent."""
    ids = tuple(str(x) for x in parent_ids)
    require(bool(ids), "H52_EMPTY_TARGET_BLOCK")
    active = active_dimensions_h52(metric_name, int(index.feature_dim))
    target_rows = np.asarray([index.parent_row_by_id[p] for p in ids], dtype=np.int32)
    target_full = (index.features[target_rows] - regime.mean) / regime.std
    target = np.ascontiguousarray(target_full[:, active], dtype=np.float64)
    support = np.ascontiguousarray(regime.normalized_support_flat[:, active], dtype=np.float64)

    b = len(ids)
    g = int(regime.group_count)
    rmax = int(regime.max_parents_per_group)
    require(g > 0 and rmax > 0, "H52_EMPTY_SUPPORT_REGIME")

    target_norm2 = np.einsum("ij,ij->i", target, target, optimize=True)
    if str(metric_name) == "FULL102":
        # Exact frozen Teacher arithmetic path.
        support_norm2 = regime.normalized_support_norm2
    else:
        support_norm2 = np.einsum("ij,ij->i", support, support, optimize=True)
    dot = target @ support.T
    sq = (target_norm2[:, None] + support_norm2[None, :] - 2.0 * dot) / float(len(active))
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
        nearest_slot[..., None],
        axis=2,
    )[..., 0]
    rank = np.broadcast_to(regime.dep_lex_rank[None, :], nearest_distance.shape)
    order = np.lexsort((rank, nearest_distance), axis=1)
    k = min(int(R11_VALIDATION_TEACHER_CONFIG.k_dependence_groups), g)
    top_dep = order[:, :k]
    top_dist = np.take_along_axis(nearest_distance, top_dep, axis=1)
    selected_rows = np.take_along_axis(nearest_parent_row, top_dep, axis=1)

    out: dict[str, dict[str, Any]] = {}
    for i, pid in enumerate(ids):
        dep_ids = tuple(str(regime.dep_ids[int(x)]) for x in top_dep[i])
        if str(metric_name) == "FULL102":
            require(full_evidence_by_parent is not None and pid in full_evidence_by_parent, f"H52_MISSING_FULL_EVIDENCE:{pid}")
            law = full_evidence_by_parent[pid].action_laws[0]
            require(canonical_hash(dep_ids) == str(law.support_dependence_group_hash), f"H52_FULL102_SUPPORT_HASH_DRIFT:{pid}")
            require(abs(float(top_dist[i, 0]) - float(law.nearest_distance)) <= 1e-12, f"H52_FULL102_NEAREST_DRIFT:{pid}")
            require(abs(float(top_dist[i, -1]) - float(law.max_distance_used)) <= 1e-12, f"H52_FULL102_MAX_DISTANCE_DRIFT:{pid}")
        out[pid] = {
            "selected_dep_ids": dep_ids,
            "selected_parent_rows": tuple(int(x) for x in selected_rows[i]),
            "nearest_distance": float(top_dist[i, 0]),
            "max_distance": float(top_dist[i, -1]),
            "active_dimension_count": int(len(active)),
        }
    return out


def select_support_all_h52(
    *, index, regime, eval_parent_ids: Sequence[str], metric_name: str,
    full_evidence: Sequence[Any] | None, block_targets: int,
) -> dict[str, dict[str, Any]]:
    ids = tuple(str(x) for x in eval_parent_ids)
    by_ev = None if full_evidence is None else {x.parent_id: x for x in full_evidence}
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), int(block_targets)):
        block = ids[start:start + int(block_targets)]
        rows = select_support_block_h52(
            index=index, regime=regime, parent_ids=block, metric_name=metric_name,
            full_evidence_by_parent=by_ev,
        )
        require(not (set(out) & set(rows)), "H52_DUPLICATE_TARGET")
        out.update(rows)
    require(set(out) == set(ids), "H52_TARGET_SET_DRIFT")
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa, bb = set(map(str, a)), set(map(str, b))
    require(bool(aa | bb), "H52_EMPTY_JACCARD")
    return float(len(aa & bb) / len(aa | bb))


def score_selected_support_h52(
    *, index, eval_parents: Mapping[str, Any], parents: Mapping[str, Any],
    selection: Mapping[str, Mapping[str, Any]], full_selection: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Read future utilities only after support selection is already materialized."""
    rows: list[dict[str, Any]] = []
    for pid in sorted(selection):
        require(pid in full_selection, f"H52_FULL_SELECTION_MISSING:{pid}")
        target_row = int(index.parent_row_by_id[pid])
        target_u = np.asarray(index.utilities[target_row], dtype=np.float64)
        require(target_u.shape == (int(index.action_count),), f"H52_TARGET_UTILITY_SHAPE:{pid}")
        selected_rows = np.asarray(selection[pid]["selected_parent_rows"], dtype=np.int32)
        require(len(selected_rows) > 0, f"H52_EMPTY_SELECTED_ROWS:{pid}")
        support_u = np.asarray(index.utilities[selected_rows], dtype=np.float64)
        require(support_u.shape == (len(selected_rows), int(index.action_count)), f"H52_SUPPORT_UTILITY_SHAPE:{pid}")
        diff = support_u - target_u[None, :]
        mse = float(np.mean(diff * diff))
        mae = float(np.mean(np.abs(diff)))
        target_best = int(np.argmax(target_u))
        best_agreement = float(np.mean(np.argmax(support_u, axis=1) == target_best))
        spear = float(np.mean([_spearman9(target_u, support_u[i]) for i in range(len(support_u))]))
        target_parent = eval_parents[pid]
        support_parent_ids = [str(index.parent_ids[int(r)]) for r in selected_rows]
        support_parents = [parents[x] for x in support_parent_ids]
        age = float(np.median([
            (int(target_parent.decision_time_ms) - int(p.decision_time_ms)) / HOUR_MS for p in support_parents
        ]))
        same_symbol = float(np.mean([str(p.symbol) == str(target_parent.symbol) for p in support_parents]))
        same_scenario = float(np.mean([str(p.scenario) == str(target_parent.scenario) for p in support_parents]))
        rows.append({
            "parent_id": pid,
            "gid": str(target_parent.dependence_group_id),
            "profile_mse": mse,
            "profile_mae": mae,
            "best_action_agreement": best_agreement,
            "spearman_action_profile": spear,
            "jaccard_vs_full102": _jaccard(selection[pid]["selected_dep_ids"], full_selection[pid]["selected_dep_ids"]),
            "support_age_hours_median": age,
            "same_symbol_fraction": same_symbol,
            "same_account_scenario_fraction": same_scenario,
            "nearest_distance": float(selection[pid]["nearest_distance"]),
        })

    def gm(key: str) -> float:
        return _group_equal_mean([(x["gid"], float(x[key])) for x in rows])

    by_gid: dict[str, list[float]] = {}
    for row in rows:
        by_gid.setdefault(row["gid"], []).append(float(row["jaccard_vs_full102"]))
    jaccard_median_group_equal = float(np.mean([np.median(v) for v in by_gid.values()]))
    return {
        "future_dependence_groups": len(by_gid),
        "parent_contexts": len(rows),
        "profile_mse": gm("profile_mse"),
        "profile_mae": gm("profile_mae"),
        "best_action_agreement": gm("best_action_agreement"),
        "spearman_action_profile": gm("spearman_action_profile"),
        "selected_support_jaccard_vs_full102_median_group_equal_mean": jaccard_median_group_equal,
        "selected_support_age_hours_median_group_equal_mean": gm("support_age_hours_median"),
        "same_symbol_fraction": gm("same_symbol_fraction"),
        "same_account_scenario_fraction": gm("same_account_scenario_fraction"),
        "nearest_distance_group_equal_mean": gm("nearest_distance"),
    }


def run_fold_h52(
    *, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any], block_targets: int = 32,
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    eval_parent_ids = tuple(sorted(eval_parents))
    index, regime, aligned, execution = _support_and_aligned_h5(
        parents=parents, samples=samples, eval_parent_ids=eval_parent_ids, block_targets=block_targets
    )
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    selections["FULL102"] = select_support_all_h52(
        index=index, regime=regime, eval_parent_ids=eval_parent_ids, metric_name="FULL102",
        full_evidence=aligned, block_targets=block_targets,
    )
    for name in ("DROP_MEDIUM48", "DROP_OPERATOR48"):
        selections[name] = select_support_all_h52(
            index=index, regime=regime, eval_parent_ids=eval_parent_ids, metric_name=name,
            full_evidence=None, block_targets=block_targets,
        )
    scores = {
        name: score_selected_support_h52(
            index=index, eval_parents=eval_parents, parents=parents,
            selection=selections[name], full_selection=selections["FULL102"],
        ) for name in METRIC_NAMES
    }
    require(abs(float(scores["FULL102"]["selected_support_jaccard_vs_full102_median_group_equal_mean"]) - 1.0) <= 1e-15,
            f"H52_FULL_JACCARD_NOT_ONE:{fold}")
    return {
        "fold": fold,
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": int(regime.group_count),
        "eval_dependence_groups": int(scores["FULL102"]["future_dependence_groups"]),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "metrics": scores,
        "full102_exact_support_identity_guard": "PASS",
        "selection_utility_access": False,
        "post_selection_truth_scoring": True,
        "teacher_output_compiled": False,
        "teacher_execution_identity": execution,
    }


def adjudicate_h52(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H52_FOLD_SET_DRIFT")
    late = [rows[3], rows[4]]
    medium_better_both = all(
        float(x["metrics"]["DROP_MEDIUM48"]["profile_mse"]) < float(x["metrics"]["FULL102"]["profile_mse"]) for x in late
    )
    operator_better_both = all(
        float(x["metrics"]["DROP_OPERATOR48"]["profile_mse"]) < float(x["metrics"]["FULL102"]["profile_mse"]) for x in late
    )
    nontrivial_both = all(
        float(x["metrics"]["DROP_MEDIUM48"]["selected_support_jaccard_vs_full102_median_group_equal_mean"]) < 0.80 for x in late
    )
    identity_all = all(str(x["full102_exact_support_identity_guard"]) == "PASS" for x in rows)
    passed = bool(medium_better_both and (not operator_better_both) and nontrivial_both and identity_all)
    return {
        "schema": "CB16_R11_M_SERIES_H5_2_MEDIUM48_SUPPORT_SELECTION_CAUSAL_AUDIT_ADJUDICATION_R0_V1",
        "overall_support_selection_mechanism_supported": passed,
        "drop_medium48_profile_mse_better_than_full102_both_late_folds": bool(medium_better_both),
        "drop_operator48_profile_mse_better_than_full102_both_late_folds": bool(operator_better_both),
        "drop_medium48_selection_nontrivial_both_late_folds": bool(nontrivial_both),
        "full102_exact_support_identity_all_folds": bool(identity_all),
        "conclusion": (
            "H5_2_MEDIUM48_SUPPORT_SELECTION_MECHANISM_SUPPORTED_ON_CONSUMED_TRAIN_ONLY_SUPPORT"
            if passed else
            "H5_2_MEDIUM48_SUPPORT_SELECTION_MECHANISM_NOT_SUPPORTED__DO_NOT_DROP_OR_REWEIGHT_MEDIUM48"
        ),
        "per_fold": [{
            "fold": int(x["fold"]),
            "full102_profile_mse": float(x["metrics"]["FULL102"]["profile_mse"]),
            "drop_medium48_profile_mse": float(x["metrics"]["DROP_MEDIUM48"]["profile_mse"]),
            "drop_operator48_profile_mse": float(x["metrics"]["DROP_OPERATOR48"]["profile_mse"]),
            "drop_medium48_jaccard_vs_full102": float(x["metrics"]["DROP_MEDIUM48"]["selected_support_jaccard_vs_full102_median_group_equal_mean"]),
            "drop_operator48_jaccard_vs_full102": float(x["metrics"]["DROP_OPERATOR48"]["selected_support_jaccard_vs_full102_median_group_equal_mean"]),
        } for x in rows],
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
    }


def selection_signature_guard_h52() -> bool:
    params = set(inspect.signature(select_support_block_h52).parameters)
    forbidden = {"utility", "utilities", "target_utility", "future_utility", "outcome"}
    return not bool(params & forbidden)


__all__ = [
    "H52_RUNTIME", "METRIC_NAMES", "active_dimensions_h52", "select_support_block_h52",
    "select_support_all_h52", "score_selected_support_h52", "run_fold_h52", "adjudicate_h52",
    "selection_signature_guard_h52",
]
