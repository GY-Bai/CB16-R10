from __future__ import annotations

"""H5.5 Teacher-free state-to-counterfactual-utility geometry transport audit."""

import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .teacher_temporal_transport_audit_h5 import H5_SHIFTS, _fold_material_h5, require, shuffled_target_feature_index_h5
from .teacher_vectorized_r11 import build_columnar_teacher_index_r11

H55_RUNTIME = "CB16_R11_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_V1"
H55_METRICS = {
    "FULL102": tuple(range(102)),
    "MARKET96": tuple(range(96)),
    "OPERATOR48": tuple(range(48)),
    "MEDIUM48": tuple(range(48, 96)),
    "ACCOUNT6": tuple(range(96, 102)),
}


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = r
        i = j
    return ranks


def spearman_h55(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(tuple(x), dtype=np.float64)
    b = np.asarray(tuple(y), dtype=np.float64)
    require(a.shape == b.shape and a.ndim == 1 and len(a) >= 3, "H55_SPEARMAN_SHAPE")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "H55_SPEARMAN_NONFINITE")
    ra = _rankdata_average(a); rb = _rankdata_average(b)
    da = ra - ra.mean(); db = rb - rb.mean()
    denom = float(np.sqrt(np.sum(da * da) * np.sum(db * db)))
    return 0.0 if denom == 0.0 else float(np.sum(da * db) / denom)


def _scenario_support_rows(index, train_parents: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    by_dep: dict[str, dict[str, int]] = {}
    dep_ts: dict[str, int] = {}
    for pid, p in train_parents.items():
        gid = str(p.dependence_group_id); scenario = str(p.scenario)
        by_dep.setdefault(gid, {})
        require(scenario not in by_dep[gid], f"H55_DUPLICATE_TRAIN_SCENARIO:{gid}:{scenario}")
        by_dep[gid][scenario] = int(index.parent_row_by_id[pid])
        dep_ts[gid] = int(p.decision_time_ms)
    dep_order = tuple(sorted(by_dep, key=lambda g: (dep_ts[g], g)))
    scenarios = sorted(next(iter(by_dep.values())).keys())
    out = {}
    for scenario in scenarios:
        rows = []
        for gid in dep_order:
            require(scenario in by_dep[gid], f"H55_MISSING_TRAIN_SCENARIO:{gid}:{scenario}")
            rows.append(by_dep[gid][scenario])
        out[scenario] = (np.asarray(rows, dtype=np.int32), dep_order)
    return out


def _centered_profiles(u: np.ndarray) -> np.ndarray:
    x = np.asarray(u, dtype=np.float64)
    require(x.ndim == 2 and x.shape[1] == 9, "H55_PROFILE_SHAPE")
    return x - x.mean(axis=1, keepdims=True)


def _metric_readout(*, index, train_rows_all: np.ndarray, eval_rows: np.ndarray, eval_parents: Mapping[str, Any], train_parents: Mapping[str, Any], metric: str) -> dict[str, float]:
    active = np.asarray(H55_METRICS[metric], dtype=np.int32)
    train_x = np.asarray(index.features[train_rows_all][:, active], dtype=np.float64)
    mean = train_x.mean(axis=0); std = train_x.std(axis=0, ddof=0); std = np.where(std < 1e-8, 1.0, std)
    support = _scenario_support_rows(index, train_parents)

    per_target = []
    raw_per_target = []
    decile_ratio = []
    grouped: dict[str, list[float]] = {}
    grouped_raw: dict[str, list[float]] = {}
    grouped_ratio: dict[str, list[float]] = {}
    for row in eval_rows:
        pid = index.parent_ids[int(row)]; p = eval_parents[pid]; scenario = str(p.scenario); gid = str(p.dependence_group_id)
        support_rows, _ = support[scenario]
        target_z = (index.features[int(row), active] - mean) / std
        support_z = (index.features[support_rows][:, active] - mean) / std
        state_d = np.sqrt(np.mean((support_z - target_z[None, :]) ** 2, axis=1))
        target_u = np.asarray(index.utilities[int(row)], dtype=np.float64)
        support_u = np.asarray(index.utilities[support_rows], dtype=np.float64)
        centered_mse = np.mean((_centered_profiles(support_u) - _centered_profiles(target_u[None, :])) ** 2, axis=1)
        raw_mse = np.mean((support_u - target_u[None, :]) ** 2, axis=1)
        rho = spearman_h55(state_d, centered_mse)
        rho_raw = spearman_h55(state_d, raw_mse)
        k = max(1, int(math.ceil(0.1 * len(state_d))))
        nearest = np.argsort(state_d, kind="stable")[:k]
        ratio = float(np.mean(centered_mse[nearest]) / max(float(np.mean(centered_mse)), 1e-30))
        per_target.append(rho); raw_per_target.append(rho_raw); decile_ratio.append(ratio)
        grouped.setdefault(gid, []).append(rho); grouped_raw.setdefault(gid, []).append(rho_raw); grouped_ratio.setdefault(gid, []).append(ratio)
    require(grouped and all(len(v) == 6 for v in grouped.values()), "H55_EVAL_SCENARIO_GROUPING_DRIFT")
    return {
        "rho": float(np.mean([np.mean(v) for v in grouped.values()])),
        "raw_rho": float(np.mean([np.mean(grouped_raw[g]) for g in grouped])),
        "nearest_decile_centered_mse_ratio": float(np.mean([np.mean(grouped_ratio[g]) for g in grouped])),
        "eval_future_groups": int(len(grouped)),
        "eval_parent_contexts": int(len(per_target)),
    }


def run_fold_h55(*, fold_spec: Mapping[str, Any], all_train_parents: Mapping[str, Any], all_train_samples: Sequence[Any]) -> dict[str, Any]:
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec, all_train_parents=all_train_parents, all_train_samples=all_train_samples
    )
    index = build_columnar_teacher_index_r11(samples)
    train_rows = np.asarray([index.parent_row_by_id[p] for p in train_parents], dtype=np.int32)
    eval_ids = tuple(sorted(eval_parents)); eval_rows = np.asarray([index.parent_row_by_id[p] for p in eval_ids], dtype=np.int32)
    aligned = {m: _metric_readout(index=index, train_rows_all=train_rows, eval_rows=eval_rows, eval_parents=eval_parents, train_parents=train_parents, metric=m) for m in H55_METRICS}
    rotated = {}
    receipts = {}
    for shift in H5_SHIFTS:
        idx2, receipt = shuffled_target_feature_index_h5(index=index, eval_parents=eval_parents, eval_parent_ids=eval_ids, shift=int(shift))
        rotated[int(shift)] = {m: _metric_readout(index=idx2, train_rows_all=train_rows, eval_rows=eval_rows, eval_parents=eval_parents, train_parents=train_parents, metric=m) for m in H55_METRICS}
        receipts[int(shift)] = receipt
    return {
        "fold": int(fold_spec["fold"]),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "aligned": aligned,
        "rotated": rotated,
        "rotation_receipts": receipts,
        "teacher_compilation_used": False,
        "teacher_kernel_used": False,
        "teacher_support_selection_used": False,
    }


def adjudicate_h55(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(folds, key=lambda x: int(x["fold"])); require(tuple(int(x["fold"]) for x in rows) == (1,2,3,4,5), "H55_FOLD_SET_DRIFT")
    metrics = {}
    for metric in H55_METRICS:
        positive_count = sum(float(x["aligned"][metric]["rho"]) > 0.0 for x in rows)
        positive_late = all(float(rows[f-1]["aligned"][metric]["rho"]) > 0.0 for f in (4,5))
        beat_med = 0; pair = 0; late_med = True; per_fold = []
        for x in rows:
            a = float(x["aligned"][metric]["rho"])
            sh = [float(x["rotated"][s][metric]["rho"]) for s in H5_SHIFTS]
            med = float(statistics.median(sh)); b = a > med
            beat_med += int(b); pair += sum(a > q for q in sh)
            if int(x["fold"]) in (4,5): late_med = late_med and b
            per_fold.append({"fold": int(x["fold"]), "aligned_rho": a, "median_rotated_rho": med, "aligned_minus_median": a-med, "pair_count": int(sum(a > q for q in sh)), "raw_rho": float(x["aligned"][metric]["raw_rho"]), "nearest_decile_ratio": float(x["aligned"][metric]["nearest_decile_centered_mse_ratio"])})
        supported = bool(positive_count >= 4 and positive_late and beat_med >= 4 and late_med and pair >= 20)
        metrics[metric] = {"transport_supported": supported, "positive_fold_count": int(positive_count), "positive_both_late": bool(positive_late), "beats_shuffle_median_fold_count": int(beat_med), "beats_shuffle_median_both_late": bool(late_med), "pairwise_shuffle_count_of_25": int(pair), "per_fold": per_fold}
    full = metrics["FULL102"]["transport_supported"]; market = metrics["MARKET96"]["transport_supported"]; op = metrics["OPERATOR48"]["transport_supported"]; med = metrics["MEDIUM48"]["transport_supported"]
    if full:
        cls = "FULL_STATE_GEOMETRY_TRANSPORT_SUPPORTED"
    elif op and not med:
        cls = "OPERATOR48_TRANSPORT_SURVIVES_MEDIUM48_SUSPECT"
    elif med and not op:
        cls = "MEDIUM48_TRANSPORT_SURVIVES_OPERATOR48_SUSPECT"
    elif (not full) and (not market) and (not op) and (not med):
        cls = "MARKET_STATE_UTILITY_GEOMETRY_TEMPORAL_NONTRANSPORT"
    else:
        cls = "MIXED_STATE_UTILITY_GEOMETRY_TRANSPORT_UNRESOLVED"
    identities = all(all(r["target_feature_multiset_preserved"] and r["train_features_byte_identical"] and r["scenario_identity_preserved"] for r in x["rotation_receipts"].values()) for x in rows)
    return {"classification": cls, "metrics": metrics, "all_rotation_identity_guards_pass": bool(identities), "market_information_verdict_change": False, "canonical_change_authorized": False, "teacher_change_authorized": False, "student_change_authorized": False, "promotion_authorized": False, "r7_evaluation_authorized": False, "final_opening_authorized": False}
