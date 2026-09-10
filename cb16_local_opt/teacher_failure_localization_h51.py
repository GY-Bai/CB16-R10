from __future__ import annotations

"""H5.1 read-only failure localization for the frozen R11 probabilistic Teacher.

No Student is loaded and no Teacher parameter/configuration is changed.  The only
interventions rotate selected target feature blocks across whole eval future groups,
preserving AccountState scenario identity, while train support and realized utilities
remain fixed.  These interventions are diagnostic controls, never candidate Teachers.
"""

from dataclasses import replace
import hashlib
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import canonical_hash
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_temporal_transport_audit_h5 import (
    H5_QUANTILES,
    H5_SCENARIOS,
    H5_SHIFTS,
    _eval_group_scenario_rows_h5,
    _fold_material_h5,
    _support_and_aligned_h5,
    pinball_score_h5,
    score_teacher_evidence_h5,
    shuffled_target_feature_index_h5,
)
from .teacher_vectorized_r11 import _compile_block_r11

H51_RUNTIME = "CB16_R11_H5_1_TEACHER_FAILURE_LOCALIZATION_R0_V1"
FEATURE_BLOCKS = {
    "OPERATOR48": (0, 48),
    "MEDIUM48": (48, 96),
    "ACCOUNT6": (96, 102),
}
CONTROL_FAMILIES = {
    "FULL_STATE": ("OPERATOR48", "MEDIUM48", "ACCOUNT6"),
    "MARKET96_ONLY": ("OPERATOR48", "MEDIUM48"),
    "OPERATOR48_ONLY": ("OPERATOR48",),
    "MEDIUM48_ONLY": ("MEDIUM48",),
    "ACCOUNT6_ONLY": ("ACCOUNT6",),
}
CENTRAL_LEVELS = (0.25, 0.50, 0.75)
TAIL_LEVELS = (0.05, 0.10, 0.90, 0.95)
HOUR_MS = 3_600_000


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def _block_indices(blocks: Sequence[str]) -> np.ndarray:
    names = tuple(str(x) for x in blocks)
    require(bool(names), "H51_EMPTY_BLOCK_SET")
    require(len(names) == len(set(names)), "H51_DUPLICATE_BLOCK")
    require(set(names).issubset(FEATURE_BLOCKS), f"H51_UNKNOWN_BLOCK:{names}")
    idx: list[int] = []
    for name in FEATURE_BLOCKS:
        if name in names:
            lo, hi = FEATURE_BLOCKS[name]
            idx.extend(range(lo, hi))
    return np.asarray(idx, dtype=np.int32)


def _row_multiset_hash(x: np.ndarray, *, domain: bytes) -> str:
    a = np.ascontiguousarray(x, dtype=np.float64)
    rows = sorted(bytes(np.ascontiguousarray(row).tobytes(order="C")) for row in a)
    h = hashlib.sha256(domain + b"\0")
    for row in rows:
        h.update(row)
    return h.hexdigest()


def shuffled_target_feature_blocks_h51(
    *,
    index,
    eval_parents: Mapping[str, Any],
    eval_parent_ids: Sequence[str],
    shift: int,
    blocks: Sequence[str],
):
    k0 = int(shift)
    require(k0 in H5_SHIFTS, f"H51_UNREGISTERED_SHIFT:{k0}")
    selected = _block_indices(blocks)
    untouched = np.asarray([i for i in range(index.feature_dim) if i not in set(selected.tolist())], dtype=np.int32)
    order, rows = _eval_group_scenario_rows_h5(
        index=index, eval_parents=eval_parents, eval_parent_ids=eval_parent_ids
    )
    n = len(order)
    require(n >= 2, "H51_TOO_FEW_EVAL_GROUPS")
    k = k0 % n
    require(k != 0, "H51_IDENTITY_SHUFFLE")
    eval_rows = np.asarray([index.parent_row_by_id[p] for p in eval_parent_ids], dtype=np.int32)
    before_selected = _row_multiset_hash(
        index.features[eval_rows][:, selected], domain=b"CB16_R11_H51_SELECTED_BLOCK_MULTISET_V1"
    )
    before_untouched = np.ascontiguousarray(index.features[eval_rows][:, untouched])
    out = np.array(index.features, copy=True)
    mapping = []
    for dst_pos, dst_gid in enumerate(order):
        src_gid = order[(dst_pos + k) % n]
        require(src_gid != dst_gid, "H51_SHUFFLE_FIXED_POINT")
        for scenario in H5_SCENARIOS:
            dst = rows[dst_gid][scenario]
            src = rows[src_gid][scenario]
            out[dst, selected] = index.features[src, selected]
        mapping.append((dst_gid, src_gid))
    after_selected = _row_multiset_hash(
        out[eval_rows][:, selected], domain=b"CB16_R11_H51_SELECTED_BLOCK_MULTISET_V1"
    )
    require(before_selected == after_selected, "H51_SELECTED_BLOCK_MULTISET_DRIFT")
    require(np.array_equal(out[eval_rows][:, untouched], before_untouched), "H51_UNSHUFFLED_BLOCK_DRIFT")
    eval_id_set = set(eval_parent_ids)
    train_rows = np.asarray([i for i, pid in enumerate(index.parent_ids) if pid not in eval_id_set], dtype=np.int32)
    require(np.array_equal(out[train_rows], index.features[train_rows]), "H51_TRAIN_FEATURES_CHANGED")
    modified = replace(index, features=np.ascontiguousarray(out))
    names = tuple(name for name in FEATURE_BLOCKS if name in set(str(x) for x in blocks))
    if names == CONTROL_FAMILIES["FULL_STATE"]:
        h5_index, _ = shuffled_target_feature_index_h5(
            index=index, eval_parents=eval_parents, eval_parent_ids=eval_parent_ids, shift=k0
        )
        require(np.array_equal(modified.features, h5_index.features), f"H51_FULL_STATE_NOT_EXACT_H5:{k0}")
    return modified, {
        "shift": k0,
        "blocks": list(names),
        "future_group_count": n,
        "scenario_identity_preserved": True,
        "selected_block_multiset_sha256_before": before_selected,
        "selected_block_multiset_sha256_after": after_selected,
        "selected_block_multiset_preserved": True,
        "unshuffled_blocks_byte_identical": True,
        "train_features_byte_identical": True,
        "eval_realized_utilities_unchanged": True,
        "fixed_points": 0,
        "mapping_hash": hashlib.sha256(repr(mapping).encode("utf-8")).hexdigest(),
    }


def compile_block_control_h51(
    *,
    index,
    regime,
    eval_parents: Mapping[str, Any],
    eval_parent_ids: Sequence[str],
    shift: int,
    blocks: Sequence[str],
    block_targets: int = 64,
):
    modified, receipt = shuffled_target_feature_blocks_h51(
        index=index,
        eval_parents=eval_parents,
        eval_parent_ids=eval_parent_ids,
        shift=shift,
        blocks=blocks,
    )
    out = []
    for start in range(0, len(eval_parent_ids), int(block_targets)):
        out.extend(_compile_block_r11(
            target_parent_ids=eval_parent_ids[start:start + int(block_targets)],
            index=modified,
            regime=regime,
            config=R11_VALIDATION_TEACHER_CONFIG,
        ))
    require(len(out) == len(eval_parent_ids), "H51_CONTROL_TARGET_COUNT_DRIFT")
    require(all(x.admission.admitted for x in out), f"H51_CONTROL_UNADMITTED:{shift}:{blocks}")
    return modified, out, receipt


def _group_equal_mean(rows: Sequence[tuple[str, float]]) -> float:
    by: dict[str, list[float]] = {}
    for gid, value in rows:
        by.setdefault(str(gid), []).append(float(value))
    require(bool(by), "H51_EMPTY_GROUP_METRIC")
    return float(np.mean([np.mean(v) for v in by.values()]))


def _action_key(direction: int, risk: float) -> str:
    side = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[int(direction)]
    return f"{side}_R{float(risk):.6f}"


def score_evidence_h51(*, evidence: Sequence[Any], index, eval_parents: Mapping[str, Any]) -> dict[str, Any]:
    levels = tuple(float(x) for x in H5_QUANTILES)
    level_rows: dict[float, list[tuple[str, float]]] = {q: [] for q in levels}
    action_rows: dict[str, list[tuple[str, float]]] = {}
    scenario_rows: dict[str, list[tuple[str, float]]] = {s: [] for s in H5_SCENARIOS}
    mse_rows: list[tuple[str, float]] = []
    c1090_rows: list[tuple[str, float]] = []
    c0595_rows: list[tuple[str, float]] = []
    overall_rows: list[tuple[str, float]] = []
    for ev in evidence:
        require(ev.admission.admitted, f"H51_SCORE_UNADMITTED:{ev.parent_id}")
        p = eval_parents[ev.parent_id]
        gid = str(p.dependence_group_id)
        scenario = str(p.scenario)
        row_idx = int(index.parent_row_by_id[ev.parent_id])
        utilities = np.asarray(index.utilities[row_idx], dtype=np.float64)
        require(len(ev.action_laws) == index.action_count, "H51_ACTION_LAW_COUNT_DRIFT")
        parent_losses = []
        parent_mse = []
        parent_c1090 = []
        parent_c0595 = []
        for a, law in enumerate(ev.action_laws):
            y = float(utilities[a])
            q = tuple(float(x) for x in law.quantiles)
            require(tuple(float(x) for x in law.quantile_levels) == levels, "H51_QUANTILE_LEVEL_DRIFT")
            per_level = [pinball_score_h5(y, (qh,), (tau,)) for qh, tau in zip(q, levels)]
            parent_losses.extend(per_level)
            for tau, loss in zip(levels, per_level):
                level_rows[tau].append((gid, loss))
            action_rows.setdefault(_action_key(law.direction, law.requested_risk), []).append((gid, float(np.mean(per_level))))
            parent_mse.append((y - float(law.mean_utility)) ** 2)
            qmap = dict(zip(levels, q))
            parent_c1090.append(float(qmap[0.10] <= y <= qmap[0.90]))
            parent_c0595.append(float(qmap[0.05] <= y <= qmap[0.95]))
        q_parent = float(np.mean(parent_losses))
        overall_rows.append((gid, q_parent))
        scenario_rows[scenario].append((gid, q_parent))
        mse_rows.append((gid, float(np.mean(parent_mse))))
        c1090_rows.append((gid, float(np.mean(parent_c1090))))
        c0595_rows.append((gid, float(np.mean(parent_c0595))))
    per_quantile = {f"Q{int(round(q*100)):02d}": _group_equal_mean(level_rows[q]) for q in levels}
    central = float(np.mean([per_quantile[f"Q{int(round(q*100)):02d}"] for q in CENTRAL_LEVELS]))
    tail = float(np.mean([per_quantile[f"Q{int(round(q*100)):02d}"] for q in TAIL_LEVELS]))
    result = {
        "qscore": _group_equal_mean(overall_rows),
        "central_qscore": central,
        "tail_qscore": tail,
        "per_quantile_qscore": per_quantile,
        "by_action_qscore": {k: _group_equal_mean(v) for k, v in sorted(action_rows.items())},
        "by_scenario_qscore": {k: _group_equal_mean(v) for k, v in scenario_rows.items()},
        "mean_utility_mse": _group_equal_mean(mse_rows),
        "coverage_10_90": _group_equal_mean(c1090_rows),
        "coverage_05_95": _group_equal_mean(c0595_rows),
        "future_dependence_groups": len({gid for gid, _ in overall_rows}),
        "parent_contexts": len(evidence),
    }
    h5 = score_teacher_evidence_h5(evidence=evidence, index=index, eval_parents=eval_parents)
    require(abs(float(result["qscore"]) - float(h5["qscore"])) <= 1e-15, "H51_H5_QSCORE_IDENTITY_DRIFT")
    return result


def reconstruct_support_h51(*, index, regime, parent_id: str, evidence, parents: Mapping[str, Any]) -> dict[str, Any]:
    row = int(index.parent_row_by_id[parent_id])
    target = (np.asarray(index.features[row], dtype=np.float64) - regime.mean) / regime.std
    g = int(regime.group_count)
    require(g > 0, "H51_EMPTY_SUPPORT_REGIME")
    rmax = int(regime.max_parents_per_group)
    target_norm2 = float(np.dot(target, target))
    dot = target @ regime.normalized_support_flat.T
    sq = (target_norm2 + regime.normalized_support_norm2 - 2.0 * dot) / float(index.feature_dim)
    sq = np.maximum(sq, 0.0)
    flat_dist = np.sqrt(sq)
    flat_dist[~regime.valid_support_flat] = np.inf
    parent_dist = flat_dist.reshape(g, rmax)
    nearest_slot = np.argmin(parent_dist, axis=1)
    nearest_distance = parent_dist[np.arange(g), nearest_slot]
    safe_parent_matrix = np.asarray(regime.support_parent_rows, dtype=np.int32).copy()
    first_valid = int(safe_parent_matrix[safe_parent_matrix >= 0][0])
    safe_parent_matrix[safe_parent_matrix < 0] = first_valid
    nearest_parent_row = safe_parent_matrix[np.arange(g), nearest_slot]
    order = np.lexsort((regime.dep_lex_rank, nearest_distance))
    k = min(int(R11_VALIDATION_TEACHER_CONFIG.k_dependence_groups), g)
    top = order[:k]
    top_dist = np.asarray(nearest_distance[top], dtype=np.float64)
    selected_rows = np.asarray(nearest_parent_row[top], dtype=np.int32)
    selected_dep_ids = tuple(str(regime.dep_ids[int(i)]) for i in top)
    w = np.exp(-0.5 * (top_dist / float(R11_VALIDATION_TEACHER_CONFIG.distance_temperature)) ** 2) + 1e-12
    w /= np.sum(w)
    eff = float(1.0 / np.sum(w * w))
    law = evidence.action_laws[0]
    require(canonical_hash(selected_dep_ids) == str(law.support_dependence_group_hash), f"H51_SUPPORT_HASH_DRIFT:{parent_id}")
    require(abs(float(top_dist[0]) - float(law.nearest_distance)) <= 1e-12, f"H51_NEAREST_DISTANCE_DRIFT:{parent_id}")
    require(abs(float(top_dist[-1]) - float(law.max_distance_used)) <= 1e-12, f"H51_MAX_DISTANCE_DRIFT:{parent_id}")
    require(abs(eff - float(law.effective_dependence_n)) <= 1e-10, f"H51_EFFECTIVE_N_DRIFT:{parent_id}")
    target_parent = parents[parent_id]
    support_parent_ids = [str(index.parent_ids[int(r)]) for r in selected_rows]
    support_parents = [parents[x] for x in support_parent_ids]
    same_symbol = float(np.mean([str(x.symbol) == str(target_parent.symbol) for x in support_parents]))
    same_scenario = float(np.mean([str(x.scenario) == str(target_parent.scenario) for x in support_parents]))
    age_hours = np.asarray([
        (int(target_parent.decision_time_ms) - int(x.decision_time_ms)) / HOUR_MS for x in support_parents
    ], dtype=np.float64)
    selected_z = (np.asarray(index.features[selected_rows], dtype=np.float64) - regime.mean) / regime.std
    delta2 = (selected_z - target[None, :]) ** 2
    weighted_mass = np.sum(w[:, None] * delta2, axis=0)
    total_mass = float(np.sum(weighted_mass))
    if total_mass <= 0.0:
        mass = {name: 0.0 for name in FEATURE_BLOCKS}
    else:
        mass = {
            name: float(np.sum(weighted_mass[lo:hi]) / total_mass)
            for name, (lo, hi) in FEATURE_BLOCKS.items()
        }
    return {
        "selected_dep_ids": selected_dep_ids,
        "nearest_distance": float(top_dist[0]),
        "max_distance_used": float(top_dist[-1]),
        "effective_dependence_n": eff,
        "same_symbol_fraction": same_symbol,
        "same_account_scenario_fraction": same_scenario,
        "support_age_hours_median": float(np.median(age_hours)),
        "distance_mass_fraction": mass,
    }


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return 1.0 if not u else float(len(sa & sb) / len(u))


def support_geometry_summary_h51(
    *,
    aligned_index,
    regime,
    aligned_evidence: Sequence[Any],
    full_control_indices: Mapping[int, Any],
    full_control_evidence: Mapping[int, Sequence[Any]],
    eval_parents: Mapping[str, Any],
    parents: Mapping[str, Any],
) -> dict[str, Any]:
    aev = {x.parent_id: x for x in aligned_evidence}
    sev = {int(s): {x.parent_id: x for x in rows} for s, rows in full_control_evidence.items()}
    target_rows = []
    for pid in sorted(aev):
        gid = str(eval_parents[pid].dependence_group_id)
        a = reconstruct_support_h51(
            index=aligned_index, regime=regime, parent_id=pid, evidence=aev[pid], parents=parents
        )
        sh = []
        for shift in H5_SHIFTS:
            sh.append(reconstruct_support_h51(
                index=full_control_indices[int(shift)], regime=regime, parent_id=pid,
                evidence=sev[int(shift)][pid], parents=parents,
            ))
        target_rows.append({
            "gid": gid,
            "aligned_nearest_distance": a["nearest_distance"],
            "median_shuffle_nearest_distance": float(statistics.median(x["nearest_distance"] for x in sh)),
            "same_symbol_fraction": a["same_symbol_fraction"],
            "same_account_scenario_fraction": a["same_account_scenario_fraction"],
            "support_age_hours_median": a["support_age_hours_median"],
            "support_jaccard_median": float(statistics.median(_jaccard(a["selected_dep_ids"], x["selected_dep_ids"]) for x in sh)),
            "operator_mass": a["distance_mass_fraction"]["OPERATOR48"],
            "medium_mass": a["distance_mass_fraction"]["MEDIUM48"],
            "account_mass": a["distance_mass_fraction"]["ACCOUNT6"],
        })
    def gm(key: str) -> float:
        return _group_equal_mean([(x["gid"], float(x[key])) for x in target_rows])
    return {
        "aligned_nearest_distance_group_equal_mean": gm("aligned_nearest_distance"),
        "median_shuffle_nearest_distance_group_equal_mean": gm("median_shuffle_nearest_distance"),
        "aligned_selected_support_same_symbol_fraction": gm("same_symbol_fraction"),
        "aligned_selected_support_same_account_scenario_fraction": gm("same_account_scenario_fraction"),
        "aligned_selected_support_age_hours_median_group_equal_mean": gm("support_age_hours_median"),
        "aligned_vs_shuffle_selected_support_jaccard_median_group_equal_mean": gm("support_jaccard_median"),
        "operator48_standardized_distance_mass_fraction": gm("operator_mass"),
        "medium48_standardized_distance_mass_fraction": gm("medium_mass"),
        "account6_standardized_distance_mass_fraction": gm("account_mass"),
        "exact_support_reconstruction_guards": "PASS",
    }


def run_fold_h51(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: Sequence[Any],
    block_targets: int = 64,
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    train_parents, eval_parents, parents, train_samples, eval_samples, samples = _fold_material_h5(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    eval_parent_ids = tuple(sorted(eval_parents))
    index, regime, aligned, execution = _support_and_aligned_h5(
        parents=parents, samples=samples, eval_parent_ids=eval_parent_ids, block_targets=block_targets
    )
    aligned_score = score_evidence_h51(evidence=aligned, index=index, eval_parents=eval_parents)
    controls: dict[str, dict[int, Any]] = {}
    receipts: dict[str, dict[int, Any]] = {}
    full_indices: dict[int, Any] = {}
    full_evidence: dict[int, Sequence[Any]] = {}
    for family, blocks in CONTROL_FAMILIES.items():
        controls[family] = {}
        receipts[family] = {}
        for shift in H5_SHIFTS:
            modified, ev, receipt = compile_block_control_h51(
                index=index,
                regime=regime,
                eval_parents=eval_parents,
                eval_parent_ids=eval_parent_ids,
                shift=int(shift),
                blocks=blocks,
                block_targets=block_targets,
            )
            controls[family][int(shift)] = score_evidence_h51(
                evidence=ev, index=index, eval_parents=eval_parents
            )
            receipts[family][int(shift)] = receipt
            if family == "FULL_STATE":
                full_indices[int(shift)] = modified
                full_evidence[int(shift)] = ev
    medians = {}
    for family in CONTROL_FAMILIES:
        medians[family] = {
            "qscore": float(statistics.median(float(controls[family][s]["qscore"]) for s in H5_SHIFTS)),
            "central_qscore": float(statistics.median(float(controls[family][s]["central_qscore"]) for s in H5_SHIFTS)),
            "tail_qscore": float(statistics.median(float(controls[family][s]["tail_qscore"]) for s in H5_SHIFTS)),
        }
    geometry = support_geometry_summary_h51(
        aligned_index=index,
        regime=regime,
        aligned_evidence=aligned,
        full_control_indices=full_indices,
        full_control_evidence=full_evidence,
        eval_parents=eval_parents,
        parents=parents,
    )
    return {
        "fold": fold,
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": int(regime.group_count),
        "eval_dependence_groups": int(aligned_score["future_dependence_groups"]),
        "train_clock_count": int(fold_spec["train_clock_count"]),
        "eval_clock_count": int(fold_spec["eval_clock_count"]),
        "train_to_eval_gap_hours": float(fold_spec["train_to_eval_gap_hours"]),
        "teacher_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "aligned": aligned_score,
        "controls": controls,
        "control_medians": medians,
        "support_geometry": geometry,
        "teacher_execution": execution,
        "control_receipts": receipts,
        "exact_h5_full_state_control_identity_guard": "PASS",
    }


def adjudicate_h51(fold_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = sorted(fold_results, key=lambda x: int(x["fold"]))
    require(tuple(int(x["fold"]) for x in rows) == (1, 2, 3, 4, 5), "H51_FOLD_SET_DRIFT")
    late = [rows[3], rows[4]]
    def harmful_both(family: str) -> bool:
        return all(float(x["aligned"]["qscore"]) > float(x["control_medians"][family]["qscore"]) for x in late)
    market_harm = harmful_both("MARKET96_ONLY")
    account_harm = harmful_both("ACCOUNT6_ONLY")
    full_harm = harmful_both("FULL_STATE")
    operator_harm = harmful_both("OPERATOR48_ONLY")
    medium_harm = harmful_both("MEDIUM48_ONLY")
    tail_primary = all(
        float(x["aligned"]["central_qscore"]) < float(x["control_medians"]["FULL_STATE"]["central_qscore"])
        and float(x["aligned"]["tail_qscore"]) > float(x["control_medians"]["FULL_STATE"]["tail_qscore"])
        for x in late
    )
    if market_harm and not account_harm:
        axis = "MARKET_STATE_TRANSPORT_FAILURE"
    elif account_harm and not market_harm:
        axis = "ACCOUNT_STATE_TRANSPORT_FAILURE"
    elif market_harm and account_harm:
        axis = "BROAD_STATE_GEOMETRY_FAILURE"
    elif (not market_harm) and (not account_harm) and full_harm:
        axis = "JOINT_MARKET_ACCOUNT_INTERACTION_FAILURE"
    else:
        axis = "H5_1_FAILURE_MODE_UNRESOLVED"
    market_sub = "NOT_APPLICABLE"
    if axis in {"MARKET_STATE_TRANSPORT_FAILURE", "BROAD_STATE_GEOMETRY_FAILURE"}:
        if operator_harm and not medium_harm:
            market_sub = "OPERATOR48_PRIMARY"
        elif medium_harm and not operator_harm:
            market_sub = "MEDIUM48_PRIMARY"
        elif operator_harm and medium_harm:
            market_sub = "OPERATOR_MEDIUM_JOINT"
        else:
            market_sub = "MARKET_SUBLOCALIZATION_UNRESOLVED"
    per_fold = []
    for x in rows:
        per_fold.append({
            "fold": int(x["fold"]),
            "aligned_qscore": float(x["aligned"]["qscore"]),
            "aligned_central_qscore": float(x["aligned"]["central_qscore"]),
            "aligned_tail_qscore": float(x["aligned"]["tail_qscore"]),
            "control_medians": x["control_medians"],
            "support_geometry": x["support_geometry"],
        })
    return {
        "schema": "CB16_R11_M_SERIES_H5_1_TEACHER_FAILURE_LOCALIZATION_ADJUDICATION_R0_V1",
        "primary_failure_axis": axis,
        "tail_calibration_primary": bool(tail_primary),
        "market_sublocalization": market_sub,
        "late_full_state_shuffle_beats_aligned_both_folds": bool(full_harm),
        "late_market96_shuffle_beats_aligned_both_folds": bool(market_harm),
        "late_account6_shuffle_beats_aligned_both_folds": bool(account_harm),
        "late_operator48_shuffle_beats_aligned_both_folds": bool(operator_harm),
        "late_medium48_shuffle_beats_aligned_both_folds": bool(medium_harm),
        "specific_failure_axis_localized": axis != "H5_1_FAILURE_MODE_UNRESOLVED",
        "conclusion": (
            f"H5_1_LOCALIZED__{axis}__TAIL_CALIBRATION_PRIMARY={str(bool(tail_primary)).upper()}__MARKET_SUB={market_sub}"
            if axis != "H5_1_FAILURE_MODE_UNRESOLVED"
            else "H5_1_FAILURE_MODE_UNRESOLVED__DO_NOT_TUNE_TEACHER_OR_STUDENT"
        ),
        "per_fold": per_fold,
        "market_information_verdict": False,
        "canonical_change_authorized": False,
        "teacher_change_authorized": False,
        "student_change_authorized": False,
        "promotion_authorized": False,
        "r7_evaluation_authorized": False,
        "final_opening_authorized": False,
        "new_direction_loss_authorized": False,
    }


__all__ = [
    "H51_RUNTIME", "FEATURE_BLOCKS", "CONTROL_FAMILIES", "CENTRAL_LEVELS", "TAIL_LEVELS",
    "shuffled_target_feature_blocks_h51", "compile_block_control_h51", "score_evidence_h51",
    "reconstruct_support_h51", "support_geometry_summary_h51", "run_fold_h51", "adjudicate_h51",
]
