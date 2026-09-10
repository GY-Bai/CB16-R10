from __future__ import annotations

"""Batch-exact support-geometry adapter for H5.1.

The frozen Teacher computes target/support distances through a target-block GEMM in
`_compile_block_r11`.  Recomputing one target at a time can alter floating-point
reduction order enough to change a near-tie lexicographic neighbor selection.  This
adapter reconstructs support using the exact same target block shape, formulas, and
ordering as the frozen Teacher.  It does not change any Teacher law, control, score,
or H5.1 adjudication rule.
"""

import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r6 import canonical_hash
from .r11_teacher_authority_candidate import R11_VALIDATION_TEACHER_CONFIG
from .teacher_failure_localization_h51 import (
    CONTROL_FAMILIES,
    FEATURE_BLOCKS,
    H5_SCENARIOS,
    H5_SHIFTS,
    HOUR_MS,
    _fold_material_h5,
    _group_equal_mean,
    _jaccard,
    _support_and_aligned_h5,
    adjudicate_h51,
    compile_block_control_h51,
    require,
    score_evidence_h51,
)

H51_BATCH_GEOMETRY_RUNTIME = "CB16_R11_H5_1_BATCH_EXACT_SUPPORT_GEOMETRY_V1"


def reconstruct_support_block_h51(
    *,
    index,
    regime,
    parent_ids: Sequence[str],
    evidence_by_parent: Mapping[str, Any],
    parents: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reproduce `_compile_block_r11` support selection for one exact target block."""
    ids = tuple(str(x) for x in parent_ids)
    require(bool(ids), "H51_BATCH_EMPTY_TARGET_BLOCK")
    target_rows = np.asarray([index.parent_row_by_id[p] for p in ids], dtype=np.int32)
    target = (index.features[target_rows] - regime.mean) / regime.std
    b = len(target_rows)
    g = int(regime.group_count)
    rmax = int(regime.max_parents_per_group)
    require(g > 0, "H51_BATCH_EMPTY_SUPPORT_REGIME")

    # Byte-for-byte algorithmic shape of teacher_vectorized_r11._compile_block_r11.
    target_norm2 = np.einsum("ij,ij->i", target, target, optimize=True)
    dot = target @ regime.normalized_support_flat.T
    sq = (
        target_norm2[:, None]
        + regime.normalized_support_norm2[None, :]
        - 2.0 * dot
    ) / float(index.feature_dim)
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
    top_local_dep = order[:, :k]
    top_distance = np.take_along_axis(nearest_distance, top_local_dep, axis=1)
    selected_parent_rows = np.take_along_axis(nearest_parent_row, top_local_dep, axis=1)
    weights = np.exp(
        -0.5 * (top_distance / float(R11_VALIDATION_TEACHER_CONFIG.distance_temperature)) ** 2
    ) + 1e-12
    weights /= np.sum(weights, axis=1, keepdims=True)
    effective_n = 1.0 / np.sum(weights * weights, axis=1)

    out: dict[str, dict[str, Any]] = {}
    for i, pid in enumerate(ids):
        ev = evidence_by_parent[pid]
        law = ev.action_laws[0]
        selected_dep_ids = [regime.dep_ids[int(x)] for x in top_local_dep[i]]
        support_hash = canonical_hash(selected_dep_ids)
        require(support_hash == str(law.support_dependence_group_hash), f"H51_SUPPORT_HASH_DRIFT:{pid}")
        require(abs(float(top_distance[i, 0]) - float(law.nearest_distance)) <= 1e-12, f"H51_NEAREST_DISTANCE_DRIFT:{pid}")
        require(abs(float(top_distance[i, -1]) - float(law.max_distance_used)) <= 1e-12, f"H51_MAX_DISTANCE_DRIFT:{pid}")
        require(abs(float(effective_n[i]) - float(law.effective_dependence_n)) <= 1e-10, f"H51_EFFECTIVE_N_DRIFT:{pid}")

        selected_rows = np.asarray(selected_parent_rows[i], dtype=np.int32)
        target_parent = parents[pid]
        support_parent_ids = [str(index.parent_ids[int(r)]) for r in selected_rows]
        support_parents = [parents[x] for x in support_parent_ids]
        same_symbol = float(np.mean([str(x.symbol) == str(target_parent.symbol) for x in support_parents]))
        same_scenario = float(np.mean([str(x.scenario) == str(target_parent.scenario) for x in support_parents]))
        age_hours = np.asarray([
            (int(target_parent.decision_time_ms) - int(x.decision_time_ms)) / HOUR_MS
            for x in support_parents
        ], dtype=np.float64)

        selected_z = (np.asarray(index.features[selected_rows], dtype=np.float64) - regime.mean) / regime.std
        delta2 = (selected_z - target[i][None, :]) ** 2
        weighted_mass = np.sum(weights[i][:, None] * delta2, axis=0)
        total_mass = float(np.sum(weighted_mass))
        if total_mass <= 0.0:
            mass = {name: 0.0 for name in FEATURE_BLOCKS}
        else:
            mass = {
                name: float(np.sum(weighted_mass[lo:hi]) / total_mass)
                for name, (lo, hi) in FEATURE_BLOCKS.items()
            }
        out[pid] = {
            "selected_dep_ids": tuple(str(x) for x in selected_dep_ids),
            "nearest_distance": float(top_distance[i, 0]),
            "max_distance_used": float(top_distance[i, -1]),
            "effective_dependence_n": float(effective_n[i]),
            "same_symbol_fraction": same_symbol,
            "same_account_scenario_fraction": same_scenario,
            "support_age_hours_median": float(np.median(age_hours)),
            "distance_mass_fraction": mass,
        }
    return out


def reconstruct_support_all_h51(
    *,
    index,
    regime,
    eval_parent_ids: Sequence[str],
    evidence: Sequence[Any],
    parents: Mapping[str, Any],
    block_targets: int,
) -> dict[str, dict[str, Any]]:
    by_ev = {x.parent_id: x for x in evidence}
    ids = tuple(str(x) for x in eval_parent_ids)
    require(set(by_ev) == set(ids), "H51_BATCH_EVIDENCE_TARGET_SET_DRIFT")
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), int(block_targets)):
        block = ids[start:start + int(block_targets)]
        rows = reconstruct_support_block_h51(
            index=index,
            regime=regime,
            parent_ids=block,
            evidence_by_parent=by_ev,
            parents=parents,
        )
        require(not (set(out) & set(rows)), "H51_BATCH_DUPLICATE_TARGET")
        out.update(rows)
    require(set(out) == set(ids), "H51_BATCH_RECONSTRUCTION_TARGET_SET_DRIFT")
    return out


def support_geometry_summary_batch_h51(
    *,
    aligned_index,
    regime,
    aligned_evidence: Sequence[Any],
    full_control_indices: Mapping[int, Any],
    full_control_evidence: Mapping[int, Sequence[Any]],
    eval_parent_ids: Sequence[str],
    eval_parents: Mapping[str, Any],
    parents: Mapping[str, Any],
    block_targets: int,
) -> dict[str, Any]:
    aligned = reconstruct_support_all_h51(
        index=aligned_index,
        regime=regime,
        eval_parent_ids=eval_parent_ids,
        evidence=aligned_evidence,
        parents=parents,
        block_targets=block_targets,
    )
    shuffled = {
        int(shift): reconstruct_support_all_h51(
            index=full_control_indices[int(shift)],
            regime=regime,
            eval_parent_ids=eval_parent_ids,
            evidence=full_control_evidence[int(shift)],
            parents=parents,
            block_targets=block_targets,
        )
        for shift in H5_SHIFTS
    }

    target_rows = []
    for pid in sorted(aligned):
        gid = str(eval_parents[pid].dependence_group_id)
        a = aligned[pid]
        sh = [shuffled[int(shift)][pid] for shift in H5_SHIFTS]
        target_rows.append({
            "gid": gid,
            "aligned_nearest_distance": a["nearest_distance"],
            "median_shuffle_nearest_distance": float(statistics.median(x["nearest_distance"] for x in sh)),
            "same_symbol_fraction": a["same_symbol_fraction"],
            "same_account_scenario_fraction": a["same_account_scenario_fraction"],
            "support_age_hours_median": a["support_age_hours_median"],
            "support_jaccard_median": float(statistics.median(
                _jaccard(a["selected_dep_ids"], x["selected_dep_ids"]) for x in sh
            )),
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
        "reconstruction_runtime": H51_BATCH_GEOMETRY_RUNTIME,
        "teacher_batch_shape_reproduced": True,
        "teacher_support_law_changed": False,
    }


def run_fold_h51_batch_geometry(
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

    geometry = support_geometry_summary_batch_h51(
        aligned_index=index,
        regime=regime,
        aligned_evidence=aligned,
        full_control_indices=full_indices,
        full_control_evidence=full_evidence,
        eval_parent_ids=eval_parent_ids,
        eval_parents=eval_parents,
        parents=parents,
        block_targets=block_targets,
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


__all__ = [
    "H51_BATCH_GEOMETRY_RUNTIME",
    "reconstruct_support_block_h51",
    "reconstruct_support_all_h51",
    "support_geometry_summary_batch_h51",
    "run_fold_h51_batch_geometry",
    "adjudicate_h51",
]
