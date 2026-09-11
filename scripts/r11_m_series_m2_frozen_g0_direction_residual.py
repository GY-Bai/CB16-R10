#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 M-series M2 — frozen-G0 Direction residual experiment."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    shuffle_reduced_targets_by_future_group_r41,
)
from cb16_local_opt.m_series_m2_frozen_g0_direction_residual import (
    M2_OBJECTIVES,
    M2_RUNTIME,
    M2_SHIFTS,
    adjudicate_m2,
    best_direction_means_from_evidence_m2,
    build_objective_target_m2,
    cache_frozen_g0_direction_m2,
    evaluate_direction_residual_m2,
    prepare_epoch_permutations_m2,
    rotate_rich_direction_means_m2,
    train_direction_residual_m2,
)
from cb16_local_opt.training_runtime_r11 import policy_hash_r11
from scripts import r11_science_g0_reduced_teacher_target_information_audit_r6 as r6

SCHEMA = "CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_RESULT_V1"
M2_PREREG_COMMIT = "27a06659a09e75f3fcf49351f174ab115bd3c564"
M2_GATE_BLOB = "8c80273998f33821514c75c774e21649e08dfd43"
M1_ADJUDICATION_BLOB = "e814c0cd3f3f9556b9477107d2f0cb00905220fb"
M0_ADJUDICATION_BLOB = "54174f86fee2ae0da5c1813af44fa84fe82962db"
R7_FREEZE_COMMIT = "62ad74a0c62ab616dfceaf70c6f7b30f7088c81f"
R6_EXECUTOR_BLOB = "72bede8dd09015c702ec26ead639a81f48efe38a"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def verify_repo_contract_m2() -> dict[str, Any]:
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", R7_FREEZE_COMMIT, "HEAD"]) == 0,
        "M2_NOT_DESCENDED_FROM_R7_FREEZE",
    )
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", M2_PREREG_COMMIT, "HEAD"]) == 0,
        "M2_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    exact = {
        "m2_gate": (
            "authority/rearchitecture_r11/CB16_R11_M_SERIES_M2_FROZEN_G0_DIRECTION_RESIDUAL_GATE_V1.json",
            M2_GATE_BLOB,
        ),
        "m1_adjudication": (
            "authority/rearchitecture_r11/CB16_R11_M_SERIES_M1_DIRECTION_RELATIVE_SUFFICIENCY_ADJUDICATION_V1.json",
            M1_ADJUDICATION_BLOB,
        ),
        "m0_adjudication": (
            "authority/rearchitecture_r11/CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_ADJUDICATION_V1.json",
            M0_ADJUDICATION_BLOB,
        ),
        "r6_executor": (
            "scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py",
            R6_EXECUTOR_BLOB,
        ),
    }
    observed = {}
    for name, (path, expected) in exact.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"M2_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "m2_preregistration_commit": M2_PREREG_COMMIT,
        "immutable_blobs": observed,
        "r6_immutable_contract": r6.verify_repo_contract(),
    }


def _teacher_best_matches_reduced(prepared, rich_means: np.ndarray, label: str) -> None:
    reduced = prepared.direction_target_probs.detach().cpu().numpy().argmax(axis=1)
    rich = np.asarray(rich_means).argmax(axis=1)
    require(np.array_equal(reduced, rich), f"M2_RICH_REDUCED_BEST_MISMATCH:{label}")


def _build_train_surfaces(campaign, train_rich_means: np.ndarray) -> dict[str, dict[str, Any]]:
    surfaces: dict[str, dict[str, Any]] = {
        "ALIGNED": {
            "prepared": campaign.train,
            "rich_means": train_rich_means,
            "control_receipt": {
                "kind": "ALIGNED",
                "whole_future_group_shuffle": False,
                "teacher_surface_multiset_preserved": True,
            },
        }
    }
    for shift in M2_SHIFTS:
        shuffled, shuffle_receipt = shuffle_reduced_targets_by_future_group_r41(
            campaign.train, shift=int(shift)
        )
        rich_rotated, rich_receipt = rotate_rich_direction_means_m2(
            rich_means=train_rich_means,
            prepared=campaign.train,
            shuffled_prepared=shuffled,
            shuffle_receipt=shuffle_receipt,
        )
        surfaces[f"SHUFFLE_{int(shift)}"] = {
            "prepared": shuffled,
            "rich_means": rich_rotated,
            "control_receipt": {
                "kind": "WHOLE_FUTURE_GROUP_TARGET_ROTATION",
                "shift": int(shift),
                "target_multiset_exactly_preserved": bool(
                    shuffle_receipt["target_multiset_exactly_preserved"]
                ),
                "scenario_identity_preserved": bool(
                    shuffle_receipt["scenario_identity_preserved"]
                ),
                "input_features_byte_identical": bool(
                    shuffle_receipt["input_features_byte_identical"]
                ),
                "group_weights_byte_identical": bool(
                    shuffle_receipt["group_weights_byte_identical"]
                ),
                "changed_target_rows": int(shuffle_receipt["changed_target_rows"]),
                "fixed_point_future_groups": int(
                    shuffle_receipt["fixed_point_future_groups"]
                ),
                "rich_surface_rotation": rich_receipt,
            },
        }
    return surfaces


def _arm_name(objective: str, control: str) -> str:
    prefix = "ABS_CE" if objective == "ABS_CE" else "SELECTIVE"
    return f"{prefix}_{control}"


def run_fold_m2(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: list[Any],
    g0_root: Path,
    device: str,
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    parents, train_evidence, eval_evidence, teacher_receipt = r6.build_fold_evidence(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    campaign = r6.r4.prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=eval_evidence,
        parents=parents,
        device=device,
    )
    train_rich = best_direction_means_from_evidence_m2(
        train_evidence, campaign.train.parent_ids
    )
    eval_rich = best_direction_means_from_evidence_m2(
        eval_evidence, campaign.validation.parent_ids
    )
    _teacher_best_matches_reduced(campaign.train, train_rich, f"TRAIN_FOLD_{fold}")
    _teacher_best_matches_reduced(campaign.validation, eval_rich, f"EVAL_FOLD_{fold}")

    g0 = r6.r4.r1.load_bootstrap_model(g0_root, device)
    g0_hash_before = policy_hash_r11(g0)
    train_cache = cache_frozen_g0_direction_m2(g0, campaign.train)
    eval_cache = cache_frozen_g0_direction_m2(g0, campaign.validation)
    require(policy_hash_r11(g0) == g0_hash_before, f"M2_G0_MUTATED_BY_CACHE:FOLD={fold}")
    require(
        all(p.grad is None and not p.requires_grad for p in g0.parameters()),
        f"M2_G0_GRADIENT_OWNERSHIP_DRIFT:FOLD={fold}",
    )

    permutations = prepare_epoch_permutations_m2(campaign.train.rows, device=device)
    surfaces = _build_train_surfaces(campaign, train_rich)
    require(
        set(surfaces) == {"ALIGNED", *{f"SHUFFLE_{s}" for s in M2_SHIFTS}},
        f"M2_CONTROL_SURFACE_SET:FOLD={fold}",
    )

    arms: dict[str, Any] = {}
    for objective in M2_OBJECTIVES:
        for control_name in ("ALIGNED", *[f"SHUFFLE_{s}" for s in M2_SHIFTS]):
            surface = surfaces[control_name]
            prepared_control = surface["prepared"]
            target, target_receipt = build_objective_target_m2(
                objective=objective,
                teacher_probs=prepared_control.direction_target_probs,
                g0_probs=train_cache.base_probs,
            )
            residual, training_receipt = train_direction_residual_m2(
                cache=train_cache,
                target_probs=target,
                group_weight=campaign.train.group_weight,
                permutations=permutations,
                device=device,
            )
            evaluation = evaluate_direction_residual_m2(
                residual=residual,
                cache=eval_cache,
                aligned_teacher_probs=campaign.validation.direction_target_probs,
                aligned_best_direction_means=eval_rich,
                group_weight=campaign.validation.group_weight,
            )
            arm_name = _arm_name(objective, control_name)
            arms[arm_name] = {
                "objective": objective,
                "control": control_name,
                "target": target_receipt,
                "training": training_receipt,
                "evaluation": evaluation,
                "control_receipt": surface["control_receipt"],
                "g0_policy_hash": g0_hash_before,
                "g0_gradient_parameters": 0,
                "sizing_gradient_parameters": 0,
            }
            del residual, target
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_arms = {
        "ABS_CE_ALIGNED",
        "SELECTIVE_ALIGNED",
        *{f"ABS_CE_SHUFFLE_{s}" for s in M2_SHIFTS},
        *{f"SELECTIVE_SHUFFLE_{s}" for s in M2_SHIFTS},
    }
    require(set(arms) == expected_arms, f"M2_ARM_SET_INCOMPLETE:FOLD={fold}")
    require(policy_hash_r11(g0) == g0_hash_before, f"M2_G0_MUTATED_AFTER_ARMS:FOLD={fold}")
    require(
        all(p.grad is None and not p.requires_grad for p in g0.parameters()),
        f"M2_G0_GRADIENT_TAINT_AFTER_ARMS:FOLD={fold}",
    )
    return {
        "fold": fold,
        "outer_support": dict(fold_spec),
        "teacher": teacher_receipt,
        "train_rows": int(campaign.train.rows),
        "eval_rows": int(campaign.validation.rows),
        "train_dependence_groups": int(len(set(campaign.train.dependence_group_ids))),
        "eval_dependence_groups": int(len(set(campaign.validation.dependence_group_ids))),
        "g0_policy_hash": g0_hash_before,
        "residual_parameter_count": 16_643,
        "arms": arms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--g0-root",
        type=Path,
        default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")),
    )
    ap.add_argument(
        "--package-root",
        type=Path,
        default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")),
    )
    ap.add_argument(
        "--r104-root",
        type=Path,
        default=Path(os.environ.get("CB16_R104_ROOT", "/cb16/runtime/r104")),
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    g0_root = args.g0_root.resolve()
    package_root = args.package_root.resolve()
    r104_root = args.r104_root.resolve()
    output = args.output.resolve()

    repo = verify_repo_contract_m2()
    runtime = r6.r4.r1.runtime_identity(args.device)
    require(
        runtime["python_series"] == [3, 10],
        f"M2_PYTHON_SERIES_DRIFT:{runtime['python_series']}",
    )
    _lineage, g0_identity = r6.r4.r1.verify_g0_authority(g0_root)
    frozen_before = r6.frozen_authority_hashes(package_root)
    g0_authority_before = dict(g0_identity["authority_file_hashes"])
    all_train_parents, all_train_samples, source = r6.load_train_only_support(r104_root)
    folds = r6.build_outer_folds_r6(all_train_parents)

    completed = []
    for fold_spec in folds:
        completed.append(
            run_fold_m2(
                fold_spec=fold_spec,
                all_train_parents=all_train_parents,
                all_train_samples=all_train_samples,
                g0_root=g0_root,
                device=args.device,
            )
        )
    require(len(completed) == 5, "M2_INCOMPLETE_FOLDS")
    summary = adjudicate_m2(completed)

    require(
        r6.frozen_authority_hashes(package_root) == frozen_before,
        "M2_FROZEN_PACKAGE_MUTATED",
    )
    _, g0_after = r6.r4.r1.verify_g0_authority(g0_root)
    require(
        g0_after["authority_file_hashes"] == g0_authority_before,
        "M2_G0_AUTHORITY_MUTATED",
    )

    result = {
        "schema": SCHEMA,
        "status": "M2_FROZEN_G0_DIRECTION_RESIDUAL_EXECUTION_COMPLETE",
        "runtime": M2_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": repo,
        "runtime_identity": runtime,
        "source": source,
        "protocol": {
            "support": "EXACT_R6_CONSUMED_TRAIN_ONLY_NESTED_FOLDS",
            "all_ten_symbols": True,
            "g0_parameters_frozen": 189_052,
            "residual_parameters_trainable": 16_643,
            "residual_architecture": "Linear256x64_SiLU_Linear64x3_ZERO_OUTPUT_INIT",
            "objectives": list(M2_OBJECTIVES),
            "controls": ["ALIGNED", *[f"SHUFFLE_{s}" for s in M2_SHIFTS]],
            "same_minibatch_permutations_across_matched_arms": True,
            "sizing_in_gradient_graph": False,
            "r7_candidate_evaluated": False,
            "final_holdout_opened": False,
            "fresh_market_data": False,
            "non_status_driving": True,
        },
        "fold_results": completed,
        "m2_summary": summary,
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "profitability_or_alpha_claimed": False,
            "r7_candidate_evaluated": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "r5_purge_support_reopened": False,
            "legacy_validation_rows_used": False,
            "raw_market_payload_used": False,
            "g0_trainable_parameters": 0,
            "sizing_gradient_parameters": 0,
            "teacher_future_physics_gradient_parameters": 0,
            "frozen_package_unchanged": True,
            "g0_authority_unchanged": True,
        },
        "next_legal_step": (
            "M2_ADJUDICATION_THEN_SEPARATE_M3_DIRECTION_CONDITIONED_SIZING_DESIGN"
            if summary["selective_alignment_pass"]
            else "M2_FAILURE_ADJUDICATION_BEFORE_ANY_EXPANSION"
        ),
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selective_alignment_conclusion": summary[
                    "selective_alignment_conclusion"
                ],
                "preservation_conclusion": summary["preservation_conclusion"],
                "selective_positive_gain_fold_count": summary[
                    "selective_positive_gain_fold_count"
                ],
                "selective_gain_gt_shuffle_median_fold_count": summary[
                    "selective_gain_gt_shuffle_median_fold_count"
                ],
                "selective_move_to_teacher_gt_shuffle_median_fold_count": summary[
                    "selective_move_to_teacher_gt_shuffle_median_fold_count"
                ],
                "selective_preservation_better_than_abs_ce_fold_count": summary[
                    "selective_preservation_better_than_abs_ce_fold_count"
                ],
                "abs_ce_gain_gt_shuffle_median_fold_count": summary[
                    "abs_ce_gain_gt_shuffle_median_fold_count"
                ],
                "next_legal_step": result["next_legal_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
