#!/usr/bin/env python3
from __future__ import annotations

"""CB16 R11 M-series M0 — policy-drift attribution on consumed R6 TRAIN-only support."""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import torch

from cb16_local_opt.m_series_m0_policy_drift_attribution import (
    M0_RUNTIME,
    adjudicate_m0,
    attribute_direction_migration_m0,
    teacher_direction_geometry_from_action_laws_m0,
)
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import policy_hash_r11
from scripts import r11_science_g0_reduced_teacher_target_information_audit_r6 as r6

SCHEMA = "CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_RESULT_V1"
M0_V2_PREREG_COMMIT = "54e1b2135acf97fdf34bcaf97e487b929d39fd17"
M0_V2_GATE_BLOB = "ef2edc3d4cdad1ba6e9e808f5ca024ece74bc1ff"
M0_V1_GATE_BLOB = "88b8664f6c8f0d8474ce73872986c6a8a33e3552"
R7_FREEZE_COMMIT = "62ad74a0c62ab616dfceaf70c6f7b30f7088c81f"
R6_EXECUTOR_BLOB = "72bede8dd09015c702ec26ead639a81f48efe38a"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"

# Reproduction guard only. These are already-observed R6 TRAIN-only aligned metrics and
# cannot create fresh status authority. M0 must reproduce the unchanged R6 update before
# attributing its per-row behavior.
R6_ALIGNED_EXPECTED = {
    1: {"loss": 1.082339882850647, "direction_loss": 0.918185830116272, "sizing_loss": 0.16415412724018097},
    2: {"loss": 1.2173532247543335, "direction_loss": 1.0563373565673828, "sizing_loss": 0.1610158234834671},
    3: {"loss": 1.203639030456543, "direction_loss": 1.0622090101242065, "sizing_loss": 0.14143000543117523},
    4: {"loss": 1.150591492652893, "direction_loss": 1.0317440032958984, "sizing_loss": 0.11884760111570358},
    5: {"loss": 1.1324816942214966, "direction_loss": 1.043397068977356, "sizing_loss": 0.08908464014530182},
}


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_repo_contract_m0() -> dict[str, Any]:
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", R7_FREEZE_COMMIT, "HEAD"]) == 0,
        "M0_NOT_DESCENDED_FROM_R7_FREEZE",
    )
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", M0_V2_PREREG_COMMIT, "HEAD"]) == 0,
        "M0_NOT_DESCENDED_FROM_V2_PREREGISTRATION",
    )
    exact = {
        "m0_gate_v2": (
            "authority/rearchitecture_r11/CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_GATE_V2.json",
            M0_V2_GATE_BLOB,
        ),
        "m0_gate_v1_preserved": (
            "authority/rearchitecture_r11/CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_GATE_V1.json",
            M0_V1_GATE_BLOB,
        ),
        "r6_executor": (
            "scripts/r11_science_g0_reduced_teacher_target_information_audit_r6.py",
            R6_EXECUTOR_BLOB,
        ),
    }
    observed: dict[str, str] = {}
    for name, (path, expected) in exact.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"M0_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    # Reuse R6's own full immutable-source verification. It pins semantic freeze,
    # Student/runtime, Teacher, R5 authority and the R6 preregistration.
    r6_contract = r6.verify_repo_contract()
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "r7_freeze_commit": R7_FREEZE_COMMIT,
        "m0_v2_preregistration_commit": M0_V2_PREREG_COMMIT,
        "m0_immutable_blobs": observed,
        "r6_immutable_contract": r6_contract,
    }


def _model_outputs(model: torch.nn.Module, prepared) -> dict[str, np.ndarray]:
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        out = model(prepared.operator48, prepared.medium48, prepared.account6)
        probs = out["direction_probs"].detach().cpu().numpy().astype(np.float64, copy=False)
        risk = out["requested_risk_raw"].detach().cpu().numpy().astype(np.float64, copy=False)
    if was_training:
        model.train()
    return {"direction_probs": probs, "requested_risk_raw": risk}


def _ordered_eval_evidence(eval_evidence, parent_ids) -> list[Any]:
    admitted = [e for e in eval_evidence if bool(e.admission.admitted)]
    by_parent = {str(e.parent_id): e for e in admitted}
    require(len(by_parent) == len(admitted), "M0_DUPLICATE_EVAL_TEACHER_PARENT")
    require(set(by_parent) == set(str(x) for x in parent_ids), "M0_EVAL_TEACHER_PARENT_SET_DRIFT")
    return [by_parent[str(pid)] for pid in parent_ids]


def _write_rows(path: Path, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False))
            fh.write("\n")
    os.replace(tmp, path)
    return {
        "file": path.name,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _assert_r6_aligned_reproduction(fold: int, receipt: Mapping[str, Any]) -> None:
    expected = R6_ALIGNED_EXPECTED[int(fold)]
    got = receipt["validation_after"]
    for key, value in expected.items():
        require(
            abs(float(got[key]) - float(value)) <= 1e-7,
            f"M0_R6_ALIGNED_REPRODUCTION_DRIFT:FOLD={fold}:{key}:{got[key]}:{value}",
        )


def run_fold_m0(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: list[Any],
    g0_root: Path,
    work_root: Path,
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
    train_audit = r6.r4.validate_teacher_targets_r11(campaign.train)
    eval_audit = r6.r4.validate_teacher_targets_r11(campaign.validation)

    g0 = r6.r4.r1.load_bootstrap_model(g0_root, device)
    g0_hash = policy_hash_r11(g0)
    evaluator = IntegratedTrainingRuntimeR11(device=device).evaluation_runtime
    g0_eval = evaluator.evaluate(g0, campaign.validation, use_cache=True)
    g0_out = _model_outputs(g0, campaign.validation)
    g0_dir = np.argmax(g0_out["direction_probs"], axis=1).astype(np.int64)

    ordered_teacher = _ordered_eval_evidence(eval_evidence, campaign.validation.parent_ids)
    teacher_probs = campaign.validation.direction_target_probs.detach().cpu().numpy().astype(np.float64, copy=False)
    geometry = teacher_direction_geometry_from_action_laws_m0(
        evidence=ordered_teacher,
        g0_direction=g0_dir,
        reduced_teacher_probs=teacher_probs,
    )

    snapshot = r6.sha256_obj(
        {
            "schema": "CB16_R11_M0_ALIGNED_REPRODUCTION_SNAPSHOT_V1",
            "fold": fold,
            "g0_policy_hash": g0_hash,
            "train_evidence_hash": campaign.train.evidence_hash,
            "eval_evidence_hash": campaign.validation.evidence_hash,
            "m0_gate_blob": M0_V2_GATE_BLOB,
            "non_status_driving": True,
        }
    )
    challenger = copy.deepcopy(g0)
    trainer = IntegratedTrainingRuntimeR11(device=device)
    receipt = trainer.train_challenger(
        model=challenger,
        campaign=campaign,
        generation=0,
        snapshot_hash=snapshot,
        receipt_dir=work_root / f"fold_{fold}" / "aligned_reproduction",
    )
    r6.validate_training_receipt(receipt, g0_eval)
    _assert_r6_aligned_reproduction(fold, receipt)
    challenger_out = _model_outputs(challenger, campaign.validation)

    rows, summary = attribute_direction_migration_m0(
        parent_ids=campaign.validation.parent_ids,
        dependence_group_ids=campaign.validation.dependence_group_ids,
        teacher_probs=teacher_probs,
        teacher_best_mean_by_direction=geometry["best_mean_by_direction"],
        teacher_best_risk_by_direction=geometry["best_risk_by_direction"],
        teacher_direction_advantage_over_g0=geometry["direction_advantage_over_g0"],
        g0_probs=g0_out["direction_probs"],
        challenger_probs=challenger_out["direction_probs"],
        g0_requested_risk_raw=g0_out["requested_risk_raw"],
        challenger_requested_risk_raw=challenger_out["requested_risk_raw"],
    )
    summary["fold"] = fold
    row_receipt = _write_rows(work_root / f"fold_{fold}" / "M0_POLICY_DRIFT_ROWS.jsonl", rows)

    require(policy_hash_r11(g0) == g0_hash, f"M0_G0_MUTATED:FOLD={fold}")
    out = {
        "fold": fold,
        "outer_support": dict(fold_spec),
        "teacher": teacher_receipt,
        "prepared_target_audit": {"train": train_audit, "eval": eval_audit},
        "g0_baseline": g0_eval,
        "aligned_training_receipt": r6._compact_training_receipt(receipt),
        "policy_drift_attribution": summary,
        "row_evidence": row_receipt,
        "r6_aligned_numeric_reproduction": True,
    }
    del challenger, trainer, g0, campaign
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _support_not_ready(
    *, output: Path, reason: str, repo: Mapping[str, Any], source: Mapping[str, Any], completed: list[Any]
) -> int:
    result = {
        "schema": SCHEMA,
        "status": "M0_SUPPORT_NOT_READY",
        "runtime": M0_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": dict(repo),
        "source": dict(source),
        "completed_folds_before_stop": completed,
        "primary_adjudication_performed": False,
        "reason": reason,
        "semantic_guards": {
            "r7_candidate_evaluated": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "r5_purge_support_reopened": False,
            "legacy_validation_rows_used": False,
            "market_information_verdict_reopened": False,
        },
        "next_legal_step": "STOP_AT_M0_SUPPORT_LIMIT__DO_NOT_REPARTITION_OR_TUNE",
    }
    atomic_json(output, result)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--r104-root", type=Path, default=Path(os.environ.get("CB16_R104_ROOT", "/cb16/runtime/r104")))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    g0_root = args.g0_root.resolve()
    package_root = args.package_root.resolve()
    r104_root = args.r104_root.resolve()
    work_root = args.work_root.resolve()
    output = args.output.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    require(work_root != g0_root and g0_root not in work_root.parents, "M0_WORK_ROOT_OVERLAPS_G0")

    repo = verify_repo_contract_m0()
    runtime = r6.r4.r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"M0_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    _lineage, g0_identity = r6.r4.r1.verify_g0_authority(g0_root)
    frozen_before = r6.frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])
    all_train_parents, all_train_samples, source = r6.load_train_only_support(r104_root)
    folds = r6.build_outer_folds_r6(all_train_parents)

    completed: list[dict[str, Any]] = []
    try:
        for fold_spec in folds:
            completed.append(
                run_fold_m0(
                    fold_spec=fold_spec,
                    all_train_parents=all_train_parents,
                    all_train_samples=all_train_samples,
                    g0_root=g0_root,
                    work_root=work_root,
                    device=args.device,
                )
            )
    except RuntimeError as exc:
        if str(exc).startswith("R11_R6_SUPPORT_NOT_READY"):
            return _support_not_ready(
                output=output,
                reason=str(exc),
                repo=repo,
                source=source,
                completed=completed,
            )
        raise

    require(len(completed) == 5, "M0_INCOMPLETE_FOLD_EXECUTION")
    fold_summaries = [x["policy_drift_attribution"] for x in completed]
    summary = adjudicate_m0(fold_summaries)

    frozen_after = r6.frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "M0_FROZEN_PACKAGE_MUTATED")
    _, g0_after = r6.r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "M0_G0_AUTHORITY_MUTATED")

    result = {
        "schema": SCHEMA,
        "status": "M0_POLICY_DRIFT_ATTRIBUTION_PASS",
        "runtime": M0_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": repo,
        "runtime_identity": runtime,
        "source": source,
        "protocol": {
            "gate": "CB16_R11_M_SERIES_M0_POLICY_DRIFT_ATTRIBUTION_GATE_V2",
            "all_ten_symbols_train_only": True,
            "outer_folds": [1, 2, 3, 4, 5],
            "aligned_r6_update_only": True,
            "whole_group_shuffles_executed_in_m0": False,
            "teacher_recompiled_with_new_semantics": False,
            "rich_eval_teacher_action_laws_used_for_exact_direction_advantage": True,
            "softmax_log_ratio_advantage_used": False,
            "continuous_g0_requested_risk_projection_used": False,
            "independent_quantile_subtraction_used": False,
            "r7_candidate_evaluated": False,
            "non_status_driving": True,
        },
        "fold_results": completed,
        "m0_summary": summary,
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "scientific_market_verdict_created": False,
            "profitability_or_alpha_claimed": False,
            "r7_candidate_evaluated": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "r5_purge_support_reopened": False,
            "legacy_validation_rows_used": False,
            "raw_market_payload_used": False,
            "frozen_package_unchanged": True,
            "g0_authority_unchanged": True,
        },
        "next_legal_step": "M0_ADJUDICATION_AND_RESEARCH_INTERPRETATION_ONLY__M1_REQUIRES_SEPARATE_PREREGISTRATION",
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "conclusion": summary["conclusion"],
                "direct_teacher_correction_majority_fold_count": summary[
                    "direct_teacher_correction_majority_fold_count"
                ],
                "non_teacher_attributable_majority_fold_count": summary[
                    "non_teacher_attributable_majority_fold_count"
                ],
                "per_fold": summary["per_fold"],
                "next_legal_step": result["next_legal_step"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
