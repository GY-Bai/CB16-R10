#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R6 — train-only reduced Teacher target information audit."""

import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    shuffle_reduced_targets_by_future_group_r41,
)
from cb16_local_opt.independent_purge_alignment_replication_r5 import (
    R5_SYMBOLS,
    compile_validation_targets_only_r5,
)
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_evidence_cache import load_teacher_samples
from cb16_local_opt.reduced_teacher_target_information_audit_r6 import (
    R6_FOLDS,
    R6_RUNTIME,
    R6_SHIFTS,
    R6_SYMBOLS,
    analytic_global_marginal_baseline_r6,
    build_outer_folds_r6,
    summarize_r6,
    target_dispersion_r6,
)
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    PreparedCampaignR11,
    policy_hash_r11,
)
from scripts import r11_science_g0_canonical_historical_learning_baseline_r4 as r4

SCHEMA = "CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_RESULT_V1"
PREREG_COMMIT = "de46ec7c436151a95a98ce0da1b841fc12daf510"
R6_GATE_BLOB = "1d1254fb03cc00283423426e07ae240a196c627e"
R5_ADJUDICATION_BLOB = "3f0305bea4edd77d091531987a36009038de4e40"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


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


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def verify_repo_contract() -> dict[str, Any]:
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0,
        "R11_R6_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    exact = {
        "r6_gate": (
            "authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_GATE_V1.json",
            R6_GATE_BLOB,
        ),
        "r5_adjudication": (
            "authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_ADJUDICATION_V1.json",
            R5_ADJUDICATION_BLOB,
        ),
        "semantic_freeze": (
            "authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json",
            r4.SEMANTIC_FREEZE_BLOB,
        ),
        "typed_brain": (
            "cb16_local_opt/typed_central_brain_r10.py",
            r4.TYPED_CENTRAL_BRAIN_BLOB,
        ),
        "training_runtime": (
            "cb16_local_opt/training_runtime_r11.py",
            r4.TRAINING_RUNTIME_BLOB,
        ),
        "training_integration": (
            "cb16_local_opt/training_integration_r11.py",
            r4.TRAINING_INTEGRATION_BLOB,
        ),
        "teacher_runtime": (
            "cb16_local_opt/teacher_runtime_r11.py",
            r4.TEACHER_RUNTIME_BLOB,
        ),
        "probabilistic_teacher": (
            "cb16_local_opt/probabilistic_teacher_r6.py",
            r4.PROBABILISTIC_TEACHER_BLOB,
        ),
        "legacy_evidence_cache": (
            "cb16_local_opt/r102_evidence_cache.py",
            "91fb73565ec60f66bf4232957f9b9b2f88cc3cfe",
        ),
    }
    observed = {}
    for name, (path, expected) in exact.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"R11_R6_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "immutable_blobs": observed,
    }


def load_train_only_support(r104_root: Path) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    manifest_path = r104_root / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    require(manifest_path.is_file(), "R11_R6_R104_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(tuple(manifest.get("symbols", ())) == tuple(R6_SYMBOLS), "R11_R6_R104_SYMBOL_SET_DRIFT")
    require(int(manifest.get("stride_hours", -1)) == 256, "R11_R6_R104_STRIDE_DRIFT")
    require(manifest.get("final_holdout_2025_09_accessed") is False, "R11_R6_FINAL_HOLDOUT_GUARD_DRIFT")
    cache_root = r104_root / "evidence_cache"
    parent_path = cache_root / Path(str(manifest["parents_file"])).name
    branch_path = cache_root / Path(str(manifest["branches_file"])).name
    require(parent_path.is_file() and branch_path.is_file(), "R11_R6_R104_EVIDENCE_FILES_MISSING")
    require(r4.r1.sha256_file(parent_path) == str(manifest["parents_sha256"]), "R11_R6_PARENT_FILE_SHA_DRIFT")
    require(r4.r1.sha256_file(branch_path) == str(manifest["branches_sha256"]), "R11_R6_BRANCH_FILE_SHA_DRIFT")

    all_parents, all_samples = load_teacher_samples(parent_path, branch_path)
    complete_parent_ids = {x.parent_id for x in all_samples}
    train_parents = {
        pid: p
        for pid, p in all_parents.items()
        if p.split == "TRAIN" and pid in complete_parent_ids and p.symbol in R6_SYMBOLS
    }
    train_ids = set(train_parents)
    train_samples = [x for x in all_samples if x.parent_id in train_ids]
    require(train_parents and train_samples, "R11_R6_EMPTY_TRAIN_ONLY_SUPPORT")
    require(all(p.split == "TRAIN" for p in train_parents.values()), "R11_R6_LEGACY_VALIDATION_ROW_RETAINED")
    require({p.symbol for p in train_parents.values()} == set(R6_SYMBOLS), "R11_R6_TRAIN_SYMBOL_SET_DRIFT")
    require(len(train_parents) == 9714, f"R11_R6_TRAIN_PARENT_COUNT_DRIFT:{len(train_parents)}")
    require(len(train_samples) == 87426, f"R11_R6_TRAIN_BRANCH_COUNT_DRIFT:{len(train_samples)}")
    require(
        len({p.dependence_group_id for p in train_parents.values()}) == 1619,
        "R11_R6_TRAIN_DEPENDENCE_GROUP_COUNT_DRIFT",
    )
    return train_parents, train_samples, {
        "manifest_path": str(manifest_path),
        "parents_file": str(parent_path),
        "branches_file": str(branch_path),
        "train_parent_contexts": len(train_parents),
        "train_branch_samples": len(train_samples),
        "train_dependence_groups": len({p.dependence_group_id for p in train_parents.values()}),
        "legacy_validation_rows_used": False,
        "r5_purge_support_used": False,
        "raw_market_payload_used": False,
    }


def _count_admitted_by_group(evidence) -> tuple[int, dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    reasons: set[str] = set()
    admitted = 0
    for e in evidence:
        if e.admission.admitted:
            admitted += 1
            counts[e.target_dependence_group_id] = counts.get(e.target_dependence_group_id, 0) + 1
        else:
            reasons.update(e.admission.reasons)
    return admitted, counts, sorted(reasons)


def _compact_training_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": receipt["schema"],
        "generation": int(receipt["generation"]),
        "generation_seed": int(receipt["generation_seed"]),
        "optimizer": receipt["optimizer"],
        "epochs": int(receipt["epochs"]),
        "batch_size": int(receipt["batch_size"]),
        "optimizer_steps": int(receipt["optimizer_steps"]),
        "lr": float(receipt["lr"]),
        "weight_decay": float(receipt["weight_decay"]),
        "parameter_l2_delta": float(receipt["parameter_l2_delta"]),
        "gradient_owner_set_last_step": list(receipt["gradient_owner_set_last_step"]),
        "gradient_group_norms_last_step": dict(receipt["gradient_group_norms_last_step"]),
        "update_group_norms": dict(receipt["update_group_norms"]),
        "challenger_semantic_sha256": receipt["challenger_semantic_sha256"],
        "train_evidence_hash": receipt["train_evidence_hash"],
        "validation_evidence_hash": receipt["validation_evidence_hash"],
        "validation_before": receipt["validation_before"],
        "validation_after": receipt["validation_after"],
    }


def validate_training_receipt(receipt: Mapping[str, Any], expected_before: Mapping[str, Any]) -> None:
    require(int(receipt["generation_seed"]) == 24680, "R11_R6_GENERATION_SEED_DRIFT")
    require(int(receipt["epochs"]) == 12 and int(receipt["batch_size"]) == 512, "R11_R6_TRAIN_RULE_DRIFT")
    require(float(receipt["lr"]) == 3e-4 and float(receipt["weight_decay"]) == 1e-4, "R11_R6_OPTIMIZER_RULE_DRIFT")
    require(receipt["optimizer"] == "AdamW_FP32" and receipt["amp"] is False, "R11_R6_NUMERIC_RULE_DRIFT")
    require(
        frozenset(receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11,
        "R11_R6_GRADIENT_OWNER_SET_DRIFT",
    )
    require(
        all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["gradient_group_norms_last_step"].values()),
        "R11_R6_NONPOSITIVE_GRADIENT_OWNER",
    )
    require(
        all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["update_group_norms"].values()),
        "R11_R6_NONPOSITIVE_UPDATE_OWNER",
    )
    for k in ("loss", "direction_loss", "sizing_loss"):
        require(
            abs(float(receipt["validation_before"][k]) - float(expected_before[k])) <= 1e-7,
            f"R11_R6_VALIDATION_BASELINE_DRIFT:{k}",
        )


def build_fold_evidence(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: list[Any],
) -> tuple[dict[str, Any], list[Any], list[Any], dict[str, Any]]:
    train_clock_set = {int(x) for x in fold_spec["train_clocks"]}
    eval_clock_set = {int(x) for x in fold_spec["eval_clocks"]}
    train_parents = {
        pid: p for pid, p in all_train_parents.items() if int(p.decision_time_ms) in train_clock_set
    }
    eval_parents_original = {
        pid: p for pid, p in all_train_parents.items() if int(p.decision_time_ms) in eval_clock_set
    }
    eval_parents = {pid: replace(p, split="VALIDATION") for pid, p in eval_parents_original.items()}
    require(train_parents and eval_parents, f"R11_R6_SUPPORT_NOT_READY_EMPTY_FOLD:{fold_spec['fold']}")
    require({p.symbol for p in train_parents.values()} == set(R6_SYMBOLS), f"R11_R6_SUPPORT_NOT_READY_TRAIN_SYMBOLS:{fold_spec['fold']}")
    require({p.symbol for p in eval_parents.values()} == set(R6_SYMBOLS), f"R11_R6_SUPPORT_NOT_READY_EVAL_SYMBOLS:{fold_spec['fold']}")

    train_ids = set(train_parents)
    eval_ids = set(eval_parents)
    train_samples = [x for x in all_train_samples if x.parent_id in train_ids]
    eval_samples = [x for x in all_train_samples if x.parent_id in eval_ids]
    require(len(train_samples) == len(train_parents) * 9, f"R11_R6_SUPPORT_NOT_READY_TRAIN_BRANCH_GRID:{fold_spec['fold']}")
    require(len(eval_samples) == len(eval_parents) * 9, f"R11_R6_SUPPORT_NOT_READY_EVAL_BRANCH_GRID:{fold_spec['fold']}")

    train_evidence, empty_validation, train_stats = r4.compile_teacher_evidence_r11(
        samples=train_samples,
        parents=train_parents,
        train_config=r4.R11_TRAIN_TEACHER_CONFIG,
        val_config=r4.R11_VALIDATION_TEACHER_CONFIG,
        workers=4,
        block_targets=32,
    )
    require(len(empty_validation) == 0, f"R11_R6_TRAIN_TEACHER_EMITTED_VALIDATION:{fold_spec['fold']}")

    combined_parents = dict(train_parents)
    combined_parents.update(eval_parents)
    combined_samples = list(train_samples) + list(eval_samples)
    eval_evidence, eval_stats = compile_validation_targets_only_r5(
        samples=combined_samples,
        parents=combined_parents,
        target_parent_ids=sorted(eval_parents),
        block_targets=32,
    )

    train_admitted, train_counts, train_reasons = _count_admitted_by_group(train_evidence)
    eval_admitted, eval_counts, eval_reasons = _count_admitted_by_group(eval_evidence)
    if train_admitted != len(train_parents) or any(v != 6 for v in train_counts.values()):
        raise RuntimeError(
            f"R11_R6_SUPPORT_NOT_READY_TRAIN_TEACHER_ADMISSION:FOLD={fold_spec['fold']}:"
            f"{train_admitted}/{len(train_parents)}:{','.join(train_reasons)}"
        )
    if eval_admitted != len(eval_parents) or any(v != 6 for v in eval_counts.values()):
        raise RuntimeError(
            f"R11_R6_SUPPORT_NOT_READY_EVAL_TEACHER_ADMISSION:FOLD={fold_spec['fold']}:"
            f"{eval_admitted}/{len(eval_parents)}:{','.join(eval_reasons)}"
        )
    require(len(train_counts) >= 32, f"R11_R6_SUPPORT_NOT_READY_TRAIN_GROUPS_AFTER_ADMISSION:{fold_spec['fold']}")
    require(len(eval_counts) >= 8, f"R11_R6_SUPPORT_NOT_READY_EVAL_GROUPS_AFTER_ADMISSION:{fold_spec['fold']}")

    return combined_parents, train_evidence, eval_evidence, {
        "train_parent_contexts": len(train_parents),
        "eval_parent_contexts": len(eval_parents),
        "train_dependence_groups": len(train_counts),
        "eval_dependence_groups": len(eval_counts),
        "train_teacher_stats": str(train_stats),
        "eval_teacher_execution": eval_stats,
        "train_teacher_protocol_hash": r4.R11_TRAIN_TEACHER_CONFIG.content_hash,
        "eval_teacher_protocol_hash": r4.R11_VALIDATION_TEACHER_CONFIG.content_hash,
        "eval_outcomes_absent_from_train_teacher_index": True,
        "future_clock_blocks_after_eval_ignored": True,
    }


def run_fold(
    *,
    fold_spec: Mapping[str, Any],
    all_train_parents: Mapping[str, Any],
    all_train_samples: list[Any],
    g0_root: Path,
    work_root: Path,
    device: str,
) -> dict[str, Any]:
    fold = int(fold_spec["fold"])
    parents, train_evidence, eval_evidence, teacher_receipt = build_fold_evidence(
        fold_spec=fold_spec,
        all_train_parents=all_train_parents,
        all_train_samples=all_train_samples,
    )
    campaign = r4.prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=eval_evidence,
        parents=parents,
        device=device,
    )
    train_audit = r4.validate_teacher_targets_r11(campaign.train)
    eval_audit = r4.validate_teacher_targets_r11(campaign.validation)
    require(
        int(train_audit["independent_dependence_groups"]) == int(fold_spec["train_group_count"]),
        f"R11_R6_PREPARED_TRAIN_GROUP_DRIFT:{fold}",
    )
    require(
        int(eval_audit["independent_dependence_groups"]) == int(fold_spec["eval_group_count"]),
        f"R11_R6_PREPARED_EVAL_GROUP_DRIFT:{fold}",
    )
    marginal = analytic_global_marginal_baseline_r6(campaign.train, campaign.validation)
    dispersion = {
        "train": target_dispersion_r6(campaign.train),
        "eval": target_dispersion_r6(campaign.validation),
    }

    g0 = r4.r1.load_bootstrap_model(g0_root, device)
    g0_hash = policy_hash_r11(g0)
    evaluator = IntegratedTrainingRuntimeR11(device=device).evaluation_runtime
    g0_eval = evaluator.evaluate(g0, campaign.validation, use_cache=True)
    base_snapshot = sha256_obj({
        "schema": "CB16_R11_R6_FOLD_SNAPSHOT_V1",
        "fold": fold,
        "g0_policy_hash": g0_hash,
        "train_evidence_hash": campaign.train.evidence_hash,
        "eval_evidence_hash": campaign.validation.evidence_hash,
        "non_status_driving": True,
    })

    arms: dict[str, Any] = {}
    aligned = copy.deepcopy(g0)
    aligned_trainer = IntegratedTrainingRuntimeR11(device=device)
    aligned_receipt = aligned_trainer.train_challenger(
        model=aligned,
        campaign=campaign,
        generation=0,
        snapshot_hash=base_snapshot,
        receipt_dir=work_root / f"fold_{fold}" / "aligned",
    )
    validate_training_receipt(aligned_receipt, g0_eval)
    arms["ALIGNED"] = _compact_training_receipt(aligned_receipt)
    del aligned, aligned_trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for shift in R6_SHIFTS:
        shuffled_train, shuffle_receipt = shuffle_reduced_targets_by_future_group_r41(
            campaign.train, shift=int(shift)
        )
        require(shuffle_receipt["target_multiset_exactly_preserved"] is True, f"R11_R6_SHUFFLE_MULTISET_DRIFT:{fold}:{shift}")
        require(shuffle_receipt["scenario_identity_preserved"] is True, f"R11_R6_SHUFFLE_SCENARIO_DRIFT:{fold}:{shift}")
        arm_campaign = PreparedCampaignR11(train=shuffled_train, validation=campaign.validation)
        model = r4.r1.load_bootstrap_model(g0_root, device)
        require(policy_hash_r11(model) == g0_hash, f"R11_R6_ARM_INITIALIZATION_DRIFT:{fold}:{shift}")
        trainer = IntegratedTrainingRuntimeR11(device=device)
        snapshot = sha256_obj({
            "schema": "CB16_R11_R6_SHUFFLE_SNAPSHOT_V1",
            "fold": fold,
            "shift": int(shift),
            "base_snapshot": base_snapshot,
            "shuffled_train_evidence_hash": shuffled_train.evidence_hash,
        })
        receipt = trainer.train_challenger(
            model=model,
            campaign=arm_campaign,
            generation=0,
            snapshot_hash=snapshot,
            receipt_dir=work_root / f"fold_{fold}" / f"shuffle_{int(shift):02d}",
        )
        validate_training_receipt(receipt, g0_eval)
        compact = _compact_training_receipt(receipt)
        compact["shuffle_receipt"] = {
            "shift": int(shift),
            "future_group_count": int(shuffle_receipt["future_group_count"]),
            "changed_target_rows": int(shuffle_receipt["changed_target_rows"]),
            "target_multiset_exactly_preserved": bool(shuffle_receipt["target_multiset_exactly_preserved"]),
            "scenario_identity_preserved": bool(shuffle_receipt["scenario_identity_preserved"]),
            "input_features_byte_identical": bool(shuffle_receipt["input_features_byte_identical"]),
            "group_weights_byte_identical": bool(shuffle_receipt["group_weights_byte_identical"]),
            "fixed_point_future_groups": int(shuffle_receipt["fixed_point_future_groups"]),
        }
        arms[f"SHUFFLE_{int(shift)}"] = compact
        del model, trainer, arm_campaign, shuffled_train
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    require(policy_hash_r11(g0) == g0_hash, f"R11_R6_G0_MUTATED:{fold}")
    return {
        "fold": fold,
        "outer_support": dict(fold_spec),
        "teacher": teacher_receipt,
        "prepared_target_audit": {"train": train_audit, "eval": eval_audit},
        "target_dispersion": dispersion,
        "g0_baseline": g0_eval,
        "marginal_baseline": marginal,
        "arms": arms,
    }


def support_not_ready(
    *,
    output: Path,
    reason: str,
    repo: Mapping[str, Any],
    source: Mapping[str, Any],
    completed_folds: list[Mapping[str, Any]],
    guards: Mapping[str, Any],
) -> int:
    result = {
        "schema": SCHEMA,
        "status": "R11_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_SUPPORT_NOT_READY",
        "runtime": R6_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": dict(repo),
        "source": dict(source),
        "support_readiness": {"ready": False, "reason": reason},
        "completed_folds_before_stop": completed_folds,
        "primary_adjudication_performed": False,
        "semantic_guards": dict(guards),
        "next_legal_step": "STOP_AT_R6_SUPPORT_LIMIT__DO_NOT_REPARTITION_OR_TUNE",
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "reason": reason}, indent=2, sort_keys=True))
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
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R6_WORK_ROOT_OVERLAPS_G0")
    require(tuple(R6_SYMBOLS) == tuple(R5_SYMBOLS), "R11_R6_SYMBOL_REGISTRY_DRIFT")
    require(tuple(R6_FOLDS) == (1, 2, 3, 4, 5), "R11_R6_FOLD_PROTOCOL_DRIFT")
    require(tuple(R6_SHIFTS) == (1, 7, 13, 23, 31), "R11_R6_SHIFT_PROTOCOL_DRIFT")

    repo = verify_repo_contract()
    runtime = r4.r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R6_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    _lineage, g0_identity = r4.r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])
    all_train_parents, all_train_samples, source = load_train_only_support(r104_root)
    folds = build_outer_folds_r6(all_train_parents)

    completed: list[dict[str, Any]] = []
    try:
        for fold_spec in folds:
            completed.append(
                run_fold(
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
            _, g0_after = r4.r1.verify_g0_authority(g0_root)
            guards = {
                "canonical_generation_advanced": False,
                "champion_promoted": False,
                "production_cutover": False,
                "market_information_verdict_reopened": False,
                "final_holdout_payload_opened": False,
                "fresh_market_data_downloaded": False,
                "r5_purge_support_reopened": False,
                "legacy_validation_rows_used": False,
                "raw_market_payload_used": False,
                "frozen_package_unchanged": frozen_authority_hashes(package_root) == frozen_before,
                "g0_authority_unchanged": g0_after["authority_file_hashes"] == g0_before,
            }
            return support_not_ready(
                output=output,
                reason=str(exc),
                repo=repo,
                source=source,
                completed_folds=completed,
                guards=guards,
            )
        raise

    summary = summarize_r6(completed)
    require(len(completed) == 5, "R11_R6_INCOMPLETE_FOLD_EXECUTION")
    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R6_FROZEN_PACKAGE_MUTATED")
    _, g0_after = r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R6_G0_AUTHORITY_MUTATED")
    result = {
        "schema": SCHEMA,
        "status": "R11_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_PASS",
        "runtime": R6_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": repo,
        "runtime_identity": runtime,
        "source": source,
        "protocol": {
            "all_ten_symbols_train_only": True,
            "outer_partition": "NP_ARRAY_SPLIT_INTO_6_CONTIGUOUS_CLOCK_BLOCKS__FOLDS_1_TO_5_EVAL",
            "outer_unit": "UNIQUE_DECISION_CLOCK_TIMESTAMP__ALL_SYMBOLS_AT_SAME_CLOCK_STAY_TOGETHER",
            "train_teacher": "R2.4_R11_TRAIN_BLOCKED_CROSSFIT__OUTER_TRAIN_ONLY",
            "eval_teacher": "R2.4_R11_VALIDATION_PREQUENTIAL__OUTER_TRAIN_SUPPORT_ONLY",
            "student": "CANONICAL_TIER1_189052__12_EPOCH_ADAMW_FP32",
            "pre_registered_shifts": list(R6_SHIFTS),
            "marginal_baseline": "ANALYTIC_GLOBAL_TRAIN_TARGET_MARGINAL__NO_STATE_FEATURES",
            "r5_purge_support_used": False,
            "legacy_validation_rows_used": False,
            "raw_market_payload_used": False,
            "non_status_driving": True,
        },
        "fold_results": completed,
        "audit_summary": summary,
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "r5_purge_support_reopened": False,
            "legacy_validation_rows_used": False,
            "raw_market_payload_used": False,
            "frozen_package_unchanged": True,
            "g0_authority_unchanged": True,
        },
        "next_legal_step": "R11_SCIENCE_G0_R6_ADJUDICATION_ONLY__NO_PRODUCTION_OR_MARKET_VERDICT_CHANGE",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "conclusion": summary["conclusion"],
        "total_lower_than_shuffle_median_fold_count": summary["total_lower_than_shuffle_median_fold_count"],
        "direction_lower_than_shuffle_median_fold_count": summary["direction_lower_than_shuffle_median_fold_count"],
        "direction_lower_than_marginal_fold_count": summary["direction_lower_than_marginal_fold_count"],
        "sizing_lower_than_shuffle_median_fold_count": summary["sizing_lower_than_shuffle_median_fold_count"],
        "aligned_lower_direction_pair_count_of_25": summary["aligned_lower_direction_pair_count_of_25"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
