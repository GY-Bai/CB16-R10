#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R7 — deterministic full-TRAIN multi-asset candidate materialization only."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from cb16_local_opt.full_train_multi_asset_candidate_r7 import (
    R7_CANDIDATE_SCHEMA,
    R7_RUNTIME,
    R7_SYMBOLS,
    candidate_identity_r7,
    state_dicts_exactly_equal_r7,
)
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    PreparedCampaignR11,
    policy_hash_r11,
)
from scripts import r11_science_g0_canonical_historical_learning_baseline_r4 as r4
from scripts import r11_science_g0_reduced_teacher_target_information_audit_r6 as r6

SCHEMA = "CB16_R11_SCIENCE_G0_FULL_TRAIN_MULTI_ASSET_CANONICAL_CANDIDATE_R7_RESULT_V1"
PREREG_COMMIT = "6bcea1ea631d1df077a70199d62acd83bbd35da1"
R7_GATE_BLOB = "40f2431e1dfea2f3198272b4f57147f97f3cd43e"
R6_ADJUDICATION_BLOB = "5a38e4d33e20c049507f3935b4de6b10c0a12912"
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


def verify_repo_contract() -> dict[str, Any]:
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0,
        "R11_R7_NOT_DESCENDED_FROM_PREREGISTRATION",
    )
    exact = {
        "r7_gate": (
            "authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_FULL_TRAIN_MULTI_ASSET_CANONICAL_CANDIDATE_R7_GATE_V1.json",
            R7_GATE_BLOB,
        ),
        "r6_adjudication": (
            "authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_REDUCED_TEACHER_TARGET_INFORMATION_AUDIT_R6_ADJUDICATION_V1.json",
            R6_ADJUDICATION_BLOB,
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
        require(got == expected, f"R11_R7_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {
        "execution_head": git("rev-parse", "HEAD"),
        "preregistration_commit": PREREG_COMMIT,
        "immutable_blobs": observed,
    }


def validate_training_receipt_r7(receipt: Mapping[str, Any], *, evidence_hash: str) -> None:
    require(receipt["optimizer"] == "AdamW_FP32", "R11_R7_OPTIMIZER_DRIFT")
    require(receipt["amp"] is False and receipt["dtype"] == "torch.float32", "R11_R7_NUMERIC_DRIFT")
    require(int(receipt["epochs"]) == 12, "R11_R7_EPOCH_DRIFT")
    require(int(receipt["batch_size"]) == 512, "R11_R7_BATCH_DRIFT")
    require(int(receipt["optimizer_steps"]) == 228, f"R11_R7_OPTIMIZER_STEP_DRIFT:{receipt['optimizer_steps']}")
    require(float(receipt["lr"]) == 3e-4, "R11_R7_LR_DRIFT")
    require(float(receipt["weight_decay"]) == 1e-4, "R11_R7_WEIGHT_DECAY_DRIFT")
    require(float(receipt["gradient_clip_max_norm"]) == 10.0, "R11_R7_CLIP_DRIFT")
    require(int(receipt["generation_seed"]) == 24680, "R11_R7_SEED_DRIFT")
    require(receipt["train_evidence_hash"] == evidence_hash, "R11_R7_TRAIN_EVIDENCE_HASH_DRIFT")
    require(receipt["validation_evidence_hash"] == evidence_hash, "R11_R7_TRAIN_FIT_SLOT_HASH_DRIFT")
    require(
        frozenset(receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11,
        "R11_R7_GRADIENT_OWNER_SET_DRIFT",
    )
    require(
        all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["gradient_group_norms_last_step"].values()),
        "R11_R7_NONPOSITIVE_GRADIENT_OWNER",
    )
    require(
        all(math.isfinite(float(v)) and float(v) > 0.0 for v in receipt["update_group_norms"].values()),
        "R11_R7_NONPOSITIVE_UPDATE_OWNER",
    )
    require(float(receipt["parameter_l2_delta"]) > 0.0, "R11_R7_ZERO_PARAMETER_DELTA")
    for phase in ("validation_before", "validation_after"):
        for key in ("loss", "direction_loss", "sizing_loss"):
            require(math.isfinite(float(receipt[phase][key])), f"R11_R7_NONFINITE_TRAIN_FIT:{phase}:{key}")


def compact_training_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": receipt["schema"],
        "optimizer": receipt["optimizer"],
        "amp": receipt["amp"],
        "dtype": receipt["dtype"],
        "epochs": int(receipt["epochs"]),
        "batch_size": int(receipt["batch_size"]),
        "optimizer_steps": int(receipt["optimizer_steps"]),
        "lr": float(receipt["lr"]),
        "weight_decay": float(receipt["weight_decay"]),
        "gradient_clip_max_norm": float(receipt["gradient_clip_max_norm"]),
        "generation_seed": int(receipt["generation_seed"]),
        "train_evidence_hash": receipt["train_evidence_hash"],
        "train_fit_slot_evidence_hash": receipt["validation_evidence_hash"],
        "parameter_l2_delta": float(receipt["parameter_l2_delta"]),
        "gradient_group_norms_last_step": dict(receipt["gradient_group_norms_last_step"]),
        "gradient_owner_set_last_step": list(receipt["gradient_owner_set_last_step"]),
        "update_group_norms": dict(receipt["update_group_norms"]),
        "candidate_policy_hash": receipt["challenger_semantic_sha256"],
        "train_fit_before": receipt["validation_before"],
        "train_fit_after": receipt["validation_after"],
        "train_fit_is_independent_evaluation": False,
        "train_fit_may_authorize_promotion": False,
    }


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
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R7_WORK_ROOT_OVERLAPS_G0")
    require(tuple(R7_SYMBOLS) == tuple(r6.R6_SYMBOLS), "R11_R7_SYMBOL_REGISTRY_DRIFT")

    repo = verify_repo_contract()
    runtime = r4.r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R7_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    _lineage, g0_identity = r4.r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    parents, samples, source = r6.load_train_only_support(r104_root)
    require(len(parents) == 9714, "R11_R7_TRAIN_PARENT_COUNT_DRIFT")
    require(len(samples) == 87426, "R11_R7_TRAIN_BRANCH_COUNT_DRIFT")
    require(len({p.dependence_group_id for p in parents.values()}) == 1619, "R11_R7_TRAIN_GROUP_COUNT_DRIFT")
    require({p.symbol for p in parents.values()} == set(R7_SYMBOLS), "R11_R7_TRAIN_SYMBOL_SET_DRIFT")
    require(all(p.split == "TRAIN" for p in parents.values()), "R11_R7_NONTRAIN_PARENT_RETAINED")

    train_evidence, empty_validation, teacher_stats = r4.compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=r4.R11_TRAIN_TEACHER_CONFIG,
        val_config=r4.R11_VALIDATION_TEACHER_CONFIG,
        workers=4,
        block_targets=32,
    )
    require(len(empty_validation) == 0, "R11_R7_TEACHER_EMITTED_VALIDATION_ROWS")
    train_summary = r4.evidence_summary(train_evidence)
    require(int(train_summary["total"]) == 9714, f"R11_R7_TEACHER_TARGET_COUNT_DRIFT:{train_summary}")
    require(int(train_summary["admitted"]) == 9714, f"R11_R7_TEACHER_ADMISSION_SHORTFALL:{train_summary}")
    require(int(train_summary["admitted_dependence_groups"]) == 1619, f"R11_R7_TEACHER_GROUP_SHORTFALL:{train_summary}")

    prepared = r4.prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=train_evidence,
        parents=parents,
        device=args.device,
    ).train
    prepared_audit = r4.validate_teacher_targets_r11(prepared)
    require(int(prepared.rows) == 9714, "R11_R7_PREPARED_ROW_COUNT_DRIFT")
    require(int(prepared_audit["independent_dependence_groups"]) == 1619, "R11_R7_PREPARED_GROUP_COUNT_DRIFT")
    campaign = PreparedCampaignR11(train=prepared, validation=prepared)

    g0_probe = r4.r1.load_bootstrap_model(g0_root, args.device)
    g0_policy_hash = policy_hash_r11(g0_probe)
    g0_state = {k: v.detach().cpu().clone() for k, v in g0_probe.state_dict().items()}
    del g0_probe

    snapshot_hash = r4.sha256_obj({
        "schema": "CB16_R11_R7_FULL_TRAIN_SNAPSHOT_V1",
        "g0_policy_hash": g0_policy_hash,
        "full_train_prepared_evidence_hash": prepared.evidence_hash,
        "train_teacher_protocol_hash": r4.R11_TRAIN_TEACHER_CONFIG.content_hash,
        "candidate_evaluation_state": "UNEVALUATED_OUTSIDE_TRAIN_FIT",
    })

    run_states: list[dict[str, torch.Tensor]] = []
    run_receipts: list[dict[str, Any]] = []
    for run_idx in (1, 2):
        model = r4.r1.load_bootstrap_model(g0_root, args.device)
        require(policy_hash_r11(model) == g0_policy_hash, f"R11_R7_G0_INITIALIZATION_DRIFT:{run_idx}")
        trainer = IntegratedTrainingRuntimeR11(device=args.device)
        receipt = trainer.train_challenger(
            model=model,
            campaign=campaign,
            generation=0,
            snapshot_hash=snapshot_hash,
            receipt_dir=work_root / f"reproduction_run_{run_idx}",
        )
        validate_training_receipt_r7(receipt, evidence_hash=prepared.evidence_hash)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        run_states.append(state)
        run_receipts.append(compact_training_receipt(receipt))
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    require(run_receipts[0]["candidate_policy_hash"] == run_receipts[1]["candidate_policy_hash"], "R11_R7_REPRODUCTION_POLICY_HASH_MISMATCH")
    require(state_dicts_exactly_equal_r7(run_states[0], run_states[1]), "R11_R7_REPRODUCTION_STATE_TENSOR_MISMATCH")
    require(not state_dicts_exactly_equal_r7(g0_state, run_states[0]), "R11_R7_CANDIDATE_IDENTICAL_TO_G0")
    candidate_policy_hash = str(run_receipts[0]["candidate_policy_hash"])

    manifest_path = Path(source["manifest_path"])
    parents_path = Path(source["parents_file"])
    branches_path = Path(source["branches_file"])
    source_hashes = {
        "manifest_sha256": r4.r1.sha256_file(manifest_path),
        "parents_sha256": r4.r1.sha256_file(parents_path),
        "branches_sha256": r4.r1.sha256_file(branches_path),
    }
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(source_hashes["parents_sha256"] == str(source_manifest["parents_sha256"]), "R11_R7_SOURCE_PARENT_HASH_DRIFT")
    require(source_hashes["branches_sha256"] == str(source_manifest["branches_sha256"]), "R11_R7_SOURCE_BRANCH_HASH_DRIFT")

    candidate_identity = candidate_identity_r7(
        g0_policy_hash=g0_policy_hash,
        full_train_evidence_hash=prepared.evidence_hash,
        train_teacher_protocol_hash=r4.R11_TRAIN_TEACHER_CONFIG.content_hash,
        candidate_policy_hash=candidate_policy_hash,
        source_manifest_sha256=source_hashes["manifest_sha256"],
        source_parents_sha256=source_hashes["parents_sha256"],
        source_branches_sha256=source_hashes["branches_sha256"],
        execution_head=repo["execution_head"],
    )

    candidate_path = work_root / "R11_R7_FROZEN_UNEVALUATED_CANDIDATE.pt"
    tmp = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    torch.save(
        {
            "schema": R7_CANDIDATE_SCHEMA,
            "candidate_identity": candidate_identity,
            "state_dict": run_states[0],
            "evaluation_state": "UNEVALUATED_OUTSIDE_TRAIN_FIT",
            "promotion_state": "NOT_AUTHORIZED",
        },
        tmp,
    )
    os.replace(tmp, candidate_path)
    checkpoint_sha256 = r4.r1.sha256_file(candidate_path)

    # Re-open saved candidate strictly as a materialization integrity check, not evaluation.
    saved = torch.load(candidate_path, map_location="cpu", weights_only=True)
    require(saved["schema"] == R7_CANDIDATE_SCHEMA, "R11_R7_CANDIDATE_CHECKPOINT_SCHEMA_DRIFT")
    require(saved["candidate_identity"] == candidate_identity, "R11_R7_CANDIDATE_IDENTITY_ROUNDTRIP_DRIFT")
    require(state_dicts_exactly_equal_r7(saved["state_dict"], run_states[0]), "R11_R7_CANDIDATE_CHECKPOINT_STATE_DRIFT")

    require(frozen_authority_hashes(package_root) == frozen_before, "R11_R7_FROZEN_PACKAGE_MUTATED")
    _, g0_after = r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R7_G0_AUTHORITY_MUTATED")

    result = {
        "schema": SCHEMA,
        "status": "R11_FULL_TRAIN_MULTI_ASSET_CANONICAL_CANDIDATE_R7_MATERIALIZED",
        "runtime": R7_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": repo,
        "runtime_identity": runtime,
        "source": {
            **source,
            **source_hashes,
            "symbols": list(R7_SYMBOLS),
            "legacy_validation_rows_used": False,
            "r5_purge_support_used": False,
            "r6_outer_eval_reused": False,
            "raw_market_payload_used": False,
        },
        "teacher": {
            "summary": train_summary,
            "stats": str(teacher_stats),
            "train_protocol_hash": r4.R11_TRAIN_TEACHER_CONFIG.content_hash,
            "prepared_evidence_hash": prepared.evidence_hash,
            "prepared_target_audit": prepared_audit,
            "all_train_parent_contexts_admitted": True,
            "realized_future_is_not_direct_student_label": True,
        },
        "materialization": {
            "g0_policy_hash": g0_policy_hash,
            "snapshot_hash": snapshot_hash,
            "reproduction_runs": run_receipts,
            "reproduction_policy_hash_match": True,
            "reproduction_state_tensors_exact_match": True,
            "candidate_policy_hash": candidate_policy_hash,
            "candidate_identity": candidate_identity,
            "candidate_checkpoint": candidate_path.name,
            "candidate_checkpoint_sha256": checkpoint_sha256,
            "candidate_evaluation_state": "UNEVALUATED_OUTSIDE_TRAIN_FIT",
            "candidate_promotion_state": "NOT_AUTHORIZED",
            "train_fit_slot_is_independent_evaluation": False,
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "market_information_verdict_reopened": False,
            "scientific_verdict_created": False,
            "profitability_or_alpha_claimed": False,
            "legacy_validation_evaluated": False,
            "r5_purge_evaluated": False,
            "r6_outer_eval_reused": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "raw_market_payload_used": False,
            "frozen_package_unchanged": True,
            "g0_authority_unchanged": True,
        },
        "next_legal_step": "STOP_AT_UNEVALUATED_CANDIDATE_BOUNDARY_UNLESS_A_GENUINELY_INDEPENDENT_SUPPORT_SURFACE_IS_SEPARATELY_AUTHORIZED",
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "candidate_policy_hash": candidate_policy_hash,
        "candidate_identity_sha256": candidate_identity["candidate_identity_sha256"],
        "full_train_evidence_hash": prepared.evidence_hash,
        "train_fit_before": run_receipts[0]["train_fit_before"]["loss"],
        "train_fit_after": run_receipts[0]["train_fit_after"]["loss"],
        "reproduction_exact": True,
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
