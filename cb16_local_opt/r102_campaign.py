from __future__ import annotations

import json
import math
import os
import shutil
import tarfile
import time
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from .r102_common import (
    CANONICAL_SYMBOLS_R102, G0_FILE_SHA256, G0_TENSOR_SEMANTIC_SHA256,
    atomic_write_json, model_state_semantic_sha256, sha256_file, sha256_obj,
)
from .r102_controls import run_f0_f1_f2_f3_controls
from .r102_evidence_cache import build_real_evidence_cache, load_teacher_samples, load_parent_physics_states
from .r102_learning import (
    compile_teacher_evidence, evidence_summary, persist_generation_snapshot,
    policy_behavior_fingerprint, soft_teacher_loss, train_challenger,
)
from .r102_market import preflight_all_ten_data
from .r102_parent_adoption import adopt_parent_r101
from .r102_physics import FrozenPhysicsRuntimeR102
from .r102_policy_trace import run_real_on_policy_trace
from .sharded_experience_lake import ShardedExperienceLake
from .typed_central_brain_r10 import build_g0_brain_r10, parameter_report_r10


FROZEN_RELATIVE_PATHS = (
    "assets/operator/runtime/kronos_model_l5.safetensors",
    "assets/operator/runtime/kronos_tokenizer_encode.safetensors",
    "assets/operator/operator_reducers_v1.npz",
    "assets/medium/runtime/timesfm_layer3.safetensors",
    "assets/medium/CANONICAL_NONLINEAR48_SEED24680_PORTABLE.npz",
    "authority/control_plane_r1/risk_supervisor_r1.py",
    "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/ACCOUNT_PHYSICS_CONTRACT_V1.json",
    "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0/runtime/account_physics_runtime_r0.py",
)


def frozen_authority_hashes(root: Path) -> dict[str, str]:
    out = {}
    for rel in FROZEN_RELATIVE_PATHS:
        p = root / rel
        if not p.is_file(): raise FileNotFoundError(p)
        out[rel] = sha256_file(p)
    return out


def _save_brain(path: Path, model, *, generation: int, role: str, parent_hash: str | None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "CB16_R10_2_CENTRAL_BRAIN_CHECKPOINT_V1", "generation": int(generation),
        "role": role, "parent_policy_semantic_sha256": parent_hash,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    torch.save(obj, path)
    return {"path": str(path), "file_sha256": sha256_file(path), "semantic_sha256": model_state_semantic_sha256(model)}


def _load_state_from_checkpoint(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj: return obj["state_dict"]
    if isinstance(obj, Mapping): return obj
    raise RuntimeError("CHECKPOINT_STATE_NOT_FOUND")


def _validation_loss(model, val_evidence, parents, device: str) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        _, metrics = soft_teacher_loss(model, val_evidence, parents, device=device)
    return metrics


def _promotion_decision(before: Mapping[str, float], after: Mapping[str, float], *, min_relative_improvement: float = 0.001) -> dict[str, Any]:
    b, a = float(before["loss"]), float(after["loss"])
    rel = (b - a) / max(abs(b), 1e-12)
    promote = bool(math.isfinite(a) and a < b and rel >= min_relative_improvement)
    return {
        "decision": "PROMOTE" if promote else "REJECT",
        "basis": "FROZEN_VALIDATION_PROBABILISTIC_TEACHER_TARGET_LOSS",
        "validation_loss_before": b, "validation_loss_after": a,
        "relative_improvement": rel, "minimum_relative_improvement": min_relative_improvement,
        "F0_F1_F2_F3_NOT_USED_FOR_PROMOTION": True,
        "realized_single_path_profit_NOT_used_as_correct_action": True,
    }


def package_return_artifacts(run_root: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    ret = run_root / "return_bundle"
    ret.mkdir(parents=True, exist_ok=True)
    wanted = [
        "PRESTART_ENVIRONMENT_R102.json", "PARENT_ADOPTION_RECEIPT_R102.json", "TEN_SYMBOL_DATA_PREFLIGHT_R102.json",
        "REAL_EVIDENCE_CACHE_MANIFEST_R102.json", "TEACHER_EVIDENCE_SUMMARY_R102.json",
        "F0_F1_F2_F3_CONTROLS_R102.json", "FINAL_RESULT_R102.json", "FROZEN_AUTHORITY_POSTCHECK_R102.json",
    ]
    for name in wanted:
        src = run_root / name
        if src.is_file():
            (ret / name).write_bytes(src.read_bytes())
    # Generation results + compact brain checkpoints are small and constitute the Champion/Challenger lineage.
    for p in sorted((run_root / "generations").glob("G*/GENERATION_RESULT.json")):
        dst = ret / f"{p.parent.name}_{p.name}"; dst.write_bytes(p.read_bytes())
    for p in sorted((run_root / "generations").glob("G*/champion_after.pt")):
        dst = ret / f"{p.parent.name}_champion_after.pt"; dst.write_bytes(p.read_bytes())
    manifest = {}
    for p in sorted(ret.iterdir()):
        if p.is_file(): manifest[p.name] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    atomic_write_json(ret / "RETURN_MANIFEST.json", {"schema": "CB16_R10_2_RETURN_MANIFEST_V1", "files": manifest})
    profile = str(result.get("profile_name", "R10_2"))
    phase = "R10_4" if profile.startswith("R10_4") else "R10_3" if profile.startswith("R10_3") else "R10_2"
    tar_path = run_root / f"CB16_{phase}_RETURN_RECEIPTS_R0.tar.gz"
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
        tf.add(ret, arcname="CB16_R10_2_RETURN_RECEIPTS_R0")
    return {"path": str(tar_path), "sha256": sha256_file(tar_path), "size": tar_path.stat().st_size}


def run_campaign(
    *, package_root: str | Path, data_root: str | Path, run_root: str | Path,
    parent_r101_root: str | Path, parent_g0: str | Path,
    device: str = "cuda", symbols: Sequence[str] = CANONICAL_SYMBOLS_R102,
    attempts: int = 5, stride_hours: int = 512, prehistory_hours: int = 96,
    epochs: int = 12, batch_size: int = 512, lr: float = 3e-4,
    verify_checksum_samples: bool = True, verify_all_cache_checksums: bool = False,
    profile_name: str = "R10_2_5GEN_QUALIFICATION", prerequisite_result: str | Path | None = None,
    start_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_root).resolve(); rr = Path(run_root).resolve(); rr.mkdir(parents=True, exist_ok=True)
    if prerequisite_result is not None:
        pr = Path(prerequisite_result)
        if not pr.is_file(): raise FileNotFoundError(pr)
        prev = json.loads(pr.read_text())
        if not str(prev.get("final_status", "")).endswith("PASS"):
            raise RuntimeError(f"PREREQUISITE_NOT_PASS:{pr}:{prev.get('final_status')}")

    prestart_path = rr / "PRESTART_ENVIRONMENT_R102.json"
    if not prestart_path.exists():
        du = shutil.disk_usage(rr)
        env = {
            "schema":"CB16_R10_2_PRESTART_ENVIRONMENT_V1", "status":"PASS",
            "python":sys.version, "platform":platform.platform(), "torch":torch.__version__,
            "cuda_available":bool(torch.cuda.is_available()), "torch_cuda":torch.version.cuda,
            "device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability":list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            "run_root":str(rr), "disk_total_bytes":du.total, "disk_free_bytes":du.free,
            "disk_hard_stop_bytes":10*(1<<30), "final_holdout_2025_09_accessed":False,
        }
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
        if du.free < 10*(1<<30): raise RuntimeError(f"SSD_HARD_STOP_LT_10GIB:free_bytes={du.free}")
        atomic_write_json(prestart_path, env)

    final_path = rr / "FINAL_RESULT_R102.json"
    if final_path.exists():
        old = json.loads(final_path.read_text())
        if old.get("profile_name") == profile_name and old.get("attempts_requested") == attempts:
            return old
        raise RuntimeError("RUN_ROOT_ALREADY_CONTAINS_DIFFERENT_FINAL_RESULT")

    # Stage 0: adopt exact installed R10.1 frozen organs and immutable G0.
    adoption_path = rr / "PARENT_ADOPTION_RECEIPT_R102.json"
    if adoption_path.exists(): adoption = json.loads(adoption_path.read_text())
    else:
        adoption = adopt_parent_r101(
            package_root=root, parent_r101_root=parent_r101_root, parent_g0=parent_g0,
            receipt_path=adoption_path,
        )
    frozen_before = frozen_authority_hashes(root)

    # Stage 1: all ten local archives structurally available; only allowed-month checksum samples are opened.
    preflight_path = rr / "TEN_SYMBOL_DATA_PREFLIGHT_R102.json"
    if preflight_path.exists(): data_preflight = json.loads(preflight_path.read_text())
    else:
        source = BinanceUSDMArchiveSourceR10(data_root)
        data_preflight = preflight_all_ten_data(source, verify_checksum_samples=verify_checksum_samples)
        atomic_write_json(preflight_path, data_preflight)

    # Stage 2: build once, reuse for all five generation attempts. This is real historical H72 truth, not synthetic utility.
    cache_dir = rr / "evidence_cache"
    cache_manifest_path = cache_dir / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    if cache_manifest_path.exists(): cache_manifest = json.loads(cache_manifest_path.read_text())
    else:
        cache_manifest = build_real_evidence_cache(
            package_root=root, data_root=data_root, out_dir=cache_dir, device=device, symbols=symbols,
            stride_hours=stride_hours, prehistory_hours=prehistory_hours,
            verify_checksums=verify_all_cache_checksums,
        )
    # Copy compact manifest to run root for easy return.
    atomic_write_json(rr / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json", cache_manifest)
    parents, samples = load_teacher_samples(cache_manifest["parents_file"], cache_manifest["branches_file"])
    parent_states = load_parent_physics_states(cache_manifest["parent_states_file"])
    physics_runtime = FrozenPhysicsRuntimeR102.load(root)
    train_evidence, val_evidence = compile_teacher_evidence(samples, parents)
    teacher_summary = {"schema": "CB16_R10_2_TEACHER_EVIDENCE_SUMMARY_V1", "train": evidence_summary(train_evidence), "validation": evidence_summary(val_evidence)}
    if teacher_summary["train"]["admitted_dependence_groups"] < 32:
        raise RuntimeError(f"TRAIN_TEACHER_SUPPORT_NOT_READY:{teacher_summary['train']}")
    if teacher_summary["validation"]["admitted_dependence_groups"] < 8:
        raise RuntimeError(f"VALIDATION_TEACHER_SUPPORT_NOT_READY:{teacher_summary['validation']}")
    atomic_write_json(rr / "TEACHER_EVIDENCE_SUMMARY_R102.json", teacher_summary)

    # Stage 3: F0/F1/F2/F3 are immutable diagnostics. Negative scientific result is never rescued and does not block mechanistic closure.
    controls_path = rr / "F0_F1_F2_F3_CONTROLS_R102.json"
    if controls_path.exists(): controls = json.loads(controls_path.read_text())
    else:
        controls = run_f0_f1_f2_f3_controls(samples, parents); atomic_write_json(controls_path, controls)

    # Stage 4: start from exact G0 unless an explicitly gated later profile supplies the previous final Champion.
    model = build_g0_brain_r10("TIER_1", seed=24680, device=device)
    if start_checkpoint is None:
        checkpoint = root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt"
        if sha256_file(checkpoint) != G0_FILE_SHA256:
            raise RuntimeError("G0_FILE_SHA_DRIFT_BEFORE_CAMPAIGN")
    else:
        checkpoint = Path(start_checkpoint)
    model.load_state_dict(_load_state_from_checkpoint(checkpoint), strict=True)
    if start_checkpoint is None and model_state_semantic_sha256(model) != G0_TENSOR_SEMANTIC_SHA256:
        raise RuntimeError("G0_SEMANTIC_DRIFT_BEFORE_CAMPAIGN")
    champion_hash = model_state_semantic_sha256(model)

    lake = ShardedExperienceLake(rr / "experience_lake", shards=4, synchronous="FULL")
    generation_results = []
    try:
        for g in range(int(attempts)):
            free = shutil.disk_usage(rr).free
            if free < 10 * (1 << 30):
                raise RuntimeError(f"SSD_HARD_STOP_LT_10GIB:free_bytes={free}")
            gd = rr / "generations" / f"G{g:02d}"; gd.mkdir(parents=True, exist_ok=True)
            gp = gd / "GENERATION_RESULT.json"
            if gp.exists():
                gr = json.loads(gp.read_text()); generation_results.append(gr)
                ckp = gd / "champion_after.pt"
                model.load_state_dict(_load_state_from_checkpoint(ckp), strict=True)
                champion_hash = model_state_semantic_sha256(model)
                continue

            parent_hash = champion_hash
            before_behavior = policy_behavior_fingerprint(model, val_evidence, parents, device=device)
            before_val = _validation_loss(model, val_evidence, parents, device)
            on_policy_trace = run_real_on_policy_trace(
                model=model, policy_hash=parent_hash, generation=g, parents=parents, parent_states=parent_states,
                cache_dir=cache_dir, physics=physics_runtime, lake=lake, device=device, max_groups=24,
            )
            if on_policy_trace["matured"] != on_policy_trace["trace_count"]:
                raise RuntimeError(f"ON_POLICY_REAL_TRACE_NOT_FULLY_MATURED:{on_policy_trace}")
            atomic_write_json(gd / "ON_POLICY_REAL_TRACE_RECEIPT.json", on_policy_trace)
            snap, refs = persist_generation_snapshot(
                lake=lake, generation=g, champion_hash=parent_hash,
                train_evidence=train_evidence, parents=parents,
            )
            # Challenger begins as exact Champion copy.
            challenger = build_g0_brain_r10("TIER_1", seed=24680, device=device)
            challenger.load_state_dict(model.state_dict(), strict=True)
            train_receipt = train_challenger(
                model=challenger, train_evidence=train_evidence, val_evidence=val_evidence,
                parents=parents, device=device, generation=g, snapshot_hash=snap.content_hash,
                receipt_dir=gd, epochs=epochs, batch_size=batch_size, lr=lr,
            )
            after_behavior = policy_behavior_fingerprint(challenger, val_evidence, parents, device=device)
            after_val = train_receipt["validation_after"]
            tournament = _promotion_decision(before_val, after_val)
            challenger_info = _save_brain(gd / "challenger.pt", challenger, generation=g + 1, role="CHALLENGER", parent_hash=parent_hash)
            if tournament["decision"] == "PROMOTE":
                model.load_state_dict(challenger.state_dict(), strict=True)
            champion_hash = model_state_semantic_sha256(model)
            champion_info = _save_brain(gd / "champion_after.pt", model, generation=g + 1, role="CHAMPION", parent_hash=parent_hash)
            frozen_now = frozen_authority_hashes(root)
            if frozen_now != frozen_before:
                raise RuntimeError("FROZEN_AUTHORITY_HASH_DRIFT_DURING_LEARNING")
            gr = {
                "schema": "CB16_R10_2_GENERATION_RESULT_V1", "generation_attempt": g,
                "parent_champion_semantic_sha256": parent_hash,
                "training_snapshot": {"snapshot_id": snap.snapshot_id, "snapshot_hash": snap.content_hash, "evidence_objects": snap.object_count},
                "on_policy_real_trace": on_policy_trace,
                "training": train_receipt, "validation_before": before_val,
                "behavior_before": before_behavior, "behavior_after": after_behavior,
                "challenger": challenger_info, "tournament": tournament,
                "champion_after": champion_info,
                "rejected_parent_rule": "IF_REJECT_NEXT_GENERATION_STILL_USES_UNCHANGED_CHAMPION",
                "frozen_authority_unchanged": True,
            }
            atomic_write_json(gp, gr); generation_results.append(gr)
            lake.checkpoint_all("PASSIVE")
    finally:
        lake_audit = lake.audit(verify_payloads=True); lake.checkpoint_all("TRUNCATE"); lake.close()

    frozen_after = frozen_authority_hashes(root)
    frozen_pass = frozen_after == frozen_before
    atomic_write_json(rr / "FROZEN_AUTHORITY_POSTCHECK_R102.json", {
        "schema": "CB16_R10_2_FROZEN_AUTHORITY_POSTCHECK_V1", "status": "PASS" if frozen_pass else "FAIL",
        "before": frozen_before, "after": frozen_after,
    })
    attempts_completed = len(generation_results)
    behavior_changed = any(x["behavior_before"].get("sha256") != x["behavior_after"].get("sha256") for x in generation_results)
    gradients_connected = all(all(v > 0 for v in x["training"]["gradient_group_norms_last_step"].values()) for x in generation_results)
    snapshots_nonempty = all(x["training_snapshot"]["evidence_objects"] > 0 for x in generation_results)
    on_policy_real_trace_ok = all(x.get("on_policy_real_trace",{}).get("matured",0) == x.get("on_policy_real_trace",{}).get("trace_count",-1) and x.get("on_policy_real_trace",{}).get("trace_count",0)>0 for x in generation_results)
    lifecycle_ok = all(
        x["champion_after"]["semantic_sha256"] == (
            x["challenger"]["semantic_sha256"] if x["tournament"]["decision"] == "PROMOTE" else x["parent_champion_semantic_sha256"]
        ) for x in generation_results
    )
    pass_mech = bool(
        attempts_completed == attempts and frozen_pass and lake_audit["pass"] and
        behavior_changed and gradients_connected and snapshots_nonempty and on_policy_real_trace_ok and lifecycle_ok
    )
    final_status = (
        "R10_2_REAL_HISTORICAL_LEARNING_PIPELINE_PASS" if pass_mech and profile_name.startswith("R10_2")
        else f"{profile_name}_PASS" if pass_mech
        else "R10_2_REAL_HISTORICAL_LEARNING_PIPELINE_NOT_READY"
    )
    result = {
        "schema": "CB16_R10_2_REAL_HISTORICAL_LEARNING_FINAL_RESULT_V1",
        "profile_name": profile_name, "final_status": final_status,
        "mechanistic_pipeline_pass": pass_mech,
        "scientific_controls_status": controls["interpretation"],
        "scientific_controls_are_not_rescued": True,
        "profitability_claimed": False, "market_alpha_claimed": False,
        "final_holdout_2025_09_accessed": False,
        "attempts_requested": int(attempts), "attempts_completed": attempts_completed,
        "promotions": sum(x["tournament"]["decision"] == "PROMOTE" for x in generation_results),
        "rejections": sum(x["tournament"]["decision"] == "REJECT" for x in generation_results),
        "final_champion_semantic_sha256": champion_hash,
        "symbols": list(symbols), "stride_hours": int(stride_hours),
        "real_market_data_used": True, "exact_frozen_v55_physics_used": True,
        "actual_binance_funding_events_used": True,
        "teacher_semantics": "PROBABILISTIC_DISTRIBUTIONAL_NO_BEST_ACTION_LABEL",
        "dependence_semantics": "SAME_SYMBOL_TIMESTAMP_FUTURE_GROUP_COUNTS_ONCE_ACROSS_ACCOUNTS_AND_ACTIONS",
        "teacher_evidence_summary": teacher_summary,
        "controls": controls,
        "experience_lake_audit": lake_audit,
        "generation_results": generation_results,
        "integrity": {
            "frozen_authority_unchanged": frozen_pass, "real_evidence_gradients_connected": gradients_connected,
            "parameter_to_behavior_changed": behavior_changed, "snapshots_nonempty": snapshots_nonempty,
            "on_policy_Brain_to_Physics_H72_trace": on_policy_real_trace_ok,
            "champion_challenger_lifecycle_correct": lifecycle_ok,
        },
    }
    atomic_write_json(final_path, result)
    return_bundle = package_return_artifacts(rr, result)
    result["return_bundle"] = return_bundle
    atomic_write_json(final_path, result)
    return result
