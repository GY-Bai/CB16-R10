from __future__ import annotations

"""R2-first campaign orchestrator.

This is a structural Infra successor to the legacy Experience Lake path:
- teacher evidence is materialized once as generation-independent immutable content;
- each generation seals only a tiny Champion/evidence-set snapshot;
- generation-specific decision/outcome events are committed as one SSD transaction;
- HDD payload packs and SSD metadata/event authority are physically separated.

Qualification only until the R2 host gates pass.
"""

import json
import math
import platform
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from .r102_common import (
    CANONICAL_SYMBOLS_R102, G0_FILE_SHA256, G0_TENSOR_SEMANTIC_SHA256,
    atomic_write_json, model_state_semantic_sha256, sha256_file,
)
from .r102_controls import run_f0_f1_f2_f3_controls
from .r102_evidence_cache import build_real_evidence_cache, load_parent_physics_states, load_teacher_samples
from .r102_learning import (
    TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102,
    evidence_summary, policy_behavior_fingerprint,
    soft_teacher_loss, train_challenger,
)
from .r102_market import preflight_all_ten_data
from .r102_parent_adoption import adopt_parent_r101
from .r102_physics import FrozenPhysicsRuntimeR102
from .r102_policy_trace import run_real_on_policy_trace
from .r102_teacher_incremental import compile_teacher_evidence_incremental
from .r2_event_journal import R2BufferedEventSink, R2EventJournal
from .r2_evidence_storage import R2EvidenceStore
from .r2_learning_bridge import materialize_training_evidence_r2, seal_generation_snapshot_r2
from .r2_sequential_audit import audit_r2_store_sequential
from .typed_central_brain_r10 import build_g0_brain_r10


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
        if not p.is_file():
            raise FileNotFoundError(p)
        out[rel] = sha256_file(p)
    return out


def _save_brain(path: Path, model, *, generation: int, role: str, parent_hash: str | None) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema":"CB16_R2_CENTRAL_BRAIN_CHECKPOINT_V1",
        "generation":int(generation), "role":role,
        "parent_policy_semantic_sha256":parent_hash,
        "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
    }
    torch.save(obj, path)
    return {"path":str(path),"file_sha256":sha256_file(path),
            "semantic_sha256":model_state_semantic_sha256(model)}


def _load_state(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, Mapping):
        return obj
    raise RuntimeError("CHECKPOINT_STATE_NOT_FOUND")


def _validation_loss(model, evidence, parents, device: str) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        _, metrics = soft_teacher_loss(model, evidence, parents, device=device)
    return metrics


def _promotion(before: Mapping[str, float], after: Mapping[str, float], minimum: float = 0.001) -> dict[str, Any]:
    b, a = float(before["loss"]), float(after["loss"])
    rel = (b-a) / max(abs(b), 1e-12)
    promote = bool(math.isfinite(a) and a < b and rel >= minimum)
    return {
        "decision":"PROMOTE" if promote else "REJECT",
        "basis":"FROZEN_VALIDATION_PROBABILISTIC_TEACHER_TARGET_LOSS",
        "validation_loss_before":b,"validation_loss_after":a,
        "relative_improvement":rel,"minimum_relative_improvement":minimum,
        "F0_F1_F2_F3_NOT_USED_FOR_PROMOTION":True,
        "realized_single_path_profit_NOT_used_as_correct_action":True,
    }


def _free_guard(path: Path, minimum_bytes: int, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < minimum_bytes:
        raise RuntimeError(f"R2_STORAGE_HARD_STOP:{label}:free_bytes={free}:minimum={minimum_bytes}")


def run_campaign_r2(
    *,
    package_root: str | Path,
    data_root: str | Path,
    run_root: str | Path,
    parent_r101_root: str | Path,
    parent_g0: str | Path,
    metadata_root: str | Path,
    payload_roots: Sequence[str | Path],
    device: str = "cuda",
    symbols: Sequence[str] = CANONICAL_SYMBOLS_R102,
    attempts: int = 5,
    stride_hours: int = 512,
    prehistory_hours: int = 96,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 3e-4,
    verify_checksum_samples: bool = True,
    verify_all_cache_checksums: bool = False,
    profile_name: str = "R2_STORAGE_QUALIFICATION",
    prerequisite_result: str | Path | None = None,
    start_checkpoint: str | Path | None = None,
    codec: str = "zstd",
    segment_target_bytes: int = 256 * 1024 * 1024,
    storage_min_free_bytes: int = 10 * (1 << 30),
    teacher_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_root).resolve()
    rr = Path(run_root).resolve(); rr.mkdir(parents=True, exist_ok=True)
    meta = Path(metadata_root).resolve(); meta.mkdir(parents=True, exist_ok=True)
    payloads = [Path(x).resolve() for x in payload_roots]
    if not payloads:
        raise RuntimeError("R2_PAYLOAD_ROOT_REQUIRED")
    for p in payloads:
        p.mkdir(parents=True, exist_ok=True)
    if prerequisite_result is not None:
        prev = json.loads(Path(prerequisite_result).read_text())
        if not str(prev.get("final_status", "")).endswith("PASS"):
            raise RuntimeError("R2_PREREQUISITE_NOT_PASS")

    _free_guard(meta, storage_min_free_bytes, "METADATA")
    for i, p in enumerate(payloads):
        _free_guard(p, storage_min_free_bytes, f"PAYLOAD_LANE_{i}")

    final_path = rr / "FINAL_RESULT_R2.json"
    if final_path.exists():
        old = json.loads(final_path.read_text())
        if old.get("profile_name") == profile_name and old.get("attempts_requested") == attempts:
            return old
        raise RuntimeError("R2_RUN_ROOT_ALREADY_CONTAINS_DIFFERENT_FINAL_RESULT")

    prestart = {
        "schema":"CB16_R2_PRESTART_ENVIRONMENT_V1","status":"PASS",
        "python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
        "cuda_available":bool(torch.cuda.is_available()),"torch_cuda":torch.version.cuda,
        "device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_root":str(rr),"metadata_root":str(meta),
        "payload_roots":[str(x) for x in payloads],"payload_lanes":len(payloads),
        "storage_model":"SSD_METADATA_PLUS_HDD_SEQUENTIAL_IMMUTABLE_PACK",
        "final_holdout_2025_09_accessed":False,
    }
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    atomic_write_json(rr / "PRESTART_ENVIRONMENT_R2.json", prestart)

    adoption_path = rr / "PARENT_ADOPTION_RECEIPT_R102.json"
    if not adoption_path.exists():
        adopt_parent_r101(package_root=root, parent_r101_root=parent_r101_root,
                          parent_g0=parent_g0, receipt_path=adoption_path)
    frozen_before = frozen_authority_hashes(root)

    preflight_path = rr / "TEN_SYMBOL_DATA_PREFLIGHT_R102.json"
    if preflight_path.exists():
        data_preflight = json.loads(preflight_path.read_text())
    else:
        data_preflight = preflight_all_ten_data(
            BinanceUSDMArchiveSourceR10(data_root), verify_checksum_samples=verify_checksum_samples
        )
        atomic_write_json(preflight_path, data_preflight)

    cache_dir = rr / "evidence_cache"
    cache_manifest_path = cache_dir / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    if cache_manifest_path.exists():
        cache_manifest = json.loads(cache_manifest_path.read_text())
    else:
        cache_manifest = build_real_evidence_cache(
            package_root=root,data_root=data_root,out_dir=cache_dir,device=device,symbols=symbols,
            stride_hours=stride_hours,prehistory_hours=prehistory_hours,
            verify_checksums=verify_all_cache_checksums,
        )
    atomic_write_json(rr / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json", cache_manifest)
    parents, samples = load_teacher_samples(cache_manifest["parents_file"], cache_manifest["branches_file"])
    parent_states = load_parent_physics_states(cache_manifest["parent_states_file"])
    physics_runtime = FrozenPhysicsRuntimeR102.load(root)

    compiled_root = (
        Path(teacher_cache_root).resolve()
        if teacher_cache_root is not None
        else (meta / "compiled_teacher_authority")
    )
    train_evidence, val_evidence, teacher_authority = compile_teacher_evidence_incremental(
        samples=samples,
        parents=parents,
        source_identity=cache_manifest,
        cache_root=compiled_root,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
    )
    atomic_write_json(rr / "COMPILED_TEACHER_AUTHORITY_RECEIPT_R102.json", teacher_authority)
    teacher_summary = {
        "schema":"CB16_R2_TEACHER_EVIDENCE_SUMMARY_V1",
        "train":evidence_summary(train_evidence),"validation":evidence_summary(val_evidence),
        "compiled_teacher_authority_hash":teacher_authority["authority_hash"],
        "compiled_teacher_authority_mode":teacher_authority["mode"],
    }
    if teacher_summary["train"]["admitted_dependence_groups"] < 32:
        raise RuntimeError("R2_TRAIN_TEACHER_SUPPORT_NOT_READY")
    if teacher_summary["validation"]["admitted_dependence_groups"] < 8:
        raise RuntimeError("R2_VALIDATION_TEACHER_SUPPORT_NOT_READY")
    atomic_write_json(rr / "TEACHER_EVIDENCE_SUMMARY_R2.json", teacher_summary)

    controls_path = rr / "F0_F1_F2_F3_CONTROLS_R102.json"
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else run_f0_f1_f2_f3_controls(samples, parents)
    if not controls_path.exists():
        atomic_write_json(controls_path, controls)

    model = build_g0_brain_r10("TIER_1", seed=24680, device=device)
    checkpoint = root / "authority/g0_parent/central_brain_g0_r10_2_parent.pt" if start_checkpoint is None else Path(start_checkpoint)
    if start_checkpoint is None and sha256_file(checkpoint) != G0_FILE_SHA256:
        raise RuntimeError("G0_FILE_SHA_DRIFT_BEFORE_R2_CAMPAIGN")
    model.load_state_dict(_load_state(checkpoint), strict=True)
    if start_checkpoint is None and model_state_semantic_sha256(model) != G0_TENSOR_SEMANTIC_SHA256:
        raise RuntimeError("G0_SEMANTIC_DRIFT_BEFORE_R2_CAMPAIGN")
    champion_hash = model_state_semantic_sha256(model)

    store = R2EvidenceStore(
        metadata_root=meta / "evidence",
        payload_roots=payloads,
        codec=codec, segment_target_bytes=segment_target_bytes,
        sqlite_synchronous="FULL", recover_on_open=True,
    )
    events = R2EventJournal(meta / "events", synchronous="FULL")
    evidence_set, materialize_receipt = materialize_training_evidence_r2(
        store=store, train_evidence=train_evidence, parents=parents,
        evidence_set_id="R2_TRAINING_EVIDENCE_SET_V1",
    )
    atomic_write_json(rr / "R2_TRAINING_EVIDENCE_MATERIALIZATION.json", asdict(materialize_receipt))

    generation_results = []
    try:
        for g in range(int(attempts)):
            _free_guard(meta, storage_min_free_bytes, "METADATA")
            for i, p in enumerate(payloads):
                _free_guard(p, storage_min_free_bytes, f"PAYLOAD_LANE_{i}")
            gd = rr / "generations" / f"G{g:02d}"; gd.mkdir(parents=True, exist_ok=True)
            gp = gd / "GENERATION_RESULT.json"
            if gp.exists():
                gr = json.loads(gp.read_text()); generation_results.append(gr)
                model.load_state_dict(_load_state(gd / "champion_after.pt"), strict=True)
                champion_hash = model_state_semantic_sha256(model)
                continue

            parent_hash = champion_hash
            before_behavior = policy_behavior_fingerprint(model,val_evidence,parents,device=device)
            before_val = _validation_loss(model,val_evidence,parents,device)

            sink = R2BufferedEventSink(events)
            on_policy = run_real_on_policy_trace(
                model=model,policy_hash=parent_hash,generation=g,parents=parents,
                parent_states=parent_states,cache_dir=cache_dir,physics=physics_runtime,
                lake=sink,device=device,max_groups=24,
            )
            if on_policy["matured"] != on_policy["trace_count"]:
                raise RuntimeError(f"R2_ON_POLICY_TRACE_NOT_FULLY_MATURED:{on_policy}")
            _event_refs, event_receipt = sink.flush()
            on_policy["r2_event_batch"] = asdict(event_receipt)
            atomic_write_json(gd / "ON_POLICY_REAL_TRACE_RECEIPT.json", on_policy)

            snap = seal_generation_snapshot_r2(
                store=store,evidence_set=evidence_set,generation=g,champion_hash=parent_hash
            )
            challenger = build_g0_brain_r10("TIER_1",seed=24680,device=device)
            challenger.load_state_dict(model.state_dict(),strict=True)
            train_receipt = train_challenger(
                model=challenger,train_evidence=train_evidence,val_evidence=val_evidence,
                parents=parents,device=device,generation=g,snapshot_hash=snap.content_hash,
                receipt_dir=gd,epochs=epochs,batch_size=batch_size,lr=lr,
            )
            after_behavior = policy_behavior_fingerprint(challenger,val_evidence,parents,device=device)
            tournament = _promotion(before_val,train_receipt["validation_after"])
            challenger_info = _save_brain(gd/"challenger.pt",challenger,generation=g+1,role="CHALLENGER",parent_hash=parent_hash)
            if tournament["decision"] == "PROMOTE":
                model.load_state_dict(challenger.state_dict(),strict=True)
            champion_hash = model_state_semantic_sha256(model)
            champion_info = _save_brain(gd/"champion_after.pt",model,generation=g+1,role="CHAMPION",parent_hash=parent_hash)
            if frozen_authority_hashes(root) != frozen_before:
                raise RuntimeError("FROZEN_AUTHORITY_HASH_DRIFT_DURING_R2_LEARNING")

            gr = {
                "schema":"CB16_R2_GENERATION_RESULT_V1","generation_attempt":g,
                "parent_champion_semantic_sha256":parent_hash,
                "training_snapshot":{
                    "snapshot_id":snap.snapshot_id,"snapshot_hash":snap.content_hash,
                    "evidence_set_hash":snap.evidence_set_hash,"evidence_objects":snap.object_count,
                    "storage_semantics":"GENERATION_REFERENCES_IMMUTABLE_EVIDENCE_SET",
                },
                "on_policy_real_trace":on_policy,"training":train_receipt,
                "validation_before":before_val,"behavior_before":before_behavior,
                "behavior_after":after_behavior,"challenger":challenger_info,
                "tournament":tournament,"champion_after":champion_info,
                "frozen_authority_unchanged":True,
            }
            atomic_write_json(gp,gr); generation_results.append(gr)
            store.checkpoint("PASSIVE"); events.checkpoint("PASSIVE")
    finally:
        storage_audit = audit_r2_store_sequential(store, verify_payloads=True)
        event_audit = events.audit()
        store.checkpoint("TRUNCATE"); events.checkpoint("TRUNCATE")
        store_stats = store.stats()
        # Qualification and downstream gates need the physical lane count.  This
        # is runtime/storage metadata only and does not enter scientific identity.
        store_stats["payload_lanes"] = len(payloads)
        store.close(); events.close()

    frozen_pass = frozen_authority_hashes(root) == frozen_before
    attempts_completed = len(generation_results)
    behavior_changed = any(x["behavior_before"].get("sha256") != x["behavior_after"].get("sha256") for x in generation_results)
    gradients_connected = all(all(v > 0 for v in x["training"]["gradient_group_norms_last_step"].values()) for x in generation_results)
    snapshots_nonempty = all(x["training_snapshot"]["evidence_objects"] > 0 for x in generation_results)
    traces_ok = all(x["on_policy_real_trace"]["matured"] == x["on_policy_real_trace"]["trace_count"] > 0 for x in generation_results)
    lifecycle_ok = all(
        x["champion_after"]["semantic_sha256"] == (
            x["challenger"]["semantic_sha256"] if x["tournament"]["decision"] == "PROMOTE"
            else x["parent_champion_semantic_sha256"]
        ) for x in generation_results
    )
    storage_reuse_ok = store_stats["payload_objects"] <= materialize_receipt.unique_payload_count
    pass_mech = bool(
        attempts_completed == attempts and frozen_pass and storage_audit["pass"] and event_audit["pass"]
        and storage_reuse_ok and behavior_changed and gradients_connected and snapshots_nonempty
        and traces_ok and lifecycle_ok
    )
    result = {
        "schema":"CB16_R2_REAL_HISTORICAL_LEARNING_FINAL_RESULT_V1",
        "profile_name":profile_name,
        "final_status":f"{profile_name}_PASS" if pass_mech else f"{profile_name}_NOT_READY",
        "mechanistic_pipeline_pass":pass_mech,
        "final_holdout_2025_09_accessed":False,
        "attempts_requested":int(attempts),"attempts_completed":attempts_completed,
        "promotions":sum(x["tournament"]["decision"]=="PROMOTE" for x in generation_results),
        "rejections":sum(x["tournament"]["decision"]=="REJECT" for x in generation_results),
        "final_champion_semantic_sha256":champion_hash,
        "teacher_semantics":"PROBABILISTIC_DISTRIBUTIONAL_NO_BEST_ACTION_LABEL",
        "compiled_teacher_authority":teacher_authority,
        "storage_semantics":"IMMUTABLE_EVIDENCE_ONCE_PLUS_TINY_GENERATION_MANIFESTS",
        "storage_materialization":asdict(materialize_receipt),
        "storage_stats":store_stats,"storage_audit":storage_audit,
        "event_journal_audit":event_audit,
        "teacher_evidence_summary":teacher_summary,"controls":controls,
        "generation_results":generation_results,
        "integrity":{
            "frozen_authority_unchanged":frozen_pass,
            "real_evidence_gradients_connected":gradients_connected,
            "parameter_to_behavior_changed":behavior_changed,
            "snapshots_nonempty":snapshots_nonempty,
            "on_policy_Brain_to_Physics_H72_trace":traces_ok,
            "champion_challenger_lifecycle_correct":lifecycle_ok,
            "evidence_payload_not_rematerialized_per_generation":storage_reuse_ok,
            "compiled_teacher_authority_verified":True,
        },
    }
    atomic_write_json(final_path,result)
    return result
