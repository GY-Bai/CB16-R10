#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 Historical Execution R1 — bounded shadow learning connectivity.

R1 deliberately does *not* advance the canonical R11 generation or run a
Champion/Challenger tournament.  It loads the immutable G0 bootstrap, creates a
disposable Challenger in the Actions work directory, and proves on real frozen
history that the qualified feedback machinery reaches legal Student gradients,
parameter updates, and measurable behavior changes.

Scientific-quality claims remain separate.  A PASS here is not an alpha claim,
not a market-information qualification, and does not supersede the frozen
TRUE_WORSE_THAN_SHUFFLE result.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from cb16_local_opt.evidence_store_r11 import EvidenceItemR11, EvidenceStoreR11
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_common import (
    FORBIDDEN_FINAL_START_MS,
    H72,
    HOUR_MS,
    TRAIN_END_MS,
    TRAIN_VALIDATION_PURGE_HOURS,
    VALIDATION_END_MS,
)
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import (
    TRAIN_TEACHER_CONFIG_R102,
    VAL_TEACHER_CONFIG_R102,
    evidence_summary,
)
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import (
    CANDIDATES_R102,
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    market_future_lineage_hash,
    simulate_h72_branch,
)
from cb16_local_opt.r102_common import model_state_semantic_sha256
from cb16_local_opt.science_feedback_diagnostics_r11 import (
    audit_training_feedback_r11,
    behavior_delta_r11,
    capture_behavior_r11,
    independent_loss_breakdown_r11,
    validate_teacher_targets_r11,
)
from cb16_local_opt.sharded_experience_lake import ShardedExperienceLake
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.trace_runtime_r11 import TraceRuntimeR11, run_real_on_policy_trace_r11
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import policy_hash_r11, prepare_evidence_campaign_r11
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


SCHEMA = "CB16_R11_SCIENCE_G0_HISTORICAL_R1_RESULT_V1"
MAIN_SEED = "a0848118b3ec39dedd9fc73368e0fbae90de1034"
SEMANTIC_FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"
DATASET_SEAL_SHA256 = "94bc3288a7e6097c4301c93d836607edee207ac3b2c7cf3e7833c85ea0f9546a"
DERIVED_CACHE_LINEAGE_SHA256 = "69e3bb82ba717104422dd45608d6a05725ccc53a08bded621c8f52095ccbb1d6"
GENESIS_IDENTITY_SHA256 = "dce5f5dc41b9b6ff712269fd0e3b87a861d2296147bd87ee62ff54e6813c798d"
ADOPTION_CANONICAL_HASH = "2e9fed060ff29160197bf4a0ab391575fe15c4bb6e21f3ce368cfddd67ffab81"
BOOTSTRAP_FILE_SHA256 = "799167bd0a2a820922c03d77563ed3fd22170666819dfbda048cc036141bccfb"
BOOTSTRAP_SEMANTIC_SHA256 = "d0e8c01cc58a1e70936f94f58eeca794eae2e4f89036ec26ccb49ae5d5219886"
FROZEN_SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError(code)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def runtime_identity(device: str) -> dict[str, Any]:
    dev = torch.device(device)
    out = {
        "python": sys.version,
        "python_series": [sys.version_info.major, sys.version_info.minor],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "requested_device": str(device),
    }
    if dev.type == "cuda":
        require(torch.cuda.is_available(), "R11_R1_CUDA_REQUESTED_BUT_UNAVAILABLE")
        idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
        out.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": torch.cuda.get_device_name(idx),
                "cuda_capability": list(torch.cuda.get_device_capability(idx)),
            }
        )
    return out


def verify_repo_seed() -> dict[str, str]:
    head = git("rev-parse", "HEAD")
    freeze = git("rev-parse", "HEAD:authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json")
    require(freeze == SEMANTIC_FREEZE_BLOB, f"R11_R1_SEMANTIC_FREEZE_DRIFT:{freeze}")
    require(
        subprocess.call(["git", "merge-base", "--is-ancestor", MAIN_SEED, head]) == 0,
        f"R11_R1_NOT_DESCENDED_FROM_IMMUTABLE_MAIN_SEED:{head}",
    )
    return {"execution_head": head, "immutable_main_seed": MAIN_SEED, "semantic_freeze_git_blob": freeze}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_g0_authority(g0_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    a = g0_root / "authority"
    genesis = _load_json(a / "CB16_R11_G0_GENESIS_SOURCE_RECORD.json")
    adoption = _load_json(a / "CB16_R11_G0_AUTHORITY_ADOPTION_RECEIPT.json")
    seal = _load_json(a / "CB16_R11_G0_DATASET_SEAL.json")
    lineage = _load_json(a / "CB16_R11_G0_DERIVED_CACHE_LINEAGE.json")
    bootstrap = g0_root / "bootstrap" / "R11_G0_INITIAL_CHAMPION.pt"

    require(genesis.get("genesis_identity_sha256") == GENESIS_IDENTITY_SHA256, "R11_R1_GENESIS_IDENTITY_DRIFT")
    require(genesis.get("r11", {}).get("generation") == 0, "R11_R1_GENESIS_GENERATION_DRIFT")
    require(genesis.get("r11", {}).get("semantic_freeze_git_blob") == SEMANTIC_FREEZE_BLOB, "R11_R1_GENESIS_FREEZE_DRIFT")
    require(genesis.get("scientific_status") == FROZEN_SCIENTIFIC_STATUS, "R11_R1_FROZEN_SCIENTIFIC_STATUS_DRIFT")
    require(genesis.get("semantic_guards", {}).get("final_holdout_payload_opened") is False, "R11_R1_GENESIS_HOLDOUT_GUARD_DRIFT")
    require(adoption.get("canonical_content_hash") == ADOPTION_CANONICAL_HASH, "R11_R1_ADOPTION_IDENTITY_DRIFT")
    require(seal.get("canonical_seal_sha256") == DATASET_SEAL_SHA256, "R11_R1_DATASET_SEAL_DRIFT")
    require(int(seal.get("holdout_payload_files_opened", -1)) == 0, "R11_R1_SEAL_HOLDOUT_OPENED")
    require(int(seal.get("network_reads", -1)) == 0, "R11_R1_SEAL_NETWORK_READ_DRIFT")
    require(lineage.get("lineage_identity_sha256") == DERIVED_CACHE_LINEAGE_SHA256, "R11_R1_CACHE_LINEAGE_DRIFT")
    require(lineage.get("dataset_seal_sha256") == DATASET_SEAL_SHA256, "R11_R1_CACHE_DATASET_BINDING_DRIFT")
    require(lineage.get("final_holdout_payload_opened") is False, "R11_R1_CACHE_HOLDOUT_GUARD_DRIFT")
    require(int(lineage.get("network_reads", -1)) == 0, "R11_R1_CACHE_NETWORK_READ_DRIFT")
    require(bootstrap.is_file(), "R11_R1_BOOTSTRAP_MISSING")
    require(sha256_file(bootstrap) == BOOTSTRAP_FILE_SHA256, "R11_R1_BOOTSTRAP_FILE_SHA_DRIFT")

    authority_files = {
        p.name: sha256_file(p)
        for p in sorted(a.glob("CB16_R11_G0_*.json"))
        if p.is_file()
    }
    authority_files[str(bootstrap.relative_to(g0_root))] = sha256_file(bootstrap)
    return lineage, {
        "genesis_identity_sha256": GENESIS_IDENTITY_SHA256,
        "adoption_canonical_content_hash": ADOPTION_CANONICAL_HASH,
        "dataset_seal_sha256": DATASET_SEAL_SHA256,
        "derived_cache_lineage_sha256": DERIVED_CACHE_LINEAGE_SHA256,
        "bootstrap_checkpoint_sha256": BOOTSTRAP_FILE_SHA256,
        "authority_file_hashes": authority_files,
    }


def _evenly_spaced(rows: Sequence[Any], count: int) -> list[Any]:
    rows = list(rows)
    if len(rows) <= count:
        return rows
    idx = np.linspace(0, len(rows) - 1, num=int(count), dtype=np.int64)
    return [rows[int(i)] for i in sorted(set(int(x) for x in idx))]


def _frame_split(t: int) -> str | None:
    t = int(t)
    if t + (H72 + TRAIN_VALIDATION_PURGE_HOURS) * HOUR_MS <= TRAIN_END_MS:
        return "TRAIN"
    if TRAIN_END_MS <= t < VALIDATION_END_MS and t + H72 * HOUR_MS <= FORBIDDEN_FINAL_START_MS:
        return "VALIDATION"
    return None


def select_candidate_frames(
    frames: Sequence[Any], *, train_target: int, validation_target: int, candidate_factor: int
) -> list[tuple[str, Any]]:
    train = [f for f in frames if _frame_split(int(f.decision_time_ms)) == "TRAIN"]
    val = [f for f in frames if _frame_split(int(f.decision_time_ms)) == "VALIDATION"]
    require(len(train) >= train_target, f"R11_R1_INSUFFICIENT_TRAIN_ANCHORS:{len(train)}<{train_target}")
    require(len(val) >= validation_target, f"R11_R1_INSUFFICIENT_VALIDATION_ANCHORS:{len(val)}<{validation_target}")
    train_candidates = _evenly_spaced(train, min(len(train), train_target * candidate_factor))
    val_candidates = _evenly_spaced(val, min(len(val), validation_target * candidate_factor))
    return [("TRAIN", f) for f in train_candidates] + [("VALIDATION", f) for f in val_candidates]


def encode_selected_frames(
    *, package_root: Path, device: str, selected: Sequence[tuple[str, Any]], batch_size: int
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    sensory = FrozenSensoryStackR10(package_root, device=device, verify_hashes=True)
    encoded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    frames = [f for _, f in selected]
    try:
        for start in range(0, len(frames), int(batch_size)):
            chunk = frames[start : start + int(batch_size)]
            batch = sensory.encode_frames(chunk)
            for i, frame in enumerate(chunk):
                encoded[int(frame.decision_time_ms)] = (
                    batch.operator48[i].copy(),
                    batch.medium48[i].copy(),
                    batch.ordered4h30[i].copy(),
                )
        assets = dict(sensory.verified_hashes)
    finally:
        del sensory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return encoded, {"frozen_sensory_assets_verified": True, "asset_receipts": assets}


def build_bounded_counterfactual_support(
    *,
    symbol: str,
    selected: Sequence[tuple[str, Any]],
    encoded: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    physics: FrozenPhysicsRuntimeR102,
    hourly_ts: np.ndarray,
    hourly_ohlcv: np.ndarray,
    funding: np.ndarray,
    train_target: int,
    validation_target: int,
) -> tuple[dict[str, ParentContextR102], dict[str, dict[str, Any]], list[CounterfactualBranchSampleR5], dict[str, Any]]:
    parents: dict[str, ParentContextR102] = {}
    states: dict[str, dict[str, Any]] = {}
    samples: list[CounterfactualBranchSampleR5] = []
    accepted = {"TRAIN": 0, "VALIDATION": 0}
    rejected = {"TRAIN": 0, "VALIDATION": 0}
    target = {"TRAIN": int(train_target), "VALIDATION": int(validation_target)}
    utility_values: list[float] = []

    for split, frame in selected:
        if accepted[split] >= target[split]:
            continue
        t = int(frame.decision_time_ms)
        require(t < FORBIDDEN_FINAL_START_MS, f"R11_R1_FORBIDDEN_DECISION_TIMESTAMP:{t}")
        require(t + (H72 - 1) * HOUR_MS < FORBIDDEN_FINAL_START_MS, f"R11_R1_FORBIDDEN_H72_FUTURE:{t}")
        op, med, riskctx = encoded[t]
        scenarios = build_parent_scenarios(
            physics,
            symbol=symbol,
            decision_time_ms=t,
            hourly_ts=hourly_ts,
            hourly_ohlcv=hourly_ohlcv,
            funding=funding,
            prehistory_hours=96,
        )
        scenario = next((x for x in scenarios if x["scenario"] == "CLEAN_FLAT_FULL"), None)
        if scenario is None or not bool(scenario["eligible_for_economic_evidence"]):
            rejected[split] += 1
            continue
        group_id = f"FUT:{symbol}:{t}"
        parent_id = f"R11R1:P:{symbol}:{t}:CLEAN_FLAT_FULL"
        future_hash = market_future_lineage_hash(symbol, t, hourly_ts, hourly_ohlcv, funding)
        parent = ParentContextR102(
            parent_id=parent_id,
            dependence_group_id=group_id,
            symbol=symbol,
            decision_time_ms=t,
            split=split,
            scenario="CLEAN_FLAT_FULL",
            operator48=tuple(float(x) for x in op),
            medium48=tuple(float(x) for x in med),
            account6=tuple(float(x) for x in scenario["account6"]),
            ordered4h30=tuple(float(x) for x in riskctx),
            current_mark=float(scenario["current_mark"]),
            snapshot_sha256=str(scenario["snapshot_sha256"]),
            eligible_for_economic_evidence=True,
            market_lineage_hash=future_hash,
        )
        state = {
            "parent_id": parent_id,
            "account_id": scenario["account_id"],
            "symbol": symbol,
            "decision_time_ms": t,
            "scenario": scenario["scenario"],
            "snapshot": scenario["snapshot"],
            "risk_authority": scenario["risk_authority"],
            "current_mark": float(scenario["current_mark"]),
            "snapshot_sha256": scenario["snapshot_sha256"],
        }
        candidate_samples: list[CounterfactualBranchSampleR5] = []
        complete = True
        for direction_v55, requested_risk in CANDIDATES_R102:
            branch = simulate_h72_branch(
                physics,
                parent=state,
                symbol=symbol,
                decision_time_ms=t,
                candidate_direction_v55=int(direction_v55),
                candidate_risk=float(requested_risk),
                hourly_ts=hourly_ts,
                hourly_ohlcv=hourly_ohlcv,
                funding=funding,
            )
            utility = branch.get("utility")
            if branch.get("status") != "MATURED" or utility is None or not math.isfinite(float(utility)):
                complete = False
                break
            sample = CounterfactualBranchSampleR5(
                parent_id=parent_id,
                student_context_object_id=parent.student_context_object_id,
                timestamp=t,
                context_features=parent.student_features,
                direction=int(direction_v55) - 1,
                requested_risk=float(requested_risk),
                realized_utility=float(utility),
                dependence_group_id=group_id,
                market_lineage_hash=future_hash,
            )
            sample.validate()
            candidate_samples.append(sample)
        if not complete or len(candidate_samples) != len(CANDIDATES_R102):
            rejected[split] += 1
            continue
        parents[parent_id] = parent
        states[parent_id] = state
        samples.extend(candidate_samples)
        utility_values.extend(float(x.realized_utility) for x in candidate_samples)
        accepted[split] += 1

    require(accepted["TRAIN"] >= train_target, f"R11_R1_TRAIN_COMPLETE_9BRANCH_SUPPORT_SHORTFALL:{accepted}")
    require(accepted["VALIDATION"] >= validation_target, f"R11_R1_VALIDATION_COMPLETE_9BRANCH_SUPPORT_SHORTFALL:{accepted}")
    require(len(samples) == (train_target + validation_target) * len(CANDIDATES_R102), "R11_R1_COUNTERFACTUAL_SAMPLE_COUNT_DRIFT")
    return parents, states, samples, {
        "candidate_action_count_per_parent": len(CANDIDATES_R102),
        "accepted_complete_parent_groups": accepted,
        "rejected_candidate_parent_groups": rejected,
        "counterfactual_branch_samples": len(samples),
        "utility_min": min(utility_values),
        "utility_max": max(utility_values),
        "utility_mean": float(np.mean(np.asarray(utility_values, dtype=np.float64))),
        "one_realization_is_not_a_correct_action_label": True,
        "all_teacher_parents_require_complete_9_branch_grid": True,
    }


def teacher_evidence_items(train_evidence: Sequence[Any], parents: Mapping[str, ParentContextR102]) -> list[EvidenceItemR11]:
    items: list[EvidenceItemR11] = []
    for e in train_evidence:
        if not bool(e.admission.admitted):
            continue
        p = parents[e.parent_id]
        payload = {
            "schema": "CB16_R10_2_EVIDENCE_PACKAGE_V1",
            "generation": 0,
            "parent_id": e.parent_id,
            "dependence_group_id": e.target_dependence_group_id,
            "student_context_object_id": e.student_context_object_id,
            "operator48": list(p.operator48),
            "medium48": list(p.medium48),
            "account6": list(p.account6),
            "direction_target_probs": list(e.direction_target_probs),
            "requested_risk_target": float(e.requested_risk_target),
            "action_laws": [asdict(x) for x in e.action_laws],
            "admission": asdict(e.admission),
            "teacher_protocol_hash": e.teacher_protocol_hash,
        }
        items.append(
            EvidenceItemR11(
                evidence_id=str(e.evidence_id),
                parent_snapshot_hash=str(p.snapshot_sha256),
                lineage_hash=str(e.content_hash),
                teacher_protocol_hash=str(e.teacher_protocol_hash),
                payload=payload,
            )
        )
    if len({x.evidence_id for x in items}) != len(items):
        raise RuntimeError("R11_R1_DUPLICATE_ADMITTED_EVIDENCE_ID")
    return items


def materialize_shadow_evidence_set(
    *, work_root: Path, g0_root: Path, train_evidence: Sequence[Any], parents: Mapping[str, ParentContextR102], base_policy_hash: str
) -> tuple[dict[str, Any], str]:
    store_root = work_root / "r11_evidence_store"
    store = EvidenceStoreR11(
        metadata_root=store_root / "metadata",
        payload_roots=[store_root / "payload0", store_root / "payload1"],
        segment_target_bytes=8 * 1024 * 1024,
        codec="zlib",
        sqlite_synchronous="FULL",
        read_only_source_roots=[g0_root],
    )
    try:
        items = teacher_evidence_items(train_evidence, parents)
        _refs, materialize = store.put_evidence(items)
        evidence_set = store.seal_evidence_set(
            "R11_G0_HISTORICAL_R1_SHADOW_TRAINING_EVIDENCE_SET",
            [x.evidence_id for x in items],
        )
        periodic = store.periodic_audit()
        require(bool(periodic.get("pass")), f"R11_R1_EVIDENCE_STORE_AUDIT_FAIL:{periodic}")
    finally:
        store.close()
    snapshot_core = {
        "schema": "CB16_R11_G0_HISTORICAL_R1_SHADOW_TRAINING_SNAPSHOT_V1",
        "canonical_generation_advanced": False,
        "base_policy_hash": base_policy_hash,
        "genesis_identity_sha256": GENESIS_IDENTITY_SHA256,
        "evidence_set_hash": evidence_set["evidence_set_hash"],
        "evidence_objects": int(evidence_set["object_count"]),
        "scientific_authority": "DIAGNOSTIC_SHADOW_ONLY",
    }
    snapshot_hash = sha256_obj(snapshot_core)
    snapshot = {**snapshot_core, "snapshot_hash": snapshot_hash}
    atomic_json(work_root / "R11_G0_HISTORICAL_R1_SHADOW_TRAINING_SNAPSHOT.json", snapshot)
    return {
        "materialization": asdict(materialize),
        "evidence_set": evidence_set,
        "periodic_audit": periodic,
        "shadow_training_snapshot": snapshot,
    }, snapshot_hash


def load_bootstrap_model(g0_root: Path, device: str):
    checkpoint = g0_root / "bootstrap" / "R11_G0_INITIAL_CHAMPION.pt"
    require(sha256_file(checkpoint) == BOOTSTRAP_FILE_SHA256, "R11_R1_BOOTSTRAP_CHANGED_BEFORE_MODEL_LOAD")
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = obj.get("state_dict") if isinstance(obj, Mapping) and "state_dict" in obj else obj
    require(isinstance(state, Mapping), "R11_R1_BOOTSTRAP_STATE_DICT_MISSING")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    model.load_state_dict(state, strict=True)
    semantic = model_state_semantic_sha256(model)
    require(semantic == BOOTSTRAP_SEMANTIC_SHA256, f"R11_R1_BOOTSTRAP_SEMANTIC_DRIFT:{semantic}")
    require(policy_hash_r11(model) == BOOTSTRAP_SEMANTIC_SHA256, "R11_R1_R11_POLICY_HASH_NOT_BOOTSTRAP_SEMANTIC")
    model.to(device)
    return model


def run_on_policy_probe(
    *, work_root: Path, model, base_policy_hash: str, parents: Mapping[str, ParentContextR102], states: Mapping[str, Mapping[str, Any]], physics, market_cache, device: str, trace_groups: int
) -> dict[str, Any]:
    trace_parents = {pid: p for pid, p in parents.items() if p.split == "TRAIN"}
    trace_states = {pid: states[pid] for pid in trace_parents}
    lake = ShardedExperienceLake(work_root / "on_policy_trace_lake", shards=2, synchronous="FULL")
    try:
        with TraceRuntimeR11(physics=physics, market_cache=market_cache, max_workers=min(4, max(1, int(trace_groups)))) as runtime:
            receipt = run_real_on_policy_trace_r11(
                model=model,
                policy_hash=base_policy_hash,
                generation=0,
                parents=trace_parents,
                parent_states=trace_states,
                trace_runtime=runtime,
                lake=lake,
                device=device,
                max_groups=int(trace_groups),
            )
        lake_audit = lake.audit(verify_payloads=True)
    finally:
        lake.close()
    require(receipt["trace_count"] == int(trace_groups), f"R11_R1_TRACE_COUNT_DRIFT:{receipt}")
    require(receipt["matured"] == receipt["trace_count"], f"R11_R1_ON_POLICY_TRACE_NOT_FULLY_MATURED:{receipt}")
    ids = [x["CausalTraceID"] for x in receipt["traces"]]
    require(len(ids) == len(set(ids)), "R11_R1_DUPLICATE_CAUSAL_TRACE_ID")
    require(bool(lake_audit.get("pass")), f"R11_R1_TRACE_LAKE_AUDIT_FAIL:{lake_audit}")
    return {
        "receipt": receipt,
        "experience_lake_audit": lake_audit,
        "causal_trace_ids_unique": True,
        "realized_outcome_is_stochastic_sample_not_correct_action_label": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--train-groups", type=int, default=64)
    ap.add_argument("--validation-groups", type=int, default=16)
    ap.add_argument("--candidate-factor", type=int, default=2)
    ap.add_argument("--sensory-batch-size", type=int, default=8)
    ap.add_argument("--teacher-workers", type=int, default=4)
    ap.add_argument("--teacher-block-targets", type=int, default=8)
    ap.add_argument("--trace-groups", type=int, default=8)
    args = ap.parse_args()

    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    g0_root = args.g0_root.resolve()
    package_root = args.package_root.resolve()
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R1_WORK_ROOT_OVERLAPS_CANONICAL_G0")
    require(int(args.train_groups) >= 32, "R11_R1_TRAIN_GROUPS_BELOW_TEACHER_MINIMUM")
    require(int(args.validation_groups) >= 8, "R11_R1_VALIDATION_GROUPS_BELOW_REQUIRED_PROBE_SUPPORT")
    require(0 < int(args.trace_groups) <= int(args.train_groups), "R11_R1_INVALID_TRACE_GROUP_COUNT")
    require(args.symbol == "BTCUSDT", "R11_R1_SCOPE_IS_INTENTIONALLY_BTCUSDT_ONLY")

    repo = verify_repo_seed()
    runtime = runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R1_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_authority_before = dict(g0_identity["authority_file_hashes"])

    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get(args.symbol)
    market_cache.assert_read_only()
    require(int(market.open_time_ms[-1]) < FORBIDDEN_FINAL_START_MS, "R11_R1_MARKET_CACHE_REACHES_FINAL_HOLDOUT")
    per_asset = next((x for x in lineage.get("per_asset", []) if x.get("symbol") == args.symbol), None)
    require(per_asset is not None, f"R11_R1_LINEAGE_SYMBOL_MISSING:{args.symbol}")
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(anchor_path.is_file(), f"R11_R1_ANCHOR_FILE_MISSING:{anchor_path}")
    require(sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R1_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = select_candidate_frames(
        frames,
        train_target=int(args.train_groups),
        validation_target=int(args.validation_groups),
        candidate_factor=int(args.candidate_factor),
    )
    encoded, sensory_receipt = encode_selected_frames(
        package_root=package_root,
        device=args.device,
        selected=selected,
        batch_size=int(args.sensory_batch_size),
    )

    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, parent_states, samples, support = build_bounded_counterfactual_support(
        symbol=args.symbol,
        selected=selected,
        encoded=encoded,
        physics=physics,
        hourly_ts=market.open_time_ms,
        hourly_ohlcv=market.ohlcv,
        funding=market.funding_rate,
        train_target=int(args.train_groups),
        validation_target=int(args.validation_groups),
    )
    del encoded
    gc.collect()

    train_evidence, validation_evidence, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=TRAIN_TEACHER_CONFIG_R102,
        val_config=VAL_TEACHER_CONFIG_R102,
        workers=int(args.teacher_workers),
        block_targets=int(args.teacher_block_targets),
    )
    train_summary = evidence_summary(train_evidence)
    validation_summary = evidence_summary(validation_evidence)
    require(train_summary["admitted_dependence_groups"] >= 32, f"R11_R1_TRAIN_TEACHER_SUPPORT_NOT_READY:{train_summary}")
    require(validation_summary["admitted_dependence_groups"] >= 8, f"R11_R1_VALIDATION_TEACHER_SUPPORT_NOT_READY:{validation_summary}")

    model = load_bootstrap_model(g0_root, args.device)
    base_policy_hash = policy_hash_r11(model)
    trace_probe = run_on_policy_probe(
        work_root=work_root,
        model=model,
        base_policy_hash=base_policy_hash,
        parents=parents,
        states=parent_states,
        physics=physics,
        market_cache=market_cache,
        device=args.device,
        trace_groups=int(args.trace_groups),
    )

    evidence_store, shadow_snapshot_hash = materialize_shadow_evidence_set(
        work_root=work_root,
        g0_root=g0_root,
        train_evidence=train_evidence,
        parents=parents,
        base_policy_hash=base_policy_hash,
    )

    campaign = prepare_evidence_campaign_r11(
        train_evidence=train_evidence,
        validation_evidence=validation_evidence,
        parents=parents,
        device=args.device,
    )
    train_target_audit = validate_teacher_targets_r11(campaign.train)
    validation_target_audit = validate_teacher_targets_r11(campaign.validation)
    independent_before = independent_loss_breakdown_r11(model, campaign.validation)
    behavior_before = capture_behavior_r11(model, campaign.validation)

    trainer = IntegratedTrainingRuntimeR11(device=args.device)
    training_receipt = trainer.train_challenger(
        model=model,
        campaign=campaign,
        generation=0,
        snapshot_hash=shadow_snapshot_hash,
        receipt_dir=work_root / "shadow_training",
    )
    independent_after = independent_loss_breakdown_r11(model, campaign.validation)
    behavior_after = capture_behavior_r11(model, campaign.validation)
    behavior_delta = behavior_delta_r11(behavior_before, behavior_after)
    feedback = audit_training_feedback_r11(
        training_receipt=training_receipt,
        independent_before=independent_before,
        independent_after=independent_after,
        behavior_delta=behavior_delta,
    )

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R1_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _lineage_after, g0_identity_after = verify_g0_authority(g0_root)
    require(g0_identity_after["authority_file_hashes"] == g0_authority_before, "R11_R1_G0_AUTHORITY_MUTATED")
    market_cache.assert_read_only()

    result = {
        "schema": SCHEMA,
        "status": "R11_G0_HISTORICAL_LEARNING_CONNECTIVITY_PASS",
        "repo": repo,
        "runtime": runtime,
        "authority": {
            **{k: v for k, v in g0_identity.items() if k != "authority_file_hashes"},
            "frozen_scientific_status_before": FROZEN_SCIENTIFIC_STATUS,
            "frozen_scientific_status_after": FROZEN_SCIENTIFIC_STATUS,
            "package_frozen_authority_hashes_before": frozen_before,
            "package_frozen_authority_hashes_after": frozen_after,
            "g0_authority_hashes_unchanged": True,
        },
        "historical_scope": {
            "symbol": args.symbol,
            "scope_role": "BOUNDED_CONNECTIVITY_DIAGNOSTIC__NOT_MARKET_RULE",
            "train_complete_parent_groups": int(args.train_groups),
            "validation_complete_parent_groups": int(args.validation_groups),
            "candidate_actions_per_parent": len(CANDIDATES_R102),
            "anchor_file": str(anchor_path),
            "anchor_sha256": str(per_asset["anchors_sha256"]),
            "market_cache_stats": asdict(market_cache.stats()),
            "final_holdout_start_ms": int(FORBIDDEN_FINAL_START_MS),
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "network_reads_by_r1": 0,
        },
        "sensory": sensory_receipt,
        "counterfactual_support": support,
        "teacher": {
            "runtime_stats": asdict(teacher_stats),
            "train": train_summary,
            "validation": validation_summary,
            "probabilistic_target_semantics": "DISTRIBUTIONAL_EVIDENCE__NOT_HINDSIGHT_BEST_ACTION_LABEL",
        },
        "on_policy_realization_probe": trace_probe,
        "r11_evidence_store": evidence_store,
        "training_targets": {
            "train": train_target_audit,
            "validation": validation_target_audit,
        },
        "training": {
            "base_policy_semantic_sha256": base_policy_hash,
            "shadow_snapshot_hash": shadow_snapshot_hash,
            "challenger_semantic_sha256": training_receipt["challenger_semantic_sha256"],
            "optimizer": training_receipt["optimizer"],
            "optimizer_steps": training_receipt["optimizer_steps"],
            "epochs": training_receipt["epochs"],
            "batch_size": training_receipt["batch_size"],
            "amp": training_receipt["amp"],
            "dtype": training_receipt["dtype"],
            "parameter_l2_delta": training_receipt["parameter_l2_delta"],
            "gradient_owner_set_last_step": training_receipt["gradient_owner_set_last_step"],
            "gradient_group_norms_last_step": training_receipt["gradient_group_norms_last_step"],
            "update_group_norms": training_receipt["update_group_norms"],
            "validation_before": training_receipt["validation_before"],
            "validation_after": training_receipt["validation_after"],
            "external_frozen_organ_gradients": training_receipt["external_frozen_organ_gradients"],
            "teacher_future_autograd": training_receipt["teacher_future_autograd"],
        },
        "independent_formula_check": {
            "formula": "weighted_soft_cross_entropy(direction_distribution)+weighted_SmoothL1(requested_risk,beta=0.05)",
            "before": independent_before,
            "after": independent_after,
            "runtime_receipt_agreement": "PASS",
        },
        "feedback_diagnostics": feedback,
        "feedback_chain": {
            "historical_market_plus_account_state": "PASS",
            "student_belief_and_intent": "PASS",
            "frozen_supervisor_and_physics_realization": "PASS",
            "counterfactual_distributional_credit_support": "PASS",
            "probabilistic_teacher_evidence": "PASS",
            "r11_immutable_evidence_materialization": "PASS",
            "shadow_frozen_training_snapshot": "PASS",
            "loss_formula": "PASS",
            "authorized_gradient_path": "PASS",
            "parameter_delta": "PASS",
            "measurable_behavior_delta": "PASS",
            "causal_trace_id_on_policy_path": "PASS",
            "single_realized_future_used_as_correct_action_label": False,
        },
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "tournament_run": False,
            "canonical_g0_state_mutated": False,
            "scientific_verdict_created": False,
            "market_information_qualified": False,
            "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "live_capital": False,
            "production_trading": False,
        },
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": result["status"],
                "execution_head": repo["execution_head"],
                "base_policy": base_policy_hash,
                "challenger": training_receipt["challenger_semantic_sha256"],
                "learning_direction": feedback["learning_direction"],
                "validation_loss_before": feedback["validation_loss_before"],
                "validation_loss_after": feedback["validation_loss_after"],
                "parameter_l2_delta": feedback["parameter_l2_delta"],
                "behavior_delta": feedback["behavior_delta"],
                "trace_count": trace_probe["receipt"]["trace_count"],
                "admitted_train_groups": train_summary["admitted_dependence_groups"],
                "admitted_validation_groups": validation_summary["admitted_dependence_groups"],
                "canonical_generation_advanced": False,
                "scientific_verdict_created": False,
                "final_holdout_payload_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
