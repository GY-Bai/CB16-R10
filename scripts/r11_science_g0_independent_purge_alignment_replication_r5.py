#!/usr/bin/env python3
from __future__ import annotations

"""R11 Science G0 R5 — independent purge-gap state-alignment replication.

The scientific gate is committed before this executor exists. R5 first recreates
six fixed training arms from the already-observed R4 training support, freezes
their parameter states, and only then evaluates them on the metadata-qualified
2024-12-28T08:00:00Z purge-gap support. Candidate outcomes never enter an
optimizer step. A negative result can falsify the R4 state-alignment story; a
positive result is mechanics-only because all ten symbols share one clock block.
"""

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    R41_SCENARIOS,
    shuffle_reduced_targets_by_future_group_r41,
)
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.independent_purge_alignment_replication_r5 import (
    R5_CANDIDATE_TIMESTAMP_MS,
    R5_RUNTIME,
    R5_SHIFTS,
    R5_SYMBOLS,
    adjudicate_alignment_replication_r5,
    compile_validation_targets_only_r5,
)
from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_campaign import frozen_authority_hashes
from cb16_local_opt.r102_evidence_cache import load_teacher_samples
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    PreparedCampaignR11,
    PreparedEvidenceR11,
    policy_hash_r11,
)
from scripts import r11_science_g0_canonical_historical_learning_baseline_r4 as r4
from scripts import r11_science_g0_geometry_r2_1 as r21

SCHEMA = "CB16_R11_SCIENCE_G0_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_RESULT_V1"
PREREG_COMMIT = "12de52ec29e3224df27558e8e157ad9ccd1d170c"
R50_ADJUDICATION_BLOB = "41506938c433e34514834bb07888c37e5a304c99"
R5_GATE_BLOB = "75b77a0184263abc6f808fb5a7fc77b8cc1dd74e"
R41_ADJUDICATION_BLOB = "3202c28c65a2f88030ac549f21e91be4d0e6403f"
FROZEN_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
R4_REFERENCE = {
    "champion": {"loss": 1.208237886428833, "direction_loss": 1.1262524127960205, "sizing_loss": 0.0819854661822319},
    "aligned": {"loss": 1.0994198322296143, "direction_loss": 1.0919666290283203, "sizing_loss": 0.007453371305018663},
    "shuffle_total": {
        1: 1.097920298576355,
        7: 1.1003360748291016,
        13: 1.1113344430923462,
        23: 1.098488450050354,
        31: 1.101085901260376,
    },
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


def metric_close(a: Mapping[str, Any], b: Mapping[str, Any], atol: float = 1e-7) -> bool:
    return all(abs(float(a[k]) - float(b[k])) <= atol for k in ("loss", "direction_loss", "sizing_loss"))


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def verify_repo_contract() -> dict[str, Any]:
    require(subprocess.call(["git", "merge-base", "--is-ancestor", PREREG_COMMIT, "HEAD"]) == 0,
            "R11_R5_NOT_DESCENDED_FROM_PREREGISTRATION")
    exact = {
        "semantic_freeze": ("authority/rearchitecture_r11/CB16_SEMANTIC_FREEZE_V1.json", r4.SEMANTIC_FREEZE_BLOB),
        "r5_0_adjudication": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_INDEPENDENT_SUPPORT_INVENTORY_R5_0_ADJUDICATION_V1.json", R50_ADJUDICATION_BLOB),
        "r5_gate": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_GATE_V1.json", R5_GATE_BLOB),
        "r4_1_adjudication": ("authority/rearchitecture_r11/CB16_R11_SCIENCE_G0_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_ADJUDICATION_V1.json", R41_ADJUDICATION_BLOB),
        "typed_brain": ("cb16_local_opt/typed_central_brain_r10.py", r4.TYPED_CENTRAL_BRAIN_BLOB),
        "training_runtime": ("cb16_local_opt/training_runtime_r11.py", r4.TRAINING_RUNTIME_BLOB),
        "training_integration": ("cb16_local_opt/training_integration_r11.py", r4.TRAINING_INTEGRATION_BLOB),
        "teacher_runtime": ("cb16_local_opt/teacher_runtime_r11.py", r4.TEACHER_RUNTIME_BLOB),
        "probabilistic_teacher": ("cb16_local_opt/probabilistic_teacher_r6.py", r4.PROBABILISTIC_TEACHER_BLOB),
        "r4_script": ("scripts/r11_science_g0_canonical_historical_learning_baseline_r4.py", "40454165a6d818a11242351b74af8d9cb4b26f77"),
        "r41_helper": ("cb16_local_opt/canonical_state_alignment_falsification_r41.py", "06117ec664adee61db916779474387230ddb4164"),
        "r21_support": ("scripts/r11_science_g0_geometry_r2_1.py", r4.R21_SCRIPT_BLOB),
        "legacy_evidence_cache": ("cb16_local_opt/r102_evidence_cache.py", "91fb73565ec60f66bf4232957f9b9b2f88cc3cfe"),
    }
    observed = {}
    for name, (path, expected) in exact.items():
        got = git("rev-parse", f"HEAD:{path}")
        require(got == expected, f"R11_R5_IMMUTABLE_BLOB_DRIFT:{name}:{got}")
        observed[name] = got
    return {"execution_head": git("rev-parse", "HEAD"), "preregistration_commit": PREREG_COMMIT, "immutable_blobs": observed}


def validate_training_receipt(receipt: Mapping[str, Any], *, expected_before: Mapping[str, Any]) -> None:
    require(int(receipt["generation_seed"]) == 24680, "R11_R5_TRAIN_SEED_DRIFT")
    require(int(receipt["epochs"]) == 12 and int(receipt["batch_size"]) == 512, "R11_R5_TRAIN_RULE_DRIFT")
    require(float(receipt["lr"]) == 3e-4 and float(receipt["weight_decay"]) == 1e-4, "R11_R5_OPTIMIZER_RULE_DRIFT")
    require(receipt["optimizer"] == "AdamW_FP32" and receipt["amp"] is False, "R11_R5_NUMERIC_RULE_DRIFT")
    require(frozenset(receipt["gradient_owner_set_last_step"]) == AUTHORIZED_GRADIENT_OWNERS_R11,
            "R11_R5_GRADIENT_OWNER_SET_DRIFT")
    require(all(math.isfinite(float(v)) and float(v) > 0 for v in receipt["gradient_group_norms_last_step"].values()),
            "R11_R5_NONPOSITIVE_GRADIENT_OWNER")
    require(all(math.isfinite(float(v)) and float(v) > 0 for v in receipt["update_group_norms"].values()),
            "R11_R5_NONPOSITIVE_UPDATE_OWNER")
    require(metric_close(receipt["validation_before"], expected_before), "R11_R5_OLD_VALIDATION_BASELINE_DRIFT")


def build_and_freeze_training_arms(*, g0_root: Path, package_root: Path, work_root: Path, device: str) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Recreate known R4/R4.1 arms; candidate support is not part of this campaign."""
    lineage, _ = r4.r1.verify_g0_authority(g0_root)
    market_cache = MarketRuntimeCacheR11(g0_root)
    market = market_cache.get("BTCUSDT")
    market_cache.assert_read_only()
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == "BTCUSDT")
    anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(r4.r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "R11_R5_R4_ANCHOR_SHA_DRIFT")
    frames = r4.load_anchor_frames("BTCUSDT", anchor_path)
    selected = r4.r1.select_candidate_frames(frames, train_target=48, validation_target=12, candidate_factor=2)
    # Explicitly prove the purge candidate is absent from the optimizer/evaluation support.
    require(all(int(frame.decision_time_ms) != R5_CANDIDATE_TIMESTAMP_MS for _, frame in selected),
            "R11_R5_CANDIDATE_LEAKED_INTO_R4_CAMPAIGN")
    encoded, sensory_receipt = r4.r1.encode_selected_frames(
        package_root=package_root, device=device, selected=selected, batch_size=8
    )
    physics = FrozenPhysicsRuntimeR102.load(package_root)
    parents, samples, support = r21.build_multi_account_counterfactual_support(
        symbol="BTCUSDT", selected=selected, encoded=encoded, physics=physics,
        hourly_ts=market.open_time_ms, hourly_ohlcv=market.ohlcv, funding=market.funding_rate,
        train_target_groups=48, validation_target_groups=12,
    )
    del encoded
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    train_evidence, validation_evidence, teacher_stats = r4.compile_teacher_evidence_r11(
        samples=samples, parents=parents,
        train_config=r4.R11_TRAIN_TEACHER_CONFIG, val_config=r4.R11_VALIDATION_TEACHER_CONFIG,
        workers=4, block_targets=32,
    )
    campaign = r4.prepare_evidence_campaign_r11(
        train_evidence=train_evidence, validation_evidence=validation_evidence, parents=parents, device=device
    )
    require(r4.validate_teacher_targets_r11(campaign.train)["independent_dependence_groups"] == 48,
            "R11_R5_R4_TRAIN_GROUP_DRIFT")
    require(r4.validate_teacher_targets_r11(campaign.validation)["independent_dependence_groups"] == 12,
            "R11_R5_R4_VALIDATION_GROUP_DRIFT")

    champion = r4.r1.load_bootstrap_model(g0_root, device)
    champion_hash = policy_hash_r11(champion)
    baseline = IntegratedTrainingRuntimeR11(device=device)
    champion_validation = baseline.evaluation_runtime.evaluate(champion, campaign.validation, use_cache=True)
    require(metric_close(champion_validation, R4_REFERENCE["champion"]), "R11_R5_R4_CHAMPION_REFERENCE_DRIFT")
    champion_state = clone_state(champion)
    snapshot = r4.build_shadow_snapshot_r4(
        champion_policy_hash=champion_hash,
        train_evidence_hash=campaign.train.evidence_hash,
        validation_evidence_hash=campaign.validation.evidence_hash,
        train_teacher_protocol_hash=r4.R11_TRAIN_TEACHER_CONFIG.content_hash,
        validation_teacher_protocol_hash=r4.R11_VALIDATION_TEACHER_CONFIG.content_hash,
        train_parent_ids=campaign.train.parent_ids,
        train_dependence_group_ids=campaign.train.dependence_group_ids,
        validation_parent_ids=campaign.validation.parent_ids,
        validation_dependence_group_ids=campaign.validation.dependence_group_ids,
    )

    states: dict[str, dict[str, torch.Tensor]] = {"G0_CHAMPION": champion_state}
    receipts: dict[str, Any] = {
        "champion_policy_hash": champion_hash,
        "champion_old_validation": champion_validation,
        "r4_support": support,
        "r4_sensory": sensory_receipt,
        "r4_teacher_stats": str(teacher_stats),
        "r4_train_evidence_hash": campaign.train.evidence_hash,
        "r4_validation_evidence_hash": campaign.validation.evidence_hash,
    }

    aligned = copy.deepcopy(champion)
    aligned_trainer = IntegratedTrainingRuntimeR11(device=device)
    aligned_receipt = aligned_trainer.train_challenger(
        model=aligned, campaign=campaign, generation=0, snapshot_hash=snapshot["snapshot_hash"],
        receipt_dir=work_root / "training_arms" / "aligned",
    )
    validate_training_receipt(aligned_receipt, expected_before=champion_validation)
    require(metric_close(aligned_receipt["validation_after"], R4_REFERENCE["aligned"]), "R11_R5_R4_ALIGNED_REFERENCE_DRIFT")
    states["ALIGNED"] = clone_state(aligned)
    receipts["ALIGNED"] = {"training_receipt": aligned_receipt, "policy_hash": policy_hash_r11(aligned)}

    for shift in R5_SHIFTS:
        shuffled_train, shuffle_receipt = shuffle_reduced_targets_by_future_group_r41(campaign.train, shift=int(shift))
        arm_campaign = PreparedCampaignR11(train=shuffled_train, validation=campaign.validation)
        model = copy.deepcopy(champion)
        trainer = IntegratedTrainingRuntimeR11(device=device)
        receipt = trainer.train_challenger(
            model=model, campaign=arm_campaign, generation=0,
            snapshot_hash=f"R5_R41_REPRO_{int(shift)}_{snapshot['snapshot_hash']}",
            receipt_dir=work_root / "training_arms" / f"shuffle_{int(shift):02d}",
        )
        validate_training_receipt(receipt, expected_before=champion_validation)
        require(abs(float(receipt["validation_after"]["loss"]) - R4_REFERENCE["shuffle_total"][int(shift)]) <= 1e-7,
                f"R11_R5_R41_SHUFFLE_REFERENCE_DRIFT:{shift}")
        name = f"SHUFFLE_{int(shift)}"
        states[name] = clone_state(model)
        receipts[name] = {"shuffle_receipt": shuffle_receipt, "training_receipt": receipt, "policy_hash": policy_hash_r11(model)}
        del model, trainer, arm_campaign, shuffled_train
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    require(set(states) == {"G0_CHAMPION", "ALIGNED", *(f"SHUFFLE_{x}" for x in R5_SHIFTS)}, "R11_R5_ARM_SET_DRIFT")
    # Freeze causal boundary: all arm tensors now reside on CPU before candidate Teacher compilation.
    arm_hashes = {name: receipts.get(name, {}).get("policy_hash", champion_hash) for name in states}
    receipts["arm_freeze"] = {
        "candidate_support_used_in_optimizer_steps": False,
        "candidate_teacher_targets_compiled_before_arm_freeze": False,
        "arm_policy_hashes": arm_hashes,
        "arm_count": len(states),
    }
    return states, receipts


def support_not_ready(*, output: Path, reason: str, repo: Mapping[str, Any], training_receipts: Mapping[str, Any], semantic_guards: Mapping[str, Any]) -> int:
    result = {
        "schema": SCHEMA,
        "status": "R11_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_SUPPORT_NOT_READY",
        "runtime": R5_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": dict(repo),
        "training_arms": training_receipts,
        "support_readiness": {"ready": False, "reason": reason},
        "alignment_adjudication_performed": False,
        "semantic_guards": dict(semantic_guards),
        "next_legal_step": "STOP_AT_SUPPORT_LIMIT__DO_NOT_TUNE_ON_OR_REPLACE_PURGE_SUPPORT_POST_HOC",
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "reason": reason}, indent=2, sort_keys=True))
    return 0


def load_legacy_train_support(r104_root: Path) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    manifest_path = r104_root / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    require(manifest_path.is_file(), "R11_R5_R104_MANIFEST_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("symbols") == list(R5_SYMBOLS), "R11_R5_R104_SYMBOL_SET_DRIFT")
    require(int(manifest.get("stride_hours", -1)) == 256, "R11_R5_R104_STRIDE_DRIFT")
    require(manifest.get("final_holdout_2025_09_accessed") is False, "R11_R5_R104_HOLDOUT_GUARD_DRIFT")
    cache_root = r104_root / "evidence_cache"
    parent_path = cache_root / Path(str(manifest["parents_file"])).name
    branch_path = cache_root / Path(str(manifest["branches_file"])).name
    require(parent_path.is_file() and branch_path.is_file(), "R11_R5_R104_EVIDENCE_FILES_MISSING")
    require(r4.r1.sha256_file(parent_path) == str(manifest["parents_sha256"]), "R11_R5_R104_PARENTS_SHA_DRIFT")
    require(r4.r1.sha256_file(branch_path) == str(manifest["branches_sha256"]), "R11_R5_R104_BRANCHES_SHA_DRIFT")
    parents, samples = load_teacher_samples(parent_path, branch_path)
    train_parents = {pid: p for pid, p in parents.items() if p.split == "TRAIN"}
    train_ids = set(train_parents)
    train_samples = [x for x in samples if x.parent_id in train_ids]
    require(bool(train_parents) and bool(train_samples), "R11_R5_EMPTY_LEGACY_TRAIN_SUPPORT")
    require(all(p.decision_time_ms < R5_CANDIDATE_TIMESTAMP_MS for p in train_parents.values()),
            "R11_R5_LEGACY_TRAIN_SUPPORT_REACHES_CANDIDATE")
    return train_parents, train_samples, {
        "manifest": str(manifest_path),
        "parent_file": str(parent_path),
        "branch_file": str(branch_path),
        "train_parent_contexts": len(train_parents),
        "train_branch_samples": len(train_samples),
        "train_dependence_groups": len({p.dependence_group_id for p in train_parents.values()}),
    }


def build_candidate_support(*, g0_root: Path, package_root: Path, lineage: Mapping[str, Any], device: str) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    frames_by_symbol: dict[str, Any] = {}
    anchor_receipts: dict[str, Any] = {}
    for symbol in R5_SYMBOLS:
        per_asset = next((x for x in lineage["per_asset"] if x["symbol"] == symbol), None)
        require(per_asset is not None, f"R11_R5_LINEAGE_SYMBOL_MISSING:{symbol}")
        anchor_path = g0_root / "market_cache" / str(per_asset["anchors_file"])
        require(r4.r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), f"R11_R5_ANCHOR_SHA_DRIFT:{symbol}")
        frames = r4.load_anchor_frames(symbol, anchor_path)
        matches = [f for f in frames if int(f.decision_time_ms) == R5_CANDIDATE_TIMESTAMP_MS]
        if len(matches) != 1:
            raise RuntimeError(f"R11_R5_SUPPORT_NOT_READY_CANDIDATE_FRAME_COUNT:{symbol}:{len(matches)}")
        frames_by_symbol[symbol] = matches[0]
        anchor_receipts[symbol] = {"anchor_file": str(anchor_path), "anchor_sha256": str(per_asset["anchors_sha256"])}

    ordered_frames = [frames_by_symbol[s] for s in R5_SYMBOLS]
    sensory = FrozenSensoryStackR10(package_root, device=device, verify_hashes=True)
    try:
        batch = sensory.encode_frames(ordered_frames)
        encoded = {
            symbol: (batch.operator48[i].copy(), batch.medium48[i].copy(), batch.ordered4h30[i].copy())
            for i, symbol in enumerate(R5_SYMBOLS)
        }
        sensory_assets = dict(sensory.verified_hashes)
    finally:
        del sensory
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    physics = FrozenPhysicsRuntimeR102.load(package_root)
    market_cache = MarketRuntimeCacheR11(g0_root)
    parents: dict[str, Any] = {}
    samples: list[Any] = []
    per_symbol_support: dict[str, Any] = {}
    for symbol in R5_SYMBOLS:
        market = market_cache.get(symbol)
        require(int(market.open_time_ms[-1]) < r4.r1.FORBIDDEN_FINAL_START_MS, f"R11_R5_MARKET_CACHE_REACHES_HOLDOUT:{symbol}")
        try:
            p, s, support = r21.build_multi_account_counterfactual_support(
                symbol=symbol,
                selected=[("VALIDATION", frames_by_symbol[symbol])],
                encoded={R5_CANDIDATE_TIMESTAMP_MS: encoded[symbol]},
                physics=physics,
                hourly_ts=market.open_time_ms,
                hourly_ohlcv=market.ohlcv,
                funding=market.funding_rate,
                train_target_groups=0,
                validation_target_groups=1,
            )
        except RuntimeError as exc:
            if str(exc).startswith("R11_R2_1_VALIDATION_GROUP_SHORTFALL"):
                raise RuntimeError(f"R11_R5_SUPPORT_NOT_READY_INCOMPLETE_FUTURE:{symbol}") from exc
            raise
        if len(p) != len(R41_SCENARIOS) or len(s) != len(R41_SCENARIOS) * 9:
            raise RuntimeError(f"R11_R5_SUPPORT_NOT_READY_SCENARIO_OR_BRANCH_COUNT:{symbol}:{len(p)}:{len(s)}")
        if {x.scenario for x in p.values()} != set(R41_SCENARIOS):
            raise RuntimeError(f"R11_R5_SUPPORT_NOT_READY_SCENARIO_SET:{symbol}")
        expected_gid = f"FUT:{symbol}:{R5_CANDIDATE_TIMESTAMP_MS}"
        if {x.dependence_group_id for x in p.values()} != {expected_gid}:
            raise RuntimeError(f"R11_R5_CANDIDATE_GROUP_ID_DRIFT:{symbol}")
        parents.update(p)
        samples.extend(s)
        per_symbol_support[symbol] = support
    market_cache.assert_read_only()
    return parents, samples, {
        "candidate_timestamp_ms": R5_CANDIDATE_TIMESTAMP_MS,
        "candidate_timestamp": "2024-12-28T08:00:00Z",
        "symbols": list(R5_SYMBOLS),
        "nominal_future_groups": len(R5_SYMBOLS),
        "macro_clock_blocks": 1,
        "parent_contexts": len(parents),
        "branch_samples": len(samples),
        "anchors": anchor_receipts,
        "sensory_assets": sensory_assets,
        "per_symbol_support": per_symbol_support,
    }


def evaluate_states(*, states: Mapping[str, Mapping[str, torch.Tensor]], g0_root: Path, candidate_evidence: list[Any], candidate_parents: Mapping[str, Any], device: str) -> dict[str, Any]:
    full = PreparedEvidenceR11.from_evidence(candidate_evidence, candidate_parents, device=device)
    require(full.rows == 60, f"R11_R5_PREPARED_CANDIDATE_ROW_COUNT_DRIFT:{full.rows}")
    per_symbol_prepared = {}
    for symbol in R5_SYMBOLS:
        subset = [e for e in candidate_evidence if candidate_parents[e.parent_id].symbol == symbol]
        require(len(subset) == 6, f"R11_R5_PER_SYMBOL_TEACHER_ROW_COUNT_DRIFT:{symbol}:{len(subset)}")
        per_symbol_prepared[symbol] = PreparedEvidenceR11.from_evidence(subset, candidate_parents, device=device)

    evaluator = IntegratedTrainingRuntimeR11(device=device).evaluation_runtime
    out: dict[str, Any] = {}
    for name in ("G0_CHAMPION", "ALIGNED", *(f"SHUFFLE_{x}" for x in R5_SHIFTS)):
        model = r4.r1.load_bootstrap_model(g0_root, device)
        model.load_state_dict(states[name], strict=True)
        full_metrics = evaluator.evaluate(model, full, use_cache=True)
        symbol_metrics = {
            symbol: evaluator.evaluate(model, per_symbol_prepared[symbol], use_cache=True)
            for symbol in R5_SYMBOLS
        }
        out[name] = {
            "policy_hash": policy_hash_r11(model),
            "candidate": full_metrics,
            "per_symbol": symbol_metrics,
        }
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--r104-root", type=Path, default=Path(os.environ.get("CB16_R104_ROOT", "/cb16/runtime/r104")))
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    g0_root = args.g0_root.resolve(); package_root = args.package_root.resolve(); r104_root = args.r104_root.resolve()
    work_root = args.work_root.resolve(); output = args.output.resolve(); work_root.mkdir(parents=True, exist_ok=True)
    require(work_root != g0_root and g0_root not in work_root.parents, "R11_R5_WORK_ROOT_OVERLAPS_G0")
    require(tuple(R5_SHIFTS) == (1, 7, 13, 23, 31), "R11_R5_SHIFT_PROTOCOL_DRIFT")
    require(len(R5_SYMBOLS) == 10, "R11_R5_SYMBOL_PROTOCOL_DRIFT")

    repo = verify_repo_contract()
    runtime = r4.r1.runtime_identity(args.device)
    require(runtime["python_series"] == [3, 10], f"R11_R5_PYTHON_SERIES_DRIFT:{runtime['python_series']}")
    lineage, g0_identity = r4.r1.verify_g0_authority(g0_root)
    frozen_before = frozen_authority_hashes(package_root)
    g0_before = dict(g0_identity["authority_file_hashes"])

    # PHASE A: all optimizer steps occur before candidate Teacher target compilation.
    states, training_receipts = build_and_freeze_training_arms(
        g0_root=g0_root, package_root=package_root, work_root=work_root, device=args.device
    )

    # PHASE B: outcome-bearing candidate support opens only under the immutable R5 gate.
    try:
        legacy_parents, legacy_samples, legacy_receipt = load_legacy_train_support(r104_root)
        for symbol in R5_SYMBOLS:
            gid = f"FUT:{symbol}:{R5_CANDIDATE_TIMESTAMP_MS}"
            if any(p.dependence_group_id == gid for p in legacy_parents.values()):
                raise RuntimeError(f"R11_R5_CANDIDATE_ALREADY_IN_LEGACY_TRAIN:{symbol}")
        candidate_parents, candidate_samples, candidate_support = build_candidate_support(
            g0_root=g0_root, package_root=package_root, lineage=lineage, device=args.device
        )
    except RuntimeError as exc:
        if str(exc).startswith("R11_R5_SUPPORT_NOT_READY_"):
            _, g0_after = r4.r1.verify_g0_authority(g0_root)
            guards = {
                "canonical_generation_advanced": False,
                "champion_promoted": False,
                "production_cutover": False,
                "scientific_verdict_created": False,
                "final_holdout_payload_opened": False,
                "fresh_market_data_downloaded": False,
                "frozen_package_unchanged": frozen_authority_hashes(package_root) == frozen_before,
                "g0_authority_unchanged": g0_after["authority_file_hashes"] == g0_before,
            }
            return support_not_ready(output=output, reason=str(exc), repo=repo, training_receipts=training_receipts, semantic_guards=guards)
        raise

    combined_parents = dict(legacy_parents); combined_parents.update(candidate_parents)
    combined_samples = list(legacy_samples) + list(candidate_samples)
    target_ids = sorted(candidate_parents)
    candidate_evidence, teacher_execution = compile_validation_targets_only_r5(
        samples=combined_samples, parents=combined_parents, target_parent_ids=target_ids, block_targets=32
    )
    admitted = [e for e in candidate_evidence if e.admission.admitted]
    if len(candidate_evidence) != 60 or len(admitted) != 60:
        reasons = sorted({reason for e in candidate_evidence for reason in e.admission.reasons})
        _, g0_after = r4.r1.verify_g0_authority(g0_root)
        guards = {
            "canonical_generation_advanced": False, "champion_promoted": False,
            "production_cutover": False, "scientific_verdict_created": False,
            "final_holdout_payload_opened": False, "fresh_market_data_downloaded": False,
            "frozen_package_unchanged": frozen_authority_hashes(package_root) == frozen_before,
            "g0_authority_unchanged": g0_after["authority_file_hashes"] == g0_before,
        }
        return support_not_ready(
            output=output,
            reason=f"R11_R5_SUPPORT_NOT_READY_TEACHER_ADMISSION:{len(admitted)}/60:{','.join(reasons)}",
            repo=repo, training_receipts=training_receipts, semantic_guards=guards,
        )

    # PHASE C: pure evaluation; no optimizer exists on candidate evidence.
    arm_results = evaluate_states(
        states=states, g0_root=g0_root, candidate_evidence=candidate_evidence,
        candidate_parents=candidate_parents, device=args.device,
    )
    shuffled_metrics = {shift: arm_results[f"SHUFFLE_{shift}"]["candidate"] for shift in R5_SHIFTS}
    adjudication = adjudicate_alignment_replication_r5(
        champion=arm_results["G0_CHAMPION"]["candidate"],
        aligned=arm_results["ALIGNED"]["candidate"],
        shuffled=shuffled_metrics,
    )

    frozen_after = frozen_authority_hashes(package_root)
    require(frozen_after == frozen_before, "R11_R5_FROZEN_PACKAGE_AUTHORITY_MUTATED")
    _, g0_after = r4.r1.verify_g0_authority(g0_root)
    require(g0_after["authority_file_hashes"] == g0_before, "R11_R5_G0_AUTHORITY_MUTATED")

    result = {
        "schema": SCHEMA,
        "status": "R11_INDEPENDENT_PURGE_ALIGNMENT_REPLICATION_R5_PASS",
        "runtime": R5_RUNTIME,
        "frozen_scientific_status_before": FROZEN_STATUS,
        "frozen_scientific_status_after": FROZEN_STATUS,
        "repo": repo,
        "runtime_identity": runtime,
        "protocol": {
            "preregistered_before_candidate_outcome_open": True,
            "candidate_timestamp": "2024-12-28T08:00:00Z",
            "candidate_symbols": list(R5_SYMBOLS),
            "nominal_future_groups": 10,
            "macro_clock_blocks": 1,
            "pre_registered_shifts": list(R5_SHIFTS),
            "candidate_support_used_in_training": False,
            "candidate_support_used_for_hyperparameter_or_architecture_tuning": False,
            "positive_rule": "ALIGNED_TOTAL_AND_DIRECTION_STRICTLY_LOWER_THAN_G0_AND_ALL_FIVE_SHUFFLES",
            "positive_result_cannot_authorize_promotion": True,
        },
        "training_arms": training_receipts,
        "legacy_train_support": legacy_receipt,
        "candidate_support": candidate_support,
        "candidate_teacher": {
            "target_count": len(candidate_evidence),
            "admitted_target_count": len(admitted),
            "execution": teacher_execution,
            "realized_future_is_not_direct_student_label": True,
        },
        "arm_results": arm_results,
        "alignment_adjudication": adjudication,
        "semantic_guards": {
            "canonical_generation_advanced": False,
            "champion_promoted": False,
            "production_cutover": False,
            "scientific_market_information_verdict_reopened": False,
            "profitability_or_alpha_claimed": False,
            "final_holdout_payload_opened": False,
            "fresh_market_data_downloaded": False,
            "candidate_optimizer_steps": 0,
            "frozen_package_unchanged": True,
            "g0_authority_unchanged": True,
        },
        "next_legal_step": (
            "ADJUDICATE_R5_MECHANISTIC_REPLICATION_ONLY__NO_PROMOTION_OR_PRODUCTION_CUTOVER"
            if adjudication["mechanistic_alignment_replication_supported"]
            else "FREEZE_R5_NEGATIVE_ALIGNMENT_RESULT__DO_NOT_TUNE_ON_PURGE_SUPPORT__REASSESS_LEARNING_TARGET_OR_INFORMATION_CONTENT"
        ),
    }
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "conclusion": adjudication["conclusion"],
        "aligned_candidate_loss": arm_results["ALIGNED"]["candidate"]["loss"],
        "champion_candidate_loss": arm_results["G0_CHAMPION"]["candidate"]["loss"],
        "aligned_lower_total_count_vs_shuffles": adjudication["aligned_lower_total_count_vs_shuffles"],
        "aligned_lower_direction_count_vs_shuffles": adjudication["aligned_lower_direction_count_vs_shuffles"],
        "next_legal_step": result["next_legal_step"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
