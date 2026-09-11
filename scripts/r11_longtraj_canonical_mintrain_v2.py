from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_market import load_anchor_frames
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import R11_TRAIN_TEACHER_CONFIG, R11_VALIDATION_TEACHER_CONFIG
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11, TrainingRuntimeR11, policy_hash_r11
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10
from scripts import r11_science_g0_geometry_r2_1 as r21
from scripts import r11_science_g0_historical_r1 as g0r1


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0-root", type=Path, default=Path(os.environ.get("CB16_G0_ROOT", "/cb16/g0")))
    ap.add_argument("--package-root", type=Path, default=Path(os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package")))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--train-groups", type=int, default=48)
    ap.add_argument("--validation-groups", type=int, default=12)
    ap.add_argument("--sensory-batch-size", type=int, default=8)
    ap.add_argument("--teacher-workers", type=int, default=4)
    ap.add_argument("--teacher-block-targets", type=int, default=32)
    args = ap.parse_args()

    require(args.symbol == "BTCUSDT", "MINTRAIN_V2_SCOPE_BTC_ONLY")
    require(args.train_groups >= 48, "MINTRAIN_V2_TRAIN_GROUP_FLOOR_48")
    require(args.validation_groups >= 8, "MINTRAIN_V2_VALIDATION_GROUP_FLOOR_8")

    lineage, g0_identity = g0r1.verify_g0_authority(args.g0_root)
    market_cache = MarketRuntimeCacheR11(args.g0_root)
    market = market_cache.get(args.symbol)
    per_asset = next(x for x in lineage["per_asset"] if x["symbol"] == args.symbol)
    anchor_path = args.g0_root / "market_cache" / str(per_asset["anchors_file"])
    require(g0r1.sha256_file(anchor_path) == str(per_asset["anchors_sha256"]), "MINTRAIN_V2_ANCHOR_SHA_DRIFT")
    frames = load_anchor_frames(args.symbol, anchor_path)
    selected = g0r1.select_candidate_frames(
        frames,
        train_target=args.train_groups,
        validation_target=args.validation_groups,
        candidate_factor=2,
    )
    encoded, sensory_receipt = g0r1.encode_selected_frames(
        package_root=args.package_root,
        device=args.device,
        selected=selected,
        batch_size=args.sensory_batch_size,
    )
    physics = FrozenPhysicsRuntimeR102.load(args.package_root)
    parents, samples, support = r21.build_multi_account_counterfactual_support(
        symbol=args.symbol,
        selected=selected,
        encoded=encoded,
        physics=physics,
        hourly_ts=market.open_time_ms,
        hourly_ohlcv=market.ohlcv,
        funding=market.funding_rate,
        train_target_groups=args.train_groups,
        validation_target_groups=args.validation_groups,
    )
    train_e, val_e, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG,
        val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=args.teacher_workers,
        block_targets=args.teacher_block_targets,
    )
    market_cache.assert_read_only()

    train_summary = evidence_summary(train_e)
    val_summary = evidence_summary(val_e)
    require(train_summary["admitted_dependence_groups"] >= 32, f"MINTRAIN_V2_TRAIN_SUPPORT:{train_summary}")
    require(val_summary["admitted"] > 0, f"MINTRAIN_V2_VALIDATION_SUPPORT:{val_summary}")
    require({e.teacher_protocol_hash for e in train_e} == {R11_TRAIN_TEACHER_CONFIG.content_hash}, "MINTRAIN_V2_TRAIN_PROTOCOL_DRIFT")
    require({e.teacher_protocol_hash for e in val_e} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}, "MINTRAIN_V2_VALIDATION_PROTOCOL_DRIFT")

    prepared_train = PreparedEvidenceR11.from_evidence(train_e, parents, device=args.device)
    prepared_val = PreparedEvidenceR11.from_evidence(val_e, parents, device=args.device)
    require(not prepared_train.packed.requires_grad and not prepared_val.packed.requires_grad, "MINTRAIN_V2_TEACHER_AUTOGRAD_FORBIDDEN")

    torch.manual_seed(24_680)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24_680)
    model = build_g0_brain_r10("TIER_1", seed=24_680, device=args.device)
    runtime = TrainingRuntimeR11(device=args.device)
    optimizer = runtime.build_optimizer(model)
    before_hash = policy_hash_r11(model)
    ids = torch.arange(min(prepared_train.rows, 512), device=prepared_train.packed.device, dtype=torch.long)
    step = runtime.train_one_step(model=model, optimizer=optimizer, prepared=prepared_train, ids=ids)
    after_hash = policy_hash_r11(model)
    require(before_hash != after_hash, "MINTRAIN_V2_STUDENT_NOT_UPDATED")
    require(math.isfinite(step.loss), "MINTRAIN_V2_NONFINITE_LOSS")
    require(len(step.gradient_owner_set) == 6, f"MINTRAIN_V2_GRADIENT_OWNER_DRIFT:{sorted(step.gradient_owner_set)}")

    result = {
        "schema": "CB16_R11_LONGTRAJ_CANONICAL_MINTRAIN_V2",
        "status": "PASS",
        "teacher": {
            "runtime_stats": str(teacher_stats),
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "future_semantics": "EXISTING_QUALIFIED_HOURLY_H72_FROZEN_PHYSICS_BINDING",
        },
        "student": {
            "before_policy_hash": before_hash,
            "after_policy_hash": after_hash,
            "loss": step.loss,
            "direction_loss": step.direction_loss,
            "sizing_loss": step.sizing_loss,
            "gradient_owner_set": sorted(step.gradient_owner_set),
            "optimizer_steps": 1,
            "prepared_train_hash": prepared_train.evidence_hash,
            "prepared_validation_hash": prepared_val.evidence_hash,
        },
        "support": support,
        "sensory": sensory_receipt,
        "g0_identity": g0_identity,
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
        "minute_feedback_binding_claimed": False,
        "scientific_verdict": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
