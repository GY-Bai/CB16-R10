from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cb16_local_opt.binance_archive_input_r10 import (
    BinanceUSDMArchiveSourceR10,
    KlineRecord,
    MINUTE_MS,
    SensoryDecisionFrameR10,
    ordered4h30_from_hourly,
    stamps_from_open_times_ms,
)
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_long_trajectory_r1 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.longtraj_infra_closure_r0 import (
    find_contiguous_prefinal_run_r0,
    funding_events_by_minute_r0,
)
from cb16_local_opt.minute_physics_binding_r2 import canonical_hash
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_common import HOUR_MS
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_learning import evidence_summary
from cb16_local_opt.r102_physics import CANDIDATES_R102, FLAT, LONG, SHORT, FrozenPhysicsRuntimeR102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
)
from cb16_local_opt.teacher_runtime_r11 import compile_teacher_evidence_r11
from cb16_local_opt.training_runtime_r11 import (
    AUTHORIZED_GRADIENT_OWNERS_R11,
    PreparedEvidenceR11,
    TrainingRuntimeR11,
    policy_hash_r11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10
from scripts import r11_longtraj_e6_feedback_binding_r2 as e6

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_TRAINING_ADMISSION_R0_SPEC_V1.json"
E6_RECEIPT = ROOT / "authority/rearchitecture_r11/CB16_R11_LONGTRAJ_E6_FEEDBACK_BINDING_R2_FREEZE_RECEIPT_V1.json"
SCHEMA = "CB16_R11_LONGTRAJ_TRAINING_ADMISSION_R0_RESULT_V1"
H72_MINUTES = 72 * 60
SENSORY_HISTORY_HOURS = 64
SENSORY_PREFIX_MINUTES = (SENSORY_HISTORY_HOURS + 1) * 60
PARENT_SPACING_MS = 73 * HOUR_MS
TRAIN_GROUPS = 48
VALIDATION_GROUPS = 12
TOTAL_GROUPS = TRAIN_GROUPS + VALIDATION_GROUPS
DIRECTION_TO_TEACHER = {SHORT: -1, FLAT: 0, LONG: 1}


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def atomic_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _aggregate_hour(rows: Sequence[KlineRecord]) -> np.ndarray:
    require(len(rows) == 60, "ADMISSION_HOUR_REQUIRES_60_ROWS")
    require(int(rows[0].open_time) % HOUR_MS == 0, "ADMISSION_HOUR_NOT_ALIGNED")
    require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(rows, rows[1:])), "ADMISSION_HOUR_GAP")
    return np.asarray([
        float(rows[0].open),
        float(max(r.high for r in rows)),
        float(min(r.low for r in rows)),
        float(rows[-1].close),
        float(sum(r.volume for r in rows)),
    ], dtype=np.float32)


def build_sensory_frame_r0(records: Sequence[KlineRecord], parent_index: int, symbol: str) -> SensoryDecisionFrameR10:
    idx = int(parent_index)
    require(int(records[idx].open_time) % HOUR_MS == HOUR_MS - MINUTE_MS, "ADMISSION_PARENT_NOT_HH59")
    current_start = idx - 59
    prior_start = current_start - SENSORY_HISTORY_HOURS * 60
    require(prior_start >= 0, "ADMISSION_SENSORY_PREHISTORY_MISSING")
    current = records[current_start:idx + 1]
    prior = records[prior_start:current_start]
    require(len(current) == 60 and len(prior) == SENSORY_HISTORY_HOURS * 60, "ADMISSION_SENSORY_SLICE_LENGTH")
    require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(prior + current, (prior + current)[1:])), "ADMISSION_SENSORY_GAP")

    hourly = np.stack([_aggregate_hour(prior[i:i + 60]) for i in range(0, len(prior), 60)], axis=0)
    hourly_open_ms = np.asarray([int(prior[i].open_time) for i in range(0, len(prior), 60)], dtype=np.int64)
    micro = np.stack([r.ohlcv() for r in current], axis=0).astype(np.float32)
    micro_ms = np.asarray([int(r.open_time) for r in current], dtype=np.int64)
    nominal_h = int(records[idx].open_time) + MINUTE_MS
    require(int(micro_ms[-1]) == int(records[idx].open_time), "ADMISSION_MICRO_VISIBILITY_DRIFT")
    require(int(hourly_open_ms[-1]) + HOUR_MS - MINUTE_MS == int(records[idx].open_time), "ADMISSION_HOURLY_VISIBILITY_DRIFT")
    return SensoryDecisionFrameR10(
        symbol=symbol,
        decision_time_ms=nominal_h,
        micro_1m_60x5=micro,
        micro_stamps_60x5=stamps_from_open_times_ms(micro_ms),
        hourly_64x5=hourly,
        hourly_stamps_64x5=stamps_from_open_times_ms(hourly_open_ms),
        ordered4h30=ordered4h30_from_hourly(hourly[-24:]),
    )


def select_parent_indices_r0(records: Sequence[KlineRecord], count: int = TOTAL_GROUPS) -> list[int]:
    eligible = [
        i for i, row in enumerate(records)
        if i >= SENSORY_PREFIX_MINUTES - 1
        and i + H72_MINUTES < len(records)
        and int(row.open_time) % HOUR_MS == HOUR_MS - MINUTE_MS
        and int(records[i + H72_MINUTES].open_time) < FINAL_HOLDOUT_START_MS
    ]
    out: list[int] = []
    last: int | None = None
    for i in eligible:
        t = int(records[i].open_time)
        if last is None or t - last >= PARENT_SPACING_MS:
            out.append(i)
            last = t
            if len(out) == int(count):
                break
    require(len(out) == int(count), f"ADMISSION_NOT_ENOUGH_NONOVERLAPPING_PARENTS:{len(out)}<{count}")
    return out


def _simulate_parent(payload: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    runtime = FrozenPhysicsRuntimeR102.load(payload["package_root"])
    branches = []
    for direction, risk in CANDIDATES_R102:
        branches.append(e6._simulate_branch(
            runtime,
            parent_state=payload["parent_state"],
            risk_authority=payload["risk_authority"],
            symbol=payload["symbol"],
            future_rows=payload["future_rows"],
            funding=payload["funding"],
            parent_id=payload["parent_id"],
            direction=int(direction),
            risk=float(risk),
        ))
    return {
        "ordinal": int(payload["ordinal"]),
        "parent_id": str(payload["parent_id"]),
        "decision_time_ms": int(payload["decision_time_ms"]),
        "first_execution_time_ms": int(payload["future_rows"][0].open_time),
        "branches": branches,
    }


def _tree_exact_equal(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, Mapping) or isinstance(b, Mapping):
        if not isinstance(a, Mapping) or not isinstance(b, Mapping) or set(a) != set(b):
            return False
        return all(_tree_exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_tree_exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def _student_checkpoint_roundtrip(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    path: Path,
) -> dict[str, Any]:
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
    payload = {
        "schema": "CB16_R11_LONGTRAJ_TRAINING_ADMISSION_CHECKPOINT_R0_V1",
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": cpu_rng,
        "cuda_rng_state_all": cuda_rng,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_g0_brain_r10("TIER_1", seed=24_680, device=device)
    restored.load_state_dict(loaded["model_state"], strict=True)
    rt = TrainingRuntimeR11(device=device)
    opt2 = rt.build_optimizer(restored)
    opt2.load_state_dict(loaded["optimizer_state"])
    torch.set_rng_state(loaded["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(loaded["cuda_rng_state_all"])
    exact = (
        policy_hash_r11(restored) == policy_hash_r11(model)
        and _tree_exact_equal(opt2.state_dict(), optimizer.state_dict())
        and torch.equal(torch.get_rng_state(), cpu_rng)
        and (not torch.cuda.is_available() or all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng)))
    )
    return {
        "exact": bool(exact),
        "checkpoint_sha256": sha256_file(path),
        "policy_hash": policy_hash_r11(restored),
        "optimizer_state_exact": bool(_tree_exact_equal(opt2.state_dict(), optimizer.state_dict())),
        "cpu_rng_exact": bool(torch.equal(torch.get_rng_state(), cpu_rng)),
        "cuda_rng_exact": bool(not torch.cuda.is_available() or all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), cuda_rng))),
    }


def _write_scientific_fail(
    output: Path,
    *,
    spec_hash: str,
    train_summary: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    reason: str,
    common: Mapping[str, Any],
) -> None:
    out = dict(common)
    out.update({
        "schema": SCHEMA,
        "status": "FAIL",
        "classification": "SCIENTIFIC_ADMISSION_FAIL__NO_LONG_GRADIENT_AUTHORITY",
        "failure_reason": reason,
        "spec_sha256": spec_hash,
        "teacher": {
            "train_summary": dict(train_summary),
            "validation_summary": dict(validation_summary),
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
        },
        "long_gradient_training_authorized": False,
        "new_scientific_verdict": False,
        "scientific_verdict": None,
    })
    atomic_json(output, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    require(spec["schema"] == "CB16_R11_LONGTRAJ_TRAINING_ADMISSION_R0_SPEC_V1", "ADMISSION_SPEC_SCHEMA")
    require(spec["status"] == "FROZEN_BEFORE_FIRST_ADMISSION_EXECUTION", "ADMISSION_SPEC_NOT_PREREGISTERED")
    require(spec["base_e6_branch_head_sha"] == "ae0e44d4a8f4c557d2b13b9689be5d9a81f8e97c", "ADMISSION_E6_BASE_DRIFT")
    require(E6_RECEIPT.is_file(), "ADMISSION_E6_FREEZE_RECEIPT_MISSING")
    e6_receipt = json.loads(E6_RECEIPT.read_text(encoding="utf-8"))
    require(e6_receipt.get("status") == "FROZEN_AND_QUALIFIED", "ADMISSION_E6_NOT_FROZEN")
    require(e6_receipt.get("qualified_code_sha") == "72005b9d7eca06be99a4cebdbdaad48126b9d061", "ADMISSION_E6_QUALIFIED_SHA_DRIFT")
    require(int(args.workers) == 8, "ADMISSION_REQUIRES_8X1_SCALAR_PHYSICS_FARM")
    require(args.symbol == "BTCUSDT", "ADMISSION_SCOPE_BTC_ONLY")
    require(args.device == "cuda", "ADMISSION_CANONICAL_DEVICE_CUDA")
    require(tuple(CANDIDATES_R102) == ((FLAT, 0.0), (SHORT, 0.25), (SHORT, 0.5), (SHORT, 0.75), (SHORT, 1.0), (LONG, 0.25), (LONG, 0.5), (LONG, 0.75), (LONG, 1.0)), "ADMISSION_CANDIDATE_GRID_DRIFT")

    source = BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    require(args.symbol in layout["symbols"], f"ADMISSION_SYMBOL_MISSING:{args.symbol}")
    required_rows = SENSORY_PREFIX_MINUTES + (TOTAL_GROUPS - 1) * 73 * 60 + H72_MINUTES + 180
    records = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=required_rows)
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in records), "ADMISSION_FINAL_TOUCHED")
    require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(records, records[1:])), "ADMISSION_ARCHIVE_NOT_CONTIGUOUS")
    indices = select_parent_indices_r0(records, TOTAL_GROUPS)
    require(len(indices) == TOTAL_GROUPS, "ADMISSION_PARENT_COUNT")

    first_data_idx = indices[0] - (SENSORY_PREFIX_MINUTES - 1)
    start_ms = int(records[first_data_idx].open_time)
    end_ms = int(records[indices[-1] + H72_MINUTES].open_time)
    funding = funding_events_by_minute_r0(source, args.symbol, start_ms=start_ms, end_ms=end_ms)
    runtime = FrozenPhysicsRuntimeR102.load(str(Path(args.package_root).resolve()))

    frames = [build_sensory_frame_r0(records, idx, args.symbol) for idx in indices]
    sensory = FrozenSensoryStackR10(args.package_root, device=args.device, verify_hashes=True)
    encoded_by_ordinal: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for start in range(0, len(frames), 8):
        chunk = frames[start:start + 8]
        enc = sensory.encode_frames(chunk)
        for j in range(len(chunk)):
            encoded_by_ordinal[start + j] = (
                enc.operator48[j].copy(), enc.medium48[j].copy(), enc.ordered4h30[j].copy()
            )
    frozen_trainable = sum(
        int(p.requires_grad)
        for module in (sensory.operator.tok, sensory.operator.model, sensory.medium.model)
        for p in module.parameters()
    )
    require(frozen_trainable == 0, "ADMISSION_FROZEN_SENSORY_TRAINABLE_PARAMETER")
    del sensory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    payloads: list[dict[str, Any]] = []
    parents: dict[str, ParentContextR102] = {}
    parent_states: dict[str, dict[str, Any]] = {}
    for ordinal, idx in enumerate(indices):
        parent_row = records[idx]
        future_rows = tuple(records[idx + 1:idx + 1 + H72_MINUTES])
        require(len(future_rows) == H72_MINUTES, "ADMISSION_H72_SLICE_LENGTH")
        require(int(future_rows[0].open_time) - int(parent_row.open_time) == MINUTE_MS, "ADMISSION_NEXT_MINUTE_DRIFT")
        require(int(future_rows[0].open_time) % HOUR_MS == 0, "ADMISSION_FIRST_EXECUTION_NOT_HOUR_BOUNDARY")
        require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in future_rows), "ADMISSION_FUTURE_FINAL_TOUCHED")

        physics_prefix_start = idx - (e6.PARENT_PREFIX_MINUTES - 1)
        physics_prefix = tuple(records[physics_prefix_start:idx + 1])
        parent_id = f"ADMR0:{args.symbol}:{int(parent_row.open_time)}"
        parent_state, risk_authority, account6 = e6._build_parent_state(
            runtime,
            symbol=args.symbol,
            account_id=parent_id,
            prefix_rows=physics_prefix,
            funding=funding,
        )
        lineage = e6._future_hash(args.symbol, int(parent_row.open_time), future_rows, funding)
        op, med, ordered = encoded_by_ordinal[ordinal]
        split = "TRAIN" if ordinal < TRAIN_GROUPS else "VALIDATION"
        dep_id = f"FUT:{args.symbol}:{int(parent_row.open_time)}:{lineage[:16]}"
        pc = ParentContextR102(
            parent_id=parent_id,
            dependence_group_id=dep_id,
            symbol=args.symbol,
            decision_time_ms=int(parent_row.open_time),
            split=split,
            scenario="FLAT_MINUTE_R2_ADMISSION",
            operator48=tuple(float(x) for x in op),
            medium48=tuple(float(x) for x in med),
            account6=tuple(float(x) for x in account6),
            ordered4h30=tuple(float(x) for x in ordered),
            current_mark=float(parent_row.close),
            snapshot_sha256=canonical_hash(parent_state),
            eligible_for_economic_evidence=True,
            market_lineage_hash=lineage,
        )
        parents[parent_id] = pc
        parent_states[parent_id] = parent_state
        payloads.append({
            "ordinal": ordinal,
            "package_root": str(Path(args.package_root).resolve()),
            "symbol": args.symbol,
            "parent_id": parent_id,
            "decision_time_ms": int(parent_row.open_time),
            "parent_state": parent_state,
            "risk_authority": risk_authority,
            "future_rows": future_rows,
            "funding": dict(funding),
        })

    pairwise_nonoverlap = all(
        int(records[b].open_time) - int(records[a].open_time) >= PARENT_SPACING_MS
        for a, b in zip(indices, indices[1:])
    )
    require(pairwise_nonoverlap, "ADMISSION_PARENT_FUTURES_OVERLAP")
    require(all(frames[i].decision_time_ms == int(records[indices[i]].open_time) + MINUTE_MS for i in range(TOTAL_GROUPS)), "ADMISSION_SENSORY_NOMINAL_TIME_DRIFT")
    require(all(int(frames[i].micro_1m_60x5.shape[0]) == 60 for i in range(TOTAL_GROUPS)), "ADMISSION_MICRO_FRAME_SHAPE")

    causal = e6._prefix_causality_canaries(
        str(Path(args.package_root).resolve()),
        args.symbol,
        payloads[0]["parent_state"],
        payloads[0]["risk_authority"],
        payloads[0]["future_rows"],
        funding,
    )
    require(all(causal.values()), f"ADMISSION_CAUSALITY_CANARY_FAIL:{causal}")

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        executed = list(pool.map(_simulate_parent, payloads))
    executed.sort(key=lambda x: int(x["ordinal"]))
    require(len(executed) == TOTAL_GROUPS, "ADMISSION_EXECUTED_PARENT_COUNT")
    require(all(len(x["branches"]) == len(CANDIDATES_R102) for x in executed), "ADMISSION_BRANCH_GRID_COUNT")
    require(all(int(x["first_execution_time_ms"]) - int(x["decision_time_ms"]) == MINUTE_MS for x in executed), "ADMISSION_NEXT_MINUTE_EXECUTION_DRIFT")

    samples: list[CounterfactualBranchSampleR5] = []
    for x in executed:
        pc = parents[str(x["parent_id"])]
        for b in x["branches"]:
            sample = CounterfactualBranchSampleR5(
                parent_id=pc.parent_id,
                student_context_object_id=pc.student_context_object_id,
                timestamp=int(pc.decision_time_ms),
                context_features=tuple(float(v) for v in pc.student_features),
                direction=DIRECTION_TO_TEACHER[int(b["direction"])],
                requested_risk=float(b["requested_risk"]),
                realized_utility=float(b["utility"]),
                dependence_group_id=pc.dependence_group_id,
                market_lineage_hash=pc.market_lineage_hash,
            )
            sample.validate()
            samples.append(sample)
    require(len(samples) == TOTAL_GROUPS * len(CANDIDATES_R102), "ADMISSION_SAMPLE_COUNT")
    require(all(len({s.dependence_group_id for s in samples if s.parent_id == pid}) == 1 for pid in parents), "ADMISSION_PARENT_DEPENDENCE_GROUP_DRIFT")
    require(all(s.student_context_object_id == parents[s.parent_id].student_context_object_id for s in samples), "ADMISSION_STUDENT_CONTEXT_BINDING_DRIFT")

    train_e, val_e, teacher_stats = compile_teacher_evidence_r11(
        samples=samples,
        parents=parents,
        train_config=R11_TRAIN_TEACHER_CONFIG,
        val_config=R11_VALIDATION_TEACHER_CONFIG,
        workers=8,
        block_targets=32,
    )
    train_summary = evidence_summary(train_e)
    val_summary = evidence_summary(val_e)
    train_hashes_exact = {e.teacher_protocol_hash for e in train_e} == {R11_TRAIN_TEACHER_CONFIG.content_hash}
    val_hashes_exact = {e.teacher_protocol_hash for e in val_e} == {R11_VALIDATION_TEACHER_CONFIG.content_hash}

    common = {
        "e6": {
            "freeze_receipt_sha256": sha256_file(E6_RECEIPT),
            "qualified_code_sha": e6_receipt.get("qualified_code_sha"),
            "qualification_run_id": e6_receipt.get("qualification_run_id"),
        },
        "archive": {
            "real_market_payload_opened": True,
            "first_ms": start_ms,
            "last_ms": end_ms,
            "contiguous_observed_1m_only": True,
            "final_holdout_touched": False,
            "fresh_market_data_downloaded": False,
        },
        "execution": {
            "train_groups": TRAIN_GROUPS,
            "validation_groups": VALIDATION_GROUPS,
            "total_groups": TOTAL_GROUPS,
            "branches_per_parent": len(CANDIDATES_R102),
            "total_branches": len(samples),
            "horizon_minutes": H72_MINUTES,
            "physics_workers": int(args.workers),
            "threads_per_worker": 1,
            "parent_spacing_hours": 73,
            "decision_to_execution_minutes": 1,
            "recurrent_time_axis_parallelized": False,
        },
        "time_binding": {
            "student_decision_event": "H-1m (HH:59)",
            "sensory_frame_nominal_time": "H",
            "sensory_max_visible_market_time": "H-1m (HH:59)",
            "action_execution": "H next-minute-open",
        },
        "causality_canaries": causal,
        "frozen_sensory_trainable_parameters": frozen_trainable,
        "outcome_as_label_used": False,
        "final_holdout_touched": False,
        "fresh_market_data_downloaded": False,
    }

    support_ok = (
        int(train_summary["admitted_dependence_groups"]) >= 32
        and int(val_summary["admitted"]) > 0
        and train_hashes_exact
        and val_hashes_exact
    )
    if not support_ok:
        _write_scientific_fail(
            output,
            spec_hash=sha256_file(SPEC),
            train_summary=train_summary,
            validation_summary=val_summary,
            reason="PRODUCTION_TEACHER_SUPPORT_OR_PROTOCOL_GATE_FAILED",
            common=common,
        )
        return 0

    prepared_train = PreparedEvidenceR11.from_evidence(train_e, parents, device=args.device)
    prepared_val = PreparedEvidenceR11.from_evidence(val_e, parents, device=args.device)
    require(not prepared_train.packed.requires_grad and not prepared_val.packed.requires_grad, "ADMISSION_TEACHER_AUTOGRAD_FORBIDDEN")

    torch.manual_seed(24_680)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(24_680)
    model = build_g0_brain_r10("TIER_1", seed=24_680, device=args.device)
    training = TrainingRuntimeR11(device=args.device)
    optimizer = training.build_optimizer(model)
    before_hash = policy_hash_r11(model)
    ids = torch.arange(min(prepared_train.rows, 512), device=prepared_train.packed.device, dtype=torch.long)
    try:
        step = training.train_one_step(model=model, optimizer=optimizer, prepared=prepared_train, ids=ids)
    except RuntimeError as exc:
        text = str(exc)
        if text.startswith("R11_GRADIENT_OWNER_SET_DRIFT") or text.startswith("R11_AUTHORIZED_GRADIENT_DISCONNECT"):
            _write_scientific_fail(
                output,
                spec_hash=sha256_file(SPEC),
                train_summary=train_summary,
                validation_summary=val_summary,
                reason=text,
                common=common,
            )
            return 0
        raise
    after_hash = policy_hash_r11(model)
    checkpoint = _student_checkpoint_roundtrip(
        model=model,
        optimizer=optimizer,
        device=args.device,
        path=output.parent / "ADMISSION_STUDENT_CHECKPOINT.pt",
    )

    gates = {
        "E6_FROZEN_LINEAGE_BOUND": True,
        "REAL_PRE_FINAL_1M_H72_FULL_GRID": len(samples) == 540,
        "NEXT_MINUTE_OPEN_EXECUTION": all(int(x["first_execution_time_ms"]) - int(x["decision_time_ms"]) == MINUTE_MS for x in executed),
        "FROZEN_SENSORY_CAUSAL_MAX_TIME_H_MINUS_1M": all(frames[i].decision_time_ms == int(records[indices[i]].open_time) + MINUTE_MS for i in range(TOTAL_GROUPS)),
        "ACCOUNT_T_BOUND_IN_STUDENT_CONTEXT": all(s.student_context_object_id == parents[s.parent_id].student_context_object_id for s in samples),
        "REAL_PARENT_FUTURES_NONOVERLAPPING": bool(pairwise_nonoverlap),
        "FUTURE_SUFFIX_MUTATION_INVARIANCE": bool(causal["future_suffix_mutation_invariance"]),
        "FUTURE_ACCOUNT_POISON_INVARIANCE": bool(causal["future_account_poison_invariance"]),
        "NEXT_BAR_ISOLATION": bool(causal["next_bar_isolation"]),
        "PRODUCTION_TEACHER_PROTOCOL_HASHES_EXACT": bool(train_hashes_exact and val_hashes_exact),
        "TRAIN_ADMITTED_DEPENDENCE_GROUPS_GE_32": int(train_summary["admitted_dependence_groups"]) >= 32,
        "VALIDATION_ADMITTED_ROWS_GT_0": int(val_summary["admitted"]) > 0,
        "TEACHER_AND_FROZEN_INPUTS_NO_GRAD": bool(not prepared_train.packed.requires_grad and not prepared_val.packed.requires_grad and frozen_trainable == 0),
        "STUDENT_POLICY_HASH_CHANGED": before_hash != after_hash,
        "FINITE_LOSS": bool(math.isfinite(step.loss) and math.isfinite(step.direction_loss) and math.isfinite(step.sizing_loss)),
        "EXACT_SIX_GRADIENT_OWNERS": step.gradient_owner_set == AUTHORIZED_GRADIENT_OWNERS_R11 and len(step.gradient_owner_set) == 6,
        "CHECKPOINT_RESTART_EXACT": bool(checkpoint["exact"]),
        "FINAL_FIREWALL_CLOSED": True,
        "FRESH_DATA_FIREWALL_CLOSED": True,
    }
    require(all(gates.values()), f"ADMISSION_GATE_FAIL:{gates}")

    result = dict(common)
    result.update({
        "schema": SCHEMA,
        "status": "PASS",
        "classification": "TRAINING_INTERFACE_ADMISSION_QUALIFIED__NO_LONG_GRADIENT_AUTHORITY__NO_SCIENTIFIC_VERDICT",
        "spec_sha256": sha256_file(SPEC),
        "teacher": {
            "runtime_stats": str(teacher_stats),
            "train_summary": train_summary,
            "validation_summary": val_summary,
            "train_protocol_hash": R11_TRAIN_TEACHER_CONFIG.content_hash,
            "validation_protocol_hash": R11_VALIDATION_TEACHER_CONFIG.content_hash,
            "support_thresholds_relaxed": False,
        },
        "student": {
            "prepared_train_rows": prepared_train.rows,
            "prepared_validation_rows": prepared_val.rows,
            "prepared_train_hash": prepared_train.evidence_hash,
            "prepared_validation_hash": prepared_val.evidence_hash,
            "before_policy_hash": before_hash,
            "after_policy_hash": after_hash,
            "optimizer_steps": 1,
            "loss": step.loss,
            "direction_loss": step.direction_loss,
            "sizing_loss": step.sizing_loss,
            "gradient_owner_set": sorted(step.gradient_owner_set),
            "gradient_group_norms": step.gradient_group_norms,
            "pre_clip_grad_norm": step.pre_clip_grad_norm,
            "checkpoint": checkpoint,
            "dtype": "torch.float32",
            "amp": False,
        },
        "parent_receipts": [
            {
                "parent_id": x["parent_id"],
                "decision_time_ms": x["decision_time_ms"],
                "first_execution_time_ms": x["first_execution_time_ms"],
                "split": parents[x["parent_id"]].split,
                "dependence_group_id": parents[x["parent_id"]].dependence_group_id,
                "student_context_object_id": parents[x["parent_id"]].student_context_object_id,
                "parent_snapshot_sha256": parents[x["parent_id"]].snapshot_sha256,
                "branch_utility_sha256": canonical_hash([(b["direction"], b["requested_risk"], b["utility"]) for b in x["branches"]]),
            }
            for x in executed
        ],
        "gates": gates,
        "long_gradient_training_authorized": False,
        "new_scientific_verdict": False,
        "scientific_verdict": None,
        "next_gate": "LONGTRAJ_TRAINING_CAMPAIGN_PREREGISTRATION",
    })
    atomic_json(output, result)
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "train_summary": train_summary,
        "validation_summary": val_summary,
        "student": {k: result["student"][k] for k in ("prepared_train_rows", "prepared_validation_rows", "optimizer_steps", "loss", "before_policy_hash", "after_policy_hash")},
        "gates": gates,
        "long_gradient_training_authorized": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
