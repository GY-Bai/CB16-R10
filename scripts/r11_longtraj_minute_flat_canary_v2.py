from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, MINUTE_MS
from cb16_local_opt.frozen_sensory_stack_r10 import FrozenSensoryStackR10
from cb16_local_opt.full_minute_historical_replay_r0 import FINAL_HOLDOUT_START_MS
from cb16_local_opt.full_minute_long_trajectory_r1 import MinuteEnvelopeR1
from cb16_local_opt.longtraj_infra_closure_r0 import (
    MINUTE_H72_FORBIDDEN_CODE_R0,
    MinuteFrozenPhysicsAdapterR0,
    find_contiguous_prefinal_run_r0,
    flat_real_path_snapshots_r0,
    funding_events_by_minute_r0,
    simulate_h72_minute_branch_r0,
)
from cb16_local_opt.minute_sensory_adapter_r1 import iter_minute_sensory_frames_r1
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


def require(cond: bool, code: str) -> None:
    if not cond:
        raise RuntimeError(code)


def archive_hash(rows) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(np.asarray([r.open_time, r.open, r.high, r.low, r.close, r.volume], dtype=np.float64).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=os.environ.get("CB16_RAW_ROOT", "/cb16/raw"))
    ap.add_argument("--package-root", default=os.environ.get("CB16_PACKAGE_ROOT", "/cb16/package"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--sensory-batch-size", type=int, default=8)
    args = ap.parse_args()

    source = BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    require(args.symbol in layout["symbols"], "MINUTE_CANARY_SYMBOL_MISSING")
    rows = find_contiguous_prefinal_run_r0(source, args.symbol, required_rows=64 * 60 + args.frames + 8)
    require(all(int(r.open_time) < FINAL_HOLDOUT_START_MS for r in rows), "MINUTE_CANARY_FINAL_TOUCHED")
    require(all(int(b.open_time) - int(a.open_time) == MINUTE_MS for a, b in zip(rows, rows[1:])), "MINUTE_CANARY_NONCONTIGUOUS")
    funding = funding_events_by_minute_r0(source, args.symbol, start_ms=rows[0].open_time, end_ms=rows[-1].open_time)
    envelopes = [MinuteEnvelopeR1(r, False) for r in rows]
    frames = list(itertools.islice(iter_minute_sensory_frames_r1(args.symbol, envelopes), args.frames))
    require(len(frames) == args.frames, "MINUTE_CANARY_FRAME_SHORTFALL")
    times = [int(f.decision_time_ms) for f, _ in frames]

    sensory = FrozenSensoryStackR10(args.package_root, device=args.device, verify_hashes=True)
    encoded = {}
    for start in range(0, len(frames), args.sensory_batch_size):
        chunk = frames[start:start + args.sensory_batch_size]
        batch = sensory.encode_frames([f for f, _ in chunk])
        for i, (frame, audit) in enumerate(chunk):
            require(audit.micro_latest_time_ms == audit.decision_time_ms, "MINUTE_CANARY_MICRO_FUTURE_LEAK")
            require(audit.latest_completed_hour_start_ms < audit.current_hour_start_ms, "MINUTE_CANARY_SLOW_LANE_FUTURE_LEAK")
            encoded[int(frame.decision_time_ms)] = (batch.operator48[i].copy(), batch.medium48[i].copy())

    adapter = MinuteFrozenPhysicsAdapterR0(args.package_root)
    snapshots, _, path_receipt = flat_real_path_snapshots_r0(
        adapter=adapter,
        symbol=args.symbol,
        records=rows,
        selected_times=set(times),
        funding_by_minute=funding,
        account_id="INFRA_V2_MINUTE_FLAT_CANARY",
    )
    require(path_receipt["open_position_steps"] == 0, "MINUTE_CANARY_OPEN_POSITION")
    require(path_receipt["account_reset_count"] == 0, "MINUTE_CANARY_ACCOUNT_RESET")

    op, med, acc = [], [], []
    for frame, _ in frames:
        t = int(frame.decision_time_ms)
        o, m = encoded[t]
        op.append(o)
        med.append(m)
        acc.append(adapter.account6(snapshots[t], float(frame.micro_1m_60x5[-1, 3])))
    model = build_g0_brain_r10("TIER_1", seed=24_680, device=args.device)
    with torch.inference_mode():
        out = model(
            torch.tensor(np.asarray(op), dtype=torch.float32, device=args.device),
            torch.tensor(np.asarray(med), dtype=torch.float32, device=args.device),
            torch.tensor(np.asarray(acc), dtype=torch.float32, device=args.device),
        )
    finite = bool(torch.isfinite(out["direction_logits"]).all().item() and torch.isfinite(out["requested_risk_raw"]).all().item())
    require(finite, "MINUTE_CANARY_STUDENT_NONFINITE")

    try:
        simulate_h72_minute_branch_r0()
        fail_closed = False
    except RuntimeError as exc:
        fail_closed = str(exc) == MINUTE_H72_FORBIDDEN_CODE_R0
    require(fail_closed, "MINUTE_CANARY_H72_NOT_FAIL_CLOSED")

    result = {
        "schema": "CB16_R11_LONGTRAJ_MINUTE_FLAT_CANARY_V2",
        "status": "PASS",
        "archive": {"symbol": args.symbol, "rows": len(rows), "sha256": archive_hash(rows), "final_holdout_touched": False, "halt_imputed_rows": 0},
        "frames": len(frames),
        "student_inference_finite": finite,
        "account_path": path_receipt,
        "minute_h72_open_position_path_failed_closed": fail_closed,
        "teacher_feedback_bound": False,
        "scientific_verdict": None,
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
