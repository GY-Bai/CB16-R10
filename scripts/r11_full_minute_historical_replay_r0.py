from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from cb16_local_opt.full_minute_historical_replay_r0 import (
    BRAIN_PARAMETER_TARGET,
    FINAL_HOLDOUT_START_MS,
    ReplayCentralBrain16M,
    brain_parameter_report,
    gradient_boundary_canary,
    iter_rolling_minute_samples,
)


SCHEMA = "CB16_R11_FULL_MINUTE_HISTORICAL_REPLAY_R0_CANARY_RESULT_V1"
EXPECTED_SYMBOLS = {
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
}


def archive_month(path: Path) -> str:
    # SYMBOL-1m-YYYY-MM.zip -> YYYY-MM
    stem = path.name[:-4]
    parts = stem.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected archive name: {path.name}")
    return f"{parts[-2]}-{parts[-1]}"


def inventory_pre_final(source: BinanceUSDMArchiveSourceR10) -> dict:
    layout = source.validate_layout()
    observed = set(layout["symbols"])
    missing = sorted(EXPECTED_SYMBOLS - observed)
    if missing:
        raise RuntimeError("MISSING_EXPECTED_SYMBOLS:" + ",".join(missing))

    by_symbol = {}
    for symbol in sorted(EXPECTED_SYMBOLS):
        archives = source.monthly_kline_archives(symbol)
        eligible = [p for p in archives if archive_month(p) < "2025-09"]
        if not eligible:
            raise RuntimeError(f"NO_PRE_FINAL_ARCHIVES:{symbol}")
        missing_sidecars = [str(p) for p in eligible if not Path(str(p) + ".CHECKSUM").is_file()]
        if missing_sidecars:
            raise RuntimeError(f"MISSING_CHECKSUM_SIDECAR:{symbol}:{missing_sidecars[:3]}")
        by_symbol[symbol] = {
            "eligible_monthly_archives": len(eligible),
            "first_month": archive_month(eligible[0]),
            "last_month": archive_month(eligible[-1]),
            "checksum_sidecars_complete": True,
        }
    return {
        "layout": layout,
        "expected_symbols": sorted(EXPECTED_SYMBOLS),
        "observed_symbols": sorted(observed),
        "pre_final": by_symbol,
        "zip_payload_bytes_read_for_inventory": False,
    }


def real_minute_enumeration_canary(source: BinanceUSDMArchiveSourceR10, *, rows: int = 10_000) -> dict:
    # Read only the earliest bounded BTC prefix. This is pre-FINAL historical TRAIN-side
    # data and is used solely to prove minute-by-minute rolling enumeration.
    prefix = list(itertools.islice(source.iter_1m("BTCUSDT", verify_checksums=False, strict_chronology=False), rows))
    if len(prefix) != rows:
        raise RuntimeError(f"BTC_PREFIX_TOO_SHORT:{len(prefix)}!={rows}")
    if any(r.open_time >= FINAL_HOLDOUT_START_MS for r in prefix):
        raise RuntimeError("FINAL_HOLDOUT_PAYLOAD_REACHED_IN_CANARY")
    lookback = 60
    samples = list(
        iter_rolling_minute_samples(
            "BTCUSDT",
            prefix,
            lookback_minutes=lookback,
            final_holdout_start_ms=FINAL_HOLDOUT_START_MS,
        )
    )
    # This prefix is expected to be contiguous. With N rows, lookback L and one
    # next transition, stride-one rolling must emit exactly N-L samples.
    expected = rows - lookback
    if len(samples) != expected:
        raise RuntimeError(f"STRIDE_ONE_ENUMERATION_MISMATCH:{len(samples)}!={expected}")
    decisions = [s.decision_time_ms for s in samples]
    if any(b - a != 60_000 for a, b in zip(decisions, decisions[1:])):
        raise RuntimeError("DECISION_STRIDE_NOT_ONE_MINUTE")
    return {
        "symbol": "BTCUSDT",
        "input_rows": len(prefix),
        "lookback_minutes": lookback,
        "rolling_samples": len(samples),
        "expected_rolling_samples": expected,
        "first_input_ms": prefix[0].open_time,
        "last_input_ms": prefix[-1].open_time,
        "first_decision_ms": decisions[0],
        "last_decision_ms": decisions[-1],
        "decision_stride_minutes": 1,
        "final_holdout_payload_opened": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/cb16/raw")
    ap.add_argument("--output", required=True)
    ap.add_argument("--require-raw-root", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = BinanceUSDMArchiveSourceR10(args.raw_root)

    result = {
        "schema": SCHEMA,
        "status": "STARTED",
        "classification": "R0_ENGINEERING_CANARY__NO_MARKET_VERDICT",
        "raw_root": str(Path(args.raw_root)),
        "semantic_guards": {
            "fresh_market_data_downloaded": False,
            "final_holdout_payload_opened": False,
            "frozen_physics_changed": False,
            "supervisor_permission_changed": False,
            "teacher_meaning_changed": False,
            "evidence_meaning_changed": False,
            "gradient_ownership_changed": False,
            "market_verdict_changed": False,
        },
    }

    if not Path(args.raw_root).is_dir():
        result["status"] = "EXECUTION_BLOCKED"
        result["blocker"] = "HOST_RAW_MOUNT_UNAVAILABLE"
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 2 if args.require_raw_root else 0

    try:
        result["host_inventory"] = inventory_pre_final(source)
        result["real_minute_enumeration_canary"] = real_minute_enumeration_canary(source)

        torch.manual_seed(1109)
        model = ReplayCentralBrain16M()
        report = brain_parameter_report(model)
        if report["total"] != BRAIN_PARAMETER_TARGET:
            raise RuntimeError("BRAIN_PARAMETER_COUNT_CHANGED")
        result["brain_parameter_report"] = report
        result["gradient_boundary_canary"] = gradient_boundary_canary(model, batch=4)
        if not result["gradient_boundary_canary"]["external_input_gradients_blocked"]:
            raise RuntimeError("EXTERNAL_INPUT_GRADIENT_LEAK")
        if not result["gradient_boundary_canary"]["all_six_groups_receive_gradient"]:
            raise RuntimeError("MISSING_BRAIN_GROUP_GRADIENT")

        result["status"] = "PASS_R0_HOST_AND_MINUTE_CANARY"
    except Exception as exc:
        result["status"] = "EXECUTION_BLOCKED__SEMANTIC_OR_INFRA_CANARY_FAILURE"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
