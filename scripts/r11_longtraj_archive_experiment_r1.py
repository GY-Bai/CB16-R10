from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10, MINUTE_MS
from cb16_local_opt.full_minute_long_trajectory_r1 import expand_internal_halts_r1
from cb16_local_opt.longtraj_archive_r1 import inventory_prefinal_archives_r1, iter_prefinal_observed_1m_r1

SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
)


def sha_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scan_symbol(source: BinanceUSDMArchiveSourceR10, symbol: str) -> dict:
    inv = inventory_prefinal_archives_r1(source, symbol)
    t0 = time.monotonic()
    observed = 0
    expanded = 0
    imputed = 0
    gaps = 0
    max_gap_missing = 0
    first_ts = None
    last_ts = None
    prev_observed = None
    prev_expanded = None
    zero_volume_real = 0
    gap_examples = []

    stream = iter_prefinal_observed_1m_r1(source, symbol, verify_checksums=False)
    for env in expand_internal_halts_r1(stream):
        ts = int(env.record.open_time)
        if first_ts is None:
            first_ts = ts
        if prev_expanded is not None and ts - prev_expanded != MINUTE_MS:
            raise RuntimeError(f"EXPANDED_NONCONTIGUOUS:{symbol}:{prev_expanded}->{ts}")
        prev_expanded = ts
        last_ts = ts
        expanded += 1
        if env.halt_imputed:
            imputed += 1
            r = env.record
            if not (
                r.open == r.high == r.low == r.close
                and r.volume == 0.0
                and r.quote_asset_volume == 0.0
                and r.number_of_trades == 0
                and r.taker_buy_base_asset_volume == 0.0
                and r.taker_buy_quote_asset_volume == 0.0
            ):
                raise RuntimeError(f"HALT_FILL_CONTRACT_FAILURE:{symbol}:{ts}")
        else:
            observed += 1
            if env.record.volume == 0.0:
                zero_volume_real += 1
            if prev_observed is not None:
                delta = ts - prev_observed
                if delta > MINUTE_MS:
                    missing = delta // MINUTE_MS - 1
                    gaps += 1
                    max_gap_missing = max(max_gap_missing, missing)
                    if len(gap_examples) < 20:
                        gap_examples.append({
                            "left_observed_ms": prev_observed,
                            "right_observed_ms": ts,
                            "missing_minutes": int(missing),
                        })
            prev_observed = ts

    if observed == 0:
        raise RuntimeError(f"NO_OBSERVED_PREFINAL_ROWS:{symbol}")
    expected_expanded = observed + imputed
    if expanded != expected_expanded:
        raise RuntimeError("EXPANDED_COUNT_MISMATCH")
    if first_ts is not None and last_ts is not None:
        calendar_minutes = (last_ts - first_ts) // MINUTE_MS + 1
        if expanded != calendar_minutes:
            raise RuntimeError(f"ZIPPER_NOT_CALENDAR_CONTIGUOUS:{symbol}:{expanded}!={calendar_minutes}")

    return {
        "symbol": symbol,
        "archive_inventory": {
            "count": len(inv.archives),
            "first_year_month": inv.first_year_month,
            "last_year_month": inv.last_year_month,
            "archive_names_hash": sha_json(inv.archives),
            "final_2025_09_archive_selected": any("-2025-09.zip" in x for x in inv.archives),
        },
        "first_timestamp_ms": first_ts,
        "last_timestamp_ms": last_ts,
        "observed_minutes": observed,
        "halt_imputed_minutes": imputed,
        "expanded_calendar_minutes": expanded,
        "gap_events": gaps,
        "max_gap_missing_minutes": max_gap_missing,
        "real_zero_volume_minutes": zero_volume_real,
        "gap_examples": gap_examples,
        "strict_expanded_stride_1m": True,
        "account_reset_implied": False,
        "elapsed_seconds": time.monotonic() - t0,
    }


def make_balanced_schedule(scans: list[dict], chunk_minutes: int) -> dict:
    if chunk_minutes <= 0:
        raise ValueError("chunk_minutes must be positive")
    counts = {x["symbol"]: int(x["expanded_calendar_minutes"]) for x in scans}
    target = min(counts.values())
    chunks_per_symbol = target // chunk_minutes
    if chunks_per_symbol < 1:
        raise RuntimeError("INSUFFICIENT_HISTORY_FOR_BALANCED_CHUNK")
    sequence = []
    for k in range(chunks_per_symbol):
        for s in SYMBOLS:
            sequence.append({"round": k, "symbol": s, "chunk_index_within_symbol": k})
    return {
        "semantic_boundary": "optimizer/evidence scheduling only; never reorders or resets each asset's environment trajectory",
        "chunk_minutes": chunk_minutes,
        "target_minutes_per_symbol": chunks_per_symbol * chunk_minutes,
        "chunks_per_symbol": chunks_per_symbol,
        "schedule_entries": len(sequence),
        "round_robin_prefix": sequence[:100],
        "full_schedule_hash": sha_json(sequence),
        "source_counts": counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/cb16/raw")
    ap.add_argument("--mode", choices=("BTC_ZIPPER", "TEN_NATURAL", "TEN_BALANCED"), required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--balanced-chunk-minutes", type=int, default=10080)  # one week
    args = ap.parse_args()

    source = BinanceUSDMArchiveSourceR10(args.raw_root)
    layout = source.validate_layout()
    symbols = ("BTCUSDT",) if args.mode == "BTC_ZIPPER" else SYMBOLS
    scans = [scan_symbol(source, s) for s in symbols]

    result = {
        "schema": "CB16_R11_LONGTRAJ_ARCHIVE_EXPERIMENT_R1_RESULT_V1",
        "mode": args.mode,
        "status": "PASS",
        "raw_root": args.raw_root,
        "layout_symbols": layout.get("symbols", []),
        "symbols": list(symbols),
        "scans": scans,
        "final_holdout_start_utc": "2025-09-01T00:00:00Z",
        "final_holdout_archive_filtered_before_open": True,
        "fresh_network_access": False,
        "halt_semantics": "prefix-last-close / zero-volume / real-reopen-untouched",
        "halt_flag_student_feature": False,
        "account_reset_at_gap": False,
    }

    if args.mode == "TEN_NATURAL":
        total = sum(int(x["expanded_calendar_minutes"]) for x in scans)
        result["natural_optimizer_exposure"] = {
            x["symbol"]: int(x["expanded_calendar_minutes"]) / total for x in scans
        }
    elif args.mode == "TEN_BALANCED":
        result["balanced_optimizer_schedule"] = make_balanced_schedule(scans, args.balanced_chunk_minutes)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({
        "status": result["status"],
        "mode": args.mode,
        "symbols": list(symbols),
        "expanded_minutes": {x["symbol"]: x["expanded_calendar_minutes"] for x in scans},
        "halt_minutes": {x["symbol"]: x["halt_imputed_minutes"] for x in scans},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
