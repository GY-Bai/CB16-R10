from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .binance_archive_input_r10 import (
    BinanceUSDMArchiveSourceR10, KlineRecord, FundingRecord, AggregatedKline,
    SensoryDecisionFrameR10, HOUR_MS, MINUTE_MS, _is_header, _row_to_kline,
    normalize_timestamp_ms, ordered4h30_from_hourly, stamps_from_open_times_ms,
    verify_binance_checksum,
)
from .r102_common import (
    ALL_SUPPORTED_SYMBOLS_R102, CANONICAL_SYMBOLS_R102, FORBIDDEN_FINAL_MONTH,
    FORBIDDEN_FINAL_START_MS, H72, TRAIN_END_MS, VALIDATION_END_MS,
    assert_not_forbidden_timestamp, atomic_write_json, month_key_from_ms, TRAIN_VALIDATION_PURGE_HOURS,
    sha256_file, utc_iso_from_ms,
)

_FUND_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-fundingRate-(?P<year>\d{4})-(?P<month>\d{2})\.zip$")
_KLINE_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-1m-(?P<year>\d{4})-(?P<month>\d{2})\.zip$")


def _month_of_archive(path: Path, regex: re.Pattern[str]) -> str:
    m = regex.match(path.name)
    if not m:
        raise ValueError(path)
    return f"{int(m.group('year')):04d}-{int(m.group('month')):02d}"


def bounded_kline_archives(source: BinanceUSDMArchiveSourceR10, symbol: str, *, end_month_exclusive: str = FORBIDDEN_FINAL_MONTH) -> list[Path]:
    out = []
    for p in source.monthly_kline_archives(symbol):
        month = _month_of_archive(p, _KLINE_RE)
        if month >= end_month_exclusive:
            continue
        out.append(p)
    return out


def bounded_funding_archives(source: BinanceUSDMArchiveSourceR10, symbol: str, *, end_month_exclusive: str = FORBIDDEN_FINAL_MONTH) -> list[Path]:
    d = source.funding_root / symbol
    if not d.is_dir():
        return []
    out = []
    for p in d.glob(f"{symbol}-fundingRate-????-??.zip"):
        month = _month_of_archive(p, _FUND_RE)
        if month < end_month_exclusive:
            out.append((month, p))
    return [p for _, p in sorted(out)]


def iter_1m_bounded(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[KlineRecord]:
    """Read only archive months strictly before the sealed 2025-09 holdout.

    The forbidden archive is never opened, even for a one-row boundary check.
    """
    archives = bounded_kline_archives(source, symbol)
    if not archives:
        raise RuntimeError(f"NO_ALLOWED_1M_ARCHIVES:{symbol}")
    prev = None
    for zp in archives:
        if FORBIDDEN_FINAL_MONTH in zp.name:
            raise RuntimeError(f"FINAL_HOLDOUT_ARCHIVE_SELECTED:{zp}")
        if verify_checksums:
            verify_binance_checksum(zp)
        with zipfile.ZipFile(zp, "r") as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            csv_members = [n for n in members if n.lower().endswith(".csv")]
            if len(members) == 1:
                member = members[0]
            elif len(csv_members) == 1:
                member = csv_members[0]
            else:
                raise RuntimeError(f"ZIP_MEMBER_AMBIGUITY:{zp}")
            with zf.open(member, "r") as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                for row in reader:
                    if not row or _is_header(row):
                        continue
                    rec = _row_to_kline(row)
                    assert_not_forbidden_timestamp(rec.open_time, what=f"1m:{symbol}")
                    if prev is not None:
                        if rec.open_time <= prev:
                            raise RuntimeError(f"NON_INCREASING_1M:{symbol}:{prev}->{rec.open_time}")
                        if rec.open_time - prev != MINUTE_MS:
                            raise RuntimeError(f"ONE_MINUTE_GAP:{symbol}:{prev}->{rec.open_time}")
                    prev = rec.open_time
                    yield rec


def iter_funding_bounded(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[FundingRecord]:
    prev = None
    for zp in bounded_funding_archives(source, symbol):
        if FORBIDDEN_FINAL_MONTH in zp.name:
            raise RuntimeError(f"FINAL_HOLDOUT_FUNDING_ARCHIVE_SELECTED:{zp}")
        if verify_checksums:
            verify_binance_checksum(zp)
        with zipfile.ZipFile(zp, "r") as zf:
            members = [n for n in zf.namelist() if not n.endswith("/")]
            if not members:
                continue
            with zf.open(members[0], "r") as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                header = None
                for row in reader:
                    if not row:
                        continue
                    if header is None and _is_header(row):
                        header = [x.strip() for x in row]
                        continue
                    if header:
                        obj = dict(zip(header, row))
                        tv = obj.get("calc_time") or obj.get("fundingTime") or obj.get("funding_time") or obj.get("time")
                        rv = obj.get("last_funding_rate") or obj.get("fundingRate") or obj.get("funding_rate") or obj.get("rate")
                        mp = obj.get("mark_price") or obj.get("markPrice")
                        if tv is None or rv is None:
                            raise RuntimeError(f"FUNDING_COLUMNS_UNRECOGNIZED:{header}")
                        rec = FundingRecord(normalize_timestamp_ms(tv), float(rv), None if mp in (None, "") else float(mp))
                    else:
                        if len(row) < 2:
                            continue
                        rec = FundingRecord(normalize_timestamp_ms(row[0]), float(row[1]), float(row[2]) if len(row) > 2 and row[2] else None)
                    assert_not_forbidden_timestamp(rec.funding_time, what=f"funding:{symbol}")
                    if rec.funding_time % HOUR_MS != 0:
                        raise RuntimeError(f"FUNDING_NOT_EXACT_HOUR:{symbol}:{utc_iso_from_ms(rec.funding_time)}")
                    if prev is not None and rec.funding_time <= prev:
                        raise RuntimeError(f"NON_INCREASING_FUNDING:{symbol}")
                    prev = rec.funding_time
                    yield rec


def iter_sensory_frames_bounded(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[SensoryDecisionFrameR10]:
    minute_buf: deque[KlineRecord] = deque(maxlen=60)
    hour_buf: deque[AggregatedKline] = deque(maxlen=64)
    current_hour: list[KlineRecord] = []
    current_start = None

    def finalize_hour(rows: list[KlineRecord], start: int) -> AggregatedKline:
        if len(rows) != 60 or rows[0].open_time != start or rows[-1].open_time != start + 59 * MINUTE_MS:
            raise RuntimeError(f"INCOMPLETE_1h_BUCKET:{symbol}:{start}:rows={len(rows)}")
        return AggregatedKline(
            interval="1h", open_time=start, open=rows[0].open,
            high=max(x.high for x in rows), low=min(x.low for x in rows), close=rows[-1].close,
            volume=sum(x.volume for x in rows), close_time=rows[-1].close_time,
            quote_asset_volume=sum(x.quote_asset_volume for x in rows),
            number_of_trades=sum(x.number_of_trades for x in rows),
            taker_buy_base_asset_volume=sum(x.taker_buy_base_asset_volume for x in rows),
            taker_buy_quote_asset_volume=sum(x.taker_buy_quote_asset_volume for x in rows),
            source_rows=60,
        )

    for rec in iter_1m_bounded(source, symbol, verify_checksums=verify_checksums):
        hstart = (rec.open_time // HOUR_MS) * HOUR_MS
        if current_start is None:
            current_start = hstart
        if hstart != current_start:
            if current_hour:
                if len(current_hour) == 60 and current_hour[0].open_time == current_start:
                    hour_buf.append(finalize_hour(current_hour, current_start))
                elif hour_buf:
                    raise RuntimeError(f"INTERNAL_PARTIAL_HOUR:{symbol}:{current_start}:{len(current_hour)}")
            current_hour = []
            current_start = hstart
        current_hour.append(rec)
        minute_buf.append(rec)
        if len(current_hour) == 60 and rec.open_time == current_start + 59 * MINUTE_MS:
            hb = finalize_hour(current_hour, current_start)
            hour_buf.append(hb)
            current_hour = []
            current_start = hstart + HOUR_MS
            if len(hour_buf) >= 64 and len(minute_buf) == 60:
                hrs = list(hour_buf)
                hmat = np.stack([x.ohlcv() for x in hrs]).astype(np.float32)
                hts = np.asarray([x.open_time for x in hrs], dtype=np.int64)
                mins = list(minute_buf)
                mmat = np.stack([x.ohlcv() for x in mins]).astype(np.float32)
                mts = np.asarray([x.open_time for x in mins], dtype=np.int64)
                yield SensoryDecisionFrameR10(
                    symbol=symbol, decision_time_ms=hb.open_time + HOUR_MS,
                    micro_1m_60x5=mmat, micro_stamps_60x5=stamps_from_open_times_ms(mts),
                    hourly_64x5=hmat, hourly_stamps_64x5=stamps_from_open_times_ms(hts),
                    ordered4h30=ordered4h30_from_hourly(hmat[-24:]),
                )


@dataclass(frozen=True)
class MarketCachePathsR102:
    symbol: str
    hourly_npz: Path
    frames_npz: Path
    manifest_json: Path


def _anchor_allowed(decision_ms: int, *, stride_hours: int) -> bool:
    # Global UTC phase prevents per-asset/post-result anchor drift.
    return ((decision_ms // HOUR_MS) % int(stride_hours)) == 0


def build_symbol_market_cache(
    *,
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    out_dir: str | Path,
    stride_hours: int = 512,
    prehistory_hours: int = 96,
    verify_checksums: bool = False,
) -> MarketCachePathsR102:
    if symbol not in ALL_SUPPORTED_SYMBOLS_R102:
        raise ValueError(f"UNREGISTERED_SYMBOL:{symbol}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    hourly_ts: list[int] = []
    hourly_ohlcv: list[np.ndarray] = []
    frame_rows: list[SensoryDecisionFrameR10] = []
    seen_last_hour = None

    for frame in iter_sensory_frames_bounded(source, symbol, verify_checksums=verify_checksums):
        # Last complete bar visible at decision t is the t-1h bar.
        last_ts = int(frame.decision_time_ms - HOUR_MS)
        last_ohlcv = np.asarray(frame.hourly_64x5[-1], dtype=np.float32)
        if seen_last_hour is None or last_ts > seen_last_hour:
            if seen_last_hour is not None and last_ts - seen_last_hour != HOUR_MS:
                raise RuntimeError(f"HOURLY_GAP_FROM_FRAME_STREAM:{symbol}:{seen_last_hour}->{last_ts}")
            hourly_ts.append(last_ts); hourly_ohlcv.append(last_ohlcv); seen_last_hour = last_ts
        # Holdout bytes are never read; futures also must mature before 2025-09.
        if frame.decision_time_ms + H72 * HOUR_MS > VALIDATION_END_MS:
            continue
        if not _anchor_allowed(frame.decision_time_ms, stride_hours=stride_hours):
            continue
        frame_rows.append(frame)

    if not hourly_ts or not frame_rows:
        raise RuntimeError(f"EMPTY_MARKET_CACHE:{symbol}")
    hts = np.asarray(hourly_ts, dtype=np.int64)
    hbar = np.stack(hourly_ohlcv).astype(np.float32)

    funding_map: dict[int, float] = {}
    hset = set(int(x) for x in hts.tolist())
    for f in iter_funding_bounded(source, symbol, verify_checksums=verify_checksums):
        if hts[0] <= f.funding_time <= hts[-1]:
            if f.funding_time not in hset:
                raise RuntimeError(f"IN_RANGE_FUNDING_NOT_ALIGNED_TO_HOURLY_BAR:{symbol}:{f.funding_time}")
            if f.funding_time in funding_map:
                raise RuntimeError(f"DUPLICATE_FUNDING_EVENT:{symbol}:{f.funding_time}")
            funding_map[int(f.funding_time)] = float(f.funding_rate)
    funding = np.asarray([funding_map.get(int(t), 0.0) for t in hts], dtype=np.float64)

    # Exact valid anchor filter after complete hourly sequence is known.
    idx_by_ts = {int(t): i for i, t in enumerate(hts)}
    valid_frames = []
    for f in frame_rows:
        t = int(f.decision_time_ms)
        # pre-roll bars t-prehistory ... t-1 and H72 future t ... t+71 must all exist.
        required = [t - prehistory_hours * HOUR_MS, t - HOUR_MS, t, t + (H72 - 1) * HOUR_MS]
        if all(x in idx_by_ts for x in required):
            a = idx_by_ts[t - prehistory_hours * HOUR_MS]
            b = idx_by_ts[t + (H72 - 1) * HOUR_MS]
            if b - a + 1 == prehistory_hours + H72:
                valid_frames.append(f)
    if not valid_frames:
        raise RuntimeError(f"NO_VALID_ANCHORS:{symbol}")

    hourly_path = out / f"{symbol}.hourly_r102.npz"
    np.savez_compressed(hourly_path, open_time_ms=hts, ohlcv=hbar, funding_rate=funding)
    frames_path = out / f"{symbol}.anchors_r102.npz"
    np.savez_compressed(
        frames_path,
        decision_time_ms=np.asarray([f.decision_time_ms for f in valid_frames], dtype=np.int64),
        micro_1m_60x5=np.stack([f.micro_1m_60x5 for f in valid_frames]).astype(np.float32),
        micro_stamps_60x5=np.stack([f.micro_stamps_60x5 for f in valid_frames]).astype(np.float32),
        hourly_64x5=np.stack([f.hourly_64x5 for f in valid_frames]).astype(np.float32),
        hourly_stamps_64x5=np.stack([f.hourly_stamps_64x5 for f in valid_frames]).astype(np.float32),
        ordered4h30=np.stack([f.ordered4h30 for f in valid_frames]).astype(np.float32),
    )
    n_train = sum(int(f.decision_time_ms + (H72 + TRAIN_VALIDATION_PURGE_HOURS) * HOUR_MS <= TRAIN_END_MS) for f in valid_frames)
    n_val = sum(int(TRAIN_END_MS <= f.decision_time_ms < VALIDATION_END_MS) for f in valid_frames)
    n_purge = len(valid_frames) - n_train - n_val
    manifest = {
        "schema": "CB16_R10_2_SYMBOL_MARKET_CACHE_MANIFEST_V1",
        "symbol": symbol, "status": "PASS", "forbidden_month_opened": False,
        "archive_boundary": f"month < {FORBIDDEN_FINAL_MONTH}",
        "hourly_rows": int(len(hts)), "funding_events": int(np.count_nonzero(funding)),
        "anchors": int(len(valid_frames)), "train_anchors": n_train, "validation_anchors": n_val, "purged_boundary_anchors": n_purge,
        "stride_hours": int(stride_hours), "prehistory_hours": int(prehistory_hours), "horizon_hours": H72,
        "hourly_sha256": sha256_file(hourly_path), "frames_sha256": sha256_file(frames_path),
        "first_hour": utc_iso_from_ms(int(hts[0])), "last_hour": utc_iso_from_ms(int(hts[-1])),
        "first_anchor": utc_iso_from_ms(int(valid_frames[0].decision_time_ms)),
        "last_anchor": utc_iso_from_ms(int(valid_frames[-1].decision_time_ms)),
    }
    manifest_path = out / f"{symbol}.manifest_r102.json"
    atomic_write_json(manifest_path, manifest)
    return MarketCachePathsR102(symbol, hourly_path, frames_path, manifest_path)


def load_anchor_frames(symbol: str, frames_npz: str | Path) -> list[SensoryDecisionFrameR10]:
    with np.load(frames_npz, allow_pickle=False) as z:
        out = []
        for i, t in enumerate(z["decision_time_ms"]):
            out.append(SensoryDecisionFrameR10(
                symbol=symbol, decision_time_ms=int(t),
                micro_1m_60x5=z["micro_1m_60x5"][i].copy(),
                micro_stamps_60x5=z["micro_stamps_60x5"][i].copy(),
                hourly_64x5=z["hourly_64x5"][i].copy(),
                hourly_stamps_64x5=z["hourly_stamps_64x5"][i].copy(),
                ordered4h30=z["ordered4h30"][i].copy(),
            ))
        return out


def preflight_all_ten_data(source: BinanceUSDMArchiveSourceR10, *, verify_checksum_samples: bool = True) -> dict:
    layout = source.validate_layout()
    missing = sorted(set(ALL_SUPPORTED_SYMBOLS_R102) - set(layout["symbols"]))
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_SYMBOLS:{missing}")
    rows = {}
    for sym in ALL_SUPPORTED_SYMBOLS_R102:
        ka = source.monthly_kline_archives(sym)
        fa = sorted((source.funding_root / sym).glob(f"{sym}-fundingRate-????-??.zip"))
        if not ka:
            raise RuntimeError(f"NO_KLINES:{sym}")
        # Verify a few immutable files, not the whole 1.3GB archive every launch.
        checked = []
        allowed = [p for p in ka if _month_of_archive(p, _KLINE_RE) < FORBIDDEN_FINAL_MONTH]
        if verify_checksum_samples and allowed:
            picks = sorted(set([0, len(allowed)//2, len(allowed)-1]))
            for i in picks:
                verify_binance_checksum(allowed[i]); checked.append(allowed[i].name)
        rows[sym] = {
            "kline_months_total": len(ka), "funding_months_total": len(fa),
            "first_kline": ka[0].name, "last_kline": ka[-1].name,
            "checksum_samples_passed": checked,
        }
    return {
        "schema": "CB16_R10_2_TEN_SYMBOL_DATA_PREFLIGHT_V1", "status": "PASS",
        "data_root": str(source.root), "symbols": rows,
        "forbidden_final_month": FORBIDDEN_FINAL_MONTH,
        "final_holdout_accessed": False,
    }
