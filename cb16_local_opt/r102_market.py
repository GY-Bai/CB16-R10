from __future__ import annotations

import csv
import hashlib
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
    FUNDING_CANONICAL_JITTER_TOLERANCE_MS, FUNDING_CANONICALIZATION_POLICY,
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
                    if rec.open_time % MINUTE_MS != 0:
                        raise RuntimeError(f"NON_MINUTE_ALIGNED_1M:{symbol}:{rec.open_time}")
                    if prev is not None:
                        if rec.open_time <= prev:
                            raise RuntimeError(f"NON_INCREASING_1M:{symbol}:{prev}->{rec.open_time}")
                        delta = rec.open_time - prev
                        if delta % MINUTE_MS != 0:
                            raise RuntimeError(f"NON_MINUTE_ALIGNED_1M_GAP:{symbol}:{prev}->{rec.open_time}")
                        # A positive whole-minute archive gap is real missing chronology, not a
                        # synthetic-data request. Downstream sensory state resets at the gap.
                    prev = rec.open_time
                    yield rec


@dataclass(frozen=True)
class CanonicalFundingEventR1021:
    """One archived funding event mapped to the hourly simulator clock.

    `raw_time_ms` is preserved for provenance. `canonical_hour_ms` is the nearest
    UTC hour only when the vendor timestamp differs by at most the frozen
    tolerance. This repairs timestamp spelling, not economic timing: the event is
    still applied exactly once and is never forward-filled.
    """
    raw_time_ms: int
    canonical_hour_ms: int
    jitter_ms: int
    funding_rate: float
    mark_price: float | None = None


def canonicalize_funding_event_hour_r1021(raw_time_ms: int, *, tolerance_ms: int = FUNDING_CANONICAL_JITTER_TOLERANCE_MS) -> tuple[int, int]:
    raw = int(raw_time_ms)
    tol = int(tolerance_ms)
    if tol < 0 or tol >= HOUR_MS // 2:
        raise ValueError(f"INVALID_FUNDING_JITTER_TOLERANCE_MS:{tol}")
    # Integer nearest-hour rounding; avoids Python banker's rounding.
    canonical = ((raw + HOUR_MS // 2) // HOUR_MS) * HOUR_MS
    jitter = raw - canonical
    if abs(jitter) > tol:
        raise RuntimeError(
            f"FUNDING_TIMESTAMP_JITTER_EXCEEDS_TOLERANCE:raw={utc_iso_from_ms(raw)}:"
            f"nearest_hour={utc_iso_from_ms(canonical)}:jitter_ms={jitter}:tolerance_ms={tol}"
        )
    return int(canonical), int(jitter)


def _iter_canonical_funding_events_bounded(
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    *,
    verify_checksums: bool = False,
) -> Iterator[CanonicalFundingEventR1021]:
    prev_raw = None
    prev_canonical = None
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
                        raw_ms = normalize_timestamp_ms(tv)
                        rate = float(rv)
                        mark = None if mp in (None, "") else float(mp)
                    else:
                        if len(row) < 2:
                            continue
                        raw_ms = normalize_timestamp_ms(row[0])
                        rate = float(row[1])
                        mark = float(row[2]) if len(row) > 2 and row[2] else None

                    # Holdout protection is checked against both the raw vendor timestamp
                    # and the normalized hourly key. The forbidden archive itself is never opened.
                    assert_not_forbidden_timestamp(raw_ms, what=f"funding_raw:{symbol}")
                    canonical_ms, jitter_ms = canonicalize_funding_event_hour_r1021(raw_ms)
                    assert_not_forbidden_timestamp(canonical_ms, what=f"funding_canonical:{symbol}")

                    if prev_raw is not None and raw_ms <= prev_raw:
                        raise RuntimeError(f"NON_INCREASING_FUNDING_RAW:{symbol}:{prev_raw}->{raw_ms}")
                    if prev_canonical is not None and canonical_ms <= prev_canonical:
                        raise RuntimeError(
                            f"NON_INCREASING_OR_COLLIDING_FUNDING_CANONICAL:{symbol}:"
                            f"{prev_canonical}->{canonical_ms}:raw={raw_ms}"
                        )
                    prev_raw = raw_ms
                    prev_canonical = canonical_ms
                    yield CanonicalFundingEventR1021(
                        raw_time_ms=int(raw_ms), canonical_hour_ms=int(canonical_ms),
                        jitter_ms=int(jitter_ms), funding_rate=float(rate), mark_price=mark,
                    )


def iter_funding_bounded(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[FundingRecord]:
    """Compatibility iterator returning hourly-keyed funding records.

    Exact-hour raw timestamps pass unchanged. Official Binance millisecond jitter
    within ±1000 ms is normalized to the nearest UTC hour. Anything larger fails
    closed. No rate is forward-filled.
    """
    for ev in _iter_canonical_funding_events_bounded(source, symbol, verify_checksums=verify_checksums):
        yield FundingRecord(ev.canonical_hour_ms, ev.funding_rate, ev.mark_price)


def scan_funding_timestamp_compatibility_r1021(
    source: BinanceUSDMArchiveSourceR10,
    symbols: Sequence[str],
    *,
    verify_checksums: bool = False,
) -> dict:
    rows = {}
    all_abs = []
    total = normalized = 0
    for symbol in symbols:
        n = nz = 0
        max_abs = 0
        min_jitter = 0
        max_jitter = 0
        h = hashlib.sha256()
        first_raw = last_raw = None
        for ev in _iter_canonical_funding_events_bounded(source, symbol, verify_checksums=verify_checksums):
            n += 1; total += 1
            if ev.jitter_ms != 0:
                nz += 1; normalized += 1
            a = abs(ev.jitter_ms); all_abs.append(a); max_abs = max(max_abs, a)
            min_jitter = min(min_jitter, ev.jitter_ms); max_jitter = max(max_jitter, ev.jitter_ms)
            first_raw = ev.raw_time_ms if first_raw is None else first_raw
            last_raw = ev.raw_time_ms
            h.update(f"{ev.raw_time_ms},{ev.canonical_hour_ms},{ev.jitter_ms},{ev.funding_rate:.17g}\n".encode("ascii"))
        rows[symbol] = {
            "events": n, "normalized_events": nz, "exact_hour_events": n - nz,
            "min_jitter_ms": int(min_jitter), "max_jitter_ms": int(max_jitter),
            "max_abs_jitter_ms": int(max_abs), "raw_to_canonical_mapping_sha256": h.hexdigest(),
            "first_raw_time": None if first_raw is None else utc_iso_from_ms(first_raw),
            "last_raw_time": None if last_raw is None else utc_iso_from_ms(last_raw),
        }
    a = np.asarray(all_abs, dtype=np.int64) if all_abs else np.asarray([], dtype=np.int64)
    pct = lambda q: None if len(a) == 0 else int(np.percentile(a, q, method="higher"))
    return {
        "schema": "CB16_R10_2_1_FUNDING_TIMESTAMP_COMPATIBILITY_V1",
        "status": "PASS",
        "scientific_semantics_changed": False,
        "policy": FUNDING_CANONICALIZATION_POLICY,
        "tolerance_ms": FUNDING_CANONICAL_JITTER_TOLERANCE_MS,
        "event_semantics": "ONE_ARCHIVED_EVENT_APPLIED_ONCE_AT_CANONICAL_HOURLY_SIMULATOR_STEP__NO_FORWARD_FILL",
        "raw_timestamp_preserved_in_mapping_audit": True,
        "symbols": rows,
        "events_total": total,
        "normalized_events_total": normalized,
        "max_abs_jitter_ms": None if len(a) == 0 else int(a.max()),
        "p50_abs_jitter_ms": pct(50), "p95_abs_jitter_ms": pct(95), "p99_abs_jitter_ms": pct(99),
        "final_holdout_2025_09_accessed": False,
    }


def iter_sensory_frames_bounded(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[SensoryDecisionFrameR10]:
    minute_buf: deque[KlineRecord] = deque(maxlen=60)
    hour_buf: deque[AggregatedKline] = deque(maxlen=64)
    current_hour: list[KlineRecord] = []
    current_start = None
    prev_open_time = None

    def reset_temporal_context_after_gap() -> None:
        nonlocal current_hour, current_start
        minute_buf.clear()
        hour_buf.clear()
        current_hour = []
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
        if prev_open_time is not None:
            delta = rec.open_time - prev_open_time
            if delta <= 0:
                raise RuntimeError(f"NON_INCREASING_1M_FRAME_STREAM:{symbol}:{prev_open_time}->{rec.open_time}")
            if delta % MINUTE_MS != 0:
                raise RuntimeError(f"NON_MINUTE_ALIGNED_1M_FRAME_GAP:{symbol}:{prev_open_time}->{rec.open_time}")
            if delta > MINUTE_MS:
                # Critical compatibility rule: no pre-gap temporal state may bridge
                # into the first post-gap real row. No rows are fabricated.
                reset_temporal_context_after_gap()
        prev_open_time = rec.open_time
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


def _hourly_discontinuity_ranges(open_time_ms: Sequence[int]) -> list[tuple[int, int]]:
    """Return inclusive missing-hour ranges between real observed hourly rows.

    Any non-increasing or non-hour-aligned chronology remains a hard failure.
    A range can include both a source archive gap and the deliberate post-gap
    sensory warm-up period during which no fully contextualized frame exists.
    """
    vals = [int(x) for x in open_time_ms]
    out: list[tuple[int, int]] = []
    prev = None
    for t in vals:
        if t % HOUR_MS != 0:
            raise RuntimeError(f"NON_HOUR_ALIGNED_CACHE_TIMESTAMP:{t}")
        if prev is not None:
            if t <= prev:
                raise RuntimeError(f"NON_INCREASING_HOURLY_CACHE:{prev}->{t}")
            delta = t - prev
            if delta % HOUR_MS != 0:
                raise RuntimeError(f"NON_HOUR_ALIGNED_CACHE_GAP:{prev}->{t}")
            if delta > HOUR_MS:
                out.append((prev + HOUR_MS, t - HOUR_MS))
        prev = t
    return out


def _hour_is_inside_discontinuity(hour_ms: int, ranges: Sequence[tuple[int, int]]) -> bool:
    t = int(hour_ms)
    return any(int(a) <= t <= int(b) for a, b in ranges)


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
            if last_ts % HOUR_MS != 0:
                raise RuntimeError(f"NON_HOUR_ALIGNED_FRAME_TIMESTAMP:{symbol}:{last_ts}")
            if seen_last_hour is not None:
                delta = last_ts - seen_last_hour
                if delta <= 0 or delta % HOUR_MS != 0:
                    raise RuntimeError(f"INVALID_HOURLY_FRAME_CHRONOLOGY:{symbol}:{seen_last_hour}->{last_ts}")
                # Whole-hour discontinuities are preserved as real missing chronology.
                # Valid-anchor filtering below still rejects any window that crosses one.
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

    hourly_discontinuities = _hourly_discontinuity_ranges(hts.tolist())
    funding_map: dict[int, float] = {}
    hset = set(int(x) for x in hts.tolist())
    funding_raw_to_canonical = hashlib.sha256()
    funding_events_seen = 0
    funding_events_normalized = 0
    funding_events_omitted_for_discontinuity = 0
    funding_max_abs_jitter_ms = 0
    for ev in _iter_canonical_funding_events_bounded(source, symbol, verify_checksums=verify_checksums):
        funding_events_seen += 1
        if ev.jitter_ms != 0:
            funding_events_normalized += 1
        funding_max_abs_jitter_ms = max(funding_max_abs_jitter_ms, abs(int(ev.jitter_ms)))
        funding_raw_to_canonical.update(
            f"{ev.raw_time_ms},{ev.canonical_hour_ms},{ev.jitter_ms},{ev.funding_rate:.17g}\n".encode("ascii")
        )
        if hts[0] <= ev.canonical_hour_ms <= hts[-1]:
            if ev.canonical_hour_ms not in hset:
                if _hour_is_inside_discontinuity(ev.canonical_hour_ms, hourly_discontinuities):
                    # The real hourly market row/context is absent. Omitting this event is
                    # safer than shifting it to another hour or carrying its rate forward.
                    funding_events_omitted_for_discontinuity += 1
                    continue
                raise RuntimeError(
                    f"IN_RANGE_FUNDING_NOT_ALIGNED_TO_HOURLY_BAR:{symbol}:"
                    f"raw={ev.raw_time_ms}:canonical={ev.canonical_hour_ms}"
                )
            if ev.canonical_hour_ms in funding_map:
                raise RuntimeError(f"DUPLICATE_FUNDING_EVENT:{symbol}:{ev.canonical_hour_ms}")
            funding_map[int(ev.canonical_hour_ms)] = float(ev.funding_rate)
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
        "hourly_rows": int(len(hts)),
        "hourly_discontinuities": [
            {"missing_start": utc_iso_from_ms(int(a)), "missing_end": utc_iso_from_ms(int(b)),
             "missing_hours": int((b - a) // HOUR_MS + 1)}
            for a, b in hourly_discontinuities
        ],
        "gap_policy": "ALLOW_POSITIVE_MINUTE_ALIGNED_REAL_GAPS__RESET_ALL_TEMPORAL_CONTEXT__NO_SYNTHETIC_ROWS__NO_CROSS_GAP_ANCHORS",
        "funding_events_in_hourly_range": int(len(funding_map)),
        "funding_events_omitted_for_hourly_discontinuity_or_warmup": int(funding_events_omitted_for_discontinuity),
        "funding_gap_policy": "OMIT_EVENT_WHEN_CANONICAL_HOUR_IS_ABSENT_INSIDE_OBSERVED_HOURLY_DISCONTINUITY__NO_SHIFT__NO_FORWARD_FILL",
        "funding_events_seen_before_holdout": int(funding_events_seen),
        "funding_timestamp_normalized_events": int(funding_events_normalized),
        "funding_timestamp_max_abs_jitter_ms": int(funding_max_abs_jitter_ms),
        "funding_timestamp_tolerance_ms": int(FUNDING_CANONICAL_JITTER_TOLERANCE_MS),
        "funding_timestamp_policy": FUNDING_CANONICALIZATION_POLICY,
        "funding_raw_to_canonical_mapping_sha256": funding_raw_to_canonical.hexdigest(),
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
