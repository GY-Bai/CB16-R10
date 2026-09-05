from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

MINUTE_MS = 60_000
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 24 * HOUR_MS
WEEK_MS = 7 * DAY_MS
WEEK_ANCHOR_MS = int(datetime(1970, 1, 5, tzinfo=timezone.utc).timestamp() * 1000)  # Monday 00:00 UTC

KLINE_COLUMNS = (
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
)

_INTERVAL_RE = re.compile(r"^(\d+)(m|h|d|w|M)$")
_MONTH_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)-1m-(?P<year>\d{4})-(?P<month>\d{2})\.zip$")


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def normalize_timestamp_ms(value: str | int | float) -> int:
    x = int(float(value))
    a = abs(x)
    if a >= 10**17:  # ns
        return x // 1_000_000
    if a >= 10**14:  # us
        return x // 1_000
    if a >= 10**11:  # ms
        return x
    if a >= 10**9:   # seconds
        return x * 1_000
    raise ValueError(f"timestamp magnitude unsupported: {x}")


def parse_interval(interval: str) -> tuple[str, int | None]:
    m = _INTERVAL_RE.match(interval)
    if not m:
        raise ValueError(f"unsupported interval {interval!r}")
    n = int(m.group(1)); unit = m.group(2)
    if n <= 0:
        raise ValueError("interval multiplier must be positive")
    if unit == "M":
        if n != 1:
            raise ValueError("calendar month supports only 1M")
        return "calendar_month", None
    mult = {"m": MINUTE_MS, "h": HOUR_MS, "d": DAY_MS, "w": WEEK_MS}[unit]
    return "fixed", n * mult


@dataclass(frozen=True)
class KlineRecord:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_asset_volume: float = 0.0
    number_of_trades: int = 0
    taker_buy_base_asset_volume: float = 0.0
    taker_buy_quote_asset_volume: float = 0.0

    def ohlcv(self) -> np.ndarray:
        return np.asarray([self.open, self.high, self.low, self.close, self.volume], dtype=np.float32)


@dataclass(frozen=True)
class FundingRecord:
    funding_time: int
    funding_rate: float
    mark_price: float | None = None


@dataclass(frozen=True)
class AggregatedKline:
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float
    source_rows: int

    def ohlcv(self) -> np.ndarray:
        return np.asarray([self.open, self.high, self.low, self.close, self.volume], dtype=np.float32)


@dataclass(frozen=True)
class SensoryDecisionFrameR10:
    symbol: str
    decision_time_ms: int
    micro_1m_60x5: np.ndarray
    micro_stamps_60x5: np.ndarray
    hourly_64x5: np.ndarray
    hourly_stamps_64x5: np.ndarray
    ordered4h30: np.ndarray

    @property
    def macro_1h_32x5(self) -> np.ndarray:
        return self.hourly_64x5[-32:]

    @property
    def macro_stamps_32x5(self) -> np.ndarray:
        return self.hourly_stamps_64x5[-32:]

    @property
    def medium_close_64(self) -> np.ndarray:
        return self.hourly_64x5[:, 3]


def _is_header(row: Sequence[str]) -> bool:
    if not row:
        return True
    try:
        float(row[0]); return False
    except Exception:
        return True


def _row_to_kline(row: Sequence[str]) -> KlineRecord:
    if len(row) < 6:
        raise ValueError(f"kline row too short: {len(row)}")
    vals = list(row) + ["0"] * max(0, 12 - len(row))
    ot = normalize_timestamp_ms(vals[0])
    ct = normalize_timestamp_ms(vals[6]) if vals[6] not in ("", None) else ot + MINUTE_MS - 1
    return KlineRecord(
        open_time=ot,
        open=float(vals[1]), high=float(vals[2]), low=float(vals[3]), close=float(vals[4]), volume=float(vals[5]),
        close_time=ct,
        quote_asset_volume=float(vals[7] or 0.0),
        number_of_trades=int(float(vals[8] or 0)),
        taker_buy_base_asset_volume=float(vals[9] or 0.0),
        taker_buy_quote_asset_volume=float(vals[10] or 0.0),
    )


def verify_binance_checksum(zip_path: str | Path) -> None:
    p = Path(zip_path)
    side = Path(str(p) + ".CHECKSUM")
    if not side.is_file():
        raise FileNotFoundError(side)
    text = side.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise RuntimeError(f"empty checksum: {side}")
    expected = text.split()[0].lower()
    actual = sha256_file(p)
    if actual != expected:
        raise RuntimeError(f"CHECKSUM_MISMATCH:{p.name}:{actual}!={expected}")


class BinanceUSDMArchiveSourceR10:
    """Read the user's Binance Vision USD-M monthly 1m/funding archive directly.

    Default root matches the Shanxi path supplied by the user. No network access is used.
    """

    def __init__(self, root: str | Path = "/data/cb16_hdd/binance_usdm_1m_funding_2020_2026"):
        self.root = Path(root)
        self.kline_root = self.root / "klines_1m"
        self.funding_root = self.root / "fundingRate"
        self.download_manifest = self.root / "DOWNLOAD_MANIFEST.json"

    def symbols(self) -> list[str]:
        if not self.kline_root.is_dir():
            return []
        return sorted(p.name for p in self.kline_root.iterdir() if p.is_dir())

    def validate_layout(self) -> dict:
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if not self.kline_root.is_dir():
            raise FileNotFoundError(self.kline_root)
        if not self.funding_root.is_dir():
            raise FileNotFoundError(self.funding_root)
        out = {"root": str(self.root), "symbols": self.symbols(), "download_manifest": self.download_manifest.is_file()}
        if self.download_manifest.is_file():
            try:
                m = json.loads(self.download_manifest.read_text())
                out["manifest_json_valid"] = True
                out["manifest_type"] = type(m).__name__
            except Exception as exc:
                raise RuntimeError(f"DOWNLOAD_MANIFEST_INVALID:{exc}") from exc
        return out

    def monthly_kline_archives(self, symbol: str) -> list[Path]:
        d = self.kline_root / symbol
        if not d.is_dir():
            raise FileNotFoundError(d)
        rows = []
        for p in d.glob(f"{symbol}-1m-????-??.zip"):
            m = _MONTH_RE.match(p.name)
            if m:
                rows.append((int(m.group("year")), int(m.group("month")), p))
        rows.sort()
        return [p for _, _, p in rows]

    def iter_1m(self, symbol: str, *, verify_checksums: bool = False, strict_chronology: bool = True) -> Iterator[KlineRecord]:
        archives = self.monthly_kline_archives(symbol)
        if not archives:
            raise RuntimeError(f"NO_1M_ARCHIVES:{symbol}")
        prev = None
        for zp in archives:
            if verify_checksums:
                verify_binance_checksum(zp)
            with zipfile.ZipFile(zp, "r") as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
                if len(members) != 1:
                    csv_members = [n for n in members if n.lower().endswith(".csv")]
                    if len(csv_members) != 1:
                        raise RuntimeError(f"ZIP_MEMBER_AMBIGUITY:{zp}:{members[:5]}")
                    member = csv_members[0]
                else:
                    member = members[0]
                with zf.open(member, "r") as raw:
                    txt = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    reader = csv.reader(txt)
                    for row in reader:
                        if not row or _is_header(row):
                            continue
                        rec = _row_to_kline(row)
                        if strict_chronology and prev is not None:
                            if rec.open_time <= prev:
                                raise RuntimeError(f"NON_INCREASING_1M:{symbol}:{prev}->{rec.open_time}")
                            if rec.open_time - prev != MINUTE_MS:
                                raise RuntimeError(f"ONE_MINUTE_GAP:{symbol}:{prev}->{rec.open_time}")
                        prev = rec.open_time
                        yield rec

    def iter_funding(self, symbol: str, *, verify_checksums: bool = False) -> Iterator[FundingRecord]:
        d = self.funding_root / symbol
        if not d.is_dir():
            return
        archives = sorted(d.glob(f"{symbol}-fundingRate-????-??.zip"))
        for zp in archives:
            if verify_checksums:
                verify_binance_checksum(zp)
            with zipfile.ZipFile(zp, "r") as zf:
                members = [n for n in zf.namelist() if not n.endswith("/")]
                if not members:
                    continue
                with zf.open(members[0], "r") as raw:
                    txt = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    reader = csv.reader(txt)
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
                            yield FundingRecord(normalize_timestamp_ms(tv), float(rv), None if mp in (None, "") else float(mp))
                        else:
                            if len(row) < 2:
                                continue
                            yield FundingRecord(normalize_timestamp_ms(row[0]), float(row[1]), float(row[2]) if len(row)>2 and row[2] else None)


def _fixed_bucket_start(ts: int, width_ms: int) -> int:
    if width_ms == WEEK_MS:
        return ((ts - WEEK_ANCHOR_MS) // WEEK_MS) * WEEK_MS + WEEK_ANCHOR_MS
    return (ts // width_ms) * width_ms


def _month_bucket_start(ts: int) -> int:
    dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _month_expected_rows(start_ms: int) -> int:
    dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    return calendar.monthrange(dt.year, dt.month)[1] * 1440


def aggregate_1m(records: Iterable[KlineRecord], interval: str, *, drop_boundary_partials: bool = True) -> Iterator[AggregatedKline]:
    kind, width = parse_interval(interval)
    cur_key = None; buf: list[KlineRecord] = []; bucket_index = 0

    def finalize(final_flush: bool = False):
        nonlocal buf, bucket_index
        if not buf:
            return None
        start = cur_key
        expected = _month_expected_rows(start) if kind == "calendar_month" else width // MINUTE_MS
        contiguous = all(buf[i].open_time - buf[i-1].open_time == MINUTE_MS for i in range(1, len(buf)))
        complete = (
            contiguous and len(buf) == expected and buf[0].open_time == start and
            buf[-1].open_time == start + (expected - 1) * MINUTE_MS
        )
        boundary = bucket_index == 0 or final_flush
        if not complete:
            if drop_boundary_partials and boundary:
                bucket_index += 1; buf = []; return None
            raise RuntimeError(f"INCOMPLETE_{interval}_BUCKET:start={start}:rows={len(buf)}:expected={expected}")
        x0, x1 = buf[0], buf[-1]
        out = AggregatedKline(
            interval=interval, open_time=start, open=x0.open,
            high=max(x.high for x in buf), low=min(x.low for x in buf), close=x1.close,
            volume=sum(x.volume for x in buf), close_time=x1.close_time,
            quote_asset_volume=sum(x.quote_asset_volume for x in buf),
            number_of_trades=sum(x.number_of_trades for x in buf),
            taker_buy_base_asset_volume=sum(x.taker_buy_base_asset_volume for x in buf),
            taker_buy_quote_asset_volume=sum(x.taker_buy_quote_asset_volume for x in buf),
            source_rows=len(buf),
        )
        bucket_index += 1; buf = []; return out

    for rec in records:
        key = _month_bucket_start(rec.open_time) if kind == "calendar_month" else _fixed_bucket_start(rec.open_time, width)
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            out = finalize(False)
            if out is not None:
                yield out
            cur_key = key
        buf.append(rec)
    out = finalize(True)
    if out is not None:
        yield out


def stamps_from_open_times_ms(ts: Sequence[int] | np.ndarray) -> np.ndarray:
    out = np.empty((len(ts), 5), dtype=np.float32)
    for i, t in enumerate(ts):
        d = datetime.fromtimestamp(int(t) / 1000, tz=timezone.utc)
        out[i] = (d.minute, d.hour, d.weekday(), d.day, d.month)
    return out


def ordered4h30_from_hourly(hourly_24x5: np.ndarray) -> np.ndarray:
    x = np.asarray(hourly_24x5, dtype=np.float64)
    if x.shape != (24, 5):
        raise ValueError(f"expected [24,5], got {x.shape}")
    slots = []
    for i in range(6):
        b = x[i*4:(i+1)*4]
        slots.extend([b[0,0], np.max(b[:,1]), np.min(b[:,2]), b[-1,3], np.sum(b[:,4])])
    return np.asarray(slots, dtype=np.float32)


def iter_sensory_frames(source: BinanceUSDMArchiveSourceR10, symbol: str, *, verify_checksums: bool = False) -> Iterator[SensoryDecisionFrameR10]:
    """Produce causally aligned hourly decision frames directly from 1m archives.

    A frame at decision time H uses only the 60 minutes ending at H-1m and the 64 complete
    hourly bars ending at H-1h.  No current/future hour bytes are used.
    """
    minute_buf: deque[KlineRecord] = deque(maxlen=60)
    hour_buf: deque[AggregatedKline] = deque(maxlen=64)
    current_hour: list[KlineRecord] = []
    current_start = None

    def finalize_hour(rows: list[KlineRecord], start: int) -> AggregatedKline:
        if len(rows) != 60 or rows[0].open_time != start or rows[-1].open_time != start + 59*MINUTE_MS:
            raise RuntimeError(f"INCOMPLETE_1h_BUCKET:{symbol}:{start}:rows={len(rows)}")
        return AggregatedKline(
            interval="1h", open_time=start, open=rows[0].open, high=max(x.high for x in rows),
            low=min(x.low for x in rows), close=rows[-1].close, volume=sum(x.volume for x in rows),
            close_time=rows[-1].close_time, quote_asset_volume=sum(x.quote_asset_volume for x in rows),
            number_of_trades=sum(x.number_of_trades for x in rows),
            taker_buy_base_asset_volume=sum(x.taker_buy_base_asset_volume for x in rows),
            taker_buy_quote_asset_volume=sum(x.taker_buy_quote_asset_volume for x in rows), source_rows=60,
        )

    for rec in source.iter_1m(symbol, verify_checksums=verify_checksums, strict_chronology=True):
        hstart = (rec.open_time // HOUR_MS) * HOUR_MS
        if current_start is None:
            current_start = hstart
        if hstart != current_start:
            if current_hour:
                # First listed partial hour is allowed to be dropped; after that strictness applies.
                if len(current_hour) == 60 and current_hour[0].open_time == current_start:
                    hb = finalize_hour(current_hour, current_start)
                    hour_buf.append(hb)
                elif hour_buf:
                    raise RuntimeError(f"INTERNAL_PARTIAL_HOUR:{symbol}:{current_start}:{len(current_hour)}")
            current_hour = []
            current_start = hstart
        current_hour.append(rec)
        minute_buf.append(rec)
        if len(current_hour) == 60 and rec.open_time == current_start + 59*MINUTE_MS:
            hb = finalize_hour(current_hour, current_start)
            # Avoid double-finalization at next hour transition.
            hour_buf.append(hb)
            current_hour = []
            current_start = hstart + HOUR_MS
            if len(hour_buf) >= 64 and len(minute_buf) == 60:
                hrs = list(hour_buf)
                hmat = np.stack([x.ohlcv() for x in hrs], axis=0).astype(np.float32)
                hts = np.asarray([x.open_time for x in hrs], dtype=np.int64)
                mins = list(minute_buf)
                mmat = np.stack([x.ohlcv() for x in mins], axis=0).astype(np.float32)
                mts = np.asarray([x.open_time for x in mins], dtype=np.int64)
                yield SensoryDecisionFrameR10(
                    symbol=symbol,
                    decision_time_ms=hb.open_time + HOUR_MS,
                    micro_1m_60x5=mmat,
                    micro_stamps_60x5=stamps_from_open_times_ms(mts),
                    hourly_64x5=hmat,
                    hourly_stamps_64x5=stamps_from_open_times_ms(hts),
                    ordered4h30=ordered4h30_from_hourly(hmat[-24:]),
                )
