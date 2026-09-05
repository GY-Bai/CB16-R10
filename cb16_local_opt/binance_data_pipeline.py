from __future__ import annotations

"""
OCI/Binance acquisition and deterministic canonical dataset materialization.

Network acquisition is intended for the Japan OCI node.  The Shanxi node can consume
the resulting canonical files without needing direct Binance connectivity.

Scientific identity is based on canonical rows, not Parquet byte identity.  Different
pyarrow versions may write different Parquet metadata while representing identical rows.
"""

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = pq = None


BINANCE_KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

CANONICAL_COLUMNS = ("timestamp","open","high","low","close","volume")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk_bytes), b""):
            h.update(b)
    return h.hexdigest()


def canonical_row_bytes(row: Sequence[Any]) -> bytes:
    ts, o, h, l, c, v = row
    return f"{int(ts)},{float(o):.17g},{float(h):.17g},{float(l):.17g},{float(c):.17g},{float(v):.17g}\n".encode()


def canonical_dataset_hash(rows: Iterable[Sequence[Any]]) -> tuple[str,int,int,int]:
    h = hashlib.sha256()
    n = 0
    first = last = None
    for row in rows:
        ts = int(row[0])
        if first is None:
            first = ts
        if last is not None and ts <= last:
            raise RuntimeError("CANONICAL_DATASET_CHRONOLOGY_VIOLATION")
        last = ts
        h.update(canonical_row_bytes(row))
        n += 1
    if n == 0:
        raise ValueError("empty canonical dataset")
    return h.hexdigest(), n, int(first), int(last)


def interval_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    mult = {
        "m":60_000,
        "h":3_600_000,
        "d":86_400_000,
        "w":604_800_000,
    }.get(unit)
    if mult is None:
        raise ValueError(f"unsupported fixed interval: {interval}")
    return value * mult


@dataclass(frozen=True)
class BinanceAcquisitionSpec:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    market: str = "spot"
    base_url: str = "https://api.binance.com"
    request_limit: int = 1000
    request_timeout_s: float = 30.0
    min_request_spacing_s: float = 0.15
    max_retries: int = 6

    def validate(self):
        if self.market != "spot":
            raise ValueError("R4 downloader currently implements Binance spot klines")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("bad time range")
        if not 1 <= self.request_limit <= 1000:
            raise ValueError("request_limit must be 1..1000")
        interval_ms(self.interval)


@dataclass(frozen=True)
class AcquisitionReceipt:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    rows: int
    first_timestamp: int
    last_timestamp: int
    canonical_dataset_hash: str
    raw_jsonl_path: str
    raw_jsonl_sha256: str
    completed: bool


@dataclass(frozen=True)
class MaterializationReceipt:
    symbol: str
    interval: str
    rows: int
    first_timestamp: int
    last_timestamp: int
    canonical_dataset_hash: str
    output_path: str
    output_format: str
    output_sha256: str
    row_group_size: int | None


class BinanceKlineDownloader:
    def __init__(self, spec: BinanceAcquisitionSpec):
        spec.validate()
        self.spec = spec

    def _url(self, start_ms: int) -> str:
        params = {
            "symbol": self.spec.symbol.upper(),
            "interval": self.spec.interval,
            "startTime": int(start_ms),
            "endTime": int(self.spec.end_ms - 1),
            "limit": int(self.spec.request_limit),
        }
        return self.spec.base_url.rstrip("/") + "/api/v3/klines?" + urllib.parse.urlencode(params)

    def _fetch_page(self, start_ms: int) -> list[list[Any]]:
        last_exc = None
        for attempt in range(self.spec.max_retries):
            try:
                req = urllib.request.Request(
                    self._url(start_ms),
                    headers={"User-Agent":"CB16-R4-Dataset-Builder/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self.spec.request_timeout_s) as resp:
                    payload = json.loads(resp.read())
                if not isinstance(payload, list):
                    raise RuntimeError(f"BINANCE_BAD_RESPONSE:{payload!r}")
                return payload
            except Exception as exc:
                last_exc = exc
                if attempt + 1 == self.spec.max_retries:
                    break
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
        raise RuntimeError("BINANCE_FETCH_FAILED") from last_exc

    def download_jsonl(
        self,
        output_path: str | Path,
        *,
        resume: bool = True,
    ) -> AcquisitionReceipt:
        """Download raw Binance rows into append-only JSONL with continuity checks."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        step = interval_ms(self.spec.interval)

        last_ts = None
        existing_rows = 0
        if out.exists() and resume:
            with out.open("rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ts = int(row[0])
                    if last_ts is not None and ts <= last_ts:
                        raise RuntimeError("EXISTING_RAW_CHRONOLOGY_INVALID")
                    last_ts = ts
                    existing_rows += 1
        elif out.exists():
            out.unlink()

        cursor = self.spec.start_ms if last_ts is None else last_ts + step
        mode = "ab" if out.exists() else "wb"

        with out.open(mode) as f:
            while cursor < self.spec.end_ms:
                page = self._fetch_page(cursor)
                if not page:
                    break
                wrote = 0
                for row in page:
                    ts = int(row[0])
                    if ts < cursor:
                        continue
                    if ts >= self.spec.end_ms:
                        break
                    if last_ts is not None:
                        if ts <= last_ts:
                            continue
                        expected = last_ts + step
                        if ts != expected:
                            raise RuntimeError(
                                f"BINANCE_KLINE_GAP expected={expected} got={ts}"
                            )
                    raw = json.dumps(row, separators=(",",":"), ensure_ascii=False).encode() + b"\n"
                    f.write(raw)
                    last_ts = ts
                    existing_rows += 1
                    wrote += 1
                f.flush()
                os.fsync(f.fileno())
                if wrote == 0:
                    break
                cursor = int(last_ts) + step
                time.sleep(self.spec.min_request_spacing_s)

        if existing_rows == 0:
            raise RuntimeError("NO_KLINES_DOWNLOADED")
        canon_hash, rows, first, last = canonical_dataset_hash(
            canonical_rows_from_jsonl(out)
        )
        return AcquisitionReceipt(
            symbol=self.spec.symbol.upper(),
            interval=self.spec.interval,
            start_ms=self.spec.start_ms,
            end_ms=self.spec.end_ms,
            rows=rows,
            first_timestamp=first,
            last_timestamp=last,
            canonical_dataset_hash=canon_hash,
            raw_jsonl_path=str(out),
            raw_jsonl_sha256=sha256_file(out),
            completed=(last + step >= self.spec.end_ms),
        )


def canonical_rows_from_jsonl(path: str | Path) -> Iterator[tuple[int,float,float,float,float,float]]:
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row) < 6:
                raise RuntimeError("BINANCE_RAW_ROW_TOO_SHORT")
            yield (
                int(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            )


def canonical_rows_from_csv(path: str | Path) -> Iterator[tuple[int,float,float,float,float,float]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        missing = set(CANONICAL_COLUMNS) - set(r.fieldnames or [])
        if missing:
            raise RuntimeError(f"CSV_MISSING_COLUMNS:{sorted(missing)}")
        for row in r:
            yield (
                int(row["timestamp"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )


def _validate_ohlcv(row):
    ts,o,h,l,c,v = row
    if min(o,h,l,c) <= 0 or v < 0:
        raise RuntimeError("INVALID_CANONICAL_OHLCV")
    if l > min(o,c) or h < max(o,c):
        raise RuntimeError("INVALID_CANONICAL_OHLC_ORDER")


class DeterministicDatasetBuilder:
    def __init__(self, *, row_group_size: int = 65536, compression: str = "zstd"):
        if row_group_size <= 0:
            raise ValueError("row_group_size must be positive")
        self.row_group_size = int(row_group_size)
        self.compression = compression

    def materialize_csv(
        self,
        rows: Iterable[Sequence[Any]],
        output_path: str | Path,
        *,
        symbol: str,
        interval: str,
    ) -> MaterializationReceipt:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=out.name+".", suffix=".partial", dir=out.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        hasher = hashlib.sha256()
        count = 0
        first = last = None
        try:
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(CANONICAL_COLUMNS)
                for row in rows:
                    row = (int(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]))
                    _validate_ohlcv(row)
                    if last is not None and row[0] <= last:
                        raise RuntimeError("MATERIALIZATION_CHRONOLOGY_VIOLATION")
                    if first is None: first = row[0]
                    last = row[0]
                    hasher.update(canonical_row_bytes(row))
                    w.writerow(row)
                    count += 1
                f.flush(); os.fsync(f.fileno())
            if count == 0:
                raise ValueError("empty dataset")
            os.replace(tmp,out)
        finally:
            tmp.unlink(missing_ok=True)
        return MaterializationReceipt(
            symbol=symbol.upper(), interval=interval, rows=count,
            first_timestamp=int(first), last_timestamp=int(last),
            canonical_dataset_hash=hasher.hexdigest(),
            output_path=str(out), output_format="csv",
            output_sha256=sha256_file(out), row_group_size=None,
        )

    def materialize_parquet(
        self,
        rows: Iterable[Sequence[Any]],
        output_path: str | Path,
        *,
        symbol: str,
        interval: str,
    ) -> MaterializationReceipt:
        if pa is None or pq is None:
            raise RuntimeError("PYARROW_REQUIRED_FOR_PARQUET")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".partial")
        tmp.unlink(missing_ok=True)

        schema = pa.schema([
            ("timestamp", pa.int64()),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ])
        writer = pq.ParquetWriter(
            tmp,
            schema,
            compression=self.compression,
            use_dictionary=False,
            write_statistics=True,
        )
        h = hashlib.sha256()
        count=0; first=last=None
        buf = {k:[] for k in CANONICAL_COLUMNS}
        try:
            for row in rows:
                row=(int(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]))
                _validate_ohlcv(row)
                if last is not None and row[0] <= last:
                    raise RuntimeError("MATERIALIZATION_CHRONOLOGY_VIOLATION")
                if first is None: first=row[0]
                last=row[0]
                h.update(canonical_row_bytes(row))
                for k,v in zip(CANONICAL_COLUMNS,row):
                    buf[k].append(v)
                count += 1
                if len(buf["timestamp"]) >= self.row_group_size:
                    writer.write_table(pa.table(buf,schema=schema), row_group_size=self.row_group_size)
                    buf={k:[] for k in CANONICAL_COLUMNS}
            if buf["timestamp"]:
                writer.write_table(pa.table(buf,schema=schema), row_group_size=self.row_group_size)
            writer.close()
            if count == 0:
                raise ValueError("empty dataset")
            os.replace(tmp,out)
        except Exception:
            try: writer.close()
            except Exception: pass
            tmp.unlink(missing_ok=True)
            raise

        return MaterializationReceipt(
            symbol=symbol.upper(), interval=interval, rows=count,
            first_timestamp=int(first), last_timestamp=int(last),
            canonical_dataset_hash=h.hexdigest(),
            output_path=str(out), output_format="parquet",
            output_sha256=sha256_file(out), row_group_size=self.row_group_size,
        )


def write_receipt(receipt: Any, path: str | Path) -> Path:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(asdict(receipt),indent=2,ensure_ascii=False)+"\n")
    return p
