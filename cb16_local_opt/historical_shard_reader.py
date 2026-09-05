from __future__ import annotations

"""
Out-of-core historical OHLCV shard reader.

Design goals:
- deterministic chronology;
- no need to materialize full history in RAM;
- explicit dataset/shard identity;
- Parquet/Arrow streaming when pyarrow is installed;
- CSV streaming fallback for portability/tests;
- strict no-duplicate/no-time-reversal validation;
- projection of only required columns.
"""

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

try:
    import pyarrow.dataset as pads
except Exception:  # optional dependency on the sandbox
    pads = None

CANONICAL_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    path: str
    symbol: str
    timeframe: str
    format: str
    start_timestamp: int
    end_timestamp: int
    rows: int
    sha256: str
    bytes: int

    def validate(self) -> None:
        if self.format not in {"csv", "parquet"}:
            raise ValueError("format must be csv/parquet")
        if self.start_timestamp > self.end_timestamp:
            raise ValueError("bad shard timestamp range")
        if self.rows <= 0:
            raise ValueError("rows must be positive")


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    symbols: tuple[str, ...]
    timeframe: str
    shards: tuple[ShardSpec, ...]
    canonical_columns: tuple[str, ...] = CANONICAL_COLUMNS
    source: str = "HISTORICAL_REPLAY"

    @property
    def content_hash(self) -> str:
        return canonical_hash({
            "dataset_id": self.dataset_id,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "shards": [asdict(x) for x in self.shards],
            "canonical_columns": self.canonical_columns,
            "source": self.source,
        })

    def validate(self, *, verify_files: bool = False) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id required")
        if not self.shards:
            raise ValueError("no shards")
        prev_by_symbol: dict[str, int] = {}
        ids = set()
        for s in sorted(self.shards, key=lambda x: (x.symbol, x.start_timestamp, x.shard_id)):
            s.validate()
            if s.shard_id in ids:
                raise ValueError(f"duplicate shard id: {s.shard_id}")
            ids.add(s.shard_id)
            prev = prev_by_symbol.get(s.symbol)
            if prev is not None and s.start_timestamp <= prev:
                raise ValueError(f"overlapping/non-increasing shard chronology for {s.symbol}")
            prev_by_symbol[s.symbol] = s.end_timestamp
            if verify_files:
                p = Path(s.path)
                if not p.is_file():
                    raise FileNotFoundError(p)
                if sha256_file(p) != s.sha256:
                    raise RuntimeError(f"SHARD_HASH_MISMATCH:{s.shard_id}")


@dataclass
class HistoricalBatch:
    symbol: str
    timeframe: str
    shard_id: str
    timestamp: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.timestamp))

    def validate(self, *, previous_timestamp: int | None = None) -> int:
        n = self.rows
        if n == 0:
            return previous_timestamp if previous_timestamp is not None else -1
        for name in ("open", "high", "low", "close", "volume"):
            a = np.asarray(getattr(self, name))
            if len(a) != n:
                raise ValueError(f"column length mismatch: {name}")
        ts = np.asarray(self.timestamp, dtype=np.int64)
        if np.any(np.diff(ts) <= 0):
            raise RuntimeError("NON_INCREASING_TIMESTAMP_WITHIN_BATCH")
        if previous_timestamp is not None and int(ts[0]) <= previous_timestamp:
            raise RuntimeError("NON_INCREASING_TIMESTAMP_ACROSS_BATCH")
        if np.any(self.low > np.minimum(self.open, self.close)):
            raise RuntimeError("OHLC_LOW_ORDER_VIOLATION")
        if np.any(self.high < np.maximum(self.open, self.close)):
            raise RuntimeError("OHLC_HIGH_ORDER_VIOLATION")
        if np.any(self.low <= 0) or np.any(self.open <= 0) or np.any(self.high <= 0) or np.any(self.close <= 0):
            raise RuntimeError("NONPOSITIVE_PRICE")
        if np.any(self.volume < 0):
            raise RuntimeError("NEGATIVE_VOLUME")
        return int(ts[-1])


def _to_numpy_dict_from_csv_rows(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    if not rows:
        return {}
    return {
        "timestamp": np.asarray([int(r["timestamp"]) for r in rows], dtype=np.int64),
        "open": np.asarray([float(r["open"]) for r in rows], dtype=np.float64),
        "high": np.asarray([float(r["high"]) for r in rows], dtype=np.float64),
        "low": np.asarray([float(r["low"]) for r in rows], dtype=np.float64),
        "close": np.asarray([float(r["close"]) for r in rows], dtype=np.float64),
        "volume": np.asarray([float(r["volume"]) for r in rows], dtype=np.float64),
    }


class HistoricalShardReader:
    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        batch_rows: int = 65536,
        arrow_batch_readahead: int = 2,
        arrow_fragment_readahead: int = 1,
        verify_hashes_on_open: bool = False,
    ):
        if batch_rows <= 0:
            raise ValueError("batch_rows must be positive")
        self.manifest = manifest
        self.batch_rows = int(batch_rows)
        self.arrow_batch_readahead = int(arrow_batch_readahead)
        self.arrow_fragment_readahead = int(arrow_fragment_readahead)
        manifest.validate(verify_files=verify_hashes_on_open)

    def iter_shards(self, *, symbol: str | None = None) -> Iterator[ShardSpec]:
        ss = sorted(self.manifest.shards, key=lambda x: (x.symbol, x.start_timestamp, x.shard_id))
        for s in ss:
            if symbol is None or s.symbol == symbol:
                yield s

    def _iter_csv(self, shard: ShardSpec) -> Iterator[HistoricalBatch]:
        with Path(shard.path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = set(CANONICAL_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"CSV_MISSING_COLUMNS:{sorted(missing)}")
            buf: list[dict[str, str]] = []
            for row in reader:
                buf.append(row)
                if len(buf) >= self.batch_rows:
                    d = _to_numpy_dict_from_csv_rows(buf)
                    yield HistoricalBatch(shard.symbol, shard.timeframe, shard.shard_id, **d)
                    buf.clear()
            if buf:
                d = _to_numpy_dict_from_csv_rows(buf)
                yield HistoricalBatch(shard.symbol, shard.timeframe, shard.shard_id, **d)

    def _iter_parquet(self, shard: ShardSpec) -> Iterator[HistoricalBatch]:
        if pads is None:
            raise RuntimeError(
                "PYARROW_NOT_INSTALLED: install pyarrow to stream Parquet shards"
            )
        dataset = pads.dataset(shard.path, format="parquet")
        scanner = dataset.scanner(
            columns=list(CANONICAL_COLUMNS),
            batch_size=self.batch_rows,
            batch_readahead=self.arrow_batch_readahead,
            fragment_readahead=self.arrow_fragment_readahead,
            use_threads=True,
            cache_metadata=True,
        )
        for rb in scanner.to_batches():
            cols = {}
            for name in CANONICAL_COLUMNS:
                arr = rb.column(rb.schema.get_field_index(name))
                cols[name] = arr.to_numpy(zero_copy_only=False)
            yield HistoricalBatch(
                symbol=shard.symbol,
                timeframe=shard.timeframe,
                shard_id=shard.shard_id,
                timestamp=np.asarray(cols["timestamp"], dtype=np.int64),
                open=np.asarray(cols["open"], dtype=np.float64),
                high=np.asarray(cols["high"], dtype=np.float64),
                low=np.asarray(cols["low"], dtype=np.float64),
                close=np.asarray(cols["close"], dtype=np.float64),
                volume=np.asarray(cols["volume"], dtype=np.float64),
            )

    def iter_batches(self, *, symbol: str | None = None) -> Iterator[HistoricalBatch]:
        prev_by_symbol: dict[str, int] = {}
        for shard in self.iter_shards(symbol=symbol):
            iterator = self._iter_csv(shard) if shard.format == "csv" else self._iter_parquet(shard)
            rows_seen = 0
            for batch in iterator:
                prev = prev_by_symbol.get(shard.symbol)
                last = batch.validate(previous_timestamp=prev)
                prev_by_symbol[shard.symbol] = last
                rows_seen += batch.rows
                yield batch
            if rows_seen != shard.rows:
                raise RuntimeError(
                    f"SHARD_ROW_COUNT_MISMATCH:{shard.shard_id}:manifest={shard.rows}:read={rows_seen}"
                )

    def iter_windows(
        self,
        *,
        symbol: str,
        window: int,
        stride: int = 1,
        columns: Sequence[str] = ("open", "high", "low", "close", "volume"),
    ) -> Iterator[tuple[np.ndarray, int]]:
        """Streaming rolling windows across batch boundaries.

        Returns (window_array [window,C], ending_timestamp).
        """
        if window <= 0 or stride <= 0:
            raise ValueError("window/stride must be positive")
        carry_x = np.empty((0, len(columns)), dtype=np.float64)
        carry_t = np.empty((0,), dtype=np.int64)
        global_index = 0
        for b in self.iter_batches(symbol=symbol):
            x = np.stack([np.asarray(getattr(b, c), dtype=np.float64) for c in columns], axis=1)
            t = np.asarray(b.timestamp, dtype=np.int64)
            if len(carry_x):
                x = np.concatenate([carry_x, x], axis=0)
                t = np.concatenate([carry_t, t], axis=0)
            first_new_index = max(0, len(carry_x))
            start_min = max(0, window - 1)
            for end in range(start_min, len(x)):
                # Stride is applied to the global ending-row index.
                end_global = global_index - len(carry_x) + end
                if end_global % stride != 0:
                    continue
                yield x[end-window+1:end+1].copy(), int(t[end])
            keep = min(window - 1, len(x))
            carry_x = x[-keep:].copy() if keep else np.empty((0, len(columns)))
            carry_t = t[-keep:].copy() if keep else np.empty((0,), dtype=np.int64)
            global_index += b.rows


def build_csv_shard_spec(
    *,
    path: str | Path,
    shard_id: str,
    symbol: str,
    timeframe: str,
) -> ShardSpec:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first = None
        last = None
        rows = 0
        for row in reader:
            ts = int(row["timestamp"])
            if first is None:
                first = ts
            last = ts
            rows += 1
    if rows == 0:
        raise ValueError("empty shard")
    return ShardSpec(
        shard_id=shard_id,
        path=str(path),
        symbol=symbol,
        timeframe=timeframe,
        format="csv",
        start_timestamp=int(first),
        end_timestamp=int(last),
        rows=rows,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )


def save_manifest(manifest: DatasetManifest, path: str | Path) -> Path:
    manifest.validate(verify_files=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "CB16_HISTORICAL_DATASET_MANIFEST_R3",
        "dataset_id": manifest.dataset_id,
        "symbols": list(manifest.symbols),
        "timeframe": manifest.timeframe,
        "shards": [asdict(s) for s in manifest.shards],
        "canonical_columns": list(manifest.canonical_columns),
        "source": manifest.source,
        "content_hash": manifest.content_hash,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def load_manifest(path: str | Path, *, verify_files: bool = False) -> DatasetManifest:
    obj = json.loads(Path(path).read_text())
    shards = tuple(ShardSpec(**s) for s in obj["shards"])
    manifest = DatasetManifest(
        dataset_id=obj["dataset_id"],
        symbols=tuple(obj["symbols"]),
        timeframe=obj["timeframe"],
        shards=shards,
        canonical_columns=tuple(obj.get("canonical_columns", CANONICAL_COLUMNS)),
        source=obj.get("source", "HISTORICAL_REPLAY"),
    )
    manifest.validate(verify_files=verify_files)
    if obj.get("content_hash") and obj["content_hash"] != manifest.content_hash:
        raise RuntimeError("DATASET_MANIFEST_CONTENT_HASH_MISMATCH")
    return manifest
