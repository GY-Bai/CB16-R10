from __future__ import annotations

"""Freeze and qualify one immutable real historical Dataset Snapshot for R9.

The mutable DatasetCatalog may keep receiving new OCI bundles, but a campaign must
bind one CatalogSnapshot and a mounted manifest that cannot change underneath it.

R9 additionally streams every selected symbol once to record chronology quality:
- exact row count;
- first/last timestamp;
- expected cadence;
- duplicate/time-reversal rejection (delegated to HistoricalShardReader);
- missing bar count and gap intervals;
- OHLCV validity;
- mounted shard SHA256 verification.

No normalization or model fitting is done here.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset_catalog_r8 import DatasetCatalogR8, CatalogSnapshotR8
from .historical_shard_reader import HistoricalShardReader, DatasetManifest, save_manifest


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def timeframe_seconds(tf: str) -> int:
    s = tf.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if len(s) < 2 or s[-1] not in units:
        raise ValueError(f"unsupported timeframe:{tf}")
    return int(s[:-1]) * units[s[-1]]


@dataclass(frozen=True)
class DatasetChronologyPolicyR9:
    expected_interval_seconds: int | None = None
    max_missing_bar_fraction: float = 0.0
    min_rows_per_symbol: int = 256
    required_symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    require_exact_required_symbol_set: bool = False

    def validate(self):
        if self.expected_interval_seconds is not None and self.expected_interval_seconds <= 0:
            raise ValueError("expected interval")
        if not 0.0 <= self.max_missing_bar_fraction < 1.0:
            raise ValueError("missing fraction")
        if self.min_rows_per_symbol <= 0:
            raise ValueError("min rows")

    @property
    def content_hash(self) -> str:
        self.validate()
        return canonical_hash(self)


@dataclass(frozen=True)
class SymbolChronologyReceiptR9:
    symbol: str
    rows: int
    first_timestamp: int
    last_timestamp: int
    expected_interval_seconds: int
    expected_slots: int
    missing_bars: int
    missing_bar_fraction: float
    gap_count: int
    maximum_gap_seconds: int
    status: str


@dataclass(frozen=True)
class DatasetSnapshotReceiptR9:
    snapshot_version: str
    status: str
    snapshot_id: str
    catalog_snapshot_hash: str
    scientific_dataset_hash: str
    mounted_manifest_path: str
    mounted_manifest_hash: str
    timeframe: str
    symbols: tuple[str, ...]
    bundle_sha256s: tuple[str, ...]
    chronology: tuple[SymbolChronologyReceiptR9, ...]
    total_rows: int
    policy_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def _audit_symbol(
    reader: HistoricalShardReader,
    symbol: str,
    *,
    expected_interval_seconds: int,
    min_rows: int,
    max_missing_fraction: float,
) -> SymbolChronologyReceiptR9:
    rows = 0
    first = None
    last = None
    prev = None
    missing = 0
    gap_count = 0
    max_gap = 0
    for batch in reader.iter_batches(symbol=symbol):
        ts = np.asarray(batch.timestamp, dtype=np.int64)
        if len(ts) == 0:
            continue
        if first is None:
            first = int(ts[0])
        if prev is not None:
            gap = int(ts[0]) - int(prev)
            if gap != expected_interval_seconds:
                if gap < expected_interval_seconds or gap % expected_interval_seconds != 0:
                    raise RuntimeError(f"NON_CANONICAL_TIMESTAMP_GAP:{symbol}:{gap}")
                miss = gap // expected_interval_seconds - 1
                missing += miss
                gap_count += 1
                max_gap = max(max_gap, gap)
        dif = np.diff(ts)
        bad = dif[dif != expected_interval_seconds]
        for gap in bad.tolist():
            gap = int(gap)
            if gap < expected_interval_seconds or gap % expected_interval_seconds != 0:
                raise RuntimeError(f"NON_CANONICAL_TIMESTAMP_GAP:{symbol}:{gap}")
            miss = gap // expected_interval_seconds - 1
            missing += miss
            gap_count += 1
            max_gap = max(max_gap, gap)
        rows += len(ts)
        prev = int(ts[-1])
        last = prev
    if rows < min_rows:
        raise RuntimeError(f"DATASET_SYMBOL_TOO_SHORT:{symbol}:{rows}")
    assert first is not None and last is not None
    expected_slots = (last - first) // expected_interval_seconds + 1
    frac = missing / expected_slots if expected_slots else 0.0
    if frac > max_missing_fraction:
        raise RuntimeError(
            f"DATASET_MISSING_BAR_FRACTION_EXCEEDED:{symbol}:{frac}>{max_missing_fraction}"
        )
    return SymbolChronologyReceiptR9(
        symbol=symbol,
        rows=rows,
        first_timestamp=first,
        last_timestamp=last,
        expected_interval_seconds=expected_interval_seconds,
        expected_slots=int(expected_slots),
        missing_bars=int(missing),
        missing_bar_fraction=float(frac),
        gap_count=int(gap_count),
        maximum_gap_seconds=int(max_gap),
        status="PASS",
    )


def freeze_dataset_snapshot_r9(
    *,
    catalog_db: str | Path,
    snapshot_id: str,
    output_dir: str | Path,
    bundle_sha256s: Sequence[str] | None = None,
    existing_snapshot: bool = False,
    policy: DatasetChronologyPolicyR9 | None = None,
    read_batch_rows: int = 65536,
    parquet_readahead_batches: int = 2,
) -> DatasetSnapshotReceiptR9:
    policy = policy or DatasetChronologyPolicyR9()
    policy.validate()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog = DatasetCatalogR8(catalog_db)
    try:
        snap = (
            catalog.get_snapshot(snapshot_id)
            if existing_snapshot
            else catalog.create_snapshot(snapshot_id, bundle_sha256s=bundle_sha256s)
        )
        mounted = catalog.mount_snapshot(snap, verify_files=True)
    finally:
        catalog.close()

    symbols = tuple(sorted(mounted.symbols))
    required = set(policy.required_symbols)
    actual = set(symbols)
    if required and not required.issubset(actual):
        raise RuntimeError(
            "DATASET_REQUIRED_SYMBOLS_MISSING:" + ",".join(sorted(required - actual))
        )
    if policy.require_exact_required_symbol_set and actual != required:
        raise RuntimeError("DATASET_SYMBOL_SET_NOT_EXACT_REQUIRED_SET")

    # The mounted manifest contains local absolute paths and is operational, not the
    # portable scientific identity. Its own hash is recorded only for this machine mount.
    manifest_path = out / "MOUNTED_DATASET_MANIFEST_R9.json"
    save_manifest(mounted, manifest_path)
    reader = HistoricalShardReader(
        mounted,
        batch_rows=read_batch_rows,
        arrow_batch_readahead=parquet_readahead_batches,
        verify_hashes_on_open=False,
    )
    # Historical sources in this runtime may encode Unix timestamps in seconds or
    # milliseconds. Binance acquisition uses milliseconds. Detect the unit from the
    # first shard metadata while keeping the configured interval semantically in seconds.
    base_interval_seconds = policy.expected_interval_seconds or timeframe_seconds(mounted.timeframe)
    first_ts = min(s.start_timestamp for s in mounted.shards)
    timestamp_scale = 1000 if first_ts >= 10**12 else 1
    interval = base_interval_seconds * timestamp_scale
    chronology = tuple(
        _audit_symbol(
            reader,
            s,
            expected_interval_seconds=interval,
            min_rows=policy.min_rows_per_symbol,
            max_missing_fraction=policy.max_missing_bar_fraction,
        )
        for s in symbols
    )
    receipt = DatasetSnapshotReceiptR9(
        snapshot_version="CB16_REAL_DATASET_SNAPSHOT_R9",
        status="REAL_DATASET_SNAPSHOT_QUALIFIED",
        snapshot_id=snap.snapshot_id,
        catalog_snapshot_hash=snap.content_hash,
        scientific_dataset_hash=snap.scientific_dataset_hash,
        mounted_manifest_path=str(manifest_path.resolve()),
        mounted_manifest_hash=mounted.content_hash,
        timeframe=mounted.timeframe,
        symbols=symbols,
        bundle_sha256s=snap.bundle_sha256s,
        chronology=chronology,
        total_rows=sum(x.rows for x in chronology),
        policy_hash=policy.content_hash,
    )
    (out / "DATASET_SNAPSHOT_RECEIPT_R9.json").write_text(
        json.dumps({**asdict(receipt), "content_hash": receipt.content_hash}, indent=2) + "\n"
    )
    # Copy the catalog snapshot payload separately so the scientific dataset identity is
    # visible without opening the mutable catalog database.
    (out / "CATALOG_SNAPSHOT_R9.json").write_text(
        json.dumps({**asdict(snap), "content_hash": snap.content_hash}, indent=2) + "\n"
    )
    return receipt


def load_dataset_snapshot_receipt_r9(path: str | Path) -> DatasetSnapshotReceiptR9:
    obj = json.loads(Path(path).read_text())
    claimed = obj.pop("content_hash", None)
    obj["symbols"] = tuple(obj["symbols"])
    obj["bundle_sha256s"] = tuple(obj["bundle_sha256s"])
    obj["chronology"] = tuple(SymbolChronologyReceiptR9(**x) for x in obj["chronology"])
    r = DatasetSnapshotReceiptR9(**obj)
    if claimed and claimed != r.content_hash:
        raise RuntimeError("DATASET_SNAPSHOT_RECEIPT_HASH_MISMATCH")
    return r
