from __future__ import annotations

"""Campaign-lifetime read-only market runtime cache for R11 trace execution.

This module is orchestration only.  It consumes the already-materialized R10.2
hourly cache and never reads raw 1m archives.  A symbol's compressed ``.npz`` is
opened at most once for the lifetime of this object; timestamps, OHLCV, funding,
and the timestamp index are then reused across generations.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from .r102_common import FORBIDDEN_FINAL_MONTH, FORBIDDEN_FINAL_START_MS, H72, HOUR_MS


class MarketRuntimeCacheError(RuntimeError):
    pass


def _readonly_array(value: np.ndarray, *, dtype: np.dtype | None = None) -> np.ndarray:
    # Own a private copy so closing NpzFile cannot invalidate the runtime cache.
    owner = np.array(value, dtype=dtype, copy=True, order="C")
    owner.setflags(write=False)
    # Expose a non-owning read-only view. NumPy then also refuses callers that
    # try to re-enable WRITEABLE on the public array while its base is read-only.
    view = owner.view()
    view.setflags(write=False)
    return view


@dataclass(frozen=True)
class MarketRuntimeSymbolR11:
    symbol: str
    source_path: Path
    open_time_ms: np.ndarray
    ohlcv: np.ndarray
    funding_rate: np.ndarray
    index_by_time_ms: Mapping[int, int]

    def index_of(self, timestamp_ms: int) -> int:
        try:
            return int(self.index_by_time_ms[int(timestamp_ms)])
        except KeyError as exc:
            raise MarketRuntimeCacheError(
                f"R11_MARKET_TIMESTAMP_MISSING:{self.symbol}:{int(timestamp_ms)}"
            ) from exc

    def h72_slice(self, decision_time_ms: int) -> slice:
        """Return the exact contiguous 72-hour window used by the frozen H72 oracle."""
        t0 = int(decision_time_ms)
        start = self.index_of(t0)
        stop = start + H72
        if stop > len(self.open_time_ms):
            raise MarketRuntimeCacheError(f"H72_FUTURE_MISSING:{self.symbol}:{t0}")
        ts = self.open_time_ms[start:stop]
        if len(ts) != H72:
            raise MarketRuntimeCacheError(f"H72_FUTURE_MISSING:{self.symbol}:{t0}")
        # The legacy oracle asks for t, t+1h, ..., t+71h.  Checking the entire
        # slice preserves its missing-hour failure semantics without rebuilding a map.
        expected = t0 + np.arange(H72, dtype=np.int64) * HOUR_MS
        if not np.array_equal(ts, expected):
            raise MarketRuntimeCacheError(f"H72_FUTURE_MISSING:{self.symbol}:{t0}")
        return slice(start, stop)

    def prehistory_slice(self, decision_time_ms: int, hours: int) -> slice:
        hours = int(hours)
        if hours <= 0:
            raise ValueError("R11_PREHISTORY_HOURS_MUST_BE_POSITIVE")
        t0 = int(decision_time_ms) - hours * HOUR_MS
        start = self.index_of(t0)
        stop = start + hours
        expected = t0 + np.arange(hours, dtype=np.int64) * HOUR_MS
        ts = self.open_time_ms[start:stop]
        if len(ts) != hours or not np.array_equal(ts, expected):
            raise MarketRuntimeCacheError(
                f"PREHISTORY_MISSING:{self.symbol}:{int(decision_time_ms)}"
            )
        return slice(start, stop)


@dataclass(frozen=True)
class MarketRuntimeCacheStatsR11:
    compressed_loads_total: int
    index_builds_total: int
    loaded_symbols: tuple[str, ...]
    compressed_loads_by_symbol: Mapping[str, int]


class MarketRuntimeCacheR11:
    """Read-only per-campaign cache over ``*.hourly_r102.npz`` payloads.

    The caller is expected to keep one instance alive across generations.  All
    returned NumPy arrays are write-protected and the timestamp map is immutable.
    Thread-safe lazy loading makes a concurrent first access still decompress once.
    """

    def __init__(self, cache_dir: str | Path):
        root = Path(cache_dir).resolve()
        # Accept either the campaign root or its market_cache child.
        if root.name != "market_cache" and (root / "market_cache").is_dir():
            root = (root / "market_cache").resolve()
        self._cache_dir = root
        self._symbols: dict[str, MarketRuntimeSymbolR11] = {}
        self._loads_by_symbol: dict[str, int] = {}
        self._compressed_loads_total = 0
        self._index_builds_total = 0
        self._lock = RLock()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _path_for_symbol(self, symbol: str) -> Path:
        sym = str(symbol)
        if not sym or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for ch in sym):
            raise ValueError(f"R11_INVALID_SYMBOL:{sym!r}")
        p = (self._cache_dir / f"{sym}.hourly_r102.npz").resolve()
        try:
            p.relative_to(self._cache_dir)
        except ValueError as exc:
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_PATH_ESCAPE:{p}") from exc
        # Fail before opening a payload if the configured cache root itself is
        # explicitly scoped to the unopened holdout month.
        if FORBIDDEN_FINAL_MONTH in p.parts:
            raise MarketRuntimeCacheError(f"FINAL_HOLDOUT_ACCESS_FAIL_CLOSED:market_cache:{p}")
        return p

    def _validate_manifest_sidecar_if_present(self, symbol: str) -> None:
        manifest = self._cache_dir / f"{symbol}.manifest_r102.json"
        if not manifest.is_file():
            return
        obj = json.loads(manifest.read_text(encoding="utf-8"))
        if obj.get("forbidden_month_opened") is not False:
            raise MarketRuntimeCacheError(
                f"R11_MARKET_MANIFEST_HOLDOUT_GUARD_FAILED:{symbol}:forbidden_month_opened"
            )
        boundary = str(obj.get("archive_boundary", ""))
        if boundary and FORBIDDEN_FINAL_MONTH not in boundary:
            raise MarketRuntimeCacheError(
                f"R11_MARKET_MANIFEST_BOUNDARY_UNRECOGNIZED:{symbol}:{boundary}"
            )

    @staticmethod
    def _validate_arrays(symbol: str, ts: np.ndarray, bars: np.ndarray, funding: np.ndarray) -> None:
        if ts.ndim != 1 or bars.ndim != 2 or bars.shape[1] != 5 or funding.ndim != 1:
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_SHAPE_INVALID:{symbol}")
        if len(ts) == 0 or len(ts) != len(bars) or len(ts) != len(funding):
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_LENGTH_INVALID:{symbol}")
        if np.any(ts >= FORBIDDEN_FINAL_START_MS):
            raise MarketRuntimeCacheError(
                f"FINAL_HOLDOUT_ACCESS_FAIL_CLOSED:hourly_cache:{symbol}:timestamp_at_or_after_2025_09"
            )
        if np.any(np.diff(ts) <= 0):
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_NON_INCREASING_TIMESTAMP:{symbol}")
        if np.any(ts % HOUR_MS != 0):
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_NON_HOURLY_TIMESTAMP:{symbol}")
        if not np.all(np.isfinite(bars)) or not np.all(np.isfinite(funding)):
            raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_NONFINITE:{symbol}")

    def get(self, symbol: str) -> MarketRuntimeSymbolR11:
        sym = str(symbol)
        cached = self._symbols.get(sym)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._symbols.get(sym)
            if cached is not None:
                return cached
            path = self._path_for_symbol(sym)
            self._validate_manifest_sidecar_if_present(sym)
            if not path.is_file():
                raise FileNotFoundError(path)
            # Compressed .npz cannot be mmap'ed.  Decompress exactly once into
            # campaign-lifetime RAM and share the resulting read-only arrays.
            with np.load(path, allow_pickle=False) as z:
                ts = _readonly_array(z["open_time_ms"], dtype=np.int64)
                bars = _readonly_array(z["ohlcv"])
                funding = _readonly_array(z["funding_rate"])
            self._validate_arrays(sym, ts, bars, funding)
            index = MappingProxyType({int(t): i for i, t in enumerate(ts)})
            if len(index) != len(ts):
                raise MarketRuntimeCacheError(f"R11_MARKET_CACHE_DUPLICATE_TIMESTAMP:{sym}")
            entry = MarketRuntimeSymbolR11(
                symbol=sym,
                source_path=path,
                open_time_ms=ts,
                ohlcv=bars,
                funding_rate=funding,
                index_by_time_ms=index,
            )
            self._symbols[sym] = entry
            self._loads_by_symbol[sym] = self._loads_by_symbol.get(sym, 0) + 1
            self._compressed_loads_total += 1
            self._index_builds_total += 1
            return entry

    def preload(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            self.get(str(symbol))

    def stats(self) -> MarketRuntimeCacheStatsR11:
        with self._lock:
            return MarketRuntimeCacheStatsR11(
                compressed_loads_total=int(self._compressed_loads_total),
                index_builds_total=int(self._index_builds_total),
                loaded_symbols=tuple(sorted(self._symbols)),
                compressed_loads_by_symbol=MappingProxyType(dict(self._loads_by_symbol)),
            )

    def assert_read_only(self) -> None:
        with self._lock:
            for symbol, entry in self._symbols.items():
                for name, arr in (
                    ("open_time_ms", entry.open_time_ms),
                    ("ohlcv", entry.ohlcv),
                    ("funding_rate", entry.funding_rate),
                ):
                    if bool(arr.flags.writeable):
                        raise MarketRuntimeCacheError(
                            f"R11_MARKET_CACHE_MUTABLE_FAIL_CLOSED:{symbol}:{name}"
                        )

    def close(self) -> None:
        # Arrays are ordinary in-memory NumPy buffers, not open file handles.
        # Explicit clear bounds lifetime and releases RAM between campaigns.
        with self._lock:
            self._symbols.clear()
