"""Canonical bar representation used by both synthetic and real data."""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class CanonicalBar:
    """One canonical OHLCV bar with strict validation.

    Synthetic and real data must both produce this exact shape.
    """

    symbol: str
    bar_start: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    mark_price: Optional[float] = None
    index_price: Optional[float] = None

    def __post_init__(self) -> None:
        # Normalize timezone-aware UTC; treat naive as UTC.
        if self.bar_start.tzinfo is None:
            object.__setattr__(self, "bar_start", self.bar_start.replace(tzinfo=timezone.utc))
        else:
            object.__setattr__(self, "bar_start", self.bar_start.astimezone(timezone.utc))

        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if not self.timeframe:
            raise ValueError("timeframe must be non-empty")

        for name in ("open", "high", "low", "close"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0:
                raise ValueError(f"{name} must be a finite positive number")

        if not isinstance(self.volume, (int, float)) or isinstance(self.volume, bool) or not math.isfinite(float(self.volume)) or float(self.volume) < 0:
            raise ValueError("volume must be a finite non-negative number")

        # Historical OHLCV files do not contain Binance mark/index prices.
        # Materialize the fallback here so every downstream consumer sees a
        # positive float rather than having to repeat Optional handling.
        if self.mark_price is None:
            object.__setattr__(self, "mark_price", float(self.close))
        if self.index_price is None:
            object.__setattr__(self, "index_price", float(self.close))
        for name in ("mark_price", "index_price"):
            v = getattr(self, name)
            if (
                not isinstance(v, (int, float))
                or isinstance(v, bool)
                or not math.isfinite(float(v))
                or float(v) <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")

        if not (self.low <= self.open <= self.high + 1e-12):
            raise ValueError("low <= open <= high violated")
        if not (self.low <= self.close <= self.high + 1e-12):
            raise ValueError("low <= close <= high violated")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bar_start": self.bar_start.isoformat(),
            "timeframe": self.timeframe,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "mark_price": self.mark_price,
            "index_price": self.index_price,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalBar":
        return cls(
            symbol=data["symbol"],
            bar_start=datetime.fromisoformat(data["bar_start"]),
            timeframe=data["timeframe"],
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            mark_price=(
                None
                if data.get("mark_price") is None
                else float(data["mark_price"])
            ),
            index_price=(
                None
                if data.get("index_price") is None
                else float(data["index_price"])
            ),
        )


def canonicalize_bars(bars: list["CanonicalBar"]) -> list["CanonicalBar"]:
    """Sort by time and remove only *identical* duplicate bars.

    Conflicting rows for the same ``(symbol, timestamp, timeframe)`` are a data
    integrity error, not a deduplication opportunity.  Silently keeping one of
    two different OHLCV rows makes training depend on file iteration order.
    """
    seen: dict[tuple[str, datetime, str], CanonicalBar] = {}
    out: list[CanonicalBar] = []
    for bar in sorted(bars, key=lambda b: b.bar_start):
        key = (bar.symbol, bar.bar_start, bar.timeframe)
        previous = seen.get(key)
        if previous is not None:
            if previous != bar:
                raise ValueError(
                    "conflicting duplicate canonical bar for "
                    f"symbol={bar.symbol} timeframe={bar.timeframe} "
                    f"bar_start={bar.bar_start.isoformat()}"
                )
            continue
        seen[key] = bar
        out.append(bar)
    return out


def contiguous_hour_runs(bars: Sequence[Any]) -> list[tuple[int, int]]:
    """Return half-open runs separated by anything other than exactly one hour.

    V5.3's canonical market source is 1h.  Resetting feature history at a gap
    prevents pre-gap observations from being treated as if they immediately
    preceded the first post-gap bar.
    """
    n = len(bars)
    if n == 0:
        return []
    # Numeric unit tests and low-level feature callers may supply bar-like
    # objects without timestamps.  Without a clock there is no evidence of a
    # gap, so preserve the legacy single-run behavior.
    if not hasattr(bars[0], "bar_start"):
        return [(0, n)]
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, n):
        delta = (bars[index].bar_start - bars[index - 1].bar_start).total_seconds()
        if not math.isclose(delta, 3600.0, rel_tol=0.0, abs_tol=1e-6):
            runs.append((start, index))
            start = index
    runs.append((start, n))
    return runs


def detect_gaps(
    bars: list["CanonicalBar"],
    max_gap_multiplier: float = 2.0,
) -> list[tuple[int, datetime, float]]:
    """Return (index, bar_start, gap_seconds) for suspicious gaps."""
    if len(bars) < 2:
        return []
    deltas = [
        (bars[i].bar_start - bars[i - 1].bar_start).total_seconds()
        for i in range(1, len(bars))
    ]
    if not deltas:
        return []
    sorted_deltas = sorted(deltas)
    median = sorted_deltas[len(sorted_deltas) // 2]
    threshold = max(median * max_gap_multiplier, 1.0)
    return [
        (i, bars[i].bar_start, d)
        for i, d in enumerate(deltas, start=1)
        if d > threshold
    ]
