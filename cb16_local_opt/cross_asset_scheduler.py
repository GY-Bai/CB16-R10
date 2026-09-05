from __future__ import annotations

"""
Cross-asset synchronized chronology scheduler.

No forward-fill is performed by default. A synchronized context exists only when the
configured asset set has an exact timestamp match unless `allow_partial=True` is
explicitly enabled. This prevents accidental future/nearest-neighbor leakage.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .event_source_contracts import MarketEvent


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SynchronizedMarketFrame:
    timestamp: int
    symbols: tuple[str,...]
    events: tuple[MarketEvent | None,...]
    availability: tuple[bool,...]
    frame_id: str

    @property
    def content_hash(self) -> str:
        return canonical_hash({
            "timestamp":self.timestamp,
            "symbols":self.symbols,
            "event_hashes":[None if e is None else e.content_hash for e in self.events],
            "availability":self.availability,
            "frame_id":self.frame_id,
        })


@dataclass(frozen=True)
class ChronologicalBlock:
    block_id: str
    start_timestamp: int
    end_timestamp: int
    frame_hashes: tuple[str,...]
    frame_count: int
    horizon_bars: int
    mature_end_timestamp: int | None

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class CrossAssetSynchronizer:
    def __init__(
        self,
        symbols: Sequence[str],
        *,
        allow_partial: bool = False,
        min_available: int | None = None,
    ):
        self.symbols=tuple(symbols)
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique nonempty")
        self.allow_partial=bool(allow_partial)
        self.min_available = (
            len(self.symbols) if min_available is None else int(min_available)
        )
        if not 1 <= self.min_available <= len(self.symbols):
            raise ValueError("min_available out of range")
        if not allow_partial and self.min_available != len(self.symbols):
            raise ValueError("strict mode requires all symbols")

    def synchronize(
        self,
        streams: Mapping[str, Iterable[MarketEvent]],
    ) -> Iterator[SynchronizedMarketFrame]:
        missing=set(self.symbols)-set(streams)
        if missing:
            raise ValueError(f"missing streams:{sorted(missing)}")
        iters={s:iter(streams[s]) for s in self.symbols}
        heads={}
        last={}
        for s,it in iters.items():
            try: heads[s]=next(it)
            except StopIteration: heads[s]=None

        while any(v is not None for v in heads.values()):
            active_ts=[e.timestamp for e in heads.values() if e is not None]
            ts=min(active_ts)
            events=[]
            availability=[]
            for s in self.symbols:
                e=heads[s]
                if e is not None and e.timestamp == ts:
                    if e.symbol != s:
                        raise RuntimeError("STREAM_SYMBOL_MISMATCH")
                    prev=last.get(s)
                    if prev is not None and e.timestamp <= prev:
                        raise RuntimeError("STREAM_CHRONOLOGY_VIOLATION")
                    last[s]=e.timestamp
                    events.append(e); availability.append(True)
                    try: heads[s]=next(iters[s])
                    except StopIteration: heads[s]=None
                else:
                    events.append(None); availability.append(False)

            available=sum(availability)
            if (not self.allow_partial and available == len(self.symbols)) or (
                self.allow_partial and available >= self.min_available
            ):
                frame_id=f"SYNC:{ts}:" + ",".join(self.symbols)
                yield SynchronizedMarketFrame(
                    timestamp=ts,symbols=self.symbols,events=tuple(events),
                    availability=tuple(availability),frame_id=frame_id,
                )

    @staticmethod
    def blocks(
        frames: Iterable[SynchronizedMarketFrame],
        *,
        block_frames: int,
        horizon_bars: int,
        nonoverlap: bool = True,
    ) -> Iterator[ChronologicalBlock]:
        """Streaming block construction with only block+horizon lookahead in RAM."""
        import collections
        if block_frames <= 0 or horizon_bars < 0:
            raise ValueError("bad block/horizon")
        stride=block_frames if nonoverlap else 1
        need=block_frames+horizon_bars
        q=collections.deque()
        start_index=0
        source=iter(frames)
        exhausted=False

        while True:
            while len(q)<need and not exhausted:
                try:q.append(next(source))
                except StopIteration: exhausted=True
            if len(q)<block_frames:
                break
            chunk=list(q)[:block_frames]
            mature_end = (
                list(q)[block_frames-1+horizon_bars].timestamp
                if len(q)>=need else None
            )
            bid=f"BLOCK:{start_index}:{chunk[0].timestamp}:{chunk[-1].timestamp}"
            yield ChronologicalBlock(
                block_id=bid,
                start_timestamp=chunk[0].timestamp,
                end_timestamp=chunk[-1].timestamp,
                frame_hashes=tuple(f.content_hash for f in chunk),
                frame_count=len(chunk),
                horizon_bars=horizon_bars,
                mature_end_timestamp=mature_end,
            )
            pop_n=min(stride,len(q))
            for _ in range(pop_n):q.popleft()
            start_index+=stride
            if exhausted and len(q)<block_frames:
                break


def assert_nonoverlapping_blocks(blocks: Iterable[ChronologicalBlock]) -> None:
    last_end=None
    for b in blocks:
        if last_end is not None and b.start_timestamp <= last_end:
            raise RuntimeError("CHRONOLOGICAL_BLOCK_OVERLAP")
        last_end=b.end_timestamp
