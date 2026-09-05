from __future__ import annotations

"""
Unified source-swap event contract.

Historical Replay and future Live Paper sources must emit the same event objects.
The downstream Trader/Physics/Teacher path is source-agnostic and may inspect
`source_kind` only for audit, never to change the economic semantics of an event.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Literal, Protocol

SourceKind = Literal["HISTORICAL_REPLAY", "LIVE_PAPER"]


def canonical_hash(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    source_kind: SourceKind
    symbol: str
    timeframe: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    market_lineage_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)

    def validate(self) -> None:
        if self.source_kind not in {"HISTORICAL_REPLAY", "LIVE_PAPER"}:
            raise ValueError("invalid source_kind")
        if self.timestamp < 0:
            raise ValueError("negative timestamp")
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("nonpositive price")
        if self.low > min(self.open, self.close) or self.high < max(self.open, self.close):
            raise ValueError("OHLC ordering invalid")
        if self.volume < 0:
            raise ValueError("negative volume")


@dataclass(frozen=True)
class AccountEvent:
    event_id: str
    source_kind: SourceKind
    account_id: str
    timestamp: int
    equity: float
    balance: float
    position_qty: float
    entry_price: float
    peak_equity: float
    margin_used: float
    holding_bars: int
    risk_budget_remaining: float
    account_lineage_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class DecisionEvent:
    event_id: str
    source_kind: SourceKind
    timestamp: int
    symbol: str
    account_id: str
    policy_generation: int
    policy_weight_hash: str
    market_event_hash: str
    account_event_hash: str
    requested_direction: int
    requested_risk: float
    decision_context_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)

    def validate(self) -> None:
        if self.requested_direction not in {-1, 0, 1}:
            raise ValueError("invalid direction")
        if not 0 <= self.requested_risk <= 1:
            raise ValueError("risk out of range")
        if self.requested_direction == 0 and self.requested_risk != 0:
            raise ValueError("FLAT_REQUIRES_ZERO_RISK")


@dataclass(frozen=True)
class ExecutionReceipt:
    event_id: str
    decision_event_hash: str
    timestamp: int
    executable_direction: int
    executable_risk: float
    supervisor_hash: str
    physics_hash: str
    status: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class OutcomeReceipt:
    event_id: str
    execution_receipt_hash: str
    timestamp_start: int
    timestamp_end: int
    realized_log_equity_return: float
    fees: float
    funding: float
    terminal_equity: float
    outcome_lineage_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class EventSource(Protocol):
    source_kind: SourceKind

    def iter_market_events(self) -> Iterator[MarketEvent]:
        ...


class EventSequenceGuard:
    def __init__(self):
        self._last_ts: dict[tuple[str, str], int] = {}
        self._seen_ids: dict[str, str] = {}

    def accept_market(self, event: MarketEvent) -> None:
        event.validate()
        old = self._seen_ids.get(event.event_id)
        if old is not None:
            if old != event.content_hash:
                raise RuntimeError("EVENT_ID_CONTENT_CONFLICT")
            return
        key = (event.symbol, event.timeframe)
        prev = self._last_ts.get(key)
        if prev is not None and event.timestamp <= prev:
            raise RuntimeError("MARKET_EVENT_CHRONOLOGY_VIOLATION")
        self._last_ts[key] = event.timestamp
        self._seen_ids[event.event_id] = event.content_hash


def assert_source_swap_equivalent(
    historical: Iterable[MarketEvent],
    live: Iterable[MarketEvent],
    *,
    ignore_source_kind: bool = True,
) -> None:
    """Canary: same market bytes through Historical and Live adapters must agree."""
    hs = list(historical)
    ls = list(live)
    if len(hs) != len(ls):
        raise RuntimeError("SOURCE_SWAP_LENGTH_MISMATCH")
    for h, l in zip(hs, ls):
        hd = asdict(h); ld = asdict(l)
        if ignore_source_kind:
            hd.pop("source_kind"); ld.pop("source_kind")
            # Source-specific event ids are not a scientific difference.
            hd.pop("event_id"); ld.pop("event_id")
        if hd != ld:
            raise RuntimeError("SOURCE_SWAP_SEMANTIC_MISMATCH")
