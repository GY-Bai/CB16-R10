from __future__ import annotations

"""R11 full-minute long-trajectory R1 foundation.

This module is deliberately small and semantic-first.  It does not replace the
frozen scalar Physics kernel.  It provides the prefix-only internal-gap halt
model, a causal trajectory scan contract, checkpoint hashes, and a batch-axis
parity canary used before any long market-learning experiment is admitted.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .binance_archive_input_r10 import KlineRecord, MINUTE_MS


FINAL_HOLDOUT_START_MS = 1756684800000  # 2025-09-01T00:00:00Z


@dataclass(frozen=True)
class MinuteEnvelopeR1:
    record: KlineRecord
    halt_imputed: bool = False

    @property
    def timestamp(self) -> int:
        return int(self.record.open_time)


@dataclass(frozen=True)
class CausalDecisionEventR1:
    decision_time_ms: int
    observation_max_time_ms: int
    account_before_hash: str
    action_hash: str
    transition_time_ms: int
    account_after_hash: str
    transition_halt_imputed: bool


@dataclass(frozen=True)
class TrajectoryCheckpointR1:
    next_decision_time_ms: int
    account_state: Any
    prefix_event_count: int
    prefix_hash: str


def canonical_hash(obj: Any) -> str:
    def normalize(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Mapping):
            return {str(k): normalize(v) for k, v in sorted(x.items(), key=lambda z: str(z[0]))}
        if isinstance(x, (list, tuple)):
            return [normalize(v) for v in x]
        if hasattr(x, "__dict__"):
            return normalize(vars(x))
        return x

    payload = json.dumps(normalize(obj), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_observed_pair(prev: KlineRecord, cur: KlineRecord) -> int:
    delta = int(cur.open_time) - int(prev.open_time)
    if delta <= 0:
        raise RuntimeError(f"NON_INCREASING_1M:{prev.open_time}->{cur.open_time}")
    if delta % MINUTE_MS:
        raise RuntimeError(f"NON_MINUTE_ALIGNED_GAP:{prev.open_time}->{cur.open_time}")
    return delta // MINUTE_MS


def make_halt_minute_from_prefix(prev: KlineRecord, open_time: int) -> KlineRecord:
    """Create one auditably synthetic halt minute using prefix information only."""
    if int(open_time) <= int(prev.open_time):
        raise ValueError("halt minute must be after prefix record")
    px = float(prev.close)
    if not np.isfinite(px) or px <= 0:
        raise ValueError("prefix close must be finite and positive")
    return KlineRecord(
        open_time=int(open_time),
        open=px,
        high=px,
        low=px,
        close=px,
        volume=0.0,
        close_time=int(open_time) + MINUTE_MS - 1,
        quote_asset_volume=0.0,
        number_of_trades=0,
        taker_buy_base_asset_volume=0.0,
        taker_buy_quote_asset_volume=0.0,
    )


def expand_internal_halts_r1(records: Iterable[KlineRecord]) -> Iterator[MinuteEnvelopeR1]:
    """Fill only internal observed-to-observed gaps as halt minutes.

    The reopening record is never used to construct any earlier missing row.
    There is intentionally no pre-listing or open-ended post-history fill.
    """
    it = iter(records)
    try:
        prev = next(it)
    except StopIteration:
        return
    yield MinuteEnvelopeR1(prev, False)
    for cur in it:
        steps = _validate_observed_pair(prev, cur)
        if steps > 1:
            prefix = prev
            for k in range(1, steps):
                t = int(prev.open_time) + k * MINUTE_MS
                fill = make_halt_minute_from_prefix(prefix, t)
                yield MinuteEnvelopeR1(fill, True)
                prefix = fill
        yield MinuteEnvelopeR1(cur, False)
        prev = cur


def student_visible_market_tuple(window: Sequence[MinuteEnvelopeR1]) -> tuple[tuple[float, ...], ...]:
    """Student-visible market surface; audit halt flags are intentionally excluded."""
    return tuple(
        (
            float(x.record.open), float(x.record.high), float(x.record.low),
            float(x.record.close), float(x.record.volume),
        )
        for x in window
    )


def causal_long_trajectory_scan_r1(
    minutes: Iterable[MinuteEnvelopeR1],
    *,
    lookback: int,
    initial_account: Any,
    policy: Callable[[tuple[tuple[float, ...], ...], Any, int], Any],
    transition: Callable[[Any, Any, MinuteEnvelopeR1, int], Any],
    stop_before_ms: int = FINAL_HOLDOUT_START_MS,
) -> tuple[Any, list[CausalDecisionEventR1]]:
    """Advance one path-dependent trajectory with an explicit causal call order.

    At decision t, policy receives only a window ending at t and Account[t].
    Only after policy returns is the t->t+1 MinuteEnvelope handed to transition.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    q: deque[MinuteEnvelopeR1] = deque(maxlen=lookback)
    account = initial_account
    events: list[CausalDecisionEventR1] = []

    it = iter(minutes)
    try:
        current = next(it)
    except StopIteration:
        return account, events
    if current.timestamp >= stop_before_ms:
        return account, events
    q.append(current)

    for nxt in it:
        if nxt.timestamp >= stop_before_ms:
            break
        if nxt.timestamp - current.timestamp != MINUTE_MS:
            raise RuntimeError("R1_EXPANDED_STREAM_NOT_CONTIGUOUS")
        q.append(current) if q[-1].timestamp != current.timestamp else None
        if len(q) == lookback:
            decision_time = current.timestamp
            if q[-1].timestamp != decision_time:
                raise RuntimeError("OBSERVATION_ENDPOINT_MISMATCH")
            if any(x.timestamp > decision_time for x in q):
                raise RuntimeError("FUTURE_MARKET_IN_OBSERVATION")
            account_before_hash = canonical_hash(account)
            obs = student_visible_market_tuple(tuple(q))
            # Critical ordering boundary: policy returns before nxt is exposed to transition.
            action = policy(obs, account, decision_time)
            action_hash = canonical_hash(action)
            account_after = transition(account, action, nxt, decision_time)
            events.append(CausalDecisionEventR1(
                decision_time_ms=decision_time,
                observation_max_time_ms=max(x.timestamp for x in q),
                account_before_hash=account_before_hash,
                action_hash=action_hash,
                transition_time_ms=nxt.timestamp,
                account_after_hash=canonical_hash(account_after),
                transition_halt_imputed=bool(nxt.halt_imputed),
            ))
            account = account_after
        current = nxt
        q.append(current)

    return account, events


def make_checkpoint_r1(*, next_decision_time_ms: int, account_state: Any, events: Sequence[CausalDecisionEventR1]) -> TrajectoryCheckpointR1:
    prefix = [vars(e) for e in events]
    return TrajectoryCheckpointR1(
        next_decision_time_ms=int(next_decision_time_ms),
        account_state=account_state,
        prefix_event_count=len(events),
        prefix_hash=canonical_hash(prefix),
    )


def vectorized_synthetic_account_step_r1(
    accounts: np.ndarray,
    actions: np.ndarray,
    market_returns: np.ndarray,
) -> np.ndarray:
    """Small known-answer batch-axis canary, not production Physics.

    Shape B is the permitted parallel axis.  Time remains outside this function.
    """
    a = np.asarray(accounts, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64)
    r = np.asarray(market_returns, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("accounts must be [B,3]")
    if u.shape != (a.shape[0],) or r.shape != (a.shape[0],):
        raise ValueError("batch shapes do not match")
    out = a.copy()
    # [equity, exposure, cumulative_cost]
    out[:, 0] = a[:, 0] * (1.0 + a[:, 1] * r)
    turnover = np.abs(u - a[:, 1])
    out[:, 2] = a[:, 2] + 1e-4 * turnover
    out[:, 0] -= 1e-4 * turnover
    out[:, 1] = u
    return out


def scalar_synthetic_account_step_r1(account: Sequence[float], action: float, market_return: float) -> np.ndarray:
    return vectorized_synthetic_account_step_r1(
        np.asarray([account], dtype=np.float64),
        np.asarray([action], dtype=np.float64),
        np.asarray([market_return], dtype=np.float64),
    )[0]


def batch_axis_parity_canary_r1(seed: int = 20260911, batch: int = 64) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    accounts = np.column_stack([
        rng.uniform(0.5, 2.0, batch),
        rng.uniform(-1.0, 1.0, batch),
        rng.uniform(0.0, 0.02, batch),
    ])
    actions = rng.uniform(-1.0, 1.0, batch)
    returns = rng.normal(0.0, 0.01, batch)
    batched = vectorized_synthetic_account_step_r1(accounts, actions, returns)
    scalar = np.stack([
        scalar_synthetic_account_step_r1(accounts[i], actions[i], returns[i])
        for i in range(batch)
    ])
    exact = bool(np.array_equal(batched, scalar))
    return {
        "status": "PASS" if exact else "FAIL",
        "batch": int(batch),
        "exact_array_equal": exact,
        "max_abs_diff": float(np.max(np.abs(batched - scalar))),
        "semantic_note": "Batch-axis canary only; production Frozen Physics remains scalar authority until separately parity-qualified.",
    }
