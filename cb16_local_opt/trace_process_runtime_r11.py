from __future__ import annotations

"""Linux fork-process execution plane for independent R11 H72 account groups.

This module changes only WHERE independent account groups execute.  Every account group
still runs its H72 requests in decision-time order, every H72 step still calls the frozen
Supervisor and Physics through ``_simulate_h72_branch_r11``, and output is canonically
reassembled independent of process completion order.

The parent campaign preloads immutable hourly market arrays before the process pool is
created.  Linux ``fork`` then shares those read-only NumPy data pages by copy-on-write;
workers are forbidden from loading additional symbols after fork.  This avoids the Python
GIL without changing market identity or duplicating compressed-cache I/O per worker.
"""

from concurrent.futures import Future, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
import multiprocessing as mp
import os
from threading import RLock
from typing import Mapping, Sequence

from .market_runtime_cache_r11 import MarketRuntimeCacheR11
from .r102_physics import FrozenPhysicsRuntimeR102
from .trace_runtime_r11 import (
    H72TraceExecutionR11,
    H72TraceWorkItemR11,
    _simulate_h72_branch_r11,
)


R11_FORK_TRACE_RUNTIME = "CB16_R11_STAGE2_FORK_COW_TRACE_RUNTIME_V1"

_FORK_PHYSICS_R11: FrozenPhysicsRuntimeR102 | None = None
_FORK_MARKET_CACHE_R11: MarketRuntimeCacheR11 | None = None
_FORK_ALLOWED_SYMBOLS_R11: frozenset[str] = frozenset()
_FORK_BIND_LOCK_R11 = RLock()


@dataclass(frozen=True)
class ForkTraceRuntimeStatsR11:
    workers: int
    pool_executions: int
    preloaded_symbols: tuple[str, ...]
    process_start_method: str = "fork"
    market_memory_model: str = "PARENT_PRELOAD__LINUX_FORK_COW_READONLY_NUMPY_PAGES"
    topology_in_scientific_identity: bool = False


def _execute_account_group_fork_r11(
    items: Sequence[H72TraceWorkItemR11],
) -> list[H72TraceExecutionR11]:
    physics = _FORK_PHYSICS_R11
    market_cache = _FORK_MARKET_CACHE_R11
    if physics is None or market_cache is None:
        raise RuntimeError("R11_FORK_TRACE_WORKER_NOT_BOUND")

    ordered = sorted(
        items,
        key=lambda x: (int(x.decision_time_ms), int(x.ordinal), x.causal_trace_id),
    )
    if len({x.account_id for x in ordered}) != 1:
        raise RuntimeError("R11_ACCOUNT_GROUP_MIXED_ACCOUNT_IDS")

    out: list[H72TraceExecutionR11] = []
    cached_symbols = getattr(market_cache, "_symbols", None)
    if not isinstance(cached_symbols, dict):
        raise RuntimeError("R11_FORK_TRACE_CACHE_INTERNALS_UNAVAILABLE")

    for item in ordered:
        if item.symbol not in _FORK_ALLOWED_SYMBOLS_R11:
            raise RuntimeError(f"R11_FORK_TRACE_SYMBOL_NOT_PRELOADED:{item.symbol}")
        market = cached_symbols.get(item.symbol)
        if market is None:
            # Fail closed: workers may never decompress/load a new payload post-fork.
            raise RuntimeError(f"R11_FORK_TRACE_INHERITED_SYMBOL_MISSING:{item.symbol}")
        branch, snapshots = _simulate_h72_branch_r11(
            physics,
            market,
            parent=item.parent_state,
            decision_time_ms=int(item.decision_time_ms),
            candidate_direction_v55=int(item.direction_v55),
            candidate_risk=float(item.requested_risk),
        )
        out.append(
            H72TraceExecutionR11(
                ordinal=int(item.ordinal),
                causal_trace_id=item.causal_trace_id,
                account_id=item.account_id,
                parent_id=item.parent_id,
                symbol=item.symbol,
                branch=branch,
                snapshot_sha256_sequence=snapshots,
            )
        )
    return out


class ForkTraceBatchFutureR11:
    """Fail-closed process batch with completion-order-independent output identity."""

    def __init__(self, futures: Mapping[Future, tuple[int, str]]):
        self._futures = dict(futures)

    def result(self, timeout: float | None = None) -> list[H72TraceExecutionR11]:
        rows: list[H72TraceExecutionR11] = []
        futures = tuple(self._futures)
        try:
            for future in as_completed(futures, timeout=timeout):
                rows.extend(future.result())
        except Exception as exc:
            remaining = tuple(f for f in futures if not f.done())
            for future in remaining:
                future.cancel()
            if remaining:
                wait(remaining)
            if isinstance(exc, RuntimeError) and str(exc).startswith("R11_"):
                raise RuntimeError(str(exc)) from exc
            raise RuntimeError("R11_FORK_TRACE_WORKER_EXCEPTION") from exc

        rows.sort(key=lambda x: (int(x.ordinal), x.causal_trace_id, x.parent_id))
        ordinals = [int(x.ordinal) for x in rows]
        if len(ordinals) != len(set(ordinals)):
            raise RuntimeError("R11_DUPLICATE_TRACE_ORDINAL")
        trace_ids = [x.causal_trace_id for x in rows]
        if len(trace_ids) != len(set(trace_ids)):
            raise RuntimeError("R11_DUPLICATE_CAUSAL_TRACE_ID")
        return rows


class ForkProcessTraceRuntimeR11:
    """Persistent Linux fork pool for independent H72 account groups.

    The pool is lazy: the first submit preloads exactly the required symbol set and then
    forks workers.  Later submissions must use only that same/preloaded symbol set so no
    child can observe a newer or separately decompressed market cache.
    """

    def __init__(
        self,
        *,
        physics: FrozenPhysicsRuntimeR102,
        market_cache: MarketRuntimeCacheR11,
        max_workers: int,
    ) -> None:
        workers = int(max_workers)
        if workers <= 0:
            raise ValueError("R11_FORK_TRACE_MAX_WORKERS_MUST_BE_POSITIVE")
        if os.name != "posix" or "fork" not in mp.get_all_start_methods():
            raise RuntimeError("R11_FORK_TRACE_REQUIRES_POSIX_FORK")
        self.physics = physics
        self.market_cache = market_cache
        self.max_workers = workers
        self._executor: ProcessPoolExecutor | None = None
        self._preloaded_symbols: frozenset[str] = frozenset()
        self._closed = False
        self._lock = RLock()
        self._executions = 0

    @property
    def executions(self) -> int:
        with self._lock:
            return int(self._executions)

    @property
    def executor_identity(self) -> int | None:
        with self._lock:
            return None if self._executor is None else id(self._executor)

    def _ensure_pool_r11(self, symbols: frozenset[str]) -> ProcessPoolExecutor:
        global _FORK_PHYSICS_R11, _FORK_MARKET_CACHE_R11, _FORK_ALLOWED_SYMBOLS_R11

        if self._executor is not None:
            missing = sorted(symbols - self._preloaded_symbols)
            if missing:
                raise RuntimeError(
                    f"R11_FORK_TRACE_SYMBOL_SET_EXPANSION_REQUIRES_NEW_POOL:{missing[0]}"
                )
            return self._executor

        self.market_cache.preload(sorted(symbols))
        self.market_cache.assert_read_only()
        with _FORK_BIND_LOCK_R11:
            _FORK_PHYSICS_R11 = self.physics
            _FORK_MARKET_CACHE_R11 = self.market_cache
            _FORK_ALLOWED_SYMBOLS_R11 = frozenset(symbols)
            ctx = mp.get_context("fork")
            executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
            self._executor = executor
            self._preloaded_symbols = frozenset(symbols)
            return executor

    @staticmethod
    def _validate_items_r11(rows: Sequence[H72TraceWorkItemR11]) -> None:
        ordinals = [int(x.ordinal) for x in rows]
        if len(ordinals) != len(set(ordinals)):
            raise RuntimeError("R11_DUPLICATE_TRACE_ORDINAL")
        trace_ids = [x.causal_trace_id for x in rows]
        if len(trace_ids) != len(set(trace_ids)):
            raise RuntimeError("R11_DUPLICATE_CAUSAL_TRACE_ID")

    def submit(self, items: Sequence[H72TraceWorkItemR11]) -> ForkTraceBatchFutureR11:
        with self._lock:
            if self._closed:
                raise RuntimeError("R11_FORK_TRACE_RUNTIME_CLOSED")
            rows = list(items)
            self._validate_items_r11(rows)
            if not rows:
                return ForkTraceBatchFutureR11({})
            symbols = frozenset(x.symbol for x in rows)
            executor = self._ensure_pool_r11(symbols)

            grouped: dict[str, list[H72TraceWorkItemR11]] = {}
            for row in rows:
                grouped.setdefault(row.account_id, []).append(row)
            futures: dict[Future, tuple[int, str]] = {}
            for account_id in sorted(grouped):
                group = tuple(grouped[account_id])
                first = min(int(x.ordinal) for x in group)
                future = executor.submit(_execute_account_group_fork_r11, group)
                futures[future] = (first, account_id)
            self._executions += 1
            return ForkTraceBatchFutureR11(futures)

    def run(self, items: Sequence[H72TraceWorkItemR11]) -> list[H72TraceExecutionR11]:
        if not items:
            return []
        return self.submit(items).result()

    def stats(self) -> ForkTraceRuntimeStatsR11:
        with self._lock:
            return ForkTraceRuntimeStatsR11(
                workers=self.max_workers,
                pool_executions=self._executions,
                preloaded_symbols=tuple(sorted(self._preloaded_symbols)),
            )

    def close(self) -> None:
        global _FORK_PHYSICS_R11, _FORK_MARKET_CACHE_R11, _FORK_ALLOWED_SYMBOLS_R11
        with self._lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with _FORK_BIND_LOCK_R11:
            if _FORK_MARKET_CACHE_R11 is self.market_cache:
                _FORK_PHYSICS_R11 = None
                _FORK_MARKET_CACHE_R11 = None
                _FORK_ALLOWED_SYMBOLS_R11 = frozenset()

    def __enter__(self) -> "ForkProcessTraceRuntimeR11":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
