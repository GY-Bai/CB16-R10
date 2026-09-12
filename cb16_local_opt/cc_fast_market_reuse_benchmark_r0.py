from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np

from .cc_fast_market_cache_r0 import MarketCacheKey, SharedMarketOwner


@dataclass(frozen=True)
class MarketReuseBenchmark:
    account_count: int
    market_bytes: int
    repeated_materialized_bytes: int
    shared_materialized_bytes: int
    repeated_compute_s: float
    shared_compute_s: float
    speedup_ratio: float
    memory_reduction_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _frozen_market_transform(market: np.ndarray) -> np.ndarray:
    x = np.asarray(market, dtype=np.float64)
    return np.ascontiguousarray(np.tanh(x * 1.0e-3) + np.sqrt(np.abs(x) + 1.0))


def benchmark_market_reuse(*, account_count: int = 16, rows: int = 20000, repeats: int = 3) -> MarketReuseBenchmark:
    if account_count <= 1 or rows <= 0 or repeats <= 0:
        raise ValueError("MARKET_REUSE_BENCHMARK_CONFIG_INVALID")
    market = np.column_stack((np.linspace(1.0, 1000.0, rows), np.linspace(-1.0, 1.0, rows))).astype(np.float64)

    t0 = time.perf_counter()
    repeated_checksum = 0.0
    for _ in range(repeats):
        for _account in range(account_count):
            repeated_checksum += float(_frozen_market_transform(market)[-1, 0])
    repeated_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    shared_checksum = 0.0
    for _ in range(repeats):
        transformed = _frozen_market_transform(market)
        key = MarketCacheKey("synthetic-market", f"rows:{rows}", "reuse-bench-v1", "frozen-transform-v1", "noop-normalizer-v1")
        with SharedMarketOwner(key, transformed) as owner:
            descriptors = [owner.descriptor for _ in range(account_count)]
            if len({d.shm_name for d in descriptors}) != 1:
                raise RuntimeError("MARKET_REUSE_MATERIALIZED_MULTIPLE_SHARED_SEGMENTS")
            shared_checksum += float(transformed[-1, 0]) * account_count
    shared_s = time.perf_counter() - t1

    if not np.isclose(repeated_checksum, shared_checksum, rtol=0.0, atol=1e-12):
        raise RuntimeError("MARKET_REUSE_SEMANTIC_CHECKSUM_MISMATCH")

    repeated_bytes = account_count * market.nbytes
    shared_bytes = market.nbytes
    return MarketReuseBenchmark(
        account_count=account_count,
        market_bytes=int(market.nbytes),
        repeated_materialized_bytes=int(repeated_bytes),
        shared_materialized_bytes=int(shared_bytes),
        repeated_compute_s=float(repeated_s),
        shared_compute_s=float(shared_s),
        speedup_ratio=float(repeated_s / max(shared_s, 1e-12)),
        memory_reduction_ratio=float(repeated_bytes / shared_bytes),
    )
