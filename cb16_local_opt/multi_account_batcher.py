from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class BatchSlice:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass
class AccountFeatureBatch:
    account_ids: np.ndarray
    market_latent: np.ndarray
    account_features: np.ndarray
    fused_features: np.ndarray


class MultiAccountBatcher:
    """Broadcast one market latent across many account states with stable account IDs."""

    def __init__(self, *, market_dim: int = 64, account_dim: int = 6):
        self.market_dim = int(market_dim)
        self.account_dim = int(account_dim)

    def fuse_numpy(
        self,
        market_latent: np.ndarray,
        account_features: np.ndarray,
        *,
        account_ids: np.ndarray | None = None,
        dtype=np.float32,
    ) -> AccountFeatureBatch:
        m = np.asarray(market_latent, dtype=dtype)
        a = np.asarray(account_features, dtype=dtype)
        if m.shape != (self.market_dim,):
            raise ValueError(f"market latent shape must be {(self.market_dim,)}")
        if a.ndim != 2 or a.shape[1] != self.account_dim:
            raise ValueError(f"account features must be [N,{self.account_dim}]")
        n = a.shape[0]
        if account_ids is None:
            account_ids = np.arange(n, dtype=np.int64)
        account_ids = np.asarray(account_ids)
        if len(account_ids) != n:
            raise ValueError("account id count mismatch")

        # broadcast_to is zero-copy; concatenate materializes one contiguous fused matrix.
        market = np.broadcast_to(m, (n, self.market_dim))
        fused = np.concatenate([market, a], axis=1).astype(dtype, copy=False)
        return AccountFeatureBatch(account_ids, m, a, fused)

    def iter_slices(self, n: int, batch_size: int) -> Iterator[BatchSlice]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for start in range(0, n, batch_size):
            yield BatchSlice(start, min(start + batch_size, n))

    def to_torch(
        self,
        batch: AccountFeatureBatch,
        *,
        device: str = "cpu",
        pin_memory: bool = False,
        non_blocking: bool = True,
    ):
        if torch is None:
            raise RuntimeError("torch unavailable")
        t = torch.from_numpy(np.ascontiguousarray(batch.fused_features))
        if pin_memory and torch.cuda.is_available():
            t = t.pin_memory()
        return t.to(device, non_blocking=non_blocking)

    @staticmethod
    def assert_permutation_equivalence(
        fn,
        account_ids: np.ndarray,
        features: np.ndarray,
        *,
        seed: int = 123,
        atol: float = 1e-6,
    ) -> None:
        """Check that batch ordering does not change per-account outputs."""
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(account_ids))
        y0 = np.asarray(fn(features))
        yp = np.asarray(fn(features[perm]))
        inverse = np.argsort(perm)
        if not np.allclose(y0, yp[inverse], atol=atol, rtol=0):
            raise RuntimeError("ACCOUNT_BATCH_PERMUTATION_NON_EQUIVALENT")


class DeviceBatchPlanner:
    """Memory-aware candidate selection without assuming modern GPU behavior."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim_hint: int = 256,
        dtype_bytes: int = 4,
        safety_fraction: float = 0.70,
    ):
        self.feature_dim = int(feature_dim)
        self.hidden_dim_hint = int(hidden_dim_hint)
        self.dtype_bytes = int(dtype_bytes)
        self.safety_fraction = float(safety_fraction)

    def estimate_inference_bytes(self, batch_size: int) -> int:
        # Conservative: input + several hidden buffers + logits/sizing + workspace multiplier.
        elems = batch_size * (self.feature_dim + self.hidden_dim_hint * 4 + 32)
        return int(elems * self.dtype_bytes * 2.0)

    def fit_candidates(self, vram_bytes: int, candidates: Sequence[int]) -> list[int]:
        budget = vram_bytes * self.safety_fraction
        return [int(b) for b in candidates if self.estimate_inference_bytes(int(b)) <= budget]
