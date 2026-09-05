from __future__ import annotations

"""
Mmap-friendly frozen Market-latent cache.

The cache is generation-independent because the Market Encoder is frozen. It is built once
per canonical dataset/encoder artifact and then reused by all Champion/Challenger
generations.

Layout:
    timestamp.npy
    open.npy / high.npy / low.npy / close.npy / volume.npy / funding_rate.npy
    market_latent.npy      [T,64] float32
    latent_valid.npy       [T] bool
    CACHE_RECEIPT.json

The first 31 rows are invalid for a 32-bar encoder window.
"""

import dataclasses
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .market_encoder_r5 import (
    FrozenMarketEncoderArtifact,
    LATENT_DIM,
    RAW_CHANNELS,
    WINDOW_BARS,
)
from .trajectory_compiler_r6 import MarketPathR6


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(chunk_bytes), b""):
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MarketLatentCacheReceiptR6:
    cache_version: str
    dataset_hash: str
    encoder_artifact_sha256: str
    encoder_weight_hash: str
    encoder_receipt_hash: str
    rows: int
    first_valid_index: int
    last_valid_index: int
    latent_dim: int
    window_bars: int
    files_sha256: dict[str, str]
    scientific_identity_hash: str

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


class MarketLatentCacheR6:
    FILES = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "market_latent",
        "latent_valid",
    )

    def __init__(self, root: str | Path, *, verify_hashes: bool = False):
        self.root = Path(root)
        receipt_path = self.root / "CACHE_RECEIPT.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(receipt_path)
        obj = json.loads(receipt_path.read_text())
        self.receipt = MarketLatentCacheReceiptR6(**obj)
        self.arrays = {
            name: np.load(self.root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.FILES
        }
        n = self.receipt.rows
        if any(len(a) != n for a in self.arrays.values()):
            raise RuntimeError("MARKET_CACHE_ARRAY_LENGTH_MISMATCH")
        if self.arrays["market_latent"].shape != (n, LATENT_DIM):
            raise RuntimeError("MARKET_CACHE_LATENT_SHAPE_MISMATCH")
        if verify_hashes:
            for name in self.FILES:
                actual = sha256_file(self.root / f"{name}.npy")
                if actual != self.receipt.files_sha256[name]:
                    raise RuntimeError(f"MARKET_CACHE_HASH_MISMATCH:{name}")

    def market_path(self) -> MarketPathR6:
        return MarketPathR6(
            timestamp=self.arrays["timestamp"],
            open=self.arrays["open"],
            high=self.arrays["high"],
            low=self.arrays["low"],
            close=self.arrays["close"],
            volume=self.arrays["volume"],
            funding_rate=self.arrays["funding_rate"],
        )


def build_market_latent_cache_r6(
    *,
    output_root: str | Path,
    market: MarketPathR6,
    dataset_hash: str,
    encoder: FrozenMarketEncoderArtifact,
    device: str = "cpu",
    batch_windows: int = 8192,
) -> MarketLatentCacheReceiptR6:
    market.validate()
    if market.rows < WINDOW_BARS:
        raise ValueError("market shorter than encoder window")
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)

    arrays = {
        "timestamp": np.asarray(market.timestamp, dtype=np.int64),
        "open": np.asarray(market.open, dtype=np.float64),
        "high": np.asarray(market.high, dtype=np.float64),
        "low": np.asarray(market.low, dtype=np.float64),
        "close": np.asarray(market.close, dtype=np.float64),
        "volume": np.asarray(market.volume, dtype=np.float64),
        "funding_rate": (
            np.zeros(market.rows, dtype=np.float64)
            if market.funding_rate is None
            else np.asarray(market.funding_rate, dtype=np.float64)
        ),
    }
    for name, a in arrays.items():
        np.save(out / f"{name}.npy", a, allow_pickle=False)

    latent = np.lib.format.open_memmap(
        out / "market_latent.npy",
        mode="w+",
        dtype=np.float32,
        shape=(market.rows, LATENT_DIM),
    )
    latent[:] = 0
    valid = np.lib.format.open_memmap(
        out / "latent_valid.npy",
        mode="w+",
        dtype=bool,
        shape=(market.rows,),
    )
    valid[:] = False

    ohlcv = np.stack(
        [
            arrays["open"],
            arrays["high"],
            arrays["low"],
            arrays["close"],
            arrays["volume"],
        ],
        axis=1,
    )
    # Virtual rolling view: [T-31, 32, 5], no full window duplication.
    windows = np.lib.stride_tricks.sliding_window_view(
        ohlcv,
        window_shape=WINDOW_BARS,
        axis=0,
    )
    # sliding_window_view on [T,5] with axis=0 yields [T-31,5,32].
    windows = np.moveaxis(windows, -1, 1)
    if windows.shape[1:] != (WINDOW_BARS, RAW_CHANNELS):
        raise RuntimeError(f"UNEXPECTED_ROLLING_WINDOW_SHAPE:{windows.shape}")

    for start in range(0, len(windows), batch_windows):
        stop = min(len(windows), start + batch_windows)
        z = encoder.encode_numpy(
            np.asarray(windows[start:stop]),
            device=device,
            batch_size=batch_windows,
        )
        target_start = start + WINDOW_BARS - 1
        target_stop = stop + WINDOW_BARS - 1
        latent[target_start:target_stop] = z
        valid[target_start:target_stop] = True

    latent.flush()
    valid.flush()
    del latent, valid

    files_sha = {
        name: sha256_file(out / f"{name}.npy")
        for name in MarketLatentCacheR6.FILES
    }
    identity = canonical_hash({
        "dataset_hash": dataset_hash,
        "encoder_artifact_sha256": encoder.receipt.artifact_sha256,
        "encoder_weight_hash": encoder.receipt.state_dict_weight_hash,
        "normalization_hash": encoder.receipt.normalization_hash,
        "rows": market.rows,
        "first_valid_index": WINDOW_BARS - 1,
        "last_valid_index": market.rows - 1,
        "files_sha256": files_sha,
    })
    receipt = MarketLatentCacheReceiptR6(
        cache_version="CB16_MARKET_LATENT_CACHE_R6",
        dataset_hash=dataset_hash,
        encoder_artifact_sha256=encoder.receipt.artifact_sha256,
        encoder_weight_hash=encoder.receipt.state_dict_weight_hash,
        encoder_receipt_hash=encoder.receipt.content_hash,
        rows=market.rows,
        first_valid_index=WINDOW_BARS - 1,
        last_valid_index=market.rows - 1,
        latent_dim=LATENT_DIM,
        window_bars=WINDOW_BARS,
        files_sha256=files_sha,
        scientific_identity_hash=identity,
    )
    (out / "CACHE_RECEIPT.json").write_text(
        json.dumps(asdict(receipt), indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
