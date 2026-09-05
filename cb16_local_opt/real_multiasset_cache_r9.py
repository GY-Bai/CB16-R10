from __future__ import annotations

"""R9 real multi-asset Market64 cache materialization.

This module builds one out-of-core cache per symbol from an immutable R9 Dataset
Snapshot using the exact frozen encoder installation receipt, then creates the R7
exact-timestamp synchronization index.

Important architecture boundary:
- every asset cache is built and qualified;
- one `primary_symbol` Market64 remains the status-driving Trader input in R9;
- the multi-asset synchronization artifact is operational/research infrastructure only;
- R9 does NOT silently concatenate multiple Market64 vectors into the Trader.
"""

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .dataset_snapshot_r9 import load_dataset_snapshot_receipt_r9
from .encoder_authority_r9 import (
    load_encoder_install_receipt_r9,
    resolve_model_factory_r9,
)
from .historical_shard_reader import load_manifest, HistoricalShardReader
from .market_encoder_r5 import FrozenMarketEncoderArtifact, WindowNormalizer
from .market_cache_r6 import MarketLatentCacheR6
from .streaming_market_cache_r7 import (
    build_streaming_market_cache_r7,
    build_multiasset_sync_index_r7,
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class MultiAssetCacheReceiptR9:
    cache_set_version: str
    status: str
    scientific_dataset_hash: str
    dataset_snapshot_receipt_hash: str
    encoder_install_receipt_hash: str
    encoder_weight_hash: str
    symbols: tuple[str, ...]
    primary_symbol: str
    primary_cache_root: str
    symbol_cache_roots: dict[str, str]
    symbol_cache_identity_hashes: dict[str, str]
    synchronization_root: str
    synchronization_identity_hash: str
    aligned_rows: int
    trader_multiasset_input_enabled: bool

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


def build_real_multiasset_cache_r9(
    *,
    dataset_snapshot_receipt_path: str | Path,
    encoder_install_receipt_path: str | Path,
    output_root: str | Path,
    primary_symbol: str,
    device: str = "cuda",
    read_batch_rows: int = 65536,
    encode_batch_windows: int = 8192,
    parquet_readahead_batches: int = 2,
    sync_chunk_rows: int = 262144,
) -> MultiAssetCacheReceiptR9:
    ds = load_dataset_snapshot_receipt_r9(dataset_snapshot_receipt_path)
    enc = load_encoder_install_receipt_r9(encoder_install_receipt_path)
    if enc.authority != "USER_FROZEN_ENCODER":
        raise RuntimeError("R9_REAL_CACHE_REQUIRES_USER_FROZEN_ENCODER")
    if primary_symbol not in ds.symbols:
        raise RuntimeError("R9_PRIMARY_SYMBOL_NOT_IN_DATASET")

    manifest = load_manifest(ds.mounted_manifest_path, verify_files=True)
    factory = resolve_model_factory_r9(enc.factory_spec)
    encoder = FrozenMarketEncoderArtifact(
        model=factory(),
        architecture_id=enc.architecture_id,
        artifact_path=enc.artifact_path,
        normalizer=WindowNormalizer(),
        expected_artifact_sha256=enc.artifact_sha256,
        expected_parameter_count=enc.parameter_count,
        authority="USER_FROZEN_ENCODER",
    )
    if encoder.receipt.state_dict_weight_hash != enc.state_dict_weight_hash:
        raise RuntimeError("R9_ENCODER_INSTALL_STATE_HASH_DRIFT")
    if encoder.receipt.normalization_hash != enc.normalization_hash:
        raise RuntimeError("R9_ENCODER_INSTALL_NORMALIZATION_DRIFT")

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    symbol_roots: dict[str, str] = {}
    identities: dict[str, str] = {}
    for symbol in ds.symbols:
        out = root / "symbols" / symbol
        reader = HistoricalShardReader(
            manifest,
            batch_rows=read_batch_rows,
            arrow_batch_readahead=parquet_readahead_batches,
            verify_hashes_on_open=False,
        )
        build_streaming_market_cache_r7(
            output_root=out,
            reader=reader,
            symbol=symbol,
            encoder=encoder,
            device=device,
            encode_batch_windows=encode_batch_windows,
            scientific_dataset_hash=ds.scientific_dataset_hash,
        )
        cache = MarketLatentCacheR6(out, verify_hashes=True)
        if cache.receipt.dataset_hash != ds.scientific_dataset_hash:
            raise RuntimeError(f"R9_CACHE_DATASET_IDENTITY_DRIFT:{symbol}")
        if cache.receipt.encoder_weight_hash != enc.state_dict_weight_hash:
            raise RuntimeError(f"R9_CACHE_ENCODER_IDENTITY_DRIFT:{symbol}")
        symbol_roots[symbol] = str(out.resolve())
        identities[symbol] = cache.receipt.scientific_identity_hash

    sync_root = root / "multiasset_sync"
    sync = build_multiasset_sync_index_r7(
        output_root=sync_root,
        cache_roots=symbol_roots,
        reference_symbol=primary_symbol,
        chunk_rows=sync_chunk_rows,
    )
    receipt = MultiAssetCacheReceiptR9(
        cache_set_version="CB16_REAL_MULTI_ASSET_MARKET_CACHE_R9",
        status="REAL_MULTI_ASSET_CACHE_QUALIFIED",
        scientific_dataset_hash=ds.scientific_dataset_hash,
        dataset_snapshot_receipt_hash=ds.content_hash,
        encoder_install_receipt_hash=enc.content_hash,
        encoder_weight_hash=enc.state_dict_weight_hash,
        symbols=ds.symbols,
        primary_symbol=primary_symbol,
        primary_cache_root=symbol_roots[primary_symbol],
        symbol_cache_roots=symbol_roots,
        symbol_cache_identity_hashes=identities,
        synchronization_root=str(sync_root.resolve()),
        synchronization_identity_hash=sync.identity_hash,
        aligned_rows=sync.aligned_rows,
        trader_multiasset_input_enabled=False,
    )
    (root / "MULTIASSET_CACHE_RECEIPT_R9.json").write_text(
        json.dumps({**asdict(receipt), "content_hash": receipt.content_hash}, indent=2) + "\n"
    )
    return receipt


def load_multiasset_cache_receipt_r9(path: str | Path) -> MultiAssetCacheReceiptR9:
    obj = json.loads(Path(path).read_text())
    claimed = obj.pop("content_hash", None)
    obj["symbols"] = tuple(obj["symbols"])
    r = MultiAssetCacheReceiptR9(**obj)
    if claimed and claimed != r.content_hash:
        raise RuntimeError("R9_MULTI_ASSET_CACHE_RECEIPT_HASH_MISMATCH")
    return r
