from __future__ import annotations

"""
Truly streaming / out-of-core Market64 cache builder and multi-asset synchronized cache set.

Single-asset builder:
- total row count comes from immutable shard manifest metadata;
- `.npy` arrays are preallocated once;
- historical batches are written directly into mmap arrays;
- only 31 tail rows are retained between batches;
- Encoder windows are produced for the current batch only;
- full historical OHLCV is never concatenated in RAM.

Multi-asset synchronization:
- choose one reference asset;
- scan reference timestamps in chunks;
- `searchsorted` exact matches against other mmap timestamp arrays;
- append aligned row indices to raw int64 files;
- no giant in-memory intersection matrix is required.
"""

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .historical_shard_reader import DatasetManifest, HistoricalShardReader
from .market_cache_r6 import (
    MarketLatentCacheR6,
    MarketLatentCacheReceiptR6,
    sha256_file,
)
from .market_encoder_r5 import (
    FrozenMarketEncoderArtifact,
    LATENT_DIM,
    RAW_CHANNELS,
    WINDOW_BARS,
)


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp_name=tempfile.mkstemp(
        prefix=path.name+".",suffix=".partial",dir=path.parent
    )
    os.close(fd)
    tmp=Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
        os.replace(tmp,path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class StreamingCacheBuildReceiptR7:
    symbol:str
    dataset_hash:str
    rows:int
    historical_batches:int
    peak_window_rows_in_memory:int
    cache_receipt_hash:str
    cache_identity_hash:str
    status:str

    @property
    def content_hash(self):return canonical_hash(self)


def _symbol_row_count(manifest:DatasetManifest,symbol:str)->int:
    shards=[s for s in manifest.shards if s.symbol==symbol]
    if not shards:raise ValueError(f"symbol not in manifest:{symbol}")
    return int(sum(s.rows for s in shards))


def build_streaming_market_cache_r7(
    *,
    output_root:str|Path,
    reader:HistoricalShardReader,
    symbol:str,
    encoder:FrozenMarketEncoderArtifact,
    device:str="cpu",
    encode_batch_windows:int=8192,
    scientific_dataset_hash:str|None=None,
)->StreamingCacheBuildReceiptR7:
    manifest=reader.manifest
    manifest.validate(verify_files=False)
    dataset_hash = scientific_dataset_hash or manifest.content_hash
    total=_symbol_row_count(manifest,symbol)
    if total<WINDOW_BARS:
        raise RuntimeError("SYMBOL_HISTORY_SHORTER_THAN_ENCODER_WINDOW")
    out=Path(output_root)
    out.mkdir(parents=True,exist_ok=True)

    dtypes={
        "timestamp":np.int64,
        "open":np.float64,
        "high":np.float64,
        "low":np.float64,
        "close":np.float64,
        "volume":np.float64,
        "funding_rate":np.float64,
    }
    arrays={
        name:np.lib.format.open_memmap(
            out/f"{name}.npy",mode="w+",dtype=dtype,shape=(total,)
        )
        for name,dtype in dtypes.items()
    }
    latent=np.lib.format.open_memmap(
        out/"market_latent.npy",mode="w+",dtype=np.float32,
        shape=(total,LATENT_DIM)
    )
    valid=np.lib.format.open_memmap(
        out/"latent_valid.npy",mode="w+",dtype=bool,shape=(total,)
    )
    latent[:]=0;valid[:]=False
    arrays["funding_rate"][:]=0.0

    cursor=0
    previous_ts=None
    tail=np.empty((0,RAW_CHANNELS),dtype=np.float64)
    # Carry unencoded windows across historical batch boundaries. This makes Encoder
    # GEMM batch composition identical to a full-history builder using the same
    # `encode_batch_windows`, avoiding ~1e-6 FP32 changes caused only by shard layout.
    pending_windows=np.empty((0,WINDOW_BARS,RAW_CHANNELS),dtype=np.float64)
    pending_end_indices=np.empty((0,),dtype=np.int64)
    batch_count=0
    peak_window_rows=0

    def flush_full_encoder_batches():
        nonlocal pending_windows,pending_end_indices
        while len(pending_windows)>=encode_batch_windows:
            x=pending_windows[:encode_batch_windows]
            idx=pending_end_indices[:encode_batch_windows]
            z=encoder.encode_numpy(
                x,device=device,batch_size=encode_batch_windows
            )
            latent[idx]=z
            valid[idx]=True
            pending_windows=pending_windows[encode_batch_windows:]
            pending_end_indices=pending_end_indices[encode_batch_windows:]

    for batch in reader.iter_batches(symbol=symbol):
        previous_ts=batch.validate(previous_timestamp=previous_ts)
        n=batch.rows
        if cursor+n>total:
            raise RuntimeError("MANIFEST_ROW_COUNT_UNDERESTIMATED")
        sl=slice(cursor,cursor+n)
        arrays["timestamp"][sl]=batch.timestamp
        arrays["open"][sl]=batch.open
        arrays["high"][sl]=batch.high
        arrays["low"][sl]=batch.low
        arrays["close"][sl]=batch.close
        arrays["volume"][sl]=batch.volume

        current=np.stack([
            np.asarray(batch.open,dtype=np.float64),
            np.asarray(batch.high,dtype=np.float64),
            np.asarray(batch.low,dtype=np.float64),
            np.asarray(batch.close,dtype=np.float64),
            np.asarray(batch.volume,dtype=np.float64),
        ],axis=1)
        combined=np.concatenate([tail,current],axis=0)
        peak_window_rows=max(peak_window_rows,len(combined))

        # All windows ending in the current batch. combined has at most 31 prior rows.
        if len(combined)>=WINDOW_BARS:
            view=np.lib.stride_tricks.sliding_window_view(
                combined,window_shape=WINDOW_BARS,axis=0
            )
            view=np.moveaxis(view,-1,1)  # [W,32,5]
            global_combined_start=cursor-len(tail)
            end_global=(
                global_combined_start
                + np.arange(WINDOW_BARS-1,len(combined))
            )
            mask=(end_global>=cursor)&(end_global<cursor+n)
            selected=np.asarray(view[mask])
            selected_end=end_global[mask]
            if len(selected):
                if len(pending_windows):
                    pending_windows=np.concatenate(
                        [pending_windows,selected],axis=0
                    )
                    pending_end_indices=np.concatenate(
                        [pending_end_indices,selected_end]
                    )
                else:
                    pending_windows=np.array(selected,copy=True)
                    pending_end_indices=np.array(selected_end,copy=True)
                flush_full_encoder_batches()

        tail=combined[-(WINDOW_BARS-1):].copy()
        cursor+=n
        batch_count+=1

    if cursor!=total:
        raise RuntimeError(
            f"MANIFEST_ROW_COUNT_MISMATCH expected={total} actual={cursor}"
        )
    # One deterministic final partial Encoder batch, exactly as full-history encoding.
    if len(pending_windows):
        z=encoder.encode_numpy(
            pending_windows,device=device,batch_size=encode_batch_windows
        )
        latent[pending_end_indices]=z
        valid[pending_end_indices]=True
        pending_windows=np.empty((0,WINDOW_BARS,RAW_CHANNELS),dtype=np.float64)
        pending_end_indices=np.empty((0,),dtype=np.int64)

    for a in arrays.values():a.flush()
    latent.flush();valid.flush()
    del latent,valid
    for name in list(arrays):del arrays[name]

    files_sha={
        name:sha256_file(out/f"{name}.npy")
        for name in MarketLatentCacheR6.FILES
    }
    identity=canonical_hash({
        "dataset_hash":dataset_hash,
        "symbol":symbol,
        "encoder_artifact_sha256":encoder.receipt.artifact_sha256,
        "encoder_weight_hash":encoder.receipt.state_dict_weight_hash,
        "normalization_hash":encoder.receipt.normalization_hash,
        "rows":total,
        "first_valid_index":WINDOW_BARS-1,
        "last_valid_index":total-1,
        "files_sha256":files_sha,
    })
    cache_receipt=MarketLatentCacheReceiptR6(
        cache_version="CB16_STREAMING_MARKET_LATENT_CACHE_R7",
        dataset_hash=dataset_hash,
        encoder_artifact_sha256=encoder.receipt.artifact_sha256,
        encoder_weight_hash=encoder.receipt.state_dict_weight_hash,
        encoder_receipt_hash=encoder.receipt.content_hash,
        rows=total,
        first_valid_index=WINDOW_BARS-1,
        last_valid_index=total-1,
        latent_dim=LATENT_DIM,
        window_bars=WINDOW_BARS,
        files_sha256=files_sha,
        scientific_identity_hash=identity,
    )
    atomic_json(out/"CACHE_RECEIPT.json",asdict(cache_receipt))
    receipt=StreamingCacheBuildReceiptR7(
        symbol=symbol,
        dataset_hash=dataset_hash,
        rows=total,
        historical_batches=batch_count,
        peak_window_rows_in_memory=peak_window_rows,
        cache_receipt_hash=cache_receipt.content_hash,
        cache_identity_hash=identity,
        status="PASS",
    )
    atomic_json(out/"STREAMING_BUILD_RECEIPT_R7.json",asdict(receipt))
    return receipt


@dataclass(frozen=True)
class MultiAssetCacheMemberR7:
    symbol:str
    cache_root:str
    cache_identity_hash:str
    rows:int
    encoder_weight_hash:str


@dataclass(frozen=True)
class MultiAssetSynchronizationReceiptR7:
    set_version:str
    reference_symbol:str
    symbols:tuple[str,...]
    aligned_rows:int
    members:tuple[MultiAssetCacheMemberR7,...]
    index_files:Mapping[str,str]
    index_sha256:Mapping[str,str]
    timestamp_sha256:str
    identity_hash:str

    @property
    def content_hash(self):return canonical_hash(self)


class MultiAssetMarketCacheSetR7:
    def __init__(self,root:str|Path,*,verify_hashes:bool=False):
        self.root=Path(root)
        obj=json.loads((self.root/"MULTIASSET_RECEIPT.json").read_text())
        obj["symbols"]=tuple(obj["symbols"])
        obj["members"]=tuple(MultiAssetCacheMemberR7(**x) for x in obj["members"])
        self.receipt=MultiAssetSynchronizationReceiptR7(**obj)
        self.caches={
            m.symbol:MarketLatentCacheR6(m.cache_root,verify_hashes=False)
            for m in self.receipt.members
        }
        self.indices={
            s:np.memmap(
                self.root/self.receipt.index_files[s],
                mode="r",dtype=np.int64,shape=(self.receipt.aligned_rows,)
            )
            for s in self.receipt.symbols
        }
        self.timestamp=np.memmap(
            self.root/"aligned_timestamp.i64",
            mode="r",dtype=np.int64,shape=(self.receipt.aligned_rows,)
        )
        if verify_hashes:
            for s in self.receipt.symbols:
                if sha256_file(self.root/self.receipt.index_files[s])!=self.receipt.index_sha256[s]:
                    raise RuntimeError(f"MULTIASSET_INDEX_HASH_MISMATCH:{s}")
            if sha256_file(self.root/"aligned_timestamp.i64")!=self.receipt.timestamp_sha256:
                raise RuntimeError("MULTIASSET_TIMESTAMP_HASH_MISMATCH")

    def latent_matrix(self,aligned_row:int)->np.ndarray:
        """Return [assets,64] without copying historical arrays into RAM."""
        return np.stack([
            np.asarray(
                self.caches[s].arrays["market_latent"][
                    int(self.indices[s][aligned_row])
                ],
                dtype=np.float32,
            )
            for s in self.receipt.symbols
        ],axis=0)

    def all_valid(self,aligned_row:int)->bool:
        return all(
            bool(self.caches[s].arrays["latent_valid"][
                int(self.indices[s][aligned_row])
            ])
            for s in self.receipt.symbols
        )


def build_multiasset_sync_index_r7(
    *,
    output_root:str|Path,
    cache_roots:Mapping[str,str|Path],
    reference_symbol:str|None=None,
    chunk_rows:int=262144,
)->MultiAssetSynchronizationReceiptR7:
    if len(cache_roots)<2:raise ValueError("need >=2 assets")
    symbols=tuple(sorted(cache_roots))
    reference_symbol=reference_symbol or symbols[0]
    if reference_symbol not in cache_roots:raise ValueError("bad reference")
    caches={
        s:MarketLatentCacheR6(cache_roots[s],verify_hashes=False)
        for s in symbols
    }
    # Require same dataset manifest identity when all assets came from one multi-asset bundle.
    encoder_hashes={c.receipt.encoder_weight_hash for c in caches.values()}
    if len(encoder_hashes)!=1:
        raise RuntimeError("MULTIASSET_ENCODER_WEIGHT_MISMATCH")

    out=Path(output_root);out.mkdir(parents=True,exist_ok=True)
    index_paths={s:out/f"index_{s}.i64" for s in symbols}
    handles={s:index_paths[s].open("wb") for s in symbols}
    tpath=out/"aligned_timestamp.i64"
    th=tpath.open("wb")
    count=0
    try:
        ref_ts=caches[reference_symbol].arrays["timestamp"]
        others={s:caches[s].arrays["timestamp"] for s in symbols if s!=reference_symbol}
        for start in range(0,len(ref_ts),chunk_rows):
            stop=min(len(ref_ts),start+chunk_rows)
            chunk=np.asarray(ref_ts[start:stop],dtype=np.int64)
            mask=np.ones(len(chunk),dtype=bool)
            found={reference_symbol:np.arange(start,stop,dtype=np.int64)}
            for s,ts in others.items():
                pos=np.searchsorted(ts,chunk,side="left")
                safe=np.minimum(pos,len(ts)-1)
                ok=(pos<len(ts))&(np.asarray(ts[safe],dtype=np.int64)==chunk)
                mask&=ok
                found[s]=pos.astype(np.int64,copy=False)
            if not np.any(mask):continue
            aligned_ts=chunk[mask]
            th.write(np.ascontiguousarray(aligned_ts).tobytes())
            for s in symbols:
                vals=found[s][mask]
                handles[s].write(np.ascontiguousarray(vals,dtype=np.int64).tobytes())
            count+=len(aligned_ts)
        th.flush();os.fsync(th.fileno())
        for h in handles.values():h.flush();os.fsync(h.fileno())
    finally:
        th.close()
        for h in handles.values():h.close()
    if count==0:raise RuntimeError("NO_ALIGNED_MULTI_ASSET_TIMESTAMPS")

    members=tuple(
        MultiAssetCacheMemberR7(
            symbol=s,
            cache_root=str(Path(cache_roots[s]).resolve()),
            cache_identity_hash=caches[s].receipt.scientific_identity_hash,
            rows=caches[s].receipt.rows,
            encoder_weight_hash=caches[s].receipt.encoder_weight_hash,
        )
        for s in symbols
    )
    index_files={s:index_paths[s].name for s in symbols}
    index_sha={s:sha256_file(index_paths[s]) for s in symbols}
    timestamp_sha=sha256_file(tpath)
    identity=canonical_hash({
        "reference_symbol":reference_symbol,
        "symbols":symbols,
        "aligned_rows":count,
        "members":[asdict(x) for x in members],
        "index_sha256":index_sha,
        "timestamp_sha256":timestamp_sha,
    })
    receipt=MultiAssetSynchronizationReceiptR7(
        set_version="CB16_MULTI_ASSET_MARKET_CACHE_SET_R7",
        reference_symbol=reference_symbol,
        symbols=symbols,
        aligned_rows=count,
        members=members,
        index_files=index_files,
        index_sha256=index_sha,
        timestamp_sha256=timestamp_sha,
        identity_hash=identity,
    )
    atomic_json(out/"MULTIASSET_RECEIPT.json",asdict(receipt))
    return receipt
