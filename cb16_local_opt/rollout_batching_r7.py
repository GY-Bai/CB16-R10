from __future__ import annotations

"""
Chunked cached-latent Trader rollout batching.

Market Encoder work is already frozen in the Market64 cache. At each timestamp R7 sends:

    M unique Market64 vectors
    N AccountState6 vectors
    N account->market indices

to the Trader in bounded account chunks.

`index_select` gathers market context on-device, so one shared Market64 does not need to be
materialized N times in host RAM. This is especially useful for thousands of account
replicas on a GTX1060 6GB.
"""

import dataclasses
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(
        json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RolloutBatchingConfigR7:
    device:str="cuda"
    account_chunk_rows:int=8192
    pin_host_memory:bool=True
    non_blocking:bool=True

    def validate(self):
        if self.device not in {"cpu","cuda"}:raise ValueError("device")
        if self.account_chunk_rows<=0:raise ValueError("chunk rows")


@dataclass(frozen=True)
class RolloutBatchReceiptR7:
    accounts:int
    unique_markets:int
    chunks:int
    device:str
    elapsed_ms:float
    accounts_per_second:float
    direction_sha256:str
    risk_sha256:str
    batching_config_hash:str

    @property
    def content_hash(self):return canonical_hash(self)


class CachedLatentRolloutBatcherR7:
    def __init__(self,model,config:RolloutBatchingConfigR7|None=None):
        self.model=model
        self.config=config or RolloutBatchingConfigR7()
        self.config.validate()
        self.device=torch.device(
            self.config.device
            if self.config.device=="cpu" or torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device).eval()


    def infer_policy_heads(
        self,
        *,
        unique_market_latent: np.ndarray,
        account_state6: np.ndarray,
        account_to_market: np.ndarray,
    ):
        """Return raw policy heads needed by exploration.

        Returns `(direction_logits, direction_probs, requested_risk_raw, receipt)`.
        Unlike `infer()`, requested risk is not zeroed merely because the deterministic
        argmax direction happened to be FLAT.
        """
        markets=np.asarray(unique_market_latent,dtype=np.float32)
        accounts=np.asarray(account_state6,dtype=np.float32)
        mapping=np.asarray(account_to_market,dtype=np.int64)
        if markets.ndim!=2 or markets.shape[1]!=64:
            raise ValueError("unique market latent shape")
        if accounts.ndim!=2 or accounts.shape[1]!=6:
            raise ValueError("account state shape")
        if mapping.shape!=(len(accounts),):
            raise ValueError("account_to_market shape")
        if np.any((mapping<0)|(mapping>=len(markets))):
            raise ValueError("market mapping out of range")

        mt=torch.from_numpy(np.array(markets,copy=True,order="C"))
        if self.device.type=="cuda" and self.config.pin_host_memory:
            mt=mt.pin_memory()
        mt=mt.to(
            self.device,
            non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
        )

        n=len(accounts)
        logits=np.empty((n,3),dtype=np.float32)
        probs=np.empty((n,3),dtype=np.float32)
        risk_raw=np.empty(n,dtype=np.float32)
        chunks=0
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        t0=time.perf_counter()
        with torch.inference_mode():
            for start in range(0,n,self.config.account_chunk_rows):
                stop=min(n,start+self.config.account_chunk_rows)
                a_np=np.array(accounts[start:stop],copy=True,order="C")
                i_np=np.array(mapping[start:stop],copy=True,order="C")
                at=torch.from_numpy(a_np)
                it=torch.from_numpy(i_np)
                if self.device.type=="cuda" and self.config.pin_host_memory:
                    at=at.pin_memory();it=it.pin_memory()
                at=at.to(
                    self.device,
                    non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
                )
                it=it.to(
                    self.device,
                    non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
                )
                out=self.model(mt.index_select(0,it),at)
                logits[start:stop]=(
                    out["direction_logits"].detach().cpu().numpy().astype(np.float32,copy=False)
                )
                probs[start:stop]=(
                    out["direction_probs"].detach().cpu().numpy().astype(np.float32,copy=False)
                )
                risk_raw[start:stop]=(
                    out["requested_risk_raw"].detach().cpu().numpy().astype(np.float32,copy=False)
                )
                chunks+=1
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        elapsed=(time.perf_counter()-t0)*1000.0
        deterministic_direction=probs.argmax(axis=1).astype(np.int8)-1
        deterministic_risk=np.where(
            deterministic_direction==0,0.0,np.clip(risk_raw,0,1)
        ).astype(np.float32)
        receipt=RolloutBatchReceiptR7(
            accounts=n,
            unique_markets=len(markets),
            chunks=chunks,
            device=str(self.device),
            elapsed_ms=elapsed,
            accounts_per_second=(
                0.0 if elapsed<=0 else n/(elapsed/1000.0)
            ),
            direction_sha256=hashlib.sha256(
                np.ascontiguousarray(deterministic_direction).tobytes()
            ).hexdigest(),
            risk_sha256=hashlib.sha256(
                np.ascontiguousarray(deterministic_risk).tobytes()
            ).hexdigest(),
            batching_config_hash=canonical_hash(self.config),
        )
        return logits,probs,risk_raw,receipt

    def infer(
        self,
        *,
        unique_market_latent:np.ndarray, # [M,64]
        account_state6:np.ndarray,       # [N,6]
        account_to_market:np.ndarray,    # [N]
    ):
        markets=np.asarray(unique_market_latent,dtype=np.float32)
        accounts=np.asarray(account_state6,dtype=np.float32)
        mapping=np.asarray(account_to_market,dtype=np.int64)
        if markets.ndim!=2 or markets.shape[1]!=64:
            raise ValueError("unique market latent shape")
        if accounts.ndim!=2 or accounts.shape[1]!=6:
            raise ValueError("account state shape")
        if mapping.shape!=(len(accounts),):
            raise ValueError("account_to_market shape")
        if np.any((mapping<0)|(mapping>=len(markets))):
            raise ValueError("market mapping out of range")

        # Unique market vectors are tiny: keep one tensor resident for all account chunks.
        mt=torch.from_numpy(np.array(markets,copy=True,order="C"))
        if self.device.type=="cuda" and self.config.pin_host_memory:
            mt=mt.pin_memory()
        mt=mt.to(
            self.device,
            non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
        )

        n=len(accounts)
        directions=np.empty(n,dtype=np.int8)
        risks=np.empty(n,dtype=np.float32)
        probs=np.empty((n,3),dtype=np.float32)
        chunks=0
        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        t0=time.perf_counter()

        with torch.inference_mode():
            for start in range(0,n,self.config.account_chunk_rows):
                stop=min(n,start+self.config.account_chunk_rows)
                a_np=np.array(accounts[start:stop],copy=True,order="C")
                i_np=np.array(mapping[start:stop],copy=True,order="C")
                at=torch.from_numpy(a_np)
                it=torch.from_numpy(i_np)
                if self.device.type=="cuda" and self.config.pin_host_memory:
                    at=at.pin_memory();it=it.pin_memory()
                at=at.to(
                    self.device,
                    non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
                )
                it=it.to(
                    self.device,
                    non_blocking=(self.config.non_blocking and self.device.type=="cuda"),
                )
                market_accounts=mt.index_select(0,it)
                out=self.model(market_accounts,at)
                action=self.model.compose_action(out)
                directions[start:stop]=(
                    action["direction"].detach().cpu().numpy().astype(np.int8,copy=False)
                )
                risks[start:stop]=(
                    action["requested_risk"].detach().cpu().numpy().astype(np.float32,copy=False)
                )
                probs[start:stop]=(
                    out["direction_probs"].detach().cpu().numpy().astype(np.float32,copy=False)
                )
                chunks+=1

        if self.device.type=="cuda":
            torch.cuda.synchronize(self.device)
        elapsed=(time.perf_counter()-t0)*1000.0
        receipt=RolloutBatchReceiptR7(
            accounts=n,
            unique_markets=len(markets),
            chunks=chunks,
            device=str(self.device),
            elapsed_ms=elapsed,
            accounts_per_second=(
                0.0 if elapsed<=0 else n/(elapsed/1000.0)
            ),
            direction_sha256=hashlib.sha256(
                np.ascontiguousarray(directions).tobytes()
            ).hexdigest(),
            risk_sha256=hashlib.sha256(
                np.ascontiguousarray(risks).tobytes()
            ).hexdigest(),
            batching_config_hash=canonical_hash(self.config),
        )
        return directions,risks,probs,receipt
