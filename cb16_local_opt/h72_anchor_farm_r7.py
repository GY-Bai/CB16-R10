from __future__ import annotations

"""
Spawn-safe H72 dependence-group farm.

Large anchor state/context arrays are written once to mmap `.npy` files. Worker jobs carry
only small identities (group index + paths + config), so a 3700X process farm does not
pickle/copy thousands of AccountState vectors for every job.

Typical topology:
    sequential Champion rollout
        -> H72AnchorStoreWriterR7 (SSD mmap)
        -> spawn workers
        -> VectorizedGroupH72CompilerR7
        -> one result bundle per dependence group
        -> Teacher consumes JSONL samples.

Workers are CPU-only. They must never initialize CUDA.
"""

import concurrent.futures as cf
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .h72_group_compiler_r7 import (
    GroupAnchorBatchR7,
    VectorizedGroupH72CompilerR7,
)
from .market_cache_r6 import MarketLatentCacheR6
from .trajectory_compiler_r6 import ActionCandidateR6, EconomicClockR6
from .vectorized_physics import VectorPhysicsConfig


STATE_FIELDS = (
    ("balance", np.float64),
    ("position_qty", np.float64),
    ("entry_price", np.float64),
    ("peak_equity", np.float64),
    ("realized_pnl", np.float64),
    ("margin_used", np.float64),
    ("holding_bars", np.int64),
    ("risk_budget_remaining", np.float64),
    ("risk_budget_capacity", np.float64),
    ("terminated", np.bool_),
    ("last_mark_price", np.float64),
)


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj=asdict(obj)
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
class H72AnchorStoreReceiptR7:
    store_version:str
    total_groups:int
    accounts_per_group:int
    feature_dim:int
    completed_groups:int
    store_identity_hash:str

    @property
    def content_hash(self):return canonical_hash(self)


class H72AnchorStoreWriterR7:
    def __init__(
        self,
        root:str|Path,
        *,
        total_groups:int,
        accounts_per_group:int,
        feature_dim:int,
        overwrite:bool=False,
    ):
        self.root=Path(root)
        if self.root.exists() and any(self.root.iterdir()) and not overwrite:
            raise RuntimeError("ANCHOR_STORE_EXISTS")
        self.root.mkdir(parents=True,exist_ok=True)
        (self.root/"meta").mkdir(exist_ok=True)
        if min(total_groups,accounts_per_group,feature_dim)<=0:
            raise ValueError("bad anchor store dimensions")
        self.total_groups=total_groups
        self.accounts=accounts_per_group
        self.feature_dim=feature_dim
        self.decision_index=np.lib.format.open_memmap(
            self.root/"decision_index.npy",mode="w+",dtype=np.int64,
            shape=(total_groups,)
        )
        self.written=np.lib.format.open_memmap(
            self.root/"written.npy",mode="w+",dtype=bool,shape=(total_groups,)
        )
        self.written[:]=False
        self.context=np.lib.format.open_memmap(
            self.root/"context_features.npy",mode="w+",dtype=np.float32,
            shape=(total_groups,accounts_per_group,feature_dim)
        )
        self.fields={}
        for name,dtype in STATE_FIELDS:
            self.fields[name]=np.lib.format.open_memmap(
                self.root/f"{name}.npy",mode="w+",dtype=dtype,
                shape=(total_groups,accounts_per_group)
            )
        atomic_json(self.root/"STORE_CONFIG.json",{
            "store_version":"CB16_H72_ANCHOR_STORE_R7",
            "total_groups":total_groups,
            "accounts_per_group":accounts_per_group,
            "feature_dim":feature_dim,
        })

    def write_group(self,index:int,anchor:GroupAnchorBatchR7):
        anchor.validate()
        if not 0<=index<self.total_groups:raise IndexError(index)
        if anchor.accounts!=self.accounts:
            raise RuntimeError("ANCHOR_STORE_ACCOUNT_COUNT_MISMATCH")
        if anchor.context_features.shape!=(self.accounts,self.feature_dim):
            raise RuntimeError("ANCHOR_STORE_CONTEXT_SHAPE_MISMATCH")
        if bool(self.written[index]):
            # Compare immutable identity before treating as idempotent.
            old=json.loads((self.root/"meta"/f"{index:08d}.json").read_text())
            if old["anchor_identity_hash"]!=canonical_hash({
                "dependence_group_id":anchor.dependence_group_id,
                "decision_index":anchor.decision_index,
                "parent_ids":anchor.parent_ids,
                "student_context_object_ids":anchor.student_context_object_ids,
                "market_lineage_hash":anchor.market_lineage_hash,
                "context_sha256":hashlib.sha256(
                    np.ascontiguousarray(anchor.context_features).tobytes()
                ).hexdigest(),
            }):
                raise RuntimeError("ANCHOR_STORE_GROUP_REWRITE_CONFLICT")
            return

        self.decision_index[index]=anchor.decision_index
        self.context[index]=anchor.context_features
        for name,_ in STATE_FIELDS:
            self.fields[name][index]=np.asarray(getattr(anchor,name))
        identity=canonical_hash({
            "dependence_group_id":anchor.dependence_group_id,
            "decision_index":anchor.decision_index,
            "parent_ids":anchor.parent_ids,
            "student_context_object_ids":anchor.student_context_object_ids,
            "market_lineage_hash":anchor.market_lineage_hash,
            "context_sha256":hashlib.sha256(
                np.ascontiguousarray(anchor.context_features).tobytes()
            ).hexdigest(),
        })
        atomic_json(self.root/"meta"/f"{index:08d}.json",{
            "dependence_group_id":anchor.dependence_group_id,
            "parent_ids":list(anchor.parent_ids),
            "student_context_object_ids":list(anchor.student_context_object_ids),
            "market_lineage_hash":anchor.market_lineage_hash,
            "anchor_identity_hash":identity,
        })
        self.written[index]=True

    def flush(self):
        self.decision_index.flush();self.written.flush();self.context.flush()
        for x in self.fields.values():x.flush()

    def seal(self)->H72AnchorStoreReceiptR7:
        self.flush()
        completed=int(np.count_nonzero(self.written))
        if completed!=self.total_groups:
            raise RuntimeError(
                f"ANCHOR_STORE_INCOMPLETE:{completed}/{self.total_groups}"
            )
        meta_hashes=[]
        for i in range(self.total_groups):
            p=self.root/"meta"/f"{i:08d}.json"
            if not p.is_file():raise RuntimeError("ANCHOR_STORE_META_MISSING")
            meta_hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
        identity=canonical_hash({
            "config":json.loads((self.root/"STORE_CONFIG.json").read_text()),
            "meta_hashes":meta_hashes,
            "decision_index_sha256":hashlib.sha256(
                np.ascontiguousarray(np.asarray(self.decision_index)).tobytes()
            ).hexdigest(),
        })
        receipt=H72AnchorStoreReceiptR7(
            store_version="CB16_H72_ANCHOR_STORE_R7",
            total_groups=self.total_groups,
            accounts_per_group=self.accounts,
            feature_dim=self.feature_dim,
            completed_groups=completed,
            store_identity_hash=identity,
        )
        atomic_json(self.root/"STORE_RECEIPT.json",asdict(receipt))
        return receipt


class H72AnchorStoreReaderR7:
    def __init__(self,root:str|Path):
        self.root=Path(root)
        cfg=json.loads((self.root/"STORE_CONFIG.json").read_text())
        self.total_groups=int(cfg["total_groups"])
        self.accounts=int(cfg["accounts_per_group"])
        self.feature_dim=int(cfg["feature_dim"])
        self.decision_index=np.load(
            self.root/"decision_index.npy",mmap_mode="r",allow_pickle=False
        )
        self.written=np.load(
            self.root/"written.npy",mmap_mode="r",allow_pickle=False
        )
        self.context=np.load(
            self.root/"context_features.npy",mmap_mode="r",allow_pickle=False
        )
        self.fields={
            name:np.load(
                self.root/f"{name}.npy",mmap_mode="r",allow_pickle=False
            )
            for name,_ in STATE_FIELDS
        }

    def group(self,index:int)->GroupAnchorBatchR7:
        if not bool(self.written[index]):
            raise RuntimeError("ANCHOR_GROUP_NOT_WRITTEN")
        meta=json.loads((self.root/"meta"/f"{index:08d}.json").read_text())
        kw={
            name:np.array(self.fields[name][index],copy=True)
            for name,_ in STATE_FIELDS
        }
        return GroupAnchorBatchR7(
            dependence_group_id=meta["dependence_group_id"],
            decision_index=int(self.decision_index[index]),
            parent_ids=tuple(meta["parent_ids"]),
            student_context_object_ids=tuple(meta["student_context_object_ids"]),
            context_features=np.array(self.context[index],copy=True),
            market_lineage_hash=meta["market_lineage_hash"],
            **kw,
        )


@dataclass(frozen=True)
class H72FarmConfigR7:
    workers:int=6
    start_method:str="spawn"
    cpu_threads_per_worker:int=1
    max_in_flight:int=12

    def validate(self):
        if self.workers<=0 or self.cpu_threads_per_worker<=0:
            raise ValueError("worker/thread count")
        if self.start_method not in {"spawn","forkserver"}:
            raise ValueError("plain fork unsupported")
        if self.max_in_flight<self.workers:
            raise ValueError("max_in_flight must be >= workers")


@dataclass(frozen=True)
class H72FarmJobR7:
    group_index:int
    anchor_store_root:str
    market_cache_root:str
    result_root:str
    physics_config:Mapping[str,Any]
    candidates:tuple[tuple[int,float],...]
    horizon_bars:int=72

    @property
    def identity_hash(self):return canonical_hash(self)


@dataclass(frozen=True)
class H72FarmResultR7:
    group_index:int
    dependence_group_id:str
    job_identity_hash:str
    group_receipt_hash:str
    teacher_sample_count:int
    samples_path:str
    matrix_path:str
    status:str


def _worker_init(threads:int):
    os.environ["OMP_NUM_THREADS"]=str(threads)
    os.environ["MKL_NUM_THREADS"]=str(threads)
    os.environ["OPENBLAS_NUM_THREADS"]=str(threads)


def _run_job(job:H72FarmJobR7)->H72FarmResultR7:
    store=H72AnchorStoreReaderR7(job.anchor_store_root)
    anchor=store.group(job.group_index)
    cache=MarketLatentCacheR6(job.market_cache_root)
    cfg=VectorPhysicsConfig(**dict(job.physics_config))
    candidates=tuple(ActionCandidateR6(d,float(r)) for d,r in job.candidates)
    compiled=VectorizedGroupH72CompilerR7(
        cfg,clock=EconomicClockR6(horizon_bars=job.horizon_bars)
    ).compile(cache.market_path(),anchor,candidates)
    samples=compiled.to_teacher_samples()

    out=Path(job.result_root)
    out.mkdir(parents=True,exist_ok=True)
    stem=f"group_{job.group_index:08d}_{anchor.dependence_group_id.replace(':','_')}"
    sample_path=out/f"{stem}.samples.jsonl"
    matrix_path=out/f"{stem}.matrix.npz"
    receipt_path=out/f"{stem}.receipt.json"

    # Idempotent result publication by job identity.
    if receipt_path.exists():
        old=json.loads(receipt_path.read_text())
        if old["job_identity_hash"]!=job.identity_hash:
            raise RuntimeError("H72_FARM_RESULT_IDENTITY_CONFLICT")
        return H72FarmResultR7(**old["result"])

    tmp=sample_path.with_name(sample_path.name+f".{os.getpid()}.partial")
    with tmp.open("w",encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s),sort_keys=True,separators=(",",":"))+"\n")
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,sample_path)

    np.savez_compressed(
        matrix_path,
        truth_log_utility=compiled.truth_log_utility,
        terminal_equity=compiled.terminal_equity,
        total_fee=compiled.total_fee,
        total_turnover=compiled.total_turnover,
        total_funding_cashflow=compiled.total_funding_cashflow,
        liquidated=compiled.liquidated,
        stopped=compiled.stopped,
        took_profit=compiled.took_profit,
        forced_horizon_exit=compiled.forced_horizon_exit,
    )
    result=H72FarmResultR7(
        group_index=job.group_index,
        dependence_group_id=anchor.dependence_group_id,
        job_identity_hash=job.identity_hash,
        group_receipt_hash=compiled.receipt.content_hash,
        teacher_sample_count=len(samples),
        samples_path=str(sample_path),
        matrix_path=str(matrix_path),
        status="PASS",
    )
    atomic_json(receipt_path,{
        "schema":"CB16_H72_FARM_RESULT_R7",
        "job_identity_hash":job.identity_hash,
        "compiled_receipt":asdict(compiled.receipt),
        "result":asdict(result),
    })
    return result


class H72DependenceGroupFarmR7:
    def __init__(self,config:H72FarmConfigR7):
        config.validate()
        self.config=config

    def run(self,jobs:Sequence[H72FarmJobR7])->list[H72FarmResultR7]:
        if not jobs:return []
        if len({j.group_index for j in jobs})!=len(jobs):
            raise ValueError("duplicate group index")
        ctx=mp.get_context(self.config.start_method)
        results={}
        pending={}
        next_i=0
        with cf.ProcessPoolExecutor(
            max_workers=self.config.workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(self.config.cpu_threads_per_worker,),
        ) as ex:
            while next_i<len(jobs) or pending:
                while (
                    next_i<len(jobs)
                    and len(pending)<self.config.max_in_flight
                ):
                    fut=ex.submit(_run_job,jobs[next_i])
                    pending[fut]=next_i
                    next_i+=1
                done,_=cf.wait(pending,return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    idx=pending.pop(fut)
                    results[idx]=fut.result()
        return [results[i] for i in range(len(jobs))]


def concatenate_teacher_samples_r7(
    results:Sequence[H72FarmResultR7],
    output_path:str|Path,
)->Path:
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+f".{os.getpid()}.partial")
    with tmp.open("wb") as dst:
        for r in sorted(results,key=lambda x:x.group_index):
            with Path(r.samples_path).open("rb") as src:
                while True:
                    b=src.read(1<<20)
                    if not b:break
                    dst.write(b)
        dst.flush();os.fsync(dst.fileno())
    os.replace(tmp,out)
    return out
