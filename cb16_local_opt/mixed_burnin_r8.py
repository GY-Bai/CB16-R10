from __future__ import annotations

"""Mixed CPU/GPU/storage burn-in for local qualification.

The burn-in is intentionally performance-only. It never opens a dataset holdout and never
writes scientific Evidence. It exercises the same classes of resources used by the factory:
Trader FP32 inference/training, vectorized Physics, temporary Experience writes and telemetry.
"""

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_training_policy import TrainingPolicy, build_optimizer, train_step
from .gtx1060_autotuner import _make_training_batch
from .rollout_batching_r7 import CachedLatentRolloutBatcherR7, RolloutBatchingConfigR7
from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake
from .trader_capacity_ladder import build_trader
from .vectorized_physics import AccountBatchState, MarketBar, VectorizedPhysics, VectorPhysicsConfig


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj): obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _ram_used_fraction() -> float:
    vals={}
    p=Path('/proc/meminfo')
    if not p.exists(): return 0.0
    for line in p.read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            try: vals[k]=int(v.strip().split()[0])
            except Exception: pass
    total=vals.get('MemTotal',0); avail=vals.get('MemAvailable',total)
    return 0.0 if total<=0 else 1-avail/total


def _swap_used_bytes() -> int:
    vals={}
    p=Path('/proc/meminfo')
    if p.exists():
        for line in p.read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1)
                try: vals[k]=int(v.strip().split()[0])*1024
                except Exception: pass
    return max(0,vals.get('SwapTotal',0)-vals.get('SwapFree',vals.get('SwapTotal',0)))


def _gpu_sample() -> dict[str,float|None]:
    if not shutil.which('nvidia-smi'):
        return {'temp_c':None,'util_pct':None,'mem_used_mib':None,'power_w':None}
    try:
        p=subprocess.run([
            'nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits'
        ],capture_output=True,text=True,timeout=5,check=False)
        if p.returncode!=0:return {'temp_c':None,'util_pct':None,'mem_used_mib':None,'power_w':None}
        x=[s.strip() for s in p.stdout.strip().split(',')]
        def num(v):
            try:return float(v)
            except Exception:return None
        return {'temp_c':num(x[0]),'util_pct':num(x[1]),'mem_used_mib':num(x[2]),'power_w':num(x[3])}
    except Exception:
        return {'temp_c':None,'util_pct':None,'mem_used_mib':None,'power_w':None}


@dataclass(frozen=True)
class BurnInPolicyR8:
    duration_seconds: int = 1800
    sample_interval_seconds: float = 2.0
    tier: str = 'TIER_1'
    inference_accounts: int = 8192
    inference_chunk_rows: int = 8192
    train_batch: int = 2048
    physics_accounts: int = 32768
    experience_write_every_loops: int = 10
    max_ram_used_fraction: float = 0.90
    warn_ram_used_fraction: float = 0.85
    max_gpu_temp_c: float = 90.0
    warn_gpu_temp_c: float = 82.0
    max_swap_growth_mib: float = 512.0

    @property
    def content_hash(self)->str:return canonical_hash(self)


@dataclass(frozen=True)
class BurnInSampleR8:
    elapsed_s: float
    ram_used_fraction: float
    swap_used_bytes: int
    gpu_temp_c: float | None
    gpu_util_pct: float | None
    gpu_mem_used_mib: float | None
    gpu_power_w: float | None
    loops: int


@dataclass(frozen=True)
class BurnInReceiptR8:
    schema: str
    authority: str
    status: str
    duration_s: float
    loops: int
    inference_rows: int
    training_rows: int
    physics_account_bar_steps: int
    experience_objects: int
    max_ram_used_fraction: float
    max_gpu_temp_c: float | None
    swap_growth_bytes: int
    warnings: tuple[str,...]
    failures: tuple[str,...]
    samples: tuple[BurnInSampleR8,...]
    policy_hash: str
    created_at_unix: float

    @property
    def content_hash(self)->str:
        d=asdict(self);d.pop('created_at_unix',None);return canonical_hash(d)


def run_mixed_burnin_r8(*, work_root: str|Path, policy: BurnInPolicyR8|None=None, allow_cpu_diagnostic: bool=False) -> BurnInReceiptR8:
    policy=policy or BurnInPolicyR8(); work=Path(work_root); work.mkdir(parents=True,exist_ok=True)
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device!='cuda' and not allow_cpu_diagnostic:
        raise RuntimeError('CUDA_REQUIRED_FOR_AUTHORITATIVE_BURNIN')
    authority='SHANXI_MIXED_BURNIN_AUTHORITATIVE' if device=='cuda' else 'NONAUTHORITATIVE_DIAGNOSTIC'

    model=build_trader(policy.tier)
    batcher=CachedLatentRolloutBatcherR7(model,RolloutBatchingConfigR7(device=device,account_chunk_rows=policy.inference_chunk_rows,pin_host_memory=(device=='cuda'),non_blocking=True))
    rng=np.random.default_rng(20260904)
    markets=rng.normal(size=(4,64)).astype(np.float32)
    accounts=rng.normal(size=(policy.inference_accounts,6)).astype(np.float32)
    mapping=rng.integers(0,4,size=policy.inference_accounts,dtype=np.int64)

    train_model=build_trader(policy.tier).to(device)
    tp=TrainingPolicy(device=device,dtype='fp32',amp_enabled=False,cpu_torch_threads=1)
    opt=build_optimizer(train_model,tp); train_batch=_make_training_batch(train_model,policy.train_batch)
    if device=='cuda': train_batch=train_batch.pin_memory()

    pcfg=VectorPhysicsConfig(initial_equity=10_000,max_gross_leverage=2,fee_rate=.0002,slippage_bps=0,maintenance_margin_rate=.1,max_holding_bars=200)
    phys=VectorizedPhysics(pcfg); state=AccountBatchState.empty(policy.physics_accounts,pcfg)
    d=np.where(np.arange(policy.physics_accounts)%3==0,-1,np.where(np.arange(policy.physics_accounts)%3==1,0,1)).astype(np.int8); r=np.where(d==0,0,.5).astype(np.float64)

    lake_root=work/'.r8_burnin_lake'; shutil.rmtree(lake_root,ignore_errors=True); lake=ShardedExperienceLake(lake_root,shards=2)
    swap0=_swap_used_bytes(); t0=time.perf_counter(); next_sample=0.0; samples=[]; failures=[]; warnings=[]
    loops=infer_rows=train_rows=phys_steps=exp_objs=0
    try:
        while True:
            elapsed=time.perf_counter()-t0
            if elapsed>=policy.duration_seconds: break
            batcher.infer(unique_market_latent=markets,account_state6=accounts,account_to_market=mapping); infer_rows+=policy.inference_accounts
            train_step(model=train_model,optimizer=opt,batch=train_batch,policy=tp); train_rows+=policy.train_batch
            base=100+0.001*loops; phys.step(state,MarketBar(base,base*1.002,base*.998,base*1.0005),executable_direction=d,executable_risk=r,requested_direction=d,dependence_group_count=1); phys_steps+=policy.physics_accounts
            if loops%policy.experience_write_every_loops==0:
                lake.put(ExperienceObject(object_id=f'R8B:{loops}',object_type='BURNIN',generation=0,policy_weight_hash='W',snapshot_hash='S',lineage_hash=f'L{loops}',payload={'loop':loops,'blob':'x'*1024})); exp_objs+=1
            loops+=1
            if elapsed>=next_sample:
                g=_gpu_sample(); s=BurnInSampleR8(elapsed,_ram_used_fraction(),_swap_used_bytes(),g['temp_c'],g['util_pct'],g['mem_used_mib'],g['power_w'],loops); samples.append(s); next_sample=elapsed+policy.sample_interval_seconds
                if s.ram_used_fraction>policy.max_ram_used_fraction: failures.append('RAM_HIGH_WATERMARK_EXCEEDED'); break
                if s.gpu_temp_c is not None and s.gpu_temp_c>policy.max_gpu_temp_c: failures.append('GPU_TEMPERATURE_LIMIT_EXCEEDED'); break
        max_ram=max((s.ram_used_fraction for s in samples),default=_ram_used_fraction()); temps=[s.gpu_temp_c for s in samples if s.gpu_temp_c is not None]; max_temp=max(temps) if temps else None
        if device=='cuda' and not temps:failures.append('GPU_THERMAL_TELEMETRY_UNAVAILABLE')
        swap_growth=max(0,_swap_used_bytes()-swap0)
        if max_ram>policy.warn_ram_used_fraction: warnings.append(f'RAM_HIGH:{max_ram:.3f}')
        if max_temp is not None and max_temp>policy.warn_gpu_temp_c: warnings.append(f'GPU_TEMP_HIGH:{max_temp:.1f}C')
        if swap_growth>policy.max_swap_growth_mib*1024**2: failures.append(f'SWAP_GROWTH_TOO_LARGE:{swap_growth}')
        status='FAIL' if failures else ('PASS_WITH_WARNINGS' if warnings else 'PASS')
        return BurnInReceiptR8('CB16_SHANXI_MIXED_BURNIN_R8',authority,status,time.perf_counter()-t0,loops,infer_rows,train_rows,phys_steps,exp_objs,max_ram,max_temp,swap_growth,tuple(warnings),tuple(failures),tuple(samples),policy.content_hash,time.time())
    finally:
        lake.close(); shutil.rmtree(lake_root,ignore_errors=True)


def write_burnin_receipt_r8(receipt:BurnInReceiptR8,path:str|Path)->Path:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=asdict(receipt);p['content_hash']=receipt.content_hash;path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+'\n');return path
