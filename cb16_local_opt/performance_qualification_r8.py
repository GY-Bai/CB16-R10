from __future__ import annotations

"""Performance qualification for the local R8 data-factory node.

All knobs selected here are engineering/performance knobs.  No benchmark is allowed to
change dataset split, H72 clock, Physics, Teacher law, promotion rule or holdout semantics.
"""

import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .gtx1060_autotuner import AutotunePolicy, GTX1060Autotuner
from .h72_group_compiler_r7 import VectorizedGroupH72CompilerR7
from .h72_worker_scaling_r8 import benchmark_h72_worker_scaling_r8
from .rollout_batching_r7 import CachedLatentRolloutBatcherR7, RolloutBatchingConfigR7
from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake
from .trader_capacity_ladder import build_trader
from .trajectory_compiler_r6 import DecisionAnchorR6, InitialAccountSnapshotR6, MarketPathR6, default_action_grid_r6
from .vectorized_physics import AccountBatchState, MarketBar, VectorizedPhysics, VectorPhysicsConfig


def canonical_hash(obj: Any) -> str:
    if dataclasses.is_dataclass(obj):
        obj = asdict(obj)
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DiskBenchmarkR8:
    label: str
    path: str
    bytes_tested: int
    write_mb_s: float
    read_mb_s: float
    fsync_ms: float
    status: str
    error: str | None = None


@dataclass(frozen=True)
class PhysicsBenchmarkR8:
    accounts: int
    bars: int
    elapsed_s: float
    account_bar_steps_per_s: float
    status: str


@dataclass(frozen=True)
class H72BenchmarkR8:
    accounts: int
    candidates: int
    horizon_bars: int
    elapsed_s: float
    branch_bars_per_s: float
    groups_per_s: float
    status: str


@dataclass(frozen=True)
class ExperienceBenchmarkR8:
    objects: int
    payload_bytes_each: int
    shards: int
    elapsed_s: float
    objects_per_s: float
    raw_mb_s: float
    snapshot_objects: int
    audit_pass: bool
    status: str


@dataclass(frozen=True)
class RolloutBenchmarkR8:
    tier: str
    accounts: int
    unique_markets: int
    chunk_rows: int
    device: str
    elapsed_ms: float
    accounts_per_s: float
    status: str


@dataclass(frozen=True)
class PerformancePolicyR8:
    disk_test_mib: int = 512
    physics_account_candidates: tuple[int, ...] = (4096, 16384, 65536)
    physics_bars: int = 64
    h72_account_candidates: tuple[int, ...] = (64, 256, 1024)
    h72_risk_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    rollout_account_candidates: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)
    experience_objects: int = 2000
    experience_payload_bytes: int = 2048
    experience_shards: int = 4
    gpu_autotune_tiers: tuple[str, ...] = ("TIER_1", "TIER_2")
    gpu_rollout_batches: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768)
    gpu_train_batches: tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192)
    gpu_cpu_threads: tuple[int, ...] = (1, 2)
    warmup: int = 2
    repeats: int = 5

    @property
    def content_hash(self) -> str:
        return canonical_hash(self)


@dataclass(frozen=True)
class PerformanceQualificationReceiptR8:
    schema: str
    authority: str
    status: str
    disk: tuple[DiskBenchmarkR8, ...]
    physics: tuple[PhysicsBenchmarkR8, ...]
    h72: tuple[H72BenchmarkR8, ...]
    experience: ExperienceBenchmarkR8
    rollout: tuple[RolloutBenchmarkR8, ...]
    gtx1060_autotune: dict[str, Any] | None
    h72_worker_scaling: dict[str, Any] | None
    best_h72_accounts: int
    best_rollout_chunk: int
    policy_hash: str
    errors: tuple[str, ...]
    created_at_unix: float

    @property
    def content_hash(self) -> str:
        d = asdict(self); d.pop("created_at_unix", None)
        return canonical_hash(d)


def benchmark_disk_r8(label: str, root: str | Path, mib: int) -> DiskBenchmarkR8:
    root = Path(root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        total = int(mib) * 1024 * 1024
        block = bytes((i % 251 for i in range(1024 * 1024)))
        fd, name = tempfile.mkstemp(prefix=".cb16_r8_diskbench_", dir=root)
        path = Path(name)
        try:
            t0 = time.perf_counter()
            written = 0
            with os.fdopen(fd, "wb", buffering=0) as f:
                while written < total:
                    n = min(len(block), total - written)
                    f.write(block[:n]); written += n
                fs0 = time.perf_counter(); os.fsync(f.fileno()); fs_ms = (time.perf_counter()-fs0)*1000
            write_s = time.perf_counter()-t0
            # Best effort cache eviction hint; harmless if unavailable.
            try:
                with path.open("rb") as f:
                    os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except Exception:
                pass
            t1=time.perf_counter(); read=0
            with path.open("rb", buffering=0) as f:
                for b in iter(lambda:f.read(4*1024*1024), b""):
                    read += len(b)
            read_s=time.perf_counter()-t1
            return DiskBenchmarkR8(label, str(root), total, total/1024**2/max(write_s,1e-9), read/1024**2/max(read_s,1e-9), fs_ms, "PASS")
        finally:
            path.unlink(missing_ok=True)
    except Exception as exc:
        return DiskBenchmarkR8(label, str(root), int(mib)*1024*1024, 0.0, 0.0, 0.0, "FAIL", repr(exc))


def _physics_config() -> VectorPhysicsConfig:
    return VectorPhysicsConfig(
        initial_equity=10_000.0, max_gross_leverage=2.0,
        fee_rate=0.0002, slippage_bps=0.0,
        maintenance_margin_rate=0.10, max_holding_bars=200,
    )


def benchmark_physics_r8(accounts: int, bars: int) -> PhysicsBenchmarkR8:
    cfg=_physics_config(); engine=VectorizedPhysics(cfg); state=AccountBatchState.empty(accounts,cfg)
    d=np.where(np.arange(accounts)%3==0,-1,np.where(np.arange(accounts)%3==1,0,1)).astype(np.int8)
    r=np.where(d==0,0.0,0.5).astype(np.float64)
    t0=time.perf_counter()
    for i in range(bars):
        base=100.0+0.01*i
        engine.step(state, MarketBar(base,base*1.002,base*0.998,base*1.0005), executable_direction=d, executable_risk=r, requested_direction=d, dependence_group_count=1)
    elapsed=time.perf_counter()-t0
    return PhysicsBenchmarkR8(accounts,bars,elapsed,accounts*bars/max(elapsed,1e-12),"PASS")


def _market_fixture(n: int = 96) -> MarketPathR6:
    t=np.arange(n); close=100*np.exp(0.0005*t+0.002*np.sin(t/7)); op=np.r_[close[0],close[:-1]]
    return MarketPathR6(
        timestamp=(1_700_000_000+t*3600).astype(np.int64),
        open=op, high=np.maximum(op,close)*1.002, low=np.minimum(op,close)*0.998,
        close=close, volume=(1000+t).astype(np.float64), funding_rate=np.zeros(n),
    )


def benchmark_h72_r8(accounts: int, risk_levels: Sequence[float]) -> H72BenchmarkR8:
    cfg=_physics_config(); market=_market_fixture(100); compiler=VectorizedGroupH72CompilerR7(cfg)
    anchors=[DecisionAnchorR6(
        parent_id=f"P{i}", student_context_object_id=f"CTX{i}", decision_index=10,
        context_features=tuple([0.0]*70), dependence_group_id="DG0", market_lineage_hash="M",
        initial_account=InitialAccountSnapshotR6.flat(10_000.0),
    ) for i in range(accounts)]
    group=compiler.from_decision_anchors("DG0",anchors); candidates=default_action_grid_r6(tuple(risk_levels))
    t0=time.perf_counter(); compiler.compile(market,group,candidates); elapsed=time.perf_counter()-t0
    work=accounts*len(candidates)*72
    return H72BenchmarkR8(accounts,len(candidates),72,elapsed,work/max(elapsed,1e-12),1.0/max(elapsed,1e-12),"PASS")


def benchmark_experience_r8(root: str | Path, objects: int, payload_bytes: int, shards: int) -> ExperienceBenchmarkR8:
    bench=Path(root)/".cb16_r8_expbench"
    import shutil
    shutil.rmtree(bench,ignore_errors=True)
    lake=ShardedExperienceLake(bench,shards=shards); refs=[]; payload="x"*payload_bytes
    try:
        t0=time.perf_counter()
        for i in range(objects):
            ref,_=lake.put(ExperienceObject(
                object_id=f"B{i:08d}", object_type="R8_BENCH",
                generation=i//max(1,objects//4), policy_weight_hash="W", snapshot_hash="S", lineage_hash=f"L{i}",
                payload={"i":i,"blob":payload},
            )); refs.append(ref)
        elapsed=time.perf_counter()-t0
        snap=lake.seal_snapshot(snapshot_id="R8BENCH",parent_generation=0,parent_policy_hash="W",refs=refs)
        audit=lake.audit()
        return ExperienceBenchmarkR8(objects,payload_bytes,shards,elapsed,objects/max(elapsed,1e-12),objects*payload_bytes/1024**2/max(elapsed,1e-12),snap.object_count,bool(audit["pass"]),"PASS" if audit["pass"] else "FAIL")
    finally:
        lake.close(); shutil.rmtree(bench,ignore_errors=True)


def benchmark_rollout_r8(accounts: int, chunk_rows: int, tier: str = "TIER_1", device: str = "cuda") -> RolloutBenchmarkR8:
    model=build_trader(tier)
    rng=np.random.default_rng(123); markets=rng.normal(size=(4,64)).astype(np.float32); a=rng.normal(size=(accounts,6)).astype(np.float32); m=rng.integers(0,4,size=accounts,dtype=np.int64)
    batcher=CachedLatentRolloutBatcherR7(model,RolloutBatchingConfigR7(device=device,account_chunk_rows=chunk_rows,pin_host_memory=True,non_blocking=True))
    _,_,_,rec=batcher.infer(unique_market_latent=markets,account_state6=a,account_to_market=m)
    return RolloutBenchmarkR8(tier,accounts,4,chunk_rows,rec.device,rec.elapsed_ms,rec.accounts_per_second,"PASS")


def run_performance_qualification_r8(*, ssd_root: str | Path, hdd_root: str | Path, policy: PerformancePolicyR8 | None = None, allow_cpu_diagnostic: bool = False) -> PerformanceQualificationReceiptR8:
    policy=policy or PerformancePolicyR8(); errors=[]
    disks=(benchmark_disk_r8("SSD",ssd_root,policy.disk_test_mib),benchmark_disk_r8("HDD",hdd_root,policy.disk_test_mib))
    physics=[]; h72=[]; rollout=[]
    for n in policy.physics_account_candidates:
        try: physics.append(benchmark_physics_r8(int(n),policy.physics_bars))
        except Exception as exc: errors.append(f"PHYSICS:{n}:{exc!r}")
    for n in policy.h72_account_candidates:
        try: h72.append(benchmark_h72_r8(int(n),policy.h72_risk_levels))
        except Exception as exc: errors.append(f"H72:{n}:{exc!r}")
    exp=benchmark_experience_r8(ssd_root,policy.experience_objects,policy.experience_payload_bytes,policy.experience_shards)

    autotune_payload=None; worker_scaling_payload=None; authority="NONAUTHORITATIVE_DIAGNOSTIC"
    try:
        ws=benchmark_h72_worker_scaling_r8(worker_candidates=(1,2,4,6),groups=12,accounts_per_group=64)
        worker_scaling_payload=asdict(ws)
    except Exception as exc:
        errors.append('H72_WORKER_SCALING:'+repr(exc))
    try:
        ap=AutotunePolicy(
            tiers=policy.gpu_autotune_tiers, rollout_batches=policy.gpu_rollout_batches,
            train_batches=policy.gpu_train_batches, cpu_thread_candidates=policy.gpu_cpu_threads,
            warmup=policy.warmup,repeats=policy.repeats,
        )
        auto=GTX1060Autotuner(ap).run(allow_cpu_diagnostic=allow_cpu_diagnostic)
        autotune_payload=asdict(auto); autotune_payload["content_hash"]=auto.content_hash
        authority=auto.authority
        device="cuda" if auto.authority=="GTX1060_AUTHORITATIVE_TUNING" else "cpu"
        for n in policy.rollout_account_candidates:
            try: rollout.append(benchmark_rollout_r8(int(n),int(auto.choice.rollout_batch),tier=auto.choice.tier,device=device))
            except Exception as exc: errors.append(f"ROLLOUT:{n}:{exc!r}")
    except Exception as exc:
        errors.append("GPU_AUTOTUNE:"+repr(exc))

    best_h72=max(h72,key=lambda x:x.branch_bars_per_s).accounts if h72 else 0
    best_rollout=max(rollout,key=lambda x:x.accounts_per_s).chunk_rows if rollout else int(policy.rollout_account_candidates[0])
    hard_disk=any(x.status!="PASS" for x in disks)
    status="FAIL" if hard_disk or exp.status!="PASS" or (not allow_cpu_diagnostic and authority!="GTX1060_AUTHORITATIVE_TUNING") else ("PASS_WITH_ERRORS" if errors else "PASS")
    return PerformanceQualificationReceiptR8(
        schema="CB16_SHANXI_PERFORMANCE_QUALIFICATION_R8", authority=authority, status=status,
        disk=disks, physics=tuple(physics), h72=tuple(h72), experience=exp, rollout=tuple(rollout),
        gtx1060_autotune=autotune_payload, h72_worker_scaling=worker_scaling_payload, best_h72_accounts=best_h72,best_rollout_chunk=best_rollout,
        policy_hash=policy.content_hash, errors=tuple(errors), created_at_unix=time.time(),
    )


def write_performance_receipt_r8(receipt: PerformanceQualificationReceiptR8, path: str | Path) -> Path:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);p=asdict(receipt);p["content_hash"]=receipt.content_hash
    path.write_text(json.dumps(p,indent=2,ensure_ascii=False)+"\n");return path
