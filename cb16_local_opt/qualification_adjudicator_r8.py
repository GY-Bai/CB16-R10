from __future__ import annotations

"""Adjudicate local hardware/performance receipts into a versioned runtime profile.

The adjudicator chooses only engineering knobs.  Scientific thresholds and semantics are
intentionally absent from the output schema.
"""

import hashlib, json, dataclasses
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj): obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def _load(path:str|Path)->dict[str,Any]: return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class QualificationPolicyR8:
    allow_burnin_warnings: bool = True
    min_ssd_write_mb_s: float = 50.0
    min_ssd_read_mb_s: float = 100.0
    min_hdd_write_mb_s: float = 20.0
    minimum_h72_workers: int = 2
    maximum_h72_workers: int = 6
    ram_backpressure_high: float = 0.85
    ram_hard_stop: float = 0.92
    vram_backpressure_high: float = 0.88
    disk_free_hard_stop_gib: float = 10.0

    @property
    def content_hash(self)->str:return canonical_hash(self)


@dataclass(frozen=True)
class LocalRuntimeProfileR8:
    profile_version: str
    status: str
    source_receipt_hashes: Mapping[str,str]
    gpu: Mapping[str,Any]
    cpu: Mapping[str,Any]
    io: Mapping[str,Any]
    pipeline: Mapping[str,Any]
    resource_limits: Mapping[str,Any]
    storage_placement: Mapping[str,str]
    scientific_semantics_changed: bool
    policy_hash: str

    @property
    def content_hash(self)->str:return canonical_hash(self)


@dataclass(frozen=True)
class LocalQualificationVerdictR8:
    status: str
    hard_failures: tuple[str,...]
    warnings: tuple[str,...]
    runtime_profile: LocalRuntimeProfileR8 | None


def adjudicate_local_r8(*,bringup_path:str|Path,performance_path:str|Path,burnin_path:str|Path,policy:QualificationPolicyR8|None=None)->LocalQualificationVerdictR8:
    policy=policy or QualificationPolicyR8(); b=_load(bringup_path); p=_load(performance_path); u=_load(burnin_path)
    hard=[]; warn=[]
    if b.get('status')=='FAIL': hard.extend('BRINGUP:'+x for x in b.get('hard_failures',[]))
    if b.get('authority')!='SHANXI_HARDWARE_AUTHORITATIVE': hard.append('BRINGUP_NOT_AUTHORITATIVE')
    if p.get('authority')!='GTX1060_AUTHORITATIVE_TUNING': hard.append('GPU_TUNING_NOT_AUTHORITATIVE')
    if p.get('status')=='FAIL': hard.append('PERFORMANCE_QUALIFICATION_FAIL')
    if u.get('status')=='FAIL': hard.extend('BURNIN:'+x for x in u.get('failures',[]))
    if u.get('status')=='PASS_WITH_WARNINGS' and not policy.allow_burnin_warnings: hard.append('BURNIN_WARNINGS_NOT_ALLOWED')
    warn.extend(b.get('warnings',[]));warn.extend(u.get('warnings',[]));warn.extend(p.get('errors',[]))

    disk={x['label']:x for x in p.get('disk',[])}
    if disk.get('SSD',{}).get('write_mb_s',0)<policy.min_ssd_write_mb_s: warn.append('SSD_WRITE_THROUGHPUT_LOW')
    if disk.get('SSD',{}).get('read_mb_s',0)<policy.min_ssd_read_mb_s: warn.append('SSD_READ_THROUGHPUT_LOW')
    if disk.get('HDD',{}).get('write_mb_s',0)<policy.min_hdd_write_mb_s: warn.append('HDD_WRITE_THROUGHPUT_LOW')
    storage=b.get('storage',{}); ssrc=storage.get('ssd',{}).get('source'); hsrc=storage.get('hdd',{}).get('source')
    if ssrc and hsrc and ssrc==hsrc: warn.append('SSD_AND_HDD_ARGUMENTS_RESOLVE_TO_SAME_DEVICE')

    if hard:return LocalQualificationVerdictR8('NOT_READY',tuple(hard),tuple(sorted(set(warn))),None)

    auto=(p.get('gtx1060_autotune') or {}); choice=auto.get('choice',{})
    cores=int(b.get('hardware',{}).get('cpu',{}).get('physical_cores',8)); ws=p.get('h72_worker_scaling') or {}; measured=ws.get('selected_workers'); h72_workers=int(measured) if measured else min(policy.maximum_h72_workers,max(policy.minimum_h72_workers,cores-2)); h72_workers=max(policy.minimum_h72_workers,min(policy.maximum_h72_workers,h72_workers))
    # 16 GiB should not run max H72 and Teacher farms simultaneously. Reserve two cores.
    teacher_workers=max(1,min(2,cores//4)); h72_inflight=max(h72_workers,2*h72_workers)
    rollout_chunk=int(choice.get('rollout_batch') or p.get('best_rollout_chunk') or 8192)
    train_batch=int(choice.get('train_batch') or 1024); cpu_threads=int(choice.get('cpu_torch_threads') or 1)
    profile=LocalRuntimeProfileR8(
        profile_version='CB16_SHANXI_RUNTIME_PROFILE_R8_1',status='READY_FOR_SHORT_REAL_CAMPAIGN' if not warn else 'READY_WITH_LIMITS',
        source_receipt_hashes={
            'bringup':b.get('content_hash') or canonical_hash(b),
            'performance':p.get('content_hash') or canonical_hash(p),
            'burnin':u.get('content_hash') or canonical_hash(u),
        },
        gpu={'dtype':'fp32','amp_enabled':False,'tier':choice.get('tier','TIER_1'),'rollout_chunk_rows':rollout_chunk,'train_batch':train_batch,'cpu_torch_threads':cpu_threads,'single_cuda_owner':True},
        cpu={'h72_workers':h72_workers,'h72_threads_per_worker':1,'h72_max_in_flight':h72_inflight,'teacher_workers':teacher_workers,'teacher_threads_per_worker':1},
        io={'historical_read_batch_rows':65536,'market_encode_batch_windows':int(min(8192,max(1024,rollout_chunk))),'experience_shards':4,'sqlite_synchronous':'FULL','parquet_readahead_batches':2},
        pipeline={'loader_prefetch':2,'max_stage_concurrency':2,'do_not_run_h72_and_teacher_at_full_concurrency':True,'maintenance_every_generations':10,'wal_checkpoint_every_generations':5,'retention_scan_every_generations':10},
        resource_limits={'ram_backpressure_high':policy.ram_backpressure_high,'ram_hard_stop':policy.ram_hard_stop,'vram_backpressure_high':policy.vram_backpressure_high,'disk_free_hard_stop_gib':policy.disk_free_hard_stop_gib,'swap_is_emergency_only':True},
        storage_placement={'active_dataset':'SSD','market64_cache':'SSD','experience_metadata':'SSD','active_checkpoints':'SSD','cold_trajectory_matrices':'HDD','old_checkpoints':'HDD','archives':'HDD'},
        scientific_semantics_changed=False,policy_hash=policy.content_hash,
    )
    return LocalQualificationVerdictR8(profile.status,(),tuple(sorted(set(warn))),profile)


def write_qualification_r8(verdict:LocalQualificationVerdictR8,path:str|Path)->Path:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=asdict(verdict)
    if verdict.runtime_profile is not None:payload['runtime_profile']['content_hash']=verdict.runtime_profile.content_hash
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n');return path
