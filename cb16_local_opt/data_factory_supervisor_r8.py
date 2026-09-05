from __future__ import annotations

"""Continuous local data-factory supervisor.

The supervisor is deliberately an operational control plane. It can receive immutable data
bundles, instantiate an approved pipeline, lease/retry jobs, apply resource backpressure and
run maintenance. It cannot automatically open FINAL holdouts.
"""

import dataclasses, hashlib, json, os, socket, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .factory_queue_r8 import FactoryJobSpecR8, FactoryQueueR8, ClaimedJobR8
from .factory_resource_governor_r8 import FactoryResourceGovernorR8, FactoryResourceLimitsR8
from .incoming_bundle_watcher_r8 import IncomingBundleWatcherR8, IncomingBundlePolicyR8, sha256_file
from .dataset_catalog_r8 import DatasetCatalogR8


FORBIDDEN_AUTOMATIC_STAGES={
    'FINAL_HOLDOUT','FINAL_TOURNAMENT','FINAL_CONTROLS','OPEN_FINAL_HOLDOUT','OPEN_FINAL_TOURNAMENT'
}
APPROVED_AUTOMATIC_STAGES={
    'TRANSFER_VERIFY','DATASET_VERIFY','MARKET_CACHE','MULTIASSET_SYNC','HISTORICAL_CAMPAIGN',
    'EXPERIENCE_MAINTENANCE','RETENTION','QUALIFICATION','STRESS','REPORT'
}


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


@dataclass(frozen=True)
class PipelineStageTemplateR8:
    name:str
    stage:str
    resource_class:str
    command:tuple[str,...]
    expected_outputs:tuple[str,...]=()
    priority:int=100
    max_attempts:int=3
    timeout_seconds:float=86400.0

    def validate(self):
        if self.stage in FORBIDDEN_AUTOMATIC_STAGES:raise RuntimeError('AUTOMATIC_FINAL_HOLDOUT_FORBIDDEN:'+self.stage)
        if self.stage not in APPROVED_AUTOMATIC_STAGES:raise RuntimeError('UNAPPROVED_FACTORY_STAGE:'+self.stage)
        if not self.name or not self.command:raise ValueError('stage name/command')


@dataclass(frozen=True)
class FactorySupervisorConfigR8:
    queue_db:str
    work_root:str
    ssd_root:str
    hdd_root:str
    inbox:str
    accepted_inbox:str
    rejected_inbox:str
    pipeline:tuple[PipelineStageTemplateR8,...]
    poll_seconds:float=2.0
    lease_seconds:float=300.0
    heartbeat_seconds:float=30.0
    max_parallel_jobs:int=2
    bundle_settle_seconds:float=30.0
    maintenance_command:tuple[str,...]|None=None
    catalog_db:str|None=None
    auto_catalog_snapshot:bool=True
    catalog_snapshot_dir:str|None=None
    status_jsonl:str|None=None
    maintenance_interval_seconds:float=21600.0

    def validate(self):
        if self.poll_seconds<=0 or self.lease_seconds<=0 or self.heartbeat_seconds<=0:raise ValueError('timing')
        if self.heartbeat_seconds>=self.lease_seconds/2:raise ValueError('heartbeat must be < half lease')
        if self.max_parallel_jobs<=0:raise ValueError('parallelism')
        names=set()
        for s in self.pipeline:
            s.validate()
            if s.name in names:raise ValueError('duplicate pipeline stage name')
            names.add(s.name)
        if self.maintenance_command and any(x.lower().find('final')>=0 and x.lower().find('holdout')>=0 for x in self.maintenance_command):raise RuntimeError('MAINTENANCE_CANNOT_OPEN_FINAL_HOLDOUT')

    @property
    def content_hash(self)->str:return canonical_hash(self)


def _render_token(token:str,ctx:Mapping[str,str])->str:
    try:return token.format_map(ctx)
    except KeyError as exc:raise RuntimeError('FACTORY_TEMPLATE_MISSING_KEY:'+str(exc)) from exc


def _job_id(bundle_sha:str,stage_name:str,stage_hash:str)->str:
    return f'BUNDLE:{bundle_sha[:16]}:{stage_name}:{stage_hash[:12]}'


@dataclass
class _Running:
    claim:ClaimedJobR8
    future:Future
    last_heartbeat:float


class DataFactorySupervisorR8:
    def __init__(self,config:FactorySupervisorConfigR8,*,resource_limits:FactoryResourceLimitsR8|None=None):
        config.validate();self.config=config
        self.queue=FactoryQueueR8(config.queue_db)
        self.governor=FactoryResourceGovernorR8(ssd_root=config.ssd_root,hdd_root=config.hdd_root,limits=resource_limits)
        self.watcher=IncomingBundleWatcherR8(inbox=config.inbox,accepted=config.accepted_inbox,rejected=config.rejected_inbox,policy=IncomingBundlePolicyR8(settle_seconds=config.bundle_settle_seconds))
        self.catalog=DatasetCatalogR8(config.catalog_db) if config.catalog_db else None
        self.owner=f'{socket.gethostname()}:{os.getpid()}';self.pool=ThreadPoolExecutor(max_workers=config.max_parallel_jobs,thread_name_prefix='cb16-factory');self.running:dict[str,_Running]={};self.stop_event=threading.Event();self.last_maintenance=0.0

    def close(self):
        self.stop_event.set();self.pool.shutdown(wait=True,cancel_futures=False);self.queue.close()
        if self.catalog is not None:self.catalog.close()

    def enqueue_bundle(self,receipt,*,catalog_snapshot=None,catalog_snapshot_path=None)->list[str]:
        ctx={
            'bundle_sha256':receipt.bundle_sha256,'bundle_root':receipt.accepted_root,'manifest':receipt.manifest_path,
            'work_root':self.config.work_root,'ssd_root':self.config.ssd_root,'hdd_root':self.config.hdd_root,
            'catalog_snapshot': '' if catalog_snapshot_path is None else str(catalog_snapshot_path),
            'catalog_scientific_hash': '' if catalog_snapshot is None else catalog_snapshot.scientific_dataset_hash,
            'catalog_snapshot_id': '' if catalog_snapshot is None else catalog_snapshot.snapshot_id,
        }
        prev=None;ids=[]
        for tmpl in self.config.pipeline:
            stage_hash=canonical_hash(tmpl);jid=_job_id(receipt.bundle_sha256,tmpl.name,stage_hash);cmd=tuple(_render_token(x,ctx) for x in tmpl.command);outs=tuple(_render_token(x,ctx) for x in tmpl.expected_outputs)
            spec=FactoryJobSpecR8(job_id=jid,stage=tmpl.stage,command=cmd,scientific_identity={'transport_bundle_sha256':receipt.bundle_sha256,'manifest_sha256':receipt.manifest_sha256,'pipeline_stage_hash':stage_hash,'factory_config_hash':self.config.content_hash},resource_class=tmpl.resource_class,priority=tmpl.priority,max_attempts=tmpl.max_attempts,timeout_seconds=tmpl.timeout_seconds,expected_outputs=outs)
            self.queue.enqueue(spec,dependencies=(() if prev is None else (prev,)));prev=jid;ids.append(jid)
        return ids

    def _execute(self,claim:ClaimedJobR8)->dict[str,Any]:
        if claim.spec.stage in FORBIDDEN_AUTOMATIC_STAGES:raise RuntimeError('AUTOMATIC_FINAL_HOLDOUT_FORBIDDEN')
        t0=time.time();p=subprocess.run(list(claim.spec.command),cwd=self.config.work_root,capture_output=True,text=True,timeout=claim.spec.timeout_seconds,check=False)
        if p.returncode!=0:raise RuntimeError(f'COMMAND_FAILED rc={p.returncode}\nSTDOUT:\n{p.stdout[-5000:]}\nSTDERR:\n{p.stderr[-5000:]}')
        outputs={}
        for x in claim.spec.expected_outputs:
            path=Path(x)
            if not path.exists():raise RuntimeError('EXPECTED_OUTPUT_MISSING:'+x)
            outputs[x]={'bytes':path.stat().st_size,'sha256':sha256_file(path) if path.is_file() else None}
        return {'returncode':p.returncode,'elapsed_s':time.time()-t0,'stdout_tail':p.stdout[-4000:],'stderr_tail':p.stderr[-4000:],'outputs':outputs}

    def _running_counts(self)->dict[str,int]:
        out={}
        for r in self.running.values():out[r.claim.spec.resource_class]=out.get(r.claim.spec.resource_class,0)+1
        return out

    def _reap(self):
        now=time.time()
        for jid,r in list(self.running.items()):
            if r.future.done():
                try:self.queue.succeed(r.claim,r.future.result())
                except Exception as exc:self.queue.fail(r.claim,repr(exc),retryable=True)
                self.running.pop(jid,None);continue
            if now-r.last_heartbeat>=self.config.heartbeat_seconds:
                self.queue.heartbeat(r.claim,lease_seconds=self.config.lease_seconds);r.last_heartbeat=now

    def _admissible_classes(self)->list[str]:
        s=self.governor.snapshot(self._running_counts());classes=[]
        for rc in ('GPU','CPU_HEAVY','CPU_IO','TRANSFER','MAINTENANCE'):
            if self.governor.decide(rc,s).allowed:classes.append(rc)
        return classes

    def _schedule(self):
        while len(self.running)<self.config.max_parallel_jobs:
            classes=self._admissible_classes()
            if not classes:return
            claim=self.queue.claim(owner=self.owner,lease_seconds=self.config.lease_seconds,allowed_resource_classes=classes)
            if claim is None:return
            if claim.spec.stage in FORBIDDEN_AUTOMATIC_STAGES:
                self.queue.fail(claim,'AUTOMATIC_FINAL_HOLDOUT_FORBIDDEN',retryable=False);continue
            fut=self.pool.submit(self._execute,claim);self.running[claim.spec.job_id]=_Running(claim,fut,time.time())

    def _schedule_maintenance(self):
        if not self.config.maintenance_command:return
        now=time.time()
        if now-self.last_maintenance<self.config.maintenance_interval_seconds:return
        jid=f'MAINT:R8:{int(now//self.config.maintenance_interval_seconds)}'
        spec=FactoryJobSpecR8(job_id=jid,stage='EXPERIENCE_MAINTENANCE',command=self.config.maintenance_command,scientific_identity={'factory_config_hash':self.config.content_hash,'maintenance_bucket':str(int(now//self.config.maintenance_interval_seconds))},resource_class='MAINTENANCE',priority=500,max_attempts=2,timeout_seconds=7200)
        self.queue.enqueue(spec);self.last_maintenance=now

    def tick(self)->dict[str,Any]:
        accepted=self.watcher.scan_once();jobs=[]
        for r in accepted:
            snapshot=None;snapshot_path=None
            if self.catalog is not None:
                self.catalog.register_accepted_bundle(r)
                if self.config.auto_catalog_snapshot:
                    snapshot=self.catalog.create_snapshot(f'AUTO:{r.bundle_sha256[:16]}')
                    sd=Path(self.config.catalog_snapshot_dir or (Path(self.config.work_root)/'catalog_snapshots'));sd.mkdir(parents=True,exist_ok=True);snapshot_path=sd/f'{snapshot.scientific_dataset_hash}.json';snapshot_path.write_text(json.dumps(asdict(snapshot),indent=2)+'\n')
            jobs.extend(self.enqueue_bundle(r,catalog_snapshot=snapshot,catalog_snapshot_path=snapshot_path))
        self.queue.recover_expired();self._reap();self._schedule_maintenance();self._schedule()
        snap=self.governor.snapshot(self._running_counts())
        status={'timestamp_unix':time.time(),'accepted_bundles':len(accepted),'enqueued_jobs':jobs,'queue_counts':self.queue.counts(),'running':list(self.running),'resource_snapshot':asdict(snap),'final_holdout_auto_open':False}
        if self.config.status_jsonl:
            sp=Path(self.config.status_jsonl);sp.parent.mkdir(parents=True,exist_ok=True)
            with sp.open('a',encoding='utf-8') as f:
                f.write(json.dumps(status,sort_keys=True,separators=(',',':'))+'\n')
        return status

    def run_forever(self):
        while not self.stop_event.is_set():
            self.tick();self.stop_event.wait(self.config.poll_seconds)
