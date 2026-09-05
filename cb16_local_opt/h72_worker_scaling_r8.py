from __future__ import annotations

"""3700X H72 process-farm scaling benchmark."""

import dataclasses, hashlib, json, shutil, tempfile, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from .h72_anchor_farm_r7 import H72AnchorStoreWriterR7,H72DependenceGroupFarmR7,H72FarmConfigR7,H72FarmJobR7
from .h72_group_compiler_r7 import VectorizedGroupH72CompilerR7
from .market_cache_r6 import build_market_latent_cache_r6
from .market_encoder_r5 import FrozenMarketEncoderArtifact,ReferenceGrammarEncoderR5,WindowNormalizer,create_reference_encoder_artifact
from .trajectory_compiler_r6 import DecisionAnchorR6,InitialAccountSnapshotR6,MarketPathR6
from .vectorized_physics import VectorPhysicsConfig


def canonical_hash(obj:Any)->str:
    if dataclasses.is_dataclass(obj):obj=asdict(obj)
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class WorkerScalingPointR8:
    workers:int
    groups:int
    accounts_per_group:int
    elapsed_s:float
    groups_per_s:float
    teacher_samples_per_s:float
    status:str

@dataclass(frozen=True)
class WorkerScalingReceiptR8:
    points:tuple[WorkerScalingPointR8,...]
    selected_workers:int
    policy_hash:str
    status:str


def _market(n=180):
    t=np.arange(n);c=100*np.exp(.0004*t+.001*np.sin(t/11));o=np.r_[c[0],c[:-1]]
    return MarketPathR6((1_700_000_000+3600*t).astype(np.int64),o,np.maximum(o,c)*1.002,np.minimum(o,c)*.998,c,(1000+t).astype(float),np.zeros(n))


def benchmark_h72_worker_scaling_r8(*,worker_candidates:Sequence[int]=(1,2,4,6),groups:int=16,accounts_per_group:int=128,work_root:str|Path|None=None)->WorkerScalingReceiptR8:
    own=work_root is None;root=Path(tempfile.mkdtemp(prefix='cb16_r8_h72scale_')) if own else Path(work_root);root.mkdir(parents=True,exist_ok=True)
    cfg=VectorPhysicsConfig(initial_equity=10_000,max_gross_leverage=2,fee_rate=.0002,slippage_bps=0,maintenance_margin_rate=.1,max_holding_bars=200)
    policy_hash=canonical_hash({'worker_candidates':list(worker_candidates),'groups':groups,'accounts_per_group':accounts_per_group,'physics_hash':cfg.config_hash})
    try:
        ep=root/'enc.pt';er=create_reference_encoder_artifact(ep);enc=FrozenMarketEncoderArtifact(model=ReferenceGrammarEncoderR5(),architecture_id=er.architecture_id,artifact_path=ep,normalizer=WindowNormalizer(),expected_artifact_sha256=er.artifact_sha256,expected_parameter_count=er.parameter_count,authority='REFERENCE_CONFORMANCE_ONLY')
        cache=root/'cache';build_market_latent_cache_r6(output_root=cache,market=_market(),dataset_hash='R8_H72_WORKER_SCALE_FIXTURE',encoder=enc,device='cpu',batch_windows=128)
        store=root/'anchors';writer=H72AnchorStoreWriterR7(store,total_groups=groups,accounts_per_group=accounts_per_group,feature_dim=70);gc=VectorizedGroupH72CompilerR7(cfg)
        for gi in range(groups):
            idx=36+(gi%60);anchors=[DecisionAnchorR6(f'G{gi}:A{a}',f'CTX:G{gi}:A{a}',idx,tuple([gi/100,a/max(1,accounts_per_group)]+[0]*68),f'DG{gi}','M',InitialAccountSnapshotR6.flat(10_000)) for a in range(accounts_per_group)]
            writer.write_group(gi,gc.from_decision_anchors(f'DG{gi}',anchors))
        writer.seal();jobs=[H72FarmJobR7(i,str(store),str(cache),str(root/f'results_{i}'),asdict(cfg),((-1,.25),(0,0.0),(1,.25)),72) for i in range(groups)]
        points=[]
        for workers in worker_candidates:
            # Unique result directory per worker count so execution, not receipt reuse, is measured.
            jobs2=[dataclasses.replace(j,result_root=str(root/f'w{workers}')) for j in jobs]
            t0=time.perf_counter();out=H72DependenceGroupFarmR7(H72FarmConfigR7(workers=int(workers),start_method='spawn',cpu_threads_per_worker=1,max_in_flight=max(int(workers),2*int(workers)))).run(jobs2);elapsed=time.perf_counter()-t0;count=sum(x.teacher_sample_count for x in out)
            points.append(WorkerScalingPointR8(int(workers),groups,accounts_per_group,elapsed,groups/max(elapsed,1e-12),count/max(elapsed,1e-12),'PASS'))
        best=max(points,key=lambda x:x.groups_per_s);threshold=.95*best.groups_per_s;selected=min(x.workers for x in points if x.groups_per_s>=threshold)
        return WorkerScalingReceiptR8(tuple(points),selected,policy_hash,'PASS')
    finally:
        if own:shutil.rmtree(root,ignore_errors=True)
