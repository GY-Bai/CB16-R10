#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil, sys, time, uuid
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cb16_local_opt.r10_experience_lake_opt_r0 import MODE_BATCHED,MODE_OBJECTWISE,put_many_r10_opt_r0,semantic_ref_tuple
from cb16_local_opt.sharded_experience_lake import ExperienceObject,ShardedExperienceLake
BASELINE='BASELINE_SEQUENTIAL'

def inside(child,parent):
    try: child.resolve().relative_to(parent.resolve()); return True
    except ValueError: return False

def noise(i,n):
    out=[]; j=0
    while sum(map(len,out))<n:
        out.append(hashlib.sha256(f'{i}:{j}:CB16-LAKE-R0'.encode()).hexdigest()); j+=1
    return ''.join(out)[:n]

def objects(n,payload_bytes):
    return [ExperienceObject(
        object_id=f'R102:G7:BENCH:{i:06d}',object_type='EVIDENCE_PACKAGE',generation=7,
        policy_weight_hash='p'*64,snapshot_hash=f'bench-{i%17}',
        lineage_hash=hashlib.sha256(f'lineage:{i}'.encode()).hexdigest(),
        payload={'schema':'CB16_R10_LAKE_OPT_BENCH_V1','row':i,'blob':noise(i,payload_bytes),
                 'direction_target_probs':[.2,.5,.3],'requested_risk_target':(i%101)/100},
    ) for i in range(n)]

def run(root,objs,mode,batch_size):
    lake=ShardedExperienceLake(root,shards=4,synchronous='FULL')
    try:
        t=time.perf_counter()
        if mode==BASELINE:
            refs=[]; created=0
            for o in objs:
                r,c=lake.put(o); refs.append(r); created+=int(c)
            wall=time.perf_counter()-t; receipt={'created_count':created}
        else:
            refs,rec=put_many_r10_opt_r0(lake,objs,mode=mode,max_workers=4,batch_size=batch_size)
            wall=rec.wall_seconds; receipt=asdict(rec)
        snap=lake.seal_snapshot(snapshot_id='R102_G7_TRAINING_SNAPSHOT',parent_generation=7,parent_policy_hash='p'*64,refs=refs)
        audit=lake.audit(verify_payloads=True)
        return {'mode':mode,'seconds':wall,'objects_per_second':len(objs)/wall if wall>0 else None,
                'refs':[semantic_ref_tuple(r) for r in refs],'snapshot_hash':snap.content_hash,
                'audit_pass':bool(audit.get('pass')),'receipt':receipt}
    finally: lake.close()

def replay(root,objs,mode,batch_size):
    lake=ShardedExperienceLake(root,shards=4,synchronous='FULL')
    try:
        if mode==BASELINE:
            created=0
            for o in objs: _,c=lake.put(o); created+=int(c)
        else:
            _,rec=put_many_r10_opt_r0(lake,objs,mode=mode,max_workers=4,batch_size=batch_size); created=rec.created_count
        return created
    finally: lake.close()

def main():
    ap=argparse.ArgumentParser(description='Same-filesystem R10 Experience Lake persistence benchmark R0')
    ap.add_argument('--work-root',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--canonical-run-root',default='/data/cb16_hdd/cb16_runtime/R10_4')
    ap.add_argument('--objects',type=int,default=512); ap.add_argument('--payload-bytes',type=int,default=3072)
    ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--keep-work',action='store_true'); a=ap.parse_args()
    work=Path(a.work_root).resolve(); out=Path(a.out).resolve(); canonical=Path(a.canonical_run_root).resolve()
    if inside(work,canonical) or inside(out,canonical): raise ValueError('BENCHMARK_MUST_BE_OUTSIDE_CANONICAL_ROOT')
    work.mkdir(parents=True,exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True)
    session=work/f'run-{uuid.uuid4().hex}'; session.mkdir(); objs=objects(a.objects,a.payload_bytes)
    try:
        base=run(session/'baseline',objs,BASELINE,a.batch_size)
        result={'schema':'CB16_R10_EXPERIENCE_LAKE_BENCHMARK_R0','status':'RUNNING','objects':a.objects,
                'payload_bytes':a.payload_bytes,'batch_size':a.batch_size,
                'safety':{'canonical_campaign_modified':False,'scientific_semantics_changed':False,
                          'final_holdout_2025_09_accessed':False,'status_driving':False},'modes':{}}
        result['modes'][BASELINE]={k:v for k,v in base.items() if k!='refs'}
        all_pass=True
        for mode in (MODE_OBJECTWISE,MODE_BATCHED):
            row=run(session/mode.lower(),objs,mode,a.batch_size)
            sem=row['refs']==base['refs']; snap=row['snapshot_hash']==base['snapshot_hash']
            replay_created=replay(session/mode.lower(),objs,mode,a.batch_size)
            speed=base['seconds']/row['seconds'] if row['seconds']>0 else None
            ok=bool(sem and snap and row['audit_pass'] and replay_created==0); all_pass &= ok
            result['modes'][mode]={**{k:v for k,v in row.items() if k!='refs'},
                'semantic_refs_equal_baseline':sem,'snapshot_hash_equal_baseline':snap,
                'replay_created_count':replay_created,'equivalence_pass':ok,'speedup_vs_baseline':speed}
        result['status']='PASS' if all_pass else 'FAIL'
        out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'baseline_seconds':base['seconds'],
            'objectwise_speedup':result['modes'][MODE_OBJECTWISE]['speedup_vs_baseline'],
            'batched_speedup':result['modes'][MODE_BATCHED]['speedup_vs_baseline'],**result['safety']},indent=2,sort_keys=True))
        return 0 if all_pass else 2
    finally:
        if not a.keep_work: shutil.rmtree(session,ignore_errors=True)
if __name__=='__main__': raise SystemExit(main())
