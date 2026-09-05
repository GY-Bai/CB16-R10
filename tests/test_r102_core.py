from __future__ import annotations
import csv, hashlib, io, math, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0,str(ROOT))

from cb16_local_opt.binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from cb16_local_opt.r102_common import G0_FILE_SHA256,G0_TENSOR_SEMANTIC_SHA256,HOUR_MS,sha256_file,torch_tensor_semantic_sha256
from cb16_local_opt.r102_market import bounded_kline_archives,iter_1m_bounded
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102,build_parent_scenarios,simulate_h72_branch,LONG
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_learning import compile_teacher_evidence,evidence_summary,train_challenger
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10

class TestR102Core(unittest.TestCase):
    def test_g0_authority(self):
        p=ROOT/'authority/g0_parent/central_brain_g0_r10_parent.pt'
        if not p.exists():
            self.skipTest("G0 parent checkpoint is not in public repo; verified only on Shanxi")
        self.assertEqual(sha256_file(p),G0_FILE_SHA256)
        s=torch.load(p,map_location='cpu',weights_only=True)
        self.assertEqual(torch_tensor_semantic_sha256(s),G0_TENSOR_SEMANTIC_SHA256)

    def test_exact_physics_h72(self):
        rt=FrozenPhysicsRuntimeR102.load(ROOT)
        start=1609459200000; ts=np.arange(200,dtype=np.int64)*HOUR_MS+start
        base=30000*(1+0.0002*np.arange(200))+100*np.sin(np.arange(200)/7)
        o=base;c=base*(1+0.0001*np.sin(np.arange(200)/3));h=np.maximum(o,c)*1.001;l=np.minimum(o,c)*.999;v=np.full(200,100.)
        bars=np.stack([o,h,l,c,v],1).astype(np.float32); funding=np.zeros(200);funding[::8]=.0001
        t=int(ts[96]); parents=build_parent_scenarios(rt,symbol='BTCUSDT',decision_time_ms=t,hourly_ts=ts,hourly_ohlcv=bars,funding=funding,prehistory_hours=96)
        self.assertEqual(len(parents),6); self.assertTrue(all(p['eligible_for_economic_evidence'] for p in parents))
        b=simulate_h72_branch(rt,parent=parents[0],symbol='BTCUSDT',decision_time_ms=t,candidate_direction_v55=LONG,candidate_risk=.25,hourly_ts=ts,hourly_ohlcv=bars,funding=funding)
        self.assertEqual(b['status'],'MATURED'); self.assertTrue(math.isfinite(b['utility']))

    def test_forbidden_2025_09_archive_never_selected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); d=root/'klines_1m/BTCUSDT'; d.mkdir(parents=True); (root/'fundingRate/BTCUSDT').mkdir(parents=True)
            for month, start in [('2025-08',1754006400000),('2025-09',1756684800000)]:
                zp=d/f'BTCUSDT-1m-{month}.zip'
                data=f'{start},1,1,1,1,1,{start+59999},0,0,0,0,0\n'.encode()
                with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(f'BTCUSDT-1m-{month}.csv',data)
                zp.with_suffix(zp.suffix+'.CHECKSUM').write_text(hashlib.sha256(zp.read_bytes()).hexdigest()+'  '+zp.name+'\n')
            src=BinanceUSDMArchiveSourceR10(root)
            self.assertEqual([p.name for p in bounded_kline_archives(src,'BTCUSDT')],['BTCUSDT-1m-2025-08.zip'])
            rows=list(iter_1m_bounded(src,'BTCUSDT'))
            self.assertEqual(len(rows),1); self.assertEqual(rows[0].open_time,1754006400000)

    def test_probabilistic_teacher_and_all_brain_groups(self):
        parents={};samples=[]
        for i in range(70):
            split='TRAIN' if i<55 else 'VALIDATION'; t=1_600_000_000_000+i*512*HOUR_MS
            op=tuple(math.sin(i/9+j*.01)*.1 for j in range(48)); med=tuple(math.cos(i/11+j*.01)*.1 for j in range(48)); acc=(0.,0.,1.,.01*(i%4),1.,0.)
            p=ParentContextR102(f'P{i}',f'G{i}','BTCUSDT',t,split,'CLEAN',op,med,acc,tuple([0.]*30),30000.,f'snap{i}',True,f'm{i}');parents[p.parent_id]=p
            for d,r in [(0,0.)]+[(d,r) for d in(-1,1) for r in(.25,.5,.75,1.)]:
                u=.003*math.sin(i/7)*d*r-.0002*r*r
                samples.append(CounterfactualBranchSampleR5(p.parent_id,p.student_context_object_id,t,p.student_features,d,r,u,p.dependence_group_id,p.market_lineage_hash))
        tr,va=compile_teacher_evidence(samples,parents)
        self.assertGreaterEqual(evidence_summary(tr)['admitted_dependence_groups'],32)
        self.assertGreaterEqual(evidence_summary(va)['admitted_dependence_groups'],8)
        with tempfile.TemporaryDirectory() as td:
            m=build_g0_brain_r10('TIER_1',seed=24680,device='cpu')
            rec=train_challenger(model=m,train_evidence=tr,val_evidence=va,parents=parents,device='cpu',generation=0,snapshot_hash='test-snapshot',receipt_dir=td,epochs=1,batch_size=128,lr=1e-4)
            self.assertTrue(all(v>0 for v in rec['gradient_group_norms_last_step'].values()))
            self.assertTrue(all(v>0 for v in rec['update_group_norms'].values()))
            m2=build_g0_brain_r10('TIER_1',seed=24680,device='cpu')
            rec2=train_challenger(model=m2,train_evidence=tr,val_evidence=va,parents=parents,device='cpu',generation=0,snapshot_hash='test-snapshot',receipt_dir=td,epochs=1,batch_size=128,lr=1e-4)
            self.assertEqual(rec['challenger_semantic_sha256'],rec2['challenger_semantic_sha256'])

if __name__=='__main__': unittest.main()
