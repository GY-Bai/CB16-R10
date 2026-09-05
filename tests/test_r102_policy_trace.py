from __future__ import annotations
import tempfile, unittest, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from cb16_local_opt.r102_common import HOUR_MS,model_state_semantic_sha256
from cb16_local_opt.r102_physics import FrozenPhysicsRuntimeR102,build_parent_scenarios
from cb16_local_opt.r102_evidence_cache import ParentContextR102
from cb16_local_opt.r102_policy_trace import run_real_on_policy_trace
from cb16_local_opt.sharded_experience_lake import ShardedExperienceLake
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10
class TestPolicyTrace(unittest.TestCase):
 def test_brain_to_exact_physics_trace(self):
  rt=FrozenPhysicsRuntimeR102.load(ROOT); start=1609459200000; ts=np.arange(220,dtype=np.int64)*HOUR_MS+start
  base=30000+np.arange(220)*2.; bars=np.stack([base,base*1.001,base*.999,base+1,np.ones(220)*100],1).astype(np.float32); funding=np.zeros(220); t=int(ts[96])
  s=build_parent_scenarios(rt,symbol='BTCUSDT',decision_time_ms=t,hourly_ts=ts,hourly_ohlcv=bars,funding=funding)[0]
  p=ParentContextR102('P0','G0','BTCUSDT',t,'VALIDATION','CLEAN_FLAT_FULL',tuple([.1]*48),tuple([.2]*48),tuple(float(x) for x in s['account6']),tuple([0.]*30),s['current_mark'],s['snapshot_sha256'],True,'m')
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);(d/'market_cache').mkdir();np.savez_compressed(d/'market_cache/BTCUSDT.hourly_r102.npz',open_time_ms=ts,ohlcv=bars,funding_rate=funding)
   lake=ShardedExperienceLake(d/'lake',shards=2); m=build_g0_brain_r10('TIER_1',seed=24680,device='cpu')
   st={'parent_id':'P0','account_id':s['account_id'],'snapshot':s['snapshot'],'risk_authority':s['risk_authority'],'current_mark':s['current_mark']}
   r=run_real_on_policy_trace(model=m,policy_hash=model_state_semantic_sha256(m),generation=0,parents={'P0':p},parent_states={'P0':st},cache_dir=d,physics=rt,lake=lake,device='cpu',max_groups=1)
   self.assertEqual(r['Brain_to_ActionIntent_to_Supervisor_to_exact_Physics_to_H72_Outcome'],'PASS'); self.assertEqual(r['matured'],1); self.assertEqual(lake.count(object_type='DECISION_EVENT'),1); self.assertEqual(lake.count(object_type='OUTCOME_SAMPLE'),1); lake.close()
if __name__=='__main__': unittest.main()
