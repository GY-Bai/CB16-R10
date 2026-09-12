from __future__ import annotations
from dataclasses import dataclass
import math,tempfile,hashlib,json
from .cc_fast_account_kernel_r0 import AccountKernelConfig,step_account_scalar
from .cc_fast_account_state_r0 import AccountStateSoA
from .cc_fast_collector_r0 import CollectorConfig,run_collector
from .cc_fast_policy_broker_r0 import PolicySpec
from .cc_fast_workload_r0 import WorkloadConfig,build_workload

@dataclass(frozen=True)
class SemanticHarnessResult:
    verdict:str; checks:dict[str,bool]; checksum:str; numeric_tolerance:float

def run_account_known_answers(*,atol:float=1e-10):
    checks={}; state=AccountStateSoA.allocate(["acct"],initial_equity=1000.0)
    r1=step_account_scalar(state,0,account_lineage_id="acct",environment_time_before_ns=0,environment_time_after_ns=1,mark_price_before=100.0,mark_price_after=110.0,target_quantity=2.0,funding_rate=0.0,policy_decision_ref="a"*64,permitted_target_direction="LONG",permitted_target_risk=0.2,config=AccountKernelConfig(fee_rate=0.0,maintenance_margin_fraction=0.0))
    checks["open_long_qty"]=math.isclose(state.quantity[0],2.0,abs_tol=atol); checks["mtm_equity"]=math.isclose(state.equity[0],1020.0,abs_tol=atol)
    r2=step_account_scalar(state,0,account_lineage_id="acct",environment_time_before_ns=1,environment_time_after_ns=2,mark_price_before=110.0,mark_price_after=105.0,target_quantity=-1.0,funding_rate=0.0,policy_decision_ref="b"*64,permitted_target_direction="SHORT",permitted_target_risk=0.2,config=AccountKernelConfig(fee_rate=0.0,maintenance_margin_fraction=0.0))
    checks["reversal_two_legs"]=[x.kind for x in r2.transition.execution_legs]==["REVERSAL_CLOSE","REVERSAL_OPEN"]; checks["reversal_sequence"]=[x.sequence for x in r2.transition.execution_legs]==[0,1]; checks["realized_long_profit"]=math.isclose(r2.transition.realized_pnl,20.0,abs_tol=atol)
    before=float(state.equity[0]); r3=step_account_scalar(state,0,account_lineage_id="acct",environment_time_before_ns=2,environment_time_after_ns=3,mark_price_before=105.0,mark_price_after=115.0,target_quantity=3.0,funding_rate=0.001,policy_decision_ref="c"*64,permitted_target_direction="LONG",permitted_target_risk=0.2,permission_status="REJECT",permission_reason="FIXTURE_REJECT",config=AccountKernelConfig(fee_rate=0.0,maintenance_margin_fraction=0.0))
    checks["reject_keeps_position"]=math.isclose(state.quantity[0],-1.0,abs_tol=atol); checks["reject_world_continues"]=not math.isclose(float(state.equity[0]),before,abs_tol=atol); checks["funding_recorded"]=not math.isclose(r3.transition.funding,0.0,abs_tol=atol)
    checksum=hashlib.sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest(); return SemanticHarnessResult("PASS" if all(checks.values()) else "FAIL",checks,checksum,atol)

def run_collector_determinism():
    workload=build_workload(WorkloadConfig(seed=7,market_steps=96,account_count=4)); policy=PolicySpec(policy_generation=3,policy_id="fixture-policy",policy_sha256="1"*64,direction_logits=(-0.2,-1.0,0.4)); checks={}
    with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
        cfg=CollectorConfig(queue_max_bytes=1<<20,queue_max_age_s=5,writer_chunk_facts=16); x=run_collector(workload,policy,output_root=a,config=cfg); y=run_collector(workload,policy,output_root=b,config=cfg)
        checks["semantic_checksum"]=x.semantic_checksum==y.semantic_checksum; checks["final_account_checksum"]=x.final_account_checksum==y.final_account_checksum; checks["transition_count"]=x.transitions==y.transitions and x.transitions>0; checks["policy_decision_count"]=x.policy_decisions==y.policy_decisions and x.policy_decisions>0; checks["durable_chunks"]=all(r.durable for r in x.chunk_receipts+y.chunk_receipts)
    checksum=hashlib.sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest(); return SemanticHarnessResult("PASS" if all(checks.values()) else "FAIL",checks,checksum,0.0)

def run_semantic_harness():
    a=run_account_known_answers(); b=run_collector_determinism(); checks={f"account:{k}":v for k,v in a.checks.items()}; checks.update({f"collector:{k}":v for k,v in b.checks.items()}); checksum=hashlib.sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest(); return SemanticHarnessResult("PASS" if all(checks.values()) else "FAIL",checks,checksum,max(a.numeric_tolerance,b.numeric_tolerance))
