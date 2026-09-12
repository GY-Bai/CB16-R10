from __future__ import annotations
from dataclasses import asdict
import json
from .account_economics_r0 import AccountEconomicsStateR0
from .cc_clock_r0 import CCFourClockR0
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0,CCExecutionSummaryR0
from .cc_runtime_wire_r0 import CCPolicyDecisionV1,CCExecutionLegV1
from .cc_environment_advance_r0 import CCEnvironmentIntervalR0
RECOVERY_SCHEMA_R0="CB16_R11_CC_ACCOUNT_RECOVERY_V1_R0"
def _account(p): return AccountEconomicsStateR0(**p)
def seal_runtime_r0(rt,*,policy_memory_token=None):
    p={"schema":RECOVERY_SCHEMA_R0,"account_lineage_id":rt.account_lineage_id,"account":asdict(rt.account),"clocks":asdict(rt.clocks),"schedule":asdict(rt.schedule),"policy_generation":rt.policy_generation,"policy_id":rt.policy_id,"policy_sha256":rt.policy_sha256,"predecessor_token":rt.predecessor_token,"phase":rt.phase,"pending_interval":asdict(rt.pending_interval) if rt.pending_interval else None,"pending_decision":asdict(rt.pending_decision) if rt.pending_decision else None,"pending_execution":None if rt.pending_execution is None else {**asdict(rt.pending_execution),"account_after_execution":asdict(rt.pending_execution.account_after_execution)},"pre_account":asdict(rt._pre_account) if rt._pre_account else None,"pre_hash":rt._pre_hash,"policy_memory_token":policy_memory_token}; return json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False)
def restore_runtime_r0(payload,*,executor):
    p=json.loads(payload)
    if p.get("schema")!=RECOVERY_SCHEMA_R0: raise RuntimeError("CCREC_SCHEMA_MISMATCH")
    rt=CCContinuousAccountRuntimeR0(account_lineage_id=p["account_lineage_id"],account=_account(p["account"]),clocks=CCFourClockR0(**p["clocks"]),schedule=CCDecisionScheduleR0(**p["schedule"]),executor=executor,policy_generation=p["policy_generation"],policy_id=p["policy_id"],policy_sha256=p["policy_sha256"]); rt.predecessor_token=p["predecessor_token"]; rt.phase=p["phase"]; rt.pending_interval=CCEnvironmentIntervalR0(**p["pending_interval"]) if p["pending_interval"] else None; rt.pending_decision=CCPolicyDecisionV1(**p["pending_decision"]) if p["pending_decision"] else None
    if p["pending_execution"]:
        q=dict(p["pending_execution"]); q["legs"]=tuple(CCExecutionLegV1(**x) for x in q["legs"]); q["account_after_execution"]=_account(q["account_after_execution"]); rt.pending_execution=CCExecutionSummaryR0(**q)
    rt._pre_account=_account(p["pre_account"]) if p["pre_account"] else None; rt._pre_hash=p["pre_hash"]; return rt,p.get("policy_memory_token")
