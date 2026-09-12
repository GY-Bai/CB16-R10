from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
from typing import Callable,Any
from .account_economics_r0 import AccountEconomicsStateR0
from .cc_clock_r0 import CCFourClockR0
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .cc_runtime_wire_r0 import CCPolicyDecisionV1,CCEnvironmentTransitionV1,CCExecutionLegV1,canonical_sha256
from .cc_environment_advance_r0 import CCEnvironmentIntervalR0,advance_environment_r0
from .cc_environment_lifecycle_r0 import OBSERVATION_AVAILABLE,DECISION_CAPTURED,EXECUTION_APPLIED,ENVIRONMENT_APPLIED,POST_STATE_PUBLISHED
@dataclass(frozen=True)
class CCExecutionSummaryR0:
    permission_status:str; permission_reason:str; permitted_target_direction:str; permitted_target_risk:float; target_quantity:float; legs:tuple[CCExecutionLegV1,...]; account_after_execution:AccountEconomicsStateR0
Executor=Callable[[AccountEconomicsStateR0,CCPolicyDecisionV1],CCExecutionSummaryR0]
PolicyCallback=Callable[[AccountEconomicsStateR0,CCFourClockR0],CCPolicyDecisionV1]
def account_truth_sha256_r0(state:AccountEconomicsStateR0)->str:
    state.validate(); p=asdict(state); p["unrealized_pnl"]=state.unrealized_pnl; p["equity"]=state.equity
    return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def execute_via_frozen_r1_r0(account:AccountEconomicsStateR0,decision:CCPolicyDecisionV1,*,config:Any,legal_envelope_id:str="CC_A_R0",max_permitted_target_risk:float=1.0)->CCExecutionSummaryR0:
    from .action_contract_r1 import make_target_position_action_r1
    from .actor_critic_supervisor_r1 import make_supervisor_authority_r1,supervise_target_action_r1
    from .actor_critic_environment_profile_r1 import policy_neutral_environment_profile_r1,MECHANICAL_OR_RESOURCE_RULES_R1,STRATEGY_PREFERENCE_RULES_R1,HISTORICAL_ONLY_RULES_R1
    from .actor_critic_physics_adapter_r1 import execute_target_position_r1,EXECUTED,REVERSAL_OPEN_REJECTED
    decision.validate(); action=make_target_position_action_r1(action_id=f"cc-a-{decision.decision_index}",policy_id=decision.policy_id,policy_version=decision.policy_generation,target_direction=decision.nominal_direction,requested_target_risk=decision.nominal_target_risk)
    state_hash=account_truth_sha256_r0(account); authority=make_supervisor_authority_r1(authority_id="CC_A_R0_SUPERVISOR",account_id=account.account_id,account_state_sha256=state_hash,terminated=not account.economic_responsibility_open,truncated=False,legal_target_directions=("SHORT","FLAT","LONG"),max_permitted_target_risk=max_permitted_target_risk)
    permission=supervise_target_action_r1(action,authority); rules=[]
    for r in MECHANICAL_OR_RESOURCE_RULES_R1: rules.append({"rule_id":r,"category":"MARKET_EXCHANGE_MECHANIC"})
    for r in STRATEGY_PREFERENCE_RULES_R1: rules.append({"rule_id":r,"category":"STRATEGY_PREFERENCE"})
    for r in HISTORICAL_ONLY_RULES_R1: rules.append({"rule_id":r,"category":"HISTORICAL_ONLY"})
    taxonomy={"schema":"CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1","rules":rules}
    receipt=execute_target_position_r1(account=account,action=action,supervisor_authority=authority,profile=policy_neutral_environment_profile_r1(),taxonomy=taxonomy,mark_price=account.mark_price,legal_envelope_id=legal_envelope_id,config=config)
    legs=[]; q0=account.position_quantity
    if receipt.transition_kind=="REVERSAL_CLOSE_THEN_OPEN":
        legs.append(CCExecutionLegV1(0,-q0,"EXECUTED",receipt.fill_price))
        if receipt.status==EXECUTED: legs.append(CCExecutionLegV1(1,receipt.target_quantity,"EXECUTED",receipt.fill_price))
        elif receipt.status==REVERSAL_OPEN_REJECTED: legs.append(CCExecutionLegV1(1,0.0,"REJECTED",None))
    elif abs(receipt.delta_quantity)>1e-12: legs.append(CCExecutionLegV1(0,receipt.delta_quantity,receipt.status,receipt.fill_price))
    return CCExecutionSummaryR0(permission.outcome,"|".join(permission.reason_codes),permission.permitted_target_direction,permission.permitted_target_risk,receipt.target_quantity,tuple(legs),receipt.account_after)
class CCContinuousAccountRuntimeR0:
    def __init__(self,*,account_lineage_id:str,account:AccountEconomicsStateR0,clocks:CCFourClockR0,schedule:CCDecisionScheduleR0,executor:Executor,policy_generation:str,policy_id:str,policy_sha256:str):
        account.validate(); clocks.validate(); schedule.validate(); self.account_lineage_id=account_lineage_id; self.account=account; self.clocks=clocks; self.schedule=schedule; self.executor=executor; self.policy_generation=policy_generation; self.policy_id=policy_id; self.policy_sha256=policy_sha256
        self.predecessor_token=canonical_sha256({"account":account_truth_sha256_r0(account),"env":clocks.environment_time,"lineage":account_lineage_id}); self.phase=POST_STATE_PUBLISHED; self.pending_interval=None; self.pending_decision=None; self.pending_execution=None; self._pre_account=None; self._pre_hash=None
    def begin_interval(self,interval,*,expected_predecessor_token):
        if self.phase!=POST_STATE_PUBLISHED: raise RuntimeError("CCLOOP_INTERVAL_ALREADY_OPEN")
        if expected_predecessor_token!=self.predecessor_token: raise RuntimeError("CCLOOP_PREDECESSOR_MISMATCH")
        interval.validate(); self.pending_interval=interval; self._pre_account=self.account; self._pre_hash=account_truth_sha256_r0(self.account); self.phase=OBSERVATION_AVAILABLE
    def capture_decision(self,callback):
        if self.phase!=OBSERVATION_AVAILABLE: raise RuntimeError("CCLOOP_PHASE_INVALID")
        d=None
        if self.schedule.is_decision_time(self.clocks.environment_time):
            if callback is None: raise RuntimeError("CCLOOP_POLICY_CALLBACK_REQUIRED")
            d=callback(self.account,self.clocks); d.validate()
            if d.account_lineage_id!=self.account_lineage_id: raise RuntimeError("CCLOOP_ACCOUNT_LINEAGE_MISMATCH")
            if d.decision_index!=self.clocks.policy_decision_index or d.environment_time!=self.clocks.environment_time: raise RuntimeError("CCLOOP_DECISION_CLOCK_MISMATCH")
            if (d.policy_generation,d.policy_id,d.policy_sha256)!=(self.policy_generation,self.policy_id,self.policy_sha256): raise RuntimeError("CCLOOP_POLICY_IDENTITY_MISMATCH")
            self.schedule=self.schedule.record(environment_time=self.clocks.environment_time,decision_index=d.decision_index)
        self.pending_decision=d; self.phase=DECISION_CAPTURED; return d
    def execute_pending(self):
        if self.phase!=DECISION_CAPTURED: raise RuntimeError("CCLOOP_PHASE_INVALID")
        d=self.pending_decision
        self.pending_execution=CCExecutionSummaryR0("NO_DECISION","NOT_SCHEDULED","FLAT",0.0,self.account.position_quantity,(),self.account) if d is None else self.executor(self.account,d)
        self.account=self.pending_execution.account_after_execution; self.phase=EXECUTION_APPLIED; return self.pending_execution
    def advance_pending_environment(self):
        if self.phase!=EXECUTION_APPLIED: raise RuntimeError("CCLOOP_PHASE_INVALID")
        self.account=advance_environment_r0(self.account,self.pending_interval); self.phase=ENVIRONMENT_APPLIED; return self.account
    def publish_pending(self):
        if self.phase!=ENVIRONMENT_APPLIED: raise RuntimeError("CCLOOP_PHASE_INVALID")
        pre=self._pre_account; post=self.account; ex=self.pending_execution; d=self.pending_decision; interval=self.pending_interval
        tr=CCEnvironmentTransitionV1(self.account_lineage_id,d.decision_index if d else self.clocks.policy_decision_index,self.clocks.environment_time,self.clocks.environment_time+1,self._pre_hash,d.ref if d else None,ex.permission_status,ex.permission_reason,ex.permitted_target_direction,ex.permitted_target_risk,ex.target_quantity,ex.legs,post.fees_cumulative-pre.fees_cumulative,post.funding_cumulative-pre.funding_cumulative,post.realized_pnl_cumulative-pre.realized_pnl_cumulative,post.unrealized_pnl-pre.unrealized_pnl,post.liabilities-pre.liabilities,account_truth_sha256_r0(post),post.equity,interval.boundary_type,not post.economic_responsibility_open,interval.external_capital_flow_ref)
        tr.validate(); self.clocks=self.clocks.advance_environment(); self.clocks=self.clocks.advance_decision() if d is not None else self.clocks; self.predecessor_token=canonical_sha256({"prev":self.predecessor_token,"transition":asdict(tr)}); self.phase=POST_STATE_PUBLISHED; self.pending_interval=None; self.pending_decision=None; self.pending_execution=None; self._pre_account=None; self._pre_hash=None; return tr
    def step(self,interval,callback,*,expected_predecessor_token):
        self.begin_interval(interval,expected_predecessor_token=expected_predecessor_token); self.capture_decision(callback); self.execute_pending(); self.advance_pending_environment(); return self.publish_pending()
