from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

class WireValidationError(ValueError): pass
W02_SCHEMA="CCEnvironmentTransitionV1"; W03_SCHEMA="CCExperienceSequenceV1"; W05_SCHEMA="CCEconomicResultV1"
W02_FIELDS=("schema_version","environment_transition_id","account_lineage_id","generation_id","step_id","market_data_ref","source_family","source_policy_class","source_window_or_seed_ref","decision_index","decision_ts","policy_decision_ref","nominal_action","nominal_target_quantity","nominal_limit_or_market","nominal_time_in_force","nominal_cancel_replace_budget","target_quantity","clip_applied","executed_quantity","execution_price","fees_paid","slippage_cost","impact_cost","retry_count","throttle_delay_ms","pre_account_truth_hash","post_account_truth_hash","realized_pnl_delta","unrealized_pnl_delta","margin_used_post","equity_post","drawdown_post","runtime_terminal","mechanical_terminal","done","failure_code","next_observation_ref_or_null","rng_state_in_hash","rng_state_out_hash","transition_integrity_hash")
W03_FIELDS=("schema_version","sequence_id","account_lineage_id","generation_id","environment_id","source_family","source_policy_class","source_window_or_seed_ref","first_decision_index","last_decision_index","transition_refs","policy_version_refs","terminal_outcome","failure_code_or_null","rng_initial_state_hash","rng_final_state_hash","sequence_integrity_hash","eligible_for_replay","replay_ineligibility_reason_or_null","bootstrap_state_ref_or_null")
W05_FIELDS=("evaluation_id","policy_object_type","policy_identity","cohort_id","common_horizon_id","capital_denominator_id","account_results","mean_arithmetic_return","buy_hold_delta","flat_delta","failure_counts","tail_diagnostics","result_scope")
SOURCE_FAMILIES={"train","validation","final_test","synthetic"}; POLICY_OBJECT_TYPES={"frozen_checkpoint","generation_chain","baseline"}

def _exact(p:Mapping[str,Any], fields:tuple[str,...], label:str):
    got,want=set(p),set(fields)
    if got!=want: raise WireValidationError(f"{label} field mismatch missing={sorted(want-got)} extra={sorted(got-want)}")
def _nonempty(p,names,label):
    for n in names:
        if not isinstance(p[n],str) or not p[n]: raise WireValidationError(f"{label}.{n} required")
def validate_w02(p:Mapping[str,Any])->dict[str,Any]:
    _exact(p,W02_FIELDS,W02_SCHEMA)
    if p["schema_version"]!=W02_SCHEMA: raise WireValidationError("W-02 schema drift")
    _nonempty(p,("environment_transition_id","account_lineage_id","generation_id","step_id","market_data_ref","source_policy_class","source_window_or_seed_ref","policy_decision_ref","pre_account_truth_hash","post_account_truth_hash","rng_state_in_hash","rng_state_out_hash","transition_integrity_hash"),W02_SCHEMA)
    if p["source_family"] not in SOURCE_FAMILIES: raise WireValidationError("invalid source_family")
    if not isinstance(p["decision_index"],int) or p["decision_index"]<0: raise WireValidationError("invalid decision_index")
    for n in ("executed_quantity","fees_paid","slippage_cost","impact_cost","realized_pnl_delta","unrealized_pnl_delta","margin_used_post","equity_post","drawdown_post"):
        if not isinstance(p[n],(int,float)) or not isfinite(float(p[n])): raise WireValidationError(f"{n} must be finite")
    price=p["execution_price"]
    if price is not None and (not isinstance(price,(int,float)) or not isfinite(float(price))): raise WireValidationError("execution_price must be finite or null")
    if price is None and float(p["executed_quantity"])!=0.0: raise WireValidationError("filled transition requires execution_price")
    if p["failure_code"] is not None and not isinstance(p["failure_code"],str): raise WireValidationError("invalid failure_code")
    return dict(p)
def validate_w03(p:Mapping[str,Any])->dict[str,Any]:
    _exact(p,W03_FIELDS,W03_SCHEMA)
    if p["schema_version"]!=W03_SCHEMA: raise WireValidationError("W-03 schema drift")
    _nonempty(p,("sequence_id","account_lineage_id","generation_id","environment_id","source_policy_class","source_window_or_seed_ref","rng_initial_state_hash","rng_final_state_hash","sequence_integrity_hash"),W03_SCHEMA)
    if p["source_family"] not in SOURCE_FAMILIES: raise WireValidationError("invalid source_family")
    a,b=p["first_decision_index"],p["last_decision_index"]
    if not isinstance(a,int) or not isinstance(b,int) or a<0 or b<a: raise WireValidationError("invalid index range")
    refs=p["transition_refs"]
    if not isinstance(refs,list) or not refs or len(refs)!=len(set(refs)): raise WireValidationError("transition refs must be nonempty and unique")
    if not isinstance(p["policy_version_refs"],list) or not p["policy_version_refs"]: raise WireValidationError("policy_version_refs required")
    if bool(p["eligible_for_replay"]) == bool(p["replay_ineligibility_reason_or_null"]): raise WireValidationError("replay eligibility/reason inconsistent")
    return dict(p)
def validate_w05(p:Mapping[str,Any])->dict[str,Any]:
    _exact(p,W05_FIELDS,W05_SCHEMA); _nonempty(p,("evaluation_id","policy_identity","cohort_id","common_horizon_id","capital_denominator_id","result_scope"),W05_SCHEMA)
    if p["policy_object_type"] not in POLICY_OBJECT_TYPES: raise WireValidationError("invalid policy object type")
    if not isinstance(p["account_results"],list) or not p["account_results"]: raise WireValidationError("account_results required")
    if not isinstance(p["mean_arithmetic_return"],(int,float)) or not isfinite(float(p["mean_arithmetic_return"])): raise WireValidationError("finite arithmetic return required")
    return dict(p)
@dataclass(frozen=True)
class CCEnvironmentTransitionV1:
    payload:Mapping[str,Any]
    def __post_init__(self): object.__setattr__(self,"payload",validate_w02(self.payload))
@dataclass(frozen=True)
class CCExperienceSequenceV1:
    payload:Mapping[str,Any]
    def __post_init__(self): object.__setattr__(self,"payload",validate_w03(self.payload))
@dataclass(frozen=True)
class CCEconomicResultV1:
    payload:Mapping[str,Any]
    def __post_init__(self): object.__setattr__(self,"payload",validate_w05(self.payload))
