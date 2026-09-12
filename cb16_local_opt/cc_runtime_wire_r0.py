from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,math,re
from typing import Any,Mapping
CC_POLICY_DECISION_V1="CCPolicyDecisionV1"; CC_ENVIRONMENT_TRANSITION_V1="CCEnvironmentTransitionV1"; DIRECTIONS=("SHORT","FLAT","LONG"); HEX64=re.compile(r"^[0-9a-f]{64}$")
def _nonempty(v:Any,code:str)->str:
    if not isinstance(v,str) or not v.strip(): raise RuntimeError(code)
    return v
def _finite(v:Any,code:str)->float:
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)): raise RuntimeError(code)
    return float(v)
def _hash(v:str,code:str)->str:
    if not isinstance(v,str) or HEX64.fullmatch(v) is None: raise RuntimeError(code)
    return v
def canonical_sha256(payload:Mapping[str,Any])->str: return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
@dataclass(frozen=True)
class CCExecutionLegV1:
    leg_index:int; delta_quantity:float; status:str; fill_price:float|None=None
    def validate(self):
        if isinstance(self.leg_index,bool) or self.leg_index<0: raise RuntimeError("CCW02_LEG_INDEX_INVALID")
        _finite(self.delta_quantity,"CCW02_LEG_QTY_INVALID"); _nonempty(self.status,"CCW02_LEG_STATUS_INVALID")
        if self.fill_price is not None and _finite(self.fill_price,"CCW02_FILL_INVALID")<=0: raise RuntimeError("CCW02_FILL_INVALID")
@dataclass(frozen=True)
class CCPolicyDecisionV1:
    science_semantic_version:str; account_lineage_id:str; decision_index:int; environment_time:int; policy_generation:str; policy_id:str; policy_sha256:str; observation_schema:str; observation_hash:str; normalizer_id:str; nominal_direction:str; nominal_target_risk:float; log_mu:float; risk_measure_kind:str; rng_stream_id:str; rng_position_or_counter:int
    def validate(self):
        for v,c in ((self.science_semantic_version,"CCW01_SCIENCE_ID_INVALID"),(self.account_lineage_id,"CCW01_ACCOUNT_ID_INVALID"),(self.policy_generation,"CCW01_GENERATION_INVALID"),(self.policy_id,"CCW01_POLICY_ID_INVALID"),(self.observation_schema,"CCW01_OBS_SCHEMA_INVALID"),(self.normalizer_id,"CCW01_NORMALIZER_INVALID"),(self.rng_stream_id,"CCW01_RNG_STREAM_INVALID")): _nonempty(v,c)
        _hash(self.policy_sha256,"CCW01_POLICY_HASH_INVALID"); _hash(self.observation_hash,"CCW01_OBS_HASH_INVALID")
        if isinstance(self.decision_index,bool) or self.decision_index<0 or isinstance(self.environment_time,bool) or self.environment_time<0: raise RuntimeError("CCW01_CLOCK_INVALID")
        if self.nominal_direction not in DIRECTIONS: raise RuntimeError("CCW01_DIRECTION_INVALID")
        r=_finite(self.nominal_target_risk,"CCW01_RISK_INVALID")
        if not 0<=r<=1 or (self.nominal_direction=="FLAT" and r!=0): raise RuntimeError("CCW01_RISK_INVALID")
        _finite(self.log_mu,"CCW01_LOG_MU_INVALID")
        expected="point_mass" if self.nominal_direction=="FLAT" else "continuous_density"
        if self.risk_measure_kind!=expected: raise RuntimeError("CCW01_RISK_MEASURE_INVALID")
        if isinstance(self.rng_position_or_counter,bool) or self.rng_position_or_counter<0: raise RuntimeError("CCW01_RNG_COUNTER_INVALID")
    @property
    def ref(self): self.validate(); return canonical_sha256(asdict(self))
@dataclass(frozen=True)
class CCEnvironmentTransitionV1:
    account_lineage_id:str; decision_index:int; environment_time_before:int; environment_time_after:int; pre_account_truth_hash:str; policy_decision_ref:str|None; permission_status:str; permission_reason:str; permitted_target_direction:str; permitted_target_risk:float; target_quantity:float; execution_legs:tuple[CCExecutionLegV1,...]; fees:float; funding:float; realized_pnl:float; unrealized_pnl_delta:float; liability_delta:float; post_account_truth_hash:str; post_equity:float; boundary_type:str; mechanical_terminal:bool; external_capital_flow_ref_or_null:str|None
    def validate(self):
        from .cc_runtime_boundary_r0 import BOUNDARIES
        _nonempty(self.account_lineage_id,"CCW02_ACCOUNT_ID_INVALID")
        if isinstance(self.decision_index,bool) or self.decision_index<0 or self.environment_time_after<=self.environment_time_before: raise RuntimeError("CCW02_CLOCK_INVALID")
        _hash(self.pre_account_truth_hash,"CCW02_PRE_HASH_INVALID"); _hash(self.post_account_truth_hash,"CCW02_POST_HASH_INVALID")
        if self.policy_decision_ref is not None: _hash(self.policy_decision_ref,"CCW02_DECISION_REF_INVALID")
        _nonempty(self.permission_status,"CCW02_PERMISSION_INVALID"); _nonempty(self.permission_reason,"CCW02_PERMISSION_REASON_INVALID")
        if self.permitted_target_direction not in DIRECTIONS: raise RuntimeError("CCW02_DIRECTION_INVALID")
        r=_finite(self.permitted_target_risk,"CCW02_RISK_INVALID")
        if not 0<=r<=1 or (self.permitted_target_direction=="FLAT" and r!=0): raise RuntimeError("CCW02_RISK_INVALID")
        for i,leg in enumerate(self.execution_legs):
            leg.validate()
            if leg.leg_index!=i: raise RuntimeError("CCW02_LEGS_UNORDERED")
        for v,c in ((self.target_quantity,"CCW02_TARGET_QTY_INVALID"),(self.fees,"CCW02_FEES_INVALID"),(self.funding,"CCW02_FUNDING_INVALID"),(self.realized_pnl,"CCW02_REALIZED_INVALID"),(self.unrealized_pnl_delta,"CCW02_UNREALIZED_INVALID"),(self.liability_delta,"CCW02_LIABILITY_INVALID"),(self.post_equity,"CCW02_EQUITY_INVALID")): _finite(v,c)
        if self.fees<0: raise RuntimeError("CCW02_FEES_INVALID")
        if self.boundary_type not in BOUNDARIES: raise RuntimeError("CCW02_BOUNDARY_INVALID")
        if not isinstance(self.mechanical_terminal,bool): raise RuntimeError("CCW02_TERMINAL_INVALID")
        if self.external_capital_flow_ref_or_null is not None: _nonempty(self.external_capital_flow_ref_or_null,"CCW02_CAPITAL_FLOW_REF_INVALID")
