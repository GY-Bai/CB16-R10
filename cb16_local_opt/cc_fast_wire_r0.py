from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,math,re
from typing import Any
import numpy as np
SCIENCE_SEMANTIC_VERSION="CB16_R11_CC_SCIENCE_SEMANTIC_V1"; POLICY_DECISION_SCHEMA="CCPolicyDecisionV1"; ENVIRONMENT_TRANSITION_SCHEMA="CCEnvironmentTransitionV1"; EXPERIENCE_SEQUENCE_SCHEMA="CCExperienceSequenceV1"
SHORT=-1; FLAT=0; LONG=1; DIRECTION_TO_CODE={"SHORT":SHORT,"FLAT":FLAT,"LONG":LONG}; CODE_TO_DIRECTION={v:k for k,v in DIRECTION_TO_CODE.items()}
POINT_MASS=0; CONTINUOUS_DENSITY=1; RISK_MEASURE_TO_CODE={"point_mass":POINT_MASS,"continuous_density":CONTINUOUS_DENSITY}; CODE_TO_RISK_MEASURE={v:k for k,v in RISK_MEASURE_TO_CODE.items()}
BOUNDARY_TO_CODE={"NONE":0,"COMPUTE_CHUNK":1,"OBJECTIVE_HORIZON_REACHED":2,"DATA_END_TRUNCATION":3,"ECONOMIC_TERMINAL":4,"PROCESS_FAILURE":5,"PAUSE":6,"TRADING_DISABLED_PENDING_SETTLEMENT":7}; CODE_TO_BOUNDARY={v:k for k,v in BOUNDARY_TO_CODE.items()}
ID_BYTES=32; HASH_BYTES=32
POLICY_DECISION_DTYPE=np.dtype([("account_id","S32"),("decision_index","<u8"),("environment_time_ns","<i8"),("policy_generation","<u8"),("policy_hash","S32"),("observation_hash","S32"),("direction","i1"),("target_risk","<f4"),("log_mu","<f8"),("risk_measure","u1"),("rng_stream","S32"),("rng_counter","<u8")],align=True)
TRANSITION_NUMERIC_DTYPE=np.dtype([("account_id","S32"),("decision_index","<u8"),("time_before_ns","<i8"),("time_after_ns","<i8"),("pre_hash","S32"),("post_hash","S32"),("permission_status","u1"),("permitted_direction","i1"),("permitted_risk","<f4"),("target_quantity","<f8"),("fees","<f8"),("funding","<f8"),("realized_pnl","<f8"),("unrealized_pnl_delta","<f8"),("liability_delta","<f8"),("post_equity","<f8"),("boundary","u1"),("mechanical_terminal","?")],align=True)

def _nonempty(value:Any,name:str)->str:
    if not isinstance(value,str) or not value.strip(): raise ValueError(f"{name}_EMPTY")
    return value
def _hex64(value:Any,name:str)->str:
    text=_nonempty(value,name)
    if re.fullmatch(r"[0-9a-f]{64}",text) is None: raise ValueError(f"{name}_INVALID_SHA256")
    return text
def _finite(value:Any,name:str)->float:
    v=float(value)
    if not math.isfinite(v): raise ValueError(f"{name}_NONFINITE")
    return v
def _risk(direction:str,risk:float)->float:
    r=_finite(risk,"target_risk")
    if r<0 or r>1: raise ValueError("TARGET_RISK_RANGE")
    if direction=="FLAT":
        if r!=0: raise ValueError("FLAT_RISK_MUST_BE_ZERO")
    elif r<=0 or r>=1: raise ValueError("NONFLAT_RISK_MUST_BE_INTERIOR")
    return r
def canonical_json(payload): return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def semantic_sha256(payload): return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
def fixed_id(text): _nonempty(text,"id"); return hashlib.sha256(text.encode("utf-8")).digest()
def fixed_hash(hex_value): return bytes.fromhex(_hex64(hex_value,"hash"))

@dataclass(frozen=True)
class FastPolicyDecision:
    science_semantic_version:str; account_lineage_id:str; decision_index:int; environment_time_ns:int; policy_generation:int; policy_id:str; policy_sha256:str; observation_schema:str; observation_hash:str; normalizer_id:str; nominal_direction:str; nominal_target_risk:float; log_mu:float; risk_measure_kind:str; rng_stream_id:str; rng_counter:int
    def validate(self):
        if self.science_semantic_version!=SCIENCE_SEMANTIC_VERSION: raise ValueError("SCIENCE_SEMANTIC_VERSION_MISMATCH")
        _nonempty(self.account_lineage_id,"account_lineage_id")
        if int(self.decision_index)<0: raise ValueError("DECISION_INDEX_NEGATIVE")
        if int(self.policy_generation)<0: raise ValueError("POLICY_GENERATION_NEGATIVE")
        _nonempty(self.policy_id,"policy_id"); _hex64(self.policy_sha256,"policy_sha256"); _nonempty(self.observation_schema,"observation_schema"); _hex64(self.observation_hash,"observation_hash"); _nonempty(self.normalizer_id,"normalizer_id")
        if self.nominal_direction not in DIRECTION_TO_CODE: raise ValueError("DIRECTION_INVALID")
        _risk(self.nominal_direction,self.nominal_target_risk); _finite(self.log_mu,"log_mu")
        if self.risk_measure_kind not in RISK_MEASURE_TO_CODE: raise ValueError("RISK_MEASURE_INVALID")
        if self.nominal_direction=="FLAT" and self.risk_measure_kind!="point_mass": raise ValueError("FLAT_REQUIRES_POINT_MASS")
        if self.nominal_direction!="FLAT" and self.risk_measure_kind!="continuous_density": raise ValueError("NONFLAT_REQUIRES_DENSITY")
        _nonempty(self.rng_stream_id,"rng_stream_id")
        if int(self.rng_counter)<0: raise ValueError("RNG_COUNTER_NEGATIVE")
        return self
    def to_record(self):
        self.validate(); rec=np.zeros((),dtype=POLICY_DECISION_DTYPE); rec["account_id"]=fixed_id(self.account_lineage_id); rec["decision_index"]=self.decision_index; rec["environment_time_ns"]=self.environment_time_ns; rec["policy_generation"]=self.policy_generation; rec["policy_hash"]=fixed_hash(self.policy_sha256); rec["observation_hash"]=fixed_hash(self.observation_hash); rec["direction"]=DIRECTION_TO_CODE[self.nominal_direction]; rec["target_risk"]=self.nominal_target_risk; rec["log_mu"]=self.log_mu; rec["risk_measure"]=RISK_MEASURE_TO_CODE[self.risk_measure_kind]; rec["rng_stream"]=fixed_id(self.rng_stream_id); rec["rng_counter"]=self.rng_counter; return rec

@dataclass(frozen=True)
class ExecutionLeg:
    sequence:int; quantity_delta:float; price:float; fee:float; kind:str
    def validate(self):
        if int(self.sequence)<0: raise ValueError("LEG_SEQUENCE_NEGATIVE")
        _finite(self.quantity_delta,"leg_quantity_delta"); p=_finite(self.price,"leg_price")
        if p<=0: raise ValueError("LEG_PRICE_NONPOSITIVE")
        f=_finite(self.fee,"leg_fee")
        if f<0: raise ValueError("LEG_FEE_NEGATIVE")
        if self.kind not in {"OPEN","REDUCE","CLOSE","REVERSAL_CLOSE","REVERSAL_OPEN","LIQUIDATION"}: raise ValueError("LEG_KIND_INVALID")
        return self

@dataclass(frozen=True)
class FastEnvironmentTransition:
    account_lineage_id:str; decision_index:int; environment_time_before_ns:int; environment_time_after_ns:int; pre_account_truth_hash:str; policy_decision_ref:str; permission_status:str; permission_reason:str; permitted_target_direction:str; permitted_target_risk:float; target_quantity:float; execution_legs:tuple[ExecutionLeg,...]; fees:float; funding:float; realized_pnl:float; unrealized_pnl_delta:float; liability_delta:float; post_account_truth_hash:str; post_equity:float; boundary_type:str; mechanical_terminal:bool; external_capital_flow_ref_or_null:str|None=None
    def validate(self):
        _nonempty(self.account_lineage_id,"account_lineage_id")
        if self.decision_index<0: raise ValueError("DECISION_INDEX_NEGATIVE")
        if self.environment_time_after_ns<=self.environment_time_before_ns: raise ValueError("ENVIRONMENT_TIME_NONMONOTONE")
        _hex64(self.pre_account_truth_hash,"pre_account_truth_hash"); _hex64(self.post_account_truth_hash,"post_account_truth_hash"); _nonempty(self.policy_decision_ref,"policy_decision_ref")
        if self.permission_status not in {"ALLOW","REJECT","NOOP"}: raise ValueError("PERMISSION_STATUS_INVALID")
        _nonempty(self.permission_reason,"permission_reason")
        if self.permitted_target_direction not in DIRECTION_TO_CODE: raise ValueError("PERMITTED_DIRECTION_INVALID")
        if self.permitted_target_direction=="FLAT":
            if float(self.permitted_target_risk)!=0: raise ValueError("PERMITTED_FLAT_RISK")
        else:
            pr=_finite(self.permitted_target_risk,"permitted_target_risk")
            if pr<0 or pr>1: raise ValueError("PERMITTED_RISK_RANGE")
        for name in ("target_quantity","fees","funding","realized_pnl","unrealized_pnl_delta","liability_delta","post_equity"): _finite(getattr(self,name),name)
        if self.fees<0: raise ValueError("FEES_NEGATIVE")
        legs=tuple(leg.validate() for leg in self.execution_legs)
        if tuple(leg.sequence for leg in legs)!=tuple(range(len(legs))): raise ValueError("EXECUTION_LEGS_NOT_ORDERED")
        if self.boundary_type not in BOUNDARY_TO_CODE: raise ValueError("BOUNDARY_INVALID")
        return self
    @property
    def content_sha256(self):
        self.validate(); payload=asdict(self); payload["execution_legs"]=[asdict(x) for x in self.execution_legs]; return semantic_sha256(payload)

@dataclass(frozen=True)
class FastExperienceSequence:
    sequence_id:str; account_lineage_id:str; science_semantic_version:str; market_lineage_id:str; source_classification:str; transition_refs:tuple[str,...]; first_decision_index:int; last_decision_index:int; behavior_policy_identities:tuple[str,...]; normalizer_identities:tuple[str,...]; chunk_boundary_type:str; bootstrap_state_ref_or_null:str|None; raw_fact_content_sha256:str
    def validate(self):
        _nonempty(self.sequence_id,"sequence_id"); _nonempty(self.account_lineage_id,"account_lineage_id")
        if self.science_semantic_version!=SCIENCE_SEMANTIC_VERSION: raise ValueError("SCIENCE_SEMANTIC_VERSION_MISMATCH")
        _nonempty(self.market_lineage_id,"market_lineage_id"); _nonempty(self.source_classification,"source_classification")
        if not self.transition_refs: raise ValueError("TRANSITION_REFS_EMPTY")
        if len(set(self.transition_refs))!=len(self.transition_refs): raise ValueError("TRANSITION_REFS_DUPLICATE")
        if self.first_decision_index<0 or self.last_decision_index<self.first_decision_index: raise ValueError("SEQUENCE_INDEX_RANGE")
        if self.chunk_boundary_type not in BOUNDARY_TO_CODE: raise ValueError("CHUNK_BOUNDARY_INVALID")
        _hex64(self.raw_fact_content_sha256,"raw_fact_content_sha256")
        if not self.behavior_policy_identities or not self.normalizer_identities: raise ValueError("SEQUENCE_IDENTITIES_EMPTY")
        return self
