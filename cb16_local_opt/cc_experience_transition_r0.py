from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
from math import isfinite
from typing import Any,Mapping
from .cc_experience_wire_r0 import validate_w02
class TransitionValidationError(ValueError): pass
@dataclass(frozen=True)
class ExecutionLeg:
    leg_index:int; quantity:float; price:float|None; status:str; cost:float=0.0
    def __post_init__(self):
        if self.leg_index<0 or not isfinite(float(self.quantity)) or not isfinite(float(self.cost)) or not self.status: raise TransitionValidationError("invalid leg")
        if self.price is not None and not isfinite(float(self.price)): raise TransitionValidationError("invalid price")
@dataclass(frozen=True)
class CCExperienceTransitionR0:
    transition_id:str; account_lineage_id:str; generation_id:str; policy_id:str; science_semantic_id:str
    observation_id:str; observation_hash:str; nominal_action:str; log_mu:float|None; behavior_likelihood_provenance:str|None
    rng_identity:str; rng_state_in_hash:str; rng_state_out_hash:str; permission:bool; requested_risk:float; target_quantity:float; feasibility:str
    execution_legs:tuple[ExecutionLeg,...]; pre_account_truth_hash:str; post_account_truth_hash:str
    equity_delta:float; cost_delta:float; liability_delta:float; environment_clock:str; decision_clock:str; boundary:str
    market_lineage:str; source_classification:str; failure_classification:str|None; w02:Mapping[str,Any]
    def __post_init__(self):
        strings=(self.transition_id,self.account_lineage_id,self.generation_id,self.policy_id,self.science_semantic_id,self.observation_id,self.observation_hash,self.nominal_action,self.rng_identity,self.rng_state_in_hash,self.rng_state_out_hash,self.pre_account_truth_hash,self.post_account_truth_hash,self.environment_clock,self.decision_clock,self.boundary,self.market_lineage,self.source_classification,self.feasibility)
        if any(not isinstance(v,str) or not v for v in strings): raise TransitionValidationError("identity/provenance fields required")
        if self.log_mu is not None and not isfinite(float(self.log_mu)): raise TransitionValidationError("log_mu must be finite")
        if not 0<=float(self.requested_risk)<=1: raise TransitionValidationError("requested_risk outside [0,1]")
        if any(not isfinite(float(v)) for v in (self.target_quantity,self.equity_delta,self.cost_delta,self.liability_delta)): raise TransitionValidationError("nonfinite numeric")
        if tuple(x.leg_index for x in self.execution_legs)!=tuple(range(len(self.execution_legs))): raise TransitionValidationError("execution legs unordered")
        w=validate_w02(self.w02)
        if w["environment_transition_id"]!=self.transition_id or w["account_lineage_id"]!=self.account_lineage_id or w["generation_id"]!=self.generation_id: raise TransitionValidationError("W-02 identity mismatch")
        if w["nominal_action"]!=self.nominal_action or w["pre_account_truth_hash"]!=self.pre_account_truth_hash or w["post_account_truth_hash"]!=self.post_account_truth_hash: raise TransitionValidationError("W-02 semantic mismatch")
        if w["rng_state_in_hash"]!=self.rng_state_in_hash or w["rng_state_out_hash"]!=self.rng_state_out_hash: raise TransitionValidationError("W-02 RNG mismatch")
        object.__setattr__(self,"w02",dict(w))
    @property
    def replay_has_true_behavior_likelihood(self): return self.log_mu is not None and bool(self.behavior_likelihood_provenance)
    @property
    def content_hash(self): return hashlib.sha256(json.dumps(asdict(self),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    def to_dict(self): return asdict(self)
