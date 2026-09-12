from __future__ import annotations
from dataclasses import dataclass
import hashlib,math,os
from collections import defaultdict
from typing import Iterable,Sequence
import numpy as np
from .cc_fast_wire_r0 import FLAT,LONG,SHORT,CODE_TO_DIRECTION,FastPolicyDecision,SCIENCE_SEMANTIC_VERSION

BROKER_SCHEMA="CB16_R11_CC_FAST_POLICY_BROKER_V1"; REQUEST_SCHEMA="CB16_R11_CC_FAST_POLICY_REQUEST_V1"; RESPONSE_SCHEMA="CB16_R11_CC_FAST_POLICY_RESPONSE_V1"; OBSERVATION_SCHEMA="CB16_R11_CC_FAST_SYNTHETIC_OBSERVATION_V1"

@dataclass(frozen=True)
class PolicySpec:
    policy_generation:int; policy_id:str; policy_sha256:str; direction_logits:tuple[float,float,float]; risk_loc_short:float=-0.25; risk_scale_short:float=0.75; risk_loc_long:float=0.25; risk_scale_long:float=0.75
    def validate(self):
        if self.policy_generation<0 or not self.policy_id: raise ValueError("POLICY_SPEC_IDENTITY_INVALID")
        if len(self.policy_sha256)!=64: raise ValueError("POLICY_SPEC_HASH_INVALID")
        logits=np.asarray(self.direction_logits,dtype=np.float64)
        if logits.shape!=(3,) or not np.all(np.isfinite(logits)): raise ValueError("POLICY_SPEC_LOGITS_INVALID")
        for scale in (self.risk_scale_short,self.risk_scale_long):
            if not math.isfinite(scale) or scale<=0: raise ValueError("POLICY_SPEC_RISK_SCALE_INVALID")
        return self
    @property
    def batching_identity(self): return (self.policy_generation,self.policy_sha256)

@dataclass(frozen=True)
class PolicyRequest:
    account_lineage_id:str; decision_index:int; environment_time_ns:int; policy_generation:int; policy_sha256:str; observation:tuple[float,...]; observation_hash:str; normalizer_id:str; rng_stream_id:str; rng_counter:int; stochastic:bool=True
    def validate(self):
        if not self.account_lineage_id or self.decision_index<0: raise ValueError("REQUEST_IDENTITY_INVALID")
        if self.policy_generation<0 or len(self.policy_sha256)!=64: raise ValueError("REQUEST_POLICY_IDENTITY_INVALID")
        if not self.observation or not np.all(np.isfinite(np.asarray(self.observation,dtype=np.float64))): raise ValueError("REQUEST_OBSERVATION_INVALID")
        if len(self.observation_hash)!=64 or not self.normalizer_id: raise ValueError("REQUEST_OBSERVATION_IDENTITY_INVALID")
        if not self.rng_stream_id or self.rng_counter<0: raise ValueError("REQUEST_RNG_INVALID")
        if not self.stochastic: raise ValueError("CC_FAST_BROKER_REQUIRES_STOCHASTIC_MODE")
        return self

@dataclass(frozen=True)
class PolicyResponse:
    account_lineage_id:str; decision_index:int; policy_generation:int; policy_id:str; policy_sha256:str; nominal_direction:str; nominal_target_risk:float; log_mu:float; risk_measure_kind:str; rng_stream_id:str; rng_counter:int
    def to_wire(self,request:PolicyRequest):
        if request.account_lineage_id!=self.account_lineage_id or request.decision_index!=self.decision_index: raise ValueError("RESPONSE_REQUEST_IDENTITY_MISMATCH")
        return FastPolicyDecision(science_semantic_version=SCIENCE_SEMANTIC_VERSION,account_lineage_id=self.account_lineage_id,decision_index=self.decision_index,environment_time_ns=request.environment_time_ns,policy_generation=self.policy_generation,policy_id=self.policy_id,policy_sha256=self.policy_sha256,observation_schema=OBSERVATION_SCHEMA,observation_hash=request.observation_hash,normalizer_id=request.normalizer_id,nominal_direction=self.nominal_direction,nominal_target_risk=self.nominal_target_risk,log_mu=self.log_mu,risk_measure_kind=self.risk_measure_kind,rng_stream_id=self.rng_stream_id,rng_counter=self.rng_counter).validate()

def _hash_block(stream,counter,lane):
    if counter<0 or lane<0: raise ValueError("RNG_COUNTER_INVALID")
    return hashlib.sha256(f"{stream}\0{counter}\0{lane}".encode()).digest()
def counter_uniform(stream,counter,lane):
    raw=int.from_bytes(_hash_block(stream,counter,lane)[:8],"big"); return (raw+0.5)/(2**64)
def counter_standard_normal(stream,counter,lane=1):
    u1=counter_uniform(stream,counter,lane); u2=counter_uniform(stream,counter,lane+1); return math.sqrt(-2.0*math.log(u1))*math.cos(2.0*math.pi*u2)
def _log_softmax(logits):
    x=np.asarray(logits,dtype=np.float64); m=float(np.max(x)); z=m+math.log(float(np.exp(x-m).sum())); return x-z
def _categorical_sample(log_probs,u):
    cdf=np.cumsum(np.exp(log_probs)); return min(int(np.searchsorted(cdf,u,side="right")),len(cdf)-1)
def _sigmoid(z):
    if z>=0:
        e=math.exp(-z); return 1.0/(1.0+e)
    e=math.exp(z); return e/(1.0+e)
def _squashed_normal_sample_and_log_density(*,loc,scale,stream,counter):
    eps=counter_standard_normal(stream,counter,lane=11); z=loc+scale*eps; risk=_sigmoid(z); log_normal=-0.5*eps*eps-math.log(scale)-0.5*math.log(2.0*math.pi); log_jac=math.log(risk)+math.log1p(-risk); return risk,log_normal-log_jac

class BatchedPolicyBroker:
    def __init__(self,policies:Iterable[PolicySpec],*,execution_mode="CPU"):
        if execution_mode not in {"CPU","CUDA_SINGLE_OWNER"}: raise ValueError("BROKER_EXECUTION_MODE_INVALID")
        self.execution_mode=execution_mode; self.owner_pid=os.getpid(); specs=[p.validate() for p in policies]; self._policies={p.batching_identity:p for p in specs}
        if len(self._policies)!=len(specs): raise ValueError("DUPLICATE_POLICY_BATCHING_IDENTITY")
        self.batch_sizes=[]; self.cuda_initialized_by_worker=False
    def _assert_owner(self):
        if os.getpid()!=self.owner_pid: raise RuntimeError("POLICY_BROKER_OWNER_PROCESS_VIOLATION")
    def _policy_for(self,request):
        request.validate(); key=(request.policy_generation,request.policy_sha256)
        try:return self._policies[key]
        except KeyError as exc: raise RuntimeError("STALE_OR_UNKNOWN_POLICY_GENERATION") from exc
    def infer(self,requests:Sequence[PolicyRequest]):
        self._assert_owner()
        if not requests:return []
        groups=defaultdict(list)
        for i,req in enumerate(requests): groups[self._policy_for(req).batching_identity].append((i,req))
        out=[None]*len(requests)
        for key,indexed in groups.items():
            spec=self._policies[key]; self.batch_sizes.append(len(indexed)); log_dir=_log_softmax(spec.direction_logits)
            for i,req in indexed:
                direction_index=_categorical_sample(log_dir,counter_uniform(req.rng_stream_id,req.rng_counter,0)); direction=CODE_TO_DIRECTION[(SHORT,FLAT,LONG)[direction_index]]; log_mu=float(log_dir[direction_index])
                if direction=="FLAT": risk=0.0; measure="point_mass"
                else:
                    loc=spec.risk_loc_short if direction=="SHORT" else spec.risk_loc_long; scale=spec.risk_scale_short if direction=="SHORT" else spec.risk_scale_long; risk,log_density=_squashed_normal_sample_and_log_density(loc=loc,scale=scale,stream=req.rng_stream_id,counter=req.rng_counter); log_mu+=log_density; measure="continuous_density"
                out[i]=PolicyResponse(req.account_lineage_id,req.decision_index,spec.policy_generation,spec.policy_id,spec.policy_sha256,direction,risk,log_mu,measure,req.rng_stream_id,req.rng_counter)
        return [x for x in out if x is not None]

def worker_cuda_detector(): return False
