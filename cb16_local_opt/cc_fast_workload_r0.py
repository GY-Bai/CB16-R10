from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .cc_fast_wire_r0 import SCIENCE_SEMANTIC_VERSION,semantic_sha256
WORKLOAD_VERSION="CB16_R11_CC_FAST_SYNTHETIC_WORKLOAD_V1"
@dataclass(frozen=True)
class WorkloadConfig:
    seed:int=230912; market_steps:int=4096; account_count:int=32; decision_interval_min:int=1; decision_interval_max:int=4; initial_equity:float=10000.0
    def validate(self):
        if self.market_steps<64: raise ValueError("MARKET_STEPS_TOO_SMALL")
        if self.account_count<=0: raise ValueError("ACCOUNT_COUNT_INVALID")
        if self.decision_interval_min<=0 or self.decision_interval_max<self.decision_interval_min: raise ValueError("DECISION_INTERVAL_INVALID")
        if self.initial_equity<=0: raise ValueError("INITIAL_EQUITY_INVALID")
        return self
@dataclass(frozen=True)
class SyntheticWorkload:
    config:WorkloadConfig; market:np.ndarray; funding:np.ndarray; decision_intervals:np.ndarray; terminal_step:np.ndarray; lineage_ids:tuple[str,...]; workload_id:str; market_lineage_id:str
    @property
    def accesses_final_or_fresh_data(self): return False
    def ready_accounts(self,step:int):
        idx=np.arange(self.config.account_count,dtype=np.int64); return idx[(step%self.decision_intervals)==0]
def build_workload(config:WorkloadConfig=WorkloadConfig()):
    config.validate(); rng=np.random.default_rng(config.seed); shocks=rng.normal(0.0,0.0015,size=config.market_steps); regime=np.where((np.arange(config.market_steps)//257)%2==0,1.0,2.5); returns=shocks*regime+0.00005*np.sin(np.arange(config.market_steps)/17.0); market=np.ascontiguousarray(100.0*np.exp(np.cumsum(returns)),dtype=np.float64); funding=np.ascontiguousarray(0.0000025*np.sin(np.arange(config.market_steps)/11.0),dtype=np.float64); decision_intervals=rng.integers(config.decision_interval_min,config.decision_interval_max+1,size=config.account_count,dtype=np.int16); terminal_step=np.full(config.account_count,config.market_steps-1,dtype=np.int64)
    if config.account_count>=4: terminal_step[0]=config.market_steps//3; terminal_step[1]=config.market_steps//2
    lineage_ids=tuple(f"cc-d-acct-{i:04d}" for i in range(config.account_count)); market_lineage_id=semantic_sha256({"source":"SYNTHETIC_ONLY","seed":config.seed,"steps":config.market_steps,"market_sha":hashlib.sha256(market.tobytes()).hexdigest()}); workload_id=semantic_sha256({"version":WORKLOAD_VERSION,"config":config.__dict__,"market_lineage_id":market_lineage_id,"science":SCIENCE_SEMANTIC_VERSION})
    market.setflags(write=False); funding.setflags(write=False); decision_intervals.setflags(write=False); terminal_step.setflags(write=False)
    return SyntheticWorkload(config,market,funding,decision_intervals,terminal_step,lineage_ids,workload_id,market_lineage_id)
