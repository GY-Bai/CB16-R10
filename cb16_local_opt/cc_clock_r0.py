from __future__ import annotations
from dataclasses import dataclass, replace
CC_FOUR_CLOCK_SCHEMA_R0="CB16_R11_CC_FOUR_CLOCK_V1_R0"
@dataclass(frozen=True)
class CCFourClockR0:
    environment_time:int
    policy_decision_index:int
    compute_chunk_index:int
    objective_horizon_id:str
    def validate(self):
        if any(isinstance(v,bool) or v<0 for v in (self.environment_time,self.policy_decision_index,self.compute_chunk_index)): raise RuntimeError("CCCLOCK_INDEX_INVALID")
        if not self.objective_horizon_id: raise RuntimeError("CCCLOCK_HORIZON_ID_INVALID")
    def advance_environment(self,steps:int=1):
        if steps<=0: raise RuntimeError("CCCLOCK_ENV_STEP_INVALID")
        self.validate(); return replace(self,environment_time=self.environment_time+steps)
    def advance_decision(self): self.validate(); return replace(self,policy_decision_index=self.policy_decision_index+1)
    def next_chunk(self): self.validate(); return replace(self,compute_chunk_index=self.compute_chunk_index+1)
    def with_horizon(self,horizon_id:str):
        if not horizon_id: raise RuntimeError("CCCLOCK_HORIZON_ID_INVALID")
        self.validate(); return replace(self,objective_horizon_id=horizon_id)
