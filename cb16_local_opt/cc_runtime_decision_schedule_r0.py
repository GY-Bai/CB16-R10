from __future__ import annotations
from dataclasses import dataclass, replace
@dataclass(frozen=True)
class CCDecisionScheduleR0:
    every_n_environment_steps:int=1
    offset:int=0
    last_decision_environment_time:int|None=None
    last_decision_index:int|None=None
    def validate(self):
        if self.every_n_environment_steps<=0 or self.offset<0: raise RuntimeError("CCSCHED_CONFIG_INVALID")
    def is_decision_time(self,environment_time:int)->bool:
        self.validate(); return environment_time>=self.offset and (environment_time-self.offset)%self.every_n_environment_steps==0
    def record(self,*,environment_time:int,decision_index:int):
        self.validate()
        if not self.is_decision_time(environment_time): raise RuntimeError("CCSCHED_UNSCHEDULED_DECISION")
        expected=0 if self.last_decision_index is None else self.last_decision_index+1
        if decision_index!=expected: raise RuntimeError("CCSCHED_DECISION_INDEX_GAP_OR_REORDER")
        if self.last_decision_environment_time is not None and environment_time<=self.last_decision_environment_time: raise RuntimeError("CCSCHED_TIME_DUPLICATE_OR_REORDER")
        return replace(self,last_decision_environment_time=environment_time,last_decision_index=decision_index)
