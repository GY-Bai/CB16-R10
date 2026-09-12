from __future__ import annotations
from dataclasses import dataclass
ECONOMIC_TERMINAL="ECONOMIC_TERMINAL"
TRADING_DISABLED_PENDING_SETTLEMENT="TRADING_DISABLED_PENDING_SETTLEMENT"
OBJECTIVE_HORIZON_REACHED="OBJECTIVE_HORIZON_REACHED"
DATA_END_TRUNCATION="DATA_END_TRUNCATION"
COMPUTE_CHUNK="COMPUTE_CHUNK"
PAUSE="PAUSE"
PROCESS_FAILURE="PROCESS_FAILURE"
CONTINUE="CONTINUE"
BOUNDARIES={ECONOMIC_TERMINAL,TRADING_DISABLED_PENDING_SETTLEMENT,OBJECTIVE_HORIZON_REACHED,DATA_END_TRUNCATION,COMPUTE_CHUNK,PAUSE,PROCESS_FAILURE,CONTINUE}
@dataclass(frozen=True)
class CCRuntimeBoundaryR0:
    boundary_type:str
    mechanical_terminal:bool=False
    def validate(self):
        if self.boundary_type not in BOUNDARIES: raise RuntimeError("CCBOUNDARY_TYPE_INVALID")
        if self.mechanical_terminal and self.boundary_type!=ECONOMIC_TERMINAL: raise RuntimeError("CCBOUNDARY_TERMINAL_MISMATCH")
    @property
    def preserves_physical_account(self)->bool:
        self.validate(); return self.boundary_type!=ECONOMIC_TERMINAL
