from __future__ import annotations
TERMINAL={"ECONOMIC_TERMINAL","MECHANICAL_TERMINAL","AUTHORIZED_OBJECTIVE_END"}
BOOTSTRAP={"COMPUTE_CHUNK","PAUSE","UNKNOWN_DATA_END","PENDING_SETTLEMENT"}
def bootstrap_value(boundary_type:str,next_value:float)->float:
    if boundary_type in TERMINAL:return 0.0
    if boundary_type in BOOTSTRAP:return float(next_value)
    raise ValueError("UNKNOWN_BOUNDARY_TYPE")
