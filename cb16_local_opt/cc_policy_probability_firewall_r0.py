from __future__ import annotations
import math
from .cc_policy_wire_r0 import CCPolicyDecisionV1

def behavior_probability_input(decision: CCPolicyDecisionV1, executed_direction:str|None=None, executed_risk:float|None=None) -> tuple[str,float,float]:
    # executed fields are deliberately ignored for the probability term.
    return decision.nominal_direction, decision.nominal_target_risk, decision.log_mu

def importance_ratio(log_pi: float, decision: CCPolicyDecisionV1) -> float:
    x=float(log_pi)-float(decision.log_mu)
    if not math.isfinite(x): raise ValueError("NONFINITE_LOG_RATIO")
    return math.exp(max(-80.0,min(80.0,x)))
