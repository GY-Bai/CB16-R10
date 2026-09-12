from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping

REQUIRED=("joint_probability","rng_recovery","actor_critic_firewall","reward_telescoping","vtrace_math",
          "gradient_ownership","exactly_once_update","checkpoint_recovery","retention","account_toy",
          "delayed_toy","high_risk_toy","offpolicy_toy","final_firewall","fresh_data_firewall")
@dataclass(frozen=True)
class Qualification:
    status:str; strongest_evidence:str; checks:Mapping[str,bool]; claim_scope:str

def compile_qualification(checks:Mapping[str,bool])->Qualification:
    missing=[k for k in REQUIRED if k not in checks]
    failed=[k for k in REQUIRED if not checks.get(k,False)]
    if missing or failed:return Qualification("FAIL","COMPONENT",dict(checks),"SYNTHETIC_ONLY_NO_REAL_MARKET_EDGE")
    return Qualification("PASS","KNOWN_ANSWER",dict(checks),"SYNTHETIC_ONLY_NO_REAL_MARKET_EDGE")

def enforce_data_firewall(*, uses_final:bool, uses_fresh_market_data:bool)->None:
    if uses_final: raise RuntimeError("FINAL_HOLDOUT_SEALED")
    if uses_fresh_market_data: raise RuntimeError("FRESH_MARKET_DATA_FORBIDDEN")
