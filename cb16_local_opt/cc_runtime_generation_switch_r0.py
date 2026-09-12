from __future__ import annotations
from dataclasses import dataclass
from .cc_environment_lifecycle_r0 import POST_STATE_PUBLISHED
@dataclass(frozen=True)
class CCGenerationSwitchR0:
    environment_time:int
    decision_index:int
    old_policy_id:str
    old_policy_sha256:str
    new_policy_generation:str
    new_policy_id:str
    new_policy_sha256:str
def switch_generation_r0(rt,*,new_policy_generation:str,new_policy_id:str,new_policy_sha256:str)->CCGenerationSwitchR0:
    if rt.phase!=POST_STATE_PUBLISHED: raise RuntimeError("CCGEN_SWITCH_REQUIRES_BOUNDARY")
    if not new_policy_generation or not new_policy_id or len(new_policy_sha256)!=64: raise RuntimeError("CCGEN_IDENTITY_INVALID")
    rec=CCGenerationSwitchR0(rt.clocks.environment_time,rt.clocks.policy_decision_index,rt.policy_id,rt.policy_sha256,new_policy_generation,new_policy_id,new_policy_sha256)
    rt.policy_generation=new_policy_generation; rt.policy_id=new_policy_id; rt.policy_sha256=new_policy_sha256
    return rec
