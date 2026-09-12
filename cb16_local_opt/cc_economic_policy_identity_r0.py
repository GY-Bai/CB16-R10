from dataclasses import dataclass
class PolicyIdentityError(ValueError): pass
@dataclass(frozen=True)
class EconomicPolicyIdentity:
    policy_object_type:str; policy_identity:str; ordered_generation_policy_refs:tuple[tuple[str,str],...]=()
    def __post_init__(self):
        if self.policy_object_type not in {"frozen_checkpoint","generation_chain","baseline"} or not self.policy_identity: raise PolicyIdentityError("invalid policy identity")
        if self.policy_object_type=="frozen_checkpoint" and len(tuple(dict.fromkeys(self.ordered_generation_policy_refs)))>1: raise PolicyIdentityError("mixed generation cannot be final checkpoint")
def identity_from_ownership(ownership):
    unique=tuple(dict.fromkeys(tuple(ownership)))
    if not unique: raise PolicyIdentityError("ownership required")
    if len(unique)==1:return EconomicPolicyIdentity("frozen_checkpoint",f"{unique[0][0]}:{unique[0][1]}",unique)
    return EconomicPolicyIdentity("generation_chain","->".join(f"{g}:{p}" for g,p in unique),unique)
