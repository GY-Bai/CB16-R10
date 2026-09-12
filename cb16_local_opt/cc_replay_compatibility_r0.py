from dataclasses import dataclass
@dataclass(frozen=True)
class ReplaySemanticIdentity:
    science_semantic_version:str; observation_identity:str; normalizer_identity:str; action_distribution_semantics:str; execution_environment_semantics:str; boundary_bootstrap_semantics:str
@dataclass(frozen=True)
class CompatibilityResult: compatible:bool; reason_codes:tuple[str,...]
def check_replay_compatibility(stored:ReplaySemanticIdentity,required:ReplaySemanticIdentity,*,true_behavior_likelihood_available:bool,boundary_interpretable:bool=True,record_age_days:int|None=None)->CompatibilityResult:
    reasons=[f"MISMATCH_{f.upper()}" for f in stored.__dataclass_fields__ if getattr(stored,f)!=getattr(required,f)]
    if not true_behavior_likelihood_available: reasons.append("TRUE_BEHAVIOR_LIKELIHOOD_UNAVAILABLE")
    if not boundary_interpretable: reasons.append("BOUNDARY_BOOTSTRAP_UNINTERPRETABLE")
    return CompatibilityResult(not reasons,tuple(reasons))
