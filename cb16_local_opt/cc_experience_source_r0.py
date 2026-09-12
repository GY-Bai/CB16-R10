from dataclasses import dataclass
from enum import Enum
class ExperienceSource(str,Enum):
    CC_STOCHASTIC_TRAJECTORY="cc_stochastic_trajectory"; BC_R1_COMPONENT_FIXTURE="bc_r1_component_fixture"; LEGACY_H72_TEACHER_DEMONSTRATION="legacy_h72_teacher_demonstration"; DIAGNOSTIC_UNKNOWN="diagnostic_unknown"
@dataclass(frozen=True)
class SourceAssessment:
    source:ExperienceSource; readable_as_fact:bool; demonstration_eligible:bool; diagnostic_eligible:bool; vtrace_replay_eligible:bool; reason:str
def assess_source(source:ExperienceSource,*,true_behavior_likelihood_available:bool)->SourceAssessment:
    demo=source in {ExperienceSource.BC_R1_COMPONENT_FIXTURE,ExperienceSource.LEGACY_H72_TEACHER_DEMONSTRATION}
    if not true_behavior_likelihood_available:return SourceAssessment(source,True,demo,True,False,"TRUE_BEHAVIOR_LIKELIHOOD_UNAVAILABLE")
    if source is ExperienceSource.DIAGNOSTIC_UNKNOWN:return SourceAssessment(source,True,False,True,False,"UNKNOWN_SOURCE_SEMANTICS")
    if source is ExperienceSource.LEGACY_H72_TEACHER_DEMONSTRATION:return SourceAssessment(source,True,True,True,False,"LEGACY_DEMONSTRATION_NOT_CC_BEHAVIOR_POLICY")
    return SourceAssessment(source,True,demo,True,True,"COMPATIBLE_SOURCE_WITH_TRUE_BEHAVIOR_LIKELIHOOD")
