from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class SourceClass(str, Enum):
    CC_STOCHASTIC_TRAJECTORY = "CC_STOCHASTIC_TRAJECTORY"
    BC_R1_COMPONENT_FIXTURE = "BC_R1_COMPONENT_FIXTURE"
    LEGACY_H72_TEACHER_DEMONSTRATION = "LEGACY_H72_TEACHER_DEMONSTRATION"
    DIAGNOSTIC_UNKNOWN = "DIAGNOSTIC_UNKNOWN"

@dataclass(frozen=True)
class SourceEligibility:
    source: SourceClass
    true_behavior_likelihood_available: bool
    replay_eligible: bool
    reason: str


def classify(source: SourceClass, *, true_log_mu_available: bool) -> SourceEligibility:
    if source is SourceClass.CC_STOCHASTIC_TRAJECTORY and true_log_mu_available:
        return SourceEligibility(source, True, True, "TRUE_BEHAVIOR_LIKELIHOOD_AVAILABLE")
    if not true_log_mu_available:
        return SourceEligibility(source, False, False, "MISSING_TRUE_BEHAVIOR_LIKELIHOOD")
    return SourceEligibility(source, True, False, "SOURCE_NOT_ADMITTED_FOR_CC_VTRACE_REPLAY")
