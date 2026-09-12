from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ReplaySemanticIdentity:
    science_semantic_version: str
    observation_schema: str
    normalizer_id: str
    action_semantics: str
    execution_semantics: str
    environment_semantics: str
    boundary_bootstrap_semantics: str
    true_behavior_likelihood_available: bool

@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason_codes: tuple[str, ...]


def check_replay_compatibility(record: ReplaySemanticIdentity, learner: ReplaySemanticIdentity) -> CompatibilityResult:
    reasons = []
    if not record.true_behavior_likelihood_available: reasons.append("MISSING_TRUE_BEHAVIOR_LIKELIHOOD")
    for field, code in (
        ("science_semantic_version", "SCIENCE_SEMANTIC_MISMATCH"),
        ("observation_schema", "OBSERVATION_SCHEMA_MISMATCH"),
        ("normalizer_id", "NORMALIZER_MISMATCH"),
        ("action_semantics", "ACTION_SEMANTICS_MISMATCH"),
        ("execution_semantics", "EXECUTION_SEMANTICS_MISMATCH"),
        ("environment_semantics", "ENVIRONMENT_SEMANTICS_MISMATCH"),
        ("boundary_bootstrap_semantics", "BOUNDARY_BOOTSTRAP_MISMATCH"),
    ):
        if getattr(record, field) != getattr(learner, field): reasons.append(code)
    return CompatibilityResult(not reasons, tuple(reasons))
