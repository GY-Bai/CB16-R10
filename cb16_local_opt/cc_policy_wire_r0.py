from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import math

DIRECTIONS = ("SHORT", "FLAT", "LONG")
RISK_MEASURES = ("point_mass", "continuous_density")
COMMIT_STATES = ("PREPARED", "COMMITTED", "ABORTED")


def _nonempty(v: str, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name}_EMPTY")
    return v


def _finite(v: float, name: str) -> float:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"{name}_NONFINITE")
    return v

@dataclass(frozen=True)
class CCPolicyDecisionV1:
    science_semantic_version: str
    account_lineage_id: str
    decision_index: int
    environment_time: str
    policy_generation: int
    policy_id: str
    policy_sha256: str
    observation_schema: str
    observation_hash: str
    normalizer_id: str
    nominal_direction: str
    nominal_target_risk: float
    log_mu: float
    risk_measure_kind: str
    rng_stream_id: str
    rng_position_or_counter: int
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("science_semantic_version", "account_lineage_id", "environment_time", "policy_id",
                     "policy_sha256", "observation_schema", "observation_hash", "normalizer_id", "rng_stream_id"):
            _nonempty(getattr(self, name), name)
        if self.decision_index < 0 or self.policy_generation < 0 or self.rng_position_or_counter < 0:
            raise ValueError("NEGATIVE_COUNTER")
        if self.nominal_direction not in DIRECTIONS:
            raise ValueError("BAD_DIRECTION")
        risk = _finite(self.nominal_target_risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("RISK_OUT_OF_RANGE")
        if self.nominal_direction == "FLAT":
            if risk != 0.0 or self.risk_measure_kind != "point_mass":
                raise ValueError("FLAT_MUST_BE_ZERO_POINT_MASS")
        else:
            if not 0.0 < risk < 1.0 or self.risk_measure_kind != "continuous_density":
                raise ValueError("NONFLAT_RISK_MUST_BE_INTERIOR_DENSITY")
        _finite(self.log_mu, "log_mu")

@dataclass(frozen=True)
class CCExperienceSequenceV1:
    sequence_id: str
    account_lineage_id: str
    science_semantic_version: str
    market_lineage_id: str
    source_classification: str
    transition_refs: tuple[str, ...]
    first_decision_index: int
    last_decision_index: int
    behavior_policy_identities: tuple[str, ...]
    normalizer_identities: tuple[str, ...]
    chunk_boundary_type: str
    bootstrap_state_ref_or_null: str | None
    raw_fact_content_sha256: str
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("sequence_id", "account_lineage_id", "science_semantic_version", "market_lineage_id",
                     "source_classification", "chunk_boundary_type", "raw_fact_content_sha256"):
            _nonempty(getattr(self, name), name)
        if not self.transition_refs:
            raise ValueError("EMPTY_SEQUENCE")
        if self.first_decision_index < 0 or self.last_decision_index < self.first_decision_index:
            raise ValueError("BAD_DECISION_RANGE")
        if not self.behavior_policy_identities or not self.normalizer_identities:
            raise ValueError("MISSING_IDENTITY")

@dataclass(frozen=True)
class CCLearningUpdateV1:
    update_id: str
    parent_checkpoint_sha256: str
    science_semantic_version: str
    sampled_sequence_ids: tuple[str, ...]
    sampling_probabilities_or_weights: tuple[float, ...]
    behavior_support_summary: Mapping[str, Any]
    actor_loss: float
    critic_loss: float
    vtrace_summary: Mapping[str, Any]
    gradient_ownership_summary: Mapping[str, Any]
    optimizer_step_before: int
    optimizer_step_after: int
    child_checkpoint_sha256: str
    commit_status: str

    def __post_init__(self) -> None:
        for name in ("update_id", "parent_checkpoint_sha256", "science_semantic_version", "child_checkpoint_sha256"):
            _nonempty(getattr(self, name), name)
        if self.commit_status not in COMMIT_STATES:
            raise ValueError("BAD_COMMIT_STATUS")
        if len(self.sampled_sequence_ids) != len(self.sampling_probabilities_or_weights):
            raise ValueError("SAMPLING_CARDINALITY_MISMATCH")
        if any((not math.isfinite(float(x)) or float(x) < 0) for x in self.sampling_probabilities_or_weights):
            raise ValueError("BAD_SAMPLING_WEIGHT")
        if self.optimizer_step_after < self.optimizer_step_before:
            raise ValueError("OPTIMIZER_STEP_REWIND")
        _finite(self.actor_loss, "actor_loss"); _finite(self.critic_loss, "critic_loss")
