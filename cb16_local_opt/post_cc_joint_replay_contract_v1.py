from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

from .post_cc_observation_contract_v1 import PostCCObservationFactV1

JOINT_REPLAY_CONTRACT_ID = "PostCCJointReplaySampleV1"
_VALID_DIRECTIONS = {"SHORT", "FLAT", "LONG"}


@dataclass(frozen=True)
class PostCCJointReplaySampleV1:
    sequence_id: str
    transition_ref: str
    account_lineage_id: str
    decision_index: int
    environment_time: str
    observation: PostCCObservationFactV1
    nominal_direction: str
    nominal_target_risk: float
    risk_measure_kind: str
    behavior_log_mu: float
    behavior_policy_identity: str
    behavior_generation: str
    reward: float
    discount: float
    boundary_type: str
    bootstrap_state_ref_or_null: str | None
    sampling_probability_or_weight: float
    target_policy_identity: str
    source_fact_hashes: tuple[str, ...]
    consequence_context: Mapping[str, Any] | None = None

    def validate(self) -> "PostCCJointReplaySampleV1":
        for name in (
            "sequence_id",
            "transition_ref",
            "account_lineage_id",
            "environment_time",
            "risk_measure_kind",
            "behavior_policy_identity",
            "behavior_generation",
            "boundary_type",
            "target_policy_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        self.observation.validate()
        if self.observation.account_lineage_id != self.account_lineage_id:
            raise ValueError("ACCOUNT_LINEAGE_MISMATCH")
        if self.observation.decision_index != self.decision_index:
            raise ValueError("DECISION_INDEX_MISMATCH")
        if self.observation.environment_time != self.environment_time:
            raise ValueError("ENVIRONMENT_TIME_MISMATCH")
        if self.nominal_direction not in _VALID_DIRECTIONS:
            raise ValueError("invalid nominal_direction")
        if not 0.0 <= float(self.nominal_target_risk) <= 1.0:
            raise ValueError("nominal_target_risk outside [0,1]")
        if self.nominal_direction == "FLAT" and float(self.nominal_target_risk) != 0.0:
            raise ValueError("FLAT nominal_target_risk must be 0")
        for name in (
            "behavior_log_mu",
            "reward",
            "discount",
            "sampling_probability_or_weight",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.discount) < 0.0:
            raise ValueError("discount must be >= 0")
        if float(self.sampling_probability_or_weight) <= 0.0:
            raise ValueError("sampling probability/weight must be > 0")
        if not self.source_fact_hashes or any(
            not isinstance(value, str) or not value for value in self.source_fact_hashes
        ):
            raise ValueError("source_fact_hashes required")
        return self
