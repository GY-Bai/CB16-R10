from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Any
from .cc_experience_wire_r0 import CCEnvironmentTransitionV1, content_sha256

ALLOWED_DIRECTIONS = {"SHORT", "FLAT", "LONG"}

@dataclass(frozen=True)
class ImmutableExperienceTransitionV1:
    transition_id: str
    environment: CCEnvironmentTransitionV1
    science_semantic_version: str
    policy_generation: str
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
    rng_position_or_counter: str
    market_lineage_id: str
    source_classification: str
    failure_classification: str

    def validate(self) -> "ImmutableExperienceTransitionV1":
        self.environment.validate()
        for name in ("transition_id", "science_semantic_version", "policy_generation", "policy_id", "policy_sha256",
                     "observation_schema", "observation_hash", "normalizer_id", "rng_stream_id",
                     "rng_position_or_counter", "market_lineage_id", "source_classification", "failure_classification"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.nominal_direction not in ALLOWED_DIRECTIONS:
            raise ValueError("invalid nominal_direction")
        if not 0 <= self.nominal_target_risk <= 1:
            raise ValueError("nominal_target_risk outside [0,1]")
        if self.nominal_direction == "FLAT" and self.nominal_target_risk != 0:
            raise ValueError("FLAT nominal_target_risk must be 0")
        if not math.isfinite(self.log_mu):
            raise ValueError("true behavior log_mu must be finite")
        if self.risk_measure_kind not in {"point_mass", "continuous_density"}:
            raise ValueError("invalid risk_measure_kind")
        return self

    @property
    def content_sha256(self) -> str:
        return content_sha256(self.validate())
