from __future__ import annotations

from dataclasses import dataclass
import math

from .account_observation_r0 import AccountPolicyObservationR0

ACTOR_OBSERVATION_SCHEMA_R0 = "CB16_R11_BC_ACTOR_OBSERVATION_V1_R0"
ACTOR_FORBIDDEN_FIELDS_R0 = (
    "tau",
    "objective_horizon",
    "objective_end_time",
    "future_market",
    "future_return",
    "teacher_target",
)


def _features(values: tuple[float, ...], code: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if any(not math.isfinite(v) for v in out):
        raise RuntimeError(code)
    return out


@dataclass(frozen=True)
class ActorObservationR0:
    schema_version: str
    market_sensory_version: str
    market_causal_features: tuple[float, ...]
    account_projection_version: str
    account_causal_features: tuple[tuple[str, object], ...]
    legal_execution_version: str
    legal_execution_causal_features: tuple[float, ...]
    policy_memory_version: str | None
    policy_memory_causal_features: tuple[float, ...]

    def validate(self) -> None:
        if self.schema_version != ACTOR_OBSERVATION_SCHEMA_R0:
            raise RuntimeError("ACACTOBS_R0_SCHEMA_MISMATCH")
        if not self.market_sensory_version or not self.account_projection_version or not self.legal_execution_version:
            raise RuntimeError("ACACTOBS_R0_VERSION_INVALID")
        _features(self.market_causal_features, "ACACTOBS_R0_MARKET_FEATURE_INVALID")
        _features(self.legal_execution_causal_features, "ACACTOBS_R0_EXECUTION_FEATURE_INVALID")
        _features(self.policy_memory_causal_features, "ACACTOBS_R0_MEMORY_FEATURE_INVALID")
        if self.policy_memory_version is None and self.policy_memory_causal_features:
            raise RuntimeError("ACACTOBS_R0_MEMORY_VERSION_REQUIRED")
        account_keys = tuple(key for key, _ in self.account_causal_features)
        if len(set(account_keys)) != len(account_keys):
            raise RuntimeError("ACACTOBS_R0_ACCOUNT_FEATURE_DUPLICATE")
        if any(field in self.__dict__ for field in ACTOR_FORBIDDEN_FIELDS_R0):
            raise RuntimeError("ACACTOBS_R0_FORBIDDEN_FIELD")

    def model_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "market_causal_features": self.market_causal_features,
            "account_causal_features": self.account_causal_features,
            "legal_execution_causal_features": self.legal_execution_causal_features,
            "policy_memory_causal_features": self.policy_memory_causal_features,
        }


def build_actor_observation_r0(*, market_sensory_version: str, market_causal_features: tuple[float, ...], account_observation: AccountPolicyObservationR0, legal_execution_version: str, legal_execution_causal_features: tuple[float, ...], policy_memory_version: str | None = None, policy_memory_causal_features: tuple[float, ...] = ()) -> ActorObservationR0:
    account_observation.validate()
    observation = ActorObservationR0(
        schema_version=ACTOR_OBSERVATION_SCHEMA_R0,
        market_sensory_version=market_sensory_version,
        market_causal_features=_features(market_causal_features, "ACACTOBS_R0_MARKET_FEATURE_INVALID"),
        account_projection_version=account_observation.schema_version,
        account_causal_features=tuple(account_observation.model_payload().items()),
        legal_execution_version=legal_execution_version,
        legal_execution_causal_features=_features(legal_execution_causal_features, "ACACTOBS_R0_EXECUTION_FEATURE_INVALID"),
        policy_memory_version=policy_memory_version,
        policy_memory_causal_features=_features(policy_memory_causal_features, "ACACTOBS_R0_MEMORY_FEATURE_INVALID"),
    )
    observation.validate()
    return observation
