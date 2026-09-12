from __future__ import annotations

from dataclasses import dataclass
import math

from .account_observation_r0 import AccountPolicyObservationR0

CRITIC_OBSERVATION_SCHEMA_R0 = "CB16_R11_BC_CRITIC_OBSERVATION_V1_R0"
CRITIC_TAU_SEMANTICS_R0 = "KNOWN_OBJECTIVE_REMAINDER"
CRITIC_FORBIDDEN_FIELDS_R0 = (
    "future_market",
    "future_return",
    "future_outcome",
    "teacher_target",
    "teacher_value",
)


def _features(values: tuple[float, ...], code: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if any(not math.isfinite(v) for v in out):
        raise RuntimeError(code)
    return out


def _tau(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("ACCRITOBS_R0_TAU_TYPE_INVALID")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise RuntimeError("ACCRITOBS_R0_TAU_INVALID")
    return 0.0 if out == 0.0 else out


@dataclass(frozen=True)
class CriticObservationR0:
    schema_version: str
    market_sensory_version: str
    market_causal_features: tuple[float, ...]
    account_projection_version: str
    account_causal_features: tuple[tuple[str, object], ...]
    legal_execution_version: str
    legal_execution_causal_features: tuple[float, ...]
    policy_memory_version: str | None
    policy_memory_causal_features: tuple[float, ...]
    tau_semantics: str
    tau: float

    def validate(self) -> None:
        if self.schema_version != CRITIC_OBSERVATION_SCHEMA_R0:
            raise RuntimeError("ACCRITOBS_R0_SCHEMA_MISMATCH")
        if (
            not self.market_sensory_version
            or not self.account_projection_version
            or not self.legal_execution_version
        ):
            raise RuntimeError("ACCRITOBS_R0_VERSION_INVALID")
        _features(self.market_causal_features, "ACCRITOBS_R0_MARKET_FEATURE_INVALID")
        _features(
            self.legal_execution_causal_features,
            "ACCRITOBS_R0_EXECUTION_FEATURE_INVALID",
        )
        _features(self.policy_memory_causal_features, "ACCRITOBS_R0_MEMORY_FEATURE_INVALID")
        if self.policy_memory_version is None and self.policy_memory_causal_features:
            raise RuntimeError("ACCRITOBS_R0_MEMORY_VERSION_REQUIRED")
        account_keys = tuple(key for key, _ in self.account_causal_features)
        if len(set(account_keys)) != len(account_keys):
            raise RuntimeError("ACCRITOBS_R0_ACCOUNT_FEATURE_DUPLICATE")
        if self.tau_semantics != CRITIC_TAU_SEMANTICS_R0:
            raise RuntimeError("ACCRITOBS_R0_TAU_SEMANTICS_MISMATCH")
        _tau(self.tau)
        if any(field in self.__dict__ for field in CRITIC_FORBIDDEN_FIELDS_R0):
            raise RuntimeError("ACCRITOBS_R0_FORBIDDEN_FIELD")

    def model_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "market_causal_features": self.market_causal_features,
            "account_causal_features": self.account_causal_features,
            "legal_execution_causal_features": self.legal_execution_causal_features,
            "policy_memory_causal_features": self.policy_memory_causal_features,
            "tau": _tau(self.tau),
        }


def build_critic_observation_r0(
    *,
    market_sensory_version: str,
    market_causal_features: tuple[float, ...],
    account_observation: AccountPolicyObservationR0,
    legal_execution_version: str,
    legal_execution_causal_features: tuple[float, ...],
    tau: float,
    policy_memory_version: str | None = None,
    policy_memory_causal_features: tuple[float, ...] = (),
) -> CriticObservationR0:
    """Build the causal Critic input for a known objective remainder.

    ``tau`` is allowed for the Critic because value estimation may be conditioned
    on the known remaining objective horizon.  It is not future market truth and
    does not grant access to future outcomes.  Equity-reference/normalization
    semantics are intentionally left to BC-028 instead of being invented here.
    """

    account_observation.validate()
    observation = CriticObservationR0(
        schema_version=CRITIC_OBSERVATION_SCHEMA_R0,
        market_sensory_version=market_sensory_version,
        market_causal_features=_features(
            market_causal_features,
            "ACCRITOBS_R0_MARKET_FEATURE_INVALID",
        ),
        account_projection_version=account_observation.schema_version,
        account_causal_features=tuple(account_observation.model_payload().items()),
        legal_execution_version=legal_execution_version,
        legal_execution_causal_features=_features(
            legal_execution_causal_features,
            "ACCRITOBS_R0_EXECUTION_FEATURE_INVALID",
        ),
        policy_memory_version=policy_memory_version,
        policy_memory_causal_features=_features(
            policy_memory_causal_features,
            "ACCRITOBS_R0_MEMORY_FEATURE_INVALID",
        ),
        tau_semantics=CRITIC_TAU_SEMANTICS_R0,
        tau=_tau(tau),
    )
    observation.validate()
    return observation
