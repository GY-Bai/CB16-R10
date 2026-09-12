from __future__ import annotations

"""Versioned Actor policy interface for the R11 Actor-Critic lane.

AC-015 freezes the *interface boundary* only.  Direction probability math,
bounded-risk parameterization, endpoint-mass semantics, and joint action
likelihood are owned by AC-016 through AC-019 and are deliberately not invented
here.

The sampling RNG is an explicit input to ``sample``.  ``log_prob`` and
``deterministic_action`` have no RNG argument, keeping stochastic behavior
collection structurally distinct from deterministic evaluation.  Exact RNG
state/provenance is owned by AC-020.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import re
from typing import Generic, TypeVar

from .action_contract_r0 import TargetPositionActionR0
from .actor_critic_contract_r0 import ACTOR_DISTRIBUTION_VERSION_R0


ACTOR_POLICY_INTERFACE_VERSION_R0 = "CB16_R11_AC_ACTOR_POLICY_INTERFACE_V1_R0"
ACTOR_POLICY_DISTRIBUTION_DESCRIPTOR_SCHEMA_R0 = (
    "CB16_R11_AC_ACTOR_DISTRIBUTION_DESCRIPTOR_V1_R0"
)

ObservationT = TypeVar("ObservationT")
SamplingRngT = TypeVar("SamplingRngT")
LogProbT = TypeVar("LogProbT")


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _require_sha256_hex(value: object, *, code: str) -> str:
    text = _require_nonempty_string(value, code=code)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RuntimeError(code)
    return text


@dataclass(frozen=True)
class ActorPolicyDistributionR0:
    """Immutable policy/distribution identity, not AC-016+ distribution math."""

    schema_version: str
    interface_version: str
    actor_distribution_version: str
    policy_id: str
    policy_version: str
    policy_hash: str

    def validate(self) -> None:
        if self.schema_version != ACTOR_POLICY_DISTRIBUTION_DESCRIPTOR_SCHEMA_R0:
            raise RuntimeError("ACPOL_DISTRIBUTION_SCHEMA_VERSION_MISMATCH")
        if self.interface_version != ACTOR_POLICY_INTERFACE_VERSION_R0:
            raise RuntimeError("ACPOL_INTERFACE_VERSION_MISMATCH")
        if self.actor_distribution_version != ACTOR_DISTRIBUTION_VERSION_R0:
            raise RuntimeError("ACPOL_ACTOR_DISTRIBUTION_VERSION_MISMATCH")
        _require_nonempty_string(self.policy_id, code="ACPOL_POLICY_ID_INVALID")
        _require_nonempty_string(
            self.policy_version,
            code="ACPOL_POLICY_VERSION_INVALID",
        )
        _require_sha256_hex(self.policy_hash, code="ACPOL_POLICY_HASH_INVALID")

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "interface_version": self.interface_version,
            "actor_distribution_version": self.actor_distribution_version,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_hash": self.policy_hash,
        }


def make_actor_policy_distribution_r0(
    *,
    policy_id: str,
    policy_version: str,
    policy_hash: str,
) -> ActorPolicyDistributionR0:
    descriptor = ActorPolicyDistributionR0(
        schema_version=ACTOR_POLICY_DISTRIBUTION_DESCRIPTOR_SCHEMA_R0,
        interface_version=ACTOR_POLICY_INTERFACE_VERSION_R0,
        actor_distribution_version=ACTOR_DISTRIBUTION_VERSION_R0,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_hash=policy_hash,
    )
    descriptor.validate()
    return descriptor


class ActorPolicyR0(ABC, Generic[ObservationT, SamplingRngT, LogProbT]):
    """Minimal stochastic Actor API frozen by AC-015.

    ``LogProbT`` is intentionally generic in AC-015 so the interface does not
    prematurely freeze the tensor/scalar representation needed by the later
    learner.  AC-019 owns the exact joint action likelihood semantics.
    """

    @property
    @abstractmethod
    def distribution_descriptor(self) -> ActorPolicyDistributionR0:
        """Return immutable policy/distribution identity for this behavior policy."""

    @abstractmethod
    def sample(
        self,
        observation: ObservationT,
        *,
        rng: SamplingRngT,
    ) -> TargetPositionActionR0:
        """Sample one behavior action using only the explicit sampling RNG."""

    @abstractmethod
    def log_prob(
        self,
        observation: ObservationT,
        action: TargetPositionActionR0,
    ) -> LogProbT:
        """Score an action without consuming sampling RNG."""

    @abstractmethod
    def deterministic_action(
        self,
        observation: ObservationT,
    ) -> TargetPositionActionR0:
        """Return deterministic evaluation action without a sampling RNG input."""


def _bound_signature_parameters(method: object, *, code: str) -> list[inspect.Parameter]:
    if not callable(method):
        raise RuntimeError(code)
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc
    parameters = list(signature.parameters.values())
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        raise RuntimeError(code)
    return parameters


def validate_actor_policy_interface_r0(policy: object) -> ActorPolicyDistributionR0:
    """Fail closed on an Actor implementation that blurs sample/eval APIs."""

    descriptor = getattr(policy, "distribution_descriptor", None)
    if not isinstance(descriptor, ActorPolicyDistributionR0):
        raise RuntimeError("ACPOL_DISTRIBUTION_DESCRIPTOR_REQUIRED")
    descriptor.validate()

    sample_parameters = _bound_signature_parameters(
        getattr(policy, "sample", None),
        code="ACPOL_SAMPLE_INTERFACE_INVALID",
    )
    if len(sample_parameters) != 2:
        raise RuntimeError("ACPOL_SAMPLE_INTERFACE_INVALID")
    observation_parameter, rng_parameter = sample_parameters
    if observation_parameter.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise RuntimeError("ACPOL_SAMPLE_INTERFACE_INVALID")
    if rng_parameter.name != "rng" or rng_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("ACPOL_SAMPLE_RNG_INTERFACE_INVALID")

    log_prob_parameters = _bound_signature_parameters(
        getattr(policy, "log_prob", None),
        code="ACPOL_LOG_PROB_INTERFACE_INVALID",
    )
    if len(log_prob_parameters) != 2 or any(
        parameter.name == "rng" for parameter in log_prob_parameters
    ):
        raise RuntimeError("ACPOL_LOG_PROB_INTERFACE_INVALID")

    deterministic_parameters = _bound_signature_parameters(
        getattr(policy, "deterministic_action", None),
        code="ACPOL_DETERMINISTIC_INTERFACE_INVALID",
    )
    if len(deterministic_parameters) != 1 or any(
        parameter.name == "rng" for parameter in deterministic_parameters
    ):
        raise RuntimeError("ACPOL_DETERMINISTIC_INTERFACE_INVALID")

    return descriptor
