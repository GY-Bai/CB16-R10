from __future__ import annotations

"""Round-2 stochastic Actor policy interface and direction distribution.

BC-034 freezes the API and compatibility boundary.  BC-035 adds only the
categorical SHORT/FLAT/LONG direction distribution.  Bounded-risk
distributions, endpoint masses, joint likelihood math, and RNG serialization
remain owned by BC-036 through BC-039.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import inspect
import json
import re
from typing import Generic, TypeVar

import torch

from .action_contract_r1 import FLAT, LONG, SHORT, TargetPositionActionR1
from .actor_critic_contract_r1 import (
    ACTION_VERSION_R1,
    ACTOR_DISTRIBUTION_VERSION_R1,
    OBSERVATION_VERSION_R1,
)
from .actor_observation_r0 import ACTOR_OBSERVATION_SCHEMA_R0, ActorObservationR0


ACTOR_POLICY_INTERFACE_VERSION_R1 = "CB16_R11_BC_ACTOR_POLICY_INTERFACE_V1_R1"
ACTOR_POLICY_IDENTITY_SCHEMA_R1 = "CB16_R11_BC_ACTOR_POLICY_IDENTITY_V1_R1"
POLICY_TENSOR_CONTRACT_SCHEMA_R1 = "CB16_R11_BC_POLICY_TENSOR_CONTRACT_V1_R1"
DIRECTION_CATEGORICAL_SCHEMA_R1 = "CB16_R11_BC_DIRECTION_CATEGORICAL_V1_R1"
DIRECTION_ORDER_R1 = (SHORT, FLAT, LONG)
_DIRECTION_INDEX_R1 = {direction: index for index, direction in enumerate(DIRECTION_ORDER_R1)}
SUPPORTED_FLOAT_DTYPES_R1 = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

ObservationT = TypeVar("ObservationT")
SamplingRngT = TypeVar("SamplingRngT")
LogProbT = TypeVar("LogProbT")


def _nonempty(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _sha256_hex(value: object, *, code: str) -> str:
    text = _nonempty(value, code=code)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RuntimeError(code)
    return text


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _semantic_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _device_type(value: object) -> str:
    text = _nonempty(value, code="ACPOL_R1_DEVICE_TYPE_INVALID")
    try:
        return torch.device(text).type
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError("ACPOL_R1_DEVICE_TYPE_INVALID") from exc


def _device_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("ACPOL_R1_DEVICE_INDEX_INVALID")
    return int(value)


def _dtype_name(value: object) -> str:
    text = _nonempty(value, code="ACPOL_R1_DTYPE_INVALID")
    if text not in SUPPORTED_FLOAT_DTYPES_R1:
        raise RuntimeError("ACPOL_R1_DTYPE_INVALID")
    return text


def _direction_index_r1(direction: object) -> int:
    if not isinstance(direction, str) or direction not in _DIRECTION_INDEX_R1:
        raise RuntimeError("ACPOL_R1_DIRECTION_INVALID")
    return _DIRECTION_INDEX_R1[direction]


@dataclass(frozen=True)
class PolicyTensorContractR1:
    schema_version: str
    device_type: str
    device_index: int | None
    dtype: str

    def validate(self) -> None:
        if self.schema_version != POLICY_TENSOR_CONTRACT_SCHEMA_R1:
            raise RuntimeError("ACPOL_R1_TENSOR_SCHEMA_MISMATCH")
        canonical_type = _device_type(self.device_type)
        if canonical_type != self.device_type:
            raise RuntimeError("ACPOL_R1_DEVICE_TYPE_NONCANONICAL")
        _device_index(self.device_index)
        _dtype_name(self.dtype)
        if self.device_type == "cpu" and self.device_index is not None:
            raise RuntimeError("ACPOL_R1_CPU_DEVICE_INDEX_FORBIDDEN")

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _semantic_sha256(
            {
                "schema_version": self.schema_version,
                "device_type": self.device_type,
                "device_index": self.device_index,
                "dtype": self.dtype,
            }
        )


def make_policy_tensor_contract_r1(
    *,
    device_type: str,
    dtype: str,
    device_index: int | None = None,
) -> PolicyTensorContractR1:
    contract = PolicyTensorContractR1(
        schema_version=POLICY_TENSOR_CONTRACT_SCHEMA_R1,
        device_type=_device_type(device_type),
        device_index=_device_index(device_index),
        dtype=_dtype_name(dtype),
    )
    contract.validate()
    return contract


@dataclass(frozen=True)
class ActorPolicyDistributionIdentityR1:
    schema_version: str
    interface_version: str
    actor_distribution_version: str
    observation_science_version: str
    observation_schema_version: str
    action_schema_version: str
    distribution_id: str
    distribution_version: str
    policy_id: str
    policy_version: str
    policy_sha256: str
    tensor_contract: PolicyTensorContractR1

    def validate(self) -> None:
        if self.schema_version != ACTOR_POLICY_IDENTITY_SCHEMA_R1:
            raise RuntimeError("ACPOL_R1_IDENTITY_SCHEMA_MISMATCH")
        if self.interface_version != ACTOR_POLICY_INTERFACE_VERSION_R1:
            raise RuntimeError("ACPOL_R1_INTERFACE_VERSION_MISMATCH")
        if self.actor_distribution_version != ACTOR_DISTRIBUTION_VERSION_R1:
            raise RuntimeError("ACPOL_R1_DISTRIBUTION_SCIENCE_VERSION_MISMATCH")
        if self.observation_science_version != OBSERVATION_VERSION_R1:
            raise RuntimeError("ACPOL_R1_OBSERVATION_SCIENCE_VERSION_MISMATCH")
        if self.observation_schema_version != ACTOR_OBSERVATION_SCHEMA_R0:
            raise RuntimeError("ACPOL_R1_OBSERVATION_SCHEMA_VERSION_MISMATCH")
        if self.action_schema_version != ACTION_VERSION_R1:
            raise RuntimeError("ACPOL_R1_ACTION_SCHEMA_VERSION_MISMATCH")
        _nonempty(self.distribution_id, code="ACPOL_R1_DISTRIBUTION_ID_INVALID")
        _nonempty(self.distribution_version, code="ACPOL_R1_DISTRIBUTION_VERSION_INVALID")
        _nonempty(self.policy_id, code="ACPOL_R1_POLICY_ID_INVALID")
        _nonempty(self.policy_version, code="ACPOL_R1_POLICY_VERSION_INVALID")
        _sha256_hex(self.policy_sha256, code="ACPOL_R1_POLICY_HASH_INVALID")
        self.tensor_contract.validate()

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _semantic_sha256(
            {
                "schema_version": self.schema_version,
                "interface_version": self.interface_version,
                "actor_distribution_version": self.actor_distribution_version,
                "observation_science_version": self.observation_science_version,
                "observation_schema_version": self.observation_schema_version,
                "action_schema_version": self.action_schema_version,
                "distribution_id": self.distribution_id,
                "distribution_version": self.distribution_version,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "policy_sha256": self.policy_sha256,
                "tensor_contract_sha256": self.tensor_contract.semantic_sha256,
            }
        )


def make_actor_policy_distribution_identity_r1(
    *,
    distribution_id: str,
    distribution_version: str,
    policy_id: str,
    policy_version: str,
    policy_sha256: str,
    tensor_contract: PolicyTensorContractR1,
) -> ActorPolicyDistributionIdentityR1:
    identity = ActorPolicyDistributionIdentityR1(
        schema_version=ACTOR_POLICY_IDENTITY_SCHEMA_R1,
        interface_version=ACTOR_POLICY_INTERFACE_VERSION_R1,
        actor_distribution_version=ACTOR_DISTRIBUTION_VERSION_R1,
        observation_science_version=OBSERVATION_VERSION_R1,
        observation_schema_version=ACTOR_OBSERVATION_SCHEMA_R0,
        action_schema_version=ACTION_VERSION_R1,
        distribution_id=distribution_id,
        distribution_version=distribution_version,
        policy_id=policy_id,
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        tensor_contract=tensor_contract,
    )
    identity.validate()
    return identity


@dataclass(frozen=True)
class DirectionCategoricalR1:
    """Stable categorical law over the canonical SHORT/FLAT/LONG order."""

    schema_version: str
    logits: torch.Tensor
    tensor_contract: PolicyTensorContractR1

    def validate(self) -> None:
        if self.schema_version != DIRECTION_CATEGORICAL_SCHEMA_R1:
            raise RuntimeError("ACPOL_R1_DIRECTION_SCHEMA_MISMATCH")
        logits = validate_policy_tensor_r1(self.logits, self.tensor_contract)
        if logits.ndim != 1 or logits.numel() != len(DIRECTION_ORDER_R1):
            raise RuntimeError("ACPOL_R1_DIRECTION_LOGITS_SHAPE_INVALID")
        log_probs = torch.log_softmax(logits, dim=0)
        probs = torch.softmax(logits, dim=0)
        if not bool(torch.isfinite(log_probs).all().item()):
            raise RuntimeError("ACPOL_R1_DIRECTION_LOG_PROB_NONFINITE")
        if not bool(torch.isfinite(probs).all().item()):
            raise RuntimeError("ACPOL_R1_DIRECTION_PROB_NONFINITE")
        if bool((probs < 0).any().item()):
            raise RuntimeError("ACPOL_R1_DIRECTION_PROB_NEGATIVE")
        one = torch.ones((), dtype=probs.dtype, device=probs.device)
        if not bool(torch.isclose(probs.sum(), one, rtol=1e-4, atol=1e-6).item()):
            raise RuntimeError("ACPOL_R1_DIRECTION_PROB_NOT_NORMALIZED")

    @property
    def probabilities(self) -> torch.Tensor:
        self.validate()
        return torch.softmax(self.logits, dim=0)

    @property
    def log_probabilities(self) -> torch.Tensor:
        self.validate()
        return torch.log_softmax(self.logits, dim=0)

    def log_prob(self, direction: str) -> torch.Tensor:
        index = _direction_index_r1(direction)
        return self.log_probabilities[index]

    def sample_directions(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[str, ...]:
        self.validate()
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise RuntimeError("ACPOL_R1_DIRECTION_SAMPLE_COUNT_INVALID")
        if not isinstance(generator, torch.Generator):
            raise RuntimeError("ACPOL_R1_DIRECTION_GENERATOR_INVALID")
        generator_device = torch.device(generator.device)
        logits_device = self.logits.device
        if generator_device.type != logits_device.type:
            raise RuntimeError("ACPOL_R1_DIRECTION_GENERATOR_DEVICE_MISMATCH")
        if logits_device.type != "cpu" and generator_device.index != logits_device.index:
            raise RuntimeError("ACPOL_R1_DIRECTION_GENERATOR_DEVICE_MISMATCH")
        indices = torch.multinomial(
            self.probabilities,
            num_samples=count,
            replacement=True,
            generator=generator,
        )
        return tuple(DIRECTION_ORDER_R1[int(index)] for index in indices.tolist())

    def sample_direction(self, *, generator: torch.Generator) -> str:
        return self.sample_directions(1, generator=generator)[0]


def make_direction_categorical_r1(
    *,
    logits: torch.Tensor,
    tensor_contract: PolicyTensorContractR1,
) -> DirectionCategoricalR1:
    distribution = DirectionCategoricalR1(
        schema_version=DIRECTION_CATEGORICAL_SCHEMA_R1,
        logits=logits,
        tensor_contract=tensor_contract,
    )
    distribution.validate()
    return distribution


class ActorPolicyR1(ABC, Generic[ObservationT, SamplingRngT, LogProbT]):
    """Minimal stochastic Actor API for Round 2.

    ``sample`` owns stochastic behavior collection and receives an explicit RNG.
    ``log_prob`` scores a supplied action and must not consume sampling RNG.
    ``deterministic_action`` is a separate evaluation path and must not consume
    sampling RNG.  BC-035 defines direction categorical math; bounded-risk and
    joint action likelihood semantics remain deferred.
    """

    @property
    @abstractmethod
    def distribution_identity(self) -> ActorPolicyDistributionIdentityR1:
        raise NotImplementedError

    @abstractmethod
    def sample(self, observation: ObservationT, *, rng: SamplingRngT) -> TargetPositionActionR1:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, observation: ObservationT, action: TargetPositionActionR1) -> LogProbT:
        raise NotImplementedError

    @abstractmethod
    def deterministic_action(self, observation: ObservationT) -> TargetPositionActionR1:
        raise NotImplementedError


def validate_actor_policy_observation_r1(
    identity: ActorPolicyDistributionIdentityR1,
    observation: object,
) -> ActorObservationR0:
    identity.validate()
    if not isinstance(observation, ActorObservationR0):
        raise RuntimeError("ACPOL_R1_OBSERVATION_TYPE_INVALID")
    observation.validate()
    if observation.schema_version != identity.observation_schema_version:
        raise RuntimeError("ACPOL_R1_OBSERVATION_SCHEMA_INCOMPATIBLE")
    return observation


def validate_policy_tensor_r1(
    tensor: object,
    contract: PolicyTensorContractR1,
) -> torch.Tensor:
    contract.validate()
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("ACPOL_R1_TENSOR_TYPE_INVALID")
    expected_dtype = SUPPORTED_FLOAT_DTYPES_R1[contract.dtype]
    if tensor.dtype != expected_dtype:
        raise RuntimeError("ACPOL_R1_TENSOR_DTYPE_MISMATCH")
    if not tensor.is_floating_point():
        raise RuntimeError("ACPOL_R1_TENSOR_NOT_FLOATING")
    if tensor.device.type != contract.device_type or tensor.device.index != contract.device_index:
        raise RuntimeError("ACPOL_R1_TENSOR_DEVICE_MISMATCH")
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError("ACPOL_R1_TENSOR_NONFINITE")
    return tensor


def validate_policy_action_binding_r1(
    identity: ActorPolicyDistributionIdentityR1,
    action: object,
) -> TargetPositionActionR1:
    identity.validate()
    if not isinstance(action, TargetPositionActionR1):
        raise RuntimeError("ACPOL_R1_ACTION_TYPE_INVALID")
    action.validate()
    if action.schema_version != identity.action_schema_version:
        raise RuntimeError("ACPOL_R1_ACTION_SCHEMA_INCOMPATIBLE")
    if action.policy_id != identity.policy_id:
        raise RuntimeError("ACPOL_R1_ACTION_POLICY_ID_MISMATCH")
    if action.policy_version != identity.policy_version:
        raise RuntimeError("ACPOL_R1_ACTION_POLICY_VERSION_MISMATCH")
    return action


def _bound_parameters(method: object, *, code: str) -> tuple[inspect.Parameter, ...]:
    if not callable(method):
        raise RuntimeError(code)
    try:
        params = tuple(inspect.signature(method).parameters.values())
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc
    if any(
        parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in params
    ):
        raise RuntimeError(code)
    return params


def validate_actor_policy_interface_r1(policy: object) -> ActorPolicyDistributionIdentityR1:
    identity = getattr(policy, "distribution_identity", None)
    if not isinstance(identity, ActorPolicyDistributionIdentityR1):
        raise RuntimeError("ACPOL_R1_DISTRIBUTION_IDENTITY_REQUIRED")
    identity.validate()

    sample_params = _bound_parameters(
        getattr(policy, "sample", None),
        code="ACPOL_R1_SAMPLE_INTERFACE_INVALID",
    )
    if len(sample_params) != 2:
        raise RuntimeError("ACPOL_R1_SAMPLE_INTERFACE_INVALID")
    observation_param, rng_param = sample_params
    if observation_param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise RuntimeError("ACPOL_R1_SAMPLE_INTERFACE_INVALID")
    if rng_param.name != "rng" or rng_param.kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("ACPOL_R1_SAMPLE_RNG_INTERFACE_INVALID")

    log_prob_params = _bound_parameters(
        getattr(policy, "log_prob", None),
        code="ACPOL_R1_LOG_PROB_INTERFACE_INVALID",
    )
    if len(log_prob_params) != 2 or any(param.name == "rng" for param in log_prob_params):
        raise RuntimeError("ACPOL_R1_LOG_PROB_INTERFACE_INVALID")

    deterministic_params = _bound_parameters(
        getattr(policy, "deterministic_action", None),
        code="ACPOL_R1_DETERMINISTIC_INTERFACE_INVALID",
    )
    if len(deterministic_params) != 1 or any(param.name == "rng" for param in deterministic_params):
        raise RuntimeError("ACPOL_R1_DETERMINISTIC_INTERFACE_INVALID")

    return identity
