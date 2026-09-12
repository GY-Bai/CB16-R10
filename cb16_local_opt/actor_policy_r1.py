from __future__ import annotations

"""Round-2 stochastic Actor policy interface and action components.

BC-034 freezes the API and compatibility boundary. BC-035 adds the categorical
SHORT/FLAT/LONG direction distribution. BC-036 adds a direction-conditioned
sigmoid-squashed Normal risk component. BC-037 freezes endpoint semantics:
FLAT has an exact zero-risk point mass; LONG/SHORT have no point mass at risk 0
or 1, so those exact points have probability zero while interior values use a
continuous density. Joint action likelihood and RNG serialization remain owned
by BC-038 and BC-039.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
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
CONDITIONAL_RISK_SCHEMA_R1 = "CB16_R11_BC_CONDITIONAL_SQUASHED_NORMAL_RISK_V1_R1"
RISK_LIKELIHOOD_SCHEMA_R1 = "CB16_R11_BC_RISK_LIKELIHOOD_TERM_V1_R1"
RISK_ENDPOINT_POLICY_R1 = "FLAT_ZERO_POINT_MASS_NONFLAT_ENDPOINT_MASSES_DISALLOWED_V1_R1"
CONTINUOUS_DENSITY = "CONTINUOUS_DENSITY"
POINT_MASS = "POINT_MASS"
RISK_MEASURE_KINDS_R1 = (CONTINUOUS_DENSITY, POINT_MASS)
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


def _validate_generator_device_r1(generator: object, reference: torch.Tensor) -> torch.Generator:
    if not isinstance(generator, torch.Generator):
        raise RuntimeError("ACPOL_R1_GENERATOR_INVALID")
    generator_device = torch.device(generator.device)
    tensor_device = reference.device
    if generator_device.type != tensor_device.type:
        raise RuntimeError("ACPOL_R1_GENERATOR_DEVICE_MISMATCH")
    if tensor_device.type != "cpu" and generator_device.index != tensor_device.index:
        raise RuntimeError("ACPOL_R1_GENERATOR_DEVICE_MISMATCH")
    return generator


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
        checked_generator = _validate_generator_device_r1(generator, self.logits)
        indices = torch.multinomial(
            self.probabilities,
            num_samples=count,
            replacement=True,
            generator=checked_generator,
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


def _validate_scalar_tensor_allow_negative_infinity_r1(
    tensor: object,
    contract: PolicyTensorContractR1,
    *,
    code: str,
) -> torch.Tensor:
    contract.validate()
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError(code)
    if tensor.dtype != SUPPORTED_FLOAT_DTYPES_R1[contract.dtype]:
        raise RuntimeError(code)
    if tensor.device.type != contract.device_type or tensor.device.index != contract.device_index:
        raise RuntimeError(code)
    if tensor.ndim != 0:
        raise RuntimeError(code)
    if bool(torch.isnan(tensor).item()) or bool(torch.isposinf(tensor).item()):
        raise RuntimeError(code)
    return tensor


@dataclass(frozen=True)
class RiskLikelihoodTermR1:
    """One conditional risk likelihood term with an explicit reference measure."""

    schema_version: str
    endpoint_policy: str
    direction: str
    risk: torch.Tensor
    measure_kind: str
    log_likelihood: torch.Tensor
    tensor_contract: PolicyTensorContractR1

    def validate(self) -> None:
        if self.schema_version != RISK_LIKELIHOOD_SCHEMA_R1:
            raise RuntimeError("ACPOL_R1_RISK_LIKELIHOOD_SCHEMA_MISMATCH")
        if self.endpoint_policy != RISK_ENDPOINT_POLICY_R1:
            raise RuntimeError("ACPOL_R1_RISK_ENDPOINT_POLICY_MISMATCH")
        _direction_index_r1(self.direction)
        value = validate_policy_tensor_r1(self.risk, self.tensor_contract)
        if value.ndim != 0:
            raise RuntimeError("ACPOL_R1_RISK_VALUE_SHAPE_INVALID")
        if self.measure_kind not in RISK_MEASURE_KINDS_R1:
            raise RuntimeError("ACPOL_R1_RISK_MEASURE_KIND_INVALID")
        score = _validate_scalar_tensor_allow_negative_infinity_r1(
            self.log_likelihood,
            self.tensor_contract,
            code="ACPOL_R1_RISK_LIKELIHOOD_INVALID",
        )

        risk_value = value.item()
        score_value = score.item()
        if self.direction == FLAT:
            if risk_value != 0.0:
                raise RuntimeError("ACPOL_R1_FLAT_RISK_MUST_BE_ZERO")
            if self.measure_kind != POINT_MASS or score_value != 0.0:
                raise RuntimeError("ACPOL_R1_FLAT_POINT_MASS_REQUIRED")
            return

        if risk_value in (0.0, 1.0):
            if self.measure_kind != POINT_MASS or not math.isinf(score_value) or score_value >= 0.0:
                raise RuntimeError("ACPOL_R1_NONFLAT_ENDPOINT_ZERO_MASS_REQUIRED")
            return
        if not 0.0 < risk_value < 1.0:
            raise RuntimeError("ACPOL_R1_RISK_OUT_OF_BOUNDS")
        if self.measure_kind != CONTINUOUS_DENSITY or not math.isfinite(score_value):
            raise RuntimeError("ACPOL_R1_INTERIOR_DENSITY_REQUIRED")


@dataclass(frozen=True)
class ConditionalBoundedRiskR1:
    """Direction-conditioned sigmoid-Normal risk with explicit endpoint policy.

    SHORT and LONG have a continuous density only on (0,1). Their exact 0/1
    point masses are disallowed and therefore have probability zero. FLAT is
    exactly the risk-zero point mass with conditional probability one.
    """

    schema_version: str
    short_location: torch.Tensor
    short_log_scale: torch.Tensor
    long_location: torch.Tensor
    long_log_scale: torch.Tensor
    tensor_contract: PolicyTensorContractR1

    def _validated_scalar(self, tensor: torch.Tensor, *, code: str) -> torch.Tensor:
        value = validate_policy_tensor_r1(tensor, self.tensor_contract)
        if value.ndim != 0:
            raise RuntimeError(code)
        return value

    def validate(self) -> None:
        if self.schema_version != CONDITIONAL_RISK_SCHEMA_R1:
            raise RuntimeError("ACPOL_R1_RISK_SCHEMA_MISMATCH")
        self.tensor_contract.validate()
        self._validated_scalar(self.short_location, code="ACPOL_R1_RISK_LOCATION_SHAPE_INVALID")
        short_log_scale = self._validated_scalar(
            self.short_log_scale,
            code="ACPOL_R1_RISK_LOG_SCALE_SHAPE_INVALID",
        )
        self._validated_scalar(self.long_location, code="ACPOL_R1_RISK_LOCATION_SHAPE_INVALID")
        long_log_scale = self._validated_scalar(
            self.long_log_scale,
            code="ACPOL_R1_RISK_LOG_SCALE_SHAPE_INVALID",
        )
        for log_scale in (short_log_scale, long_log_scale):
            scale = torch.exp(log_scale)
            if not bool(torch.isfinite(scale).item()) or not bool((scale > 0).item()):
                raise RuntimeError("ACPOL_R1_RISK_SCALE_INVALID")

    def _parameters(self, direction: str) -> tuple[torch.Tensor, torch.Tensor]:
        _direction_index_r1(direction)
        if direction == SHORT:
            return self.short_location, self.short_log_scale
        if direction == LONG:
            return self.long_location, self.long_log_scale
        raise RuntimeError("ACPOL_R1_FLAT_HAS_NO_CONTINUOUS_RISK_PARAMETERS")

    def sample_risk(self, direction: str, *, generator: torch.Generator) -> torch.Tensor:
        self.validate()
        _direction_index_r1(direction)
        checked_generator = _validate_generator_device_r1(generator, self.short_location)
        if direction == FLAT:
            return torch.zeros_like(self.short_location)
        location, log_scale = self._parameters(direction)
        epsilon = torch.randn(
            (),
            dtype=location.dtype,
            device=location.device,
            generator=checked_generator,
        )
        latent = location + torch.exp(log_scale) * epsilon
        risk = torch.sigmoid(latent)
        validate_policy_tensor_r1(risk, self.tensor_contract)
        if not bool(((risk > 0) & (risk < 1)).item()):
            raise RuntimeError("ACPOL_R1_RISK_NUMERIC_ZERO_MASS_ENDPOINT")
        return risk

    def _interior_log_density(self, direction: str, value: torch.Tensor) -> torch.Tensor:
        location, log_scale = self._parameters(direction)
        latent = torch.log(value) - torch.log1p(-value)
        standardized = (latent - location) * torch.exp(-log_scale)
        normal_log_prob = (
            -0.5 * standardized.square()
            - log_scale
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_abs_det_inverse = -torch.log(value) - torch.log1p(-value)
        result = normal_log_prob + log_abs_det_inverse
        if not bool(torch.isfinite(result).item()):
            raise RuntimeError("ACPOL_R1_RISK_LOG_PROB_NONFINITE")
        return result

    def likelihood_term(self, direction: str, risk: torch.Tensor) -> RiskLikelihoodTermR1:
        self.validate()
        _direction_index_r1(direction)
        value = self._validated_scalar(risk, code="ACPOL_R1_RISK_VALUE_SHAPE_INVALID")
        risk_value = value.item()
        if direction == FLAT:
            if risk_value != 0.0:
                raise RuntimeError("ACPOL_R1_FLAT_RISK_MUST_BE_ZERO")
            term = RiskLikelihoodTermR1(
                schema_version=RISK_LIKELIHOOD_SCHEMA_R1,
                endpoint_policy=RISK_ENDPOINT_POLICY_R1,
                direction=direction,
                risk=value,
                measure_kind=POINT_MASS,
                log_likelihood=torch.zeros_like(value),
                tensor_contract=self.tensor_contract,
            )
        elif risk_value in (0.0, 1.0):
            term = RiskLikelihoodTermR1(
                schema_version=RISK_LIKELIHOOD_SCHEMA_R1,
                endpoint_policy=RISK_ENDPOINT_POLICY_R1,
                direction=direction,
                risk=value,
                measure_kind=POINT_MASS,
                log_likelihood=torch.full_like(value, float("-inf")),
                tensor_contract=self.tensor_contract,
            )
        elif 0.0 < risk_value < 1.0:
            term = RiskLikelihoodTermR1(
                schema_version=RISK_LIKELIHOOD_SCHEMA_R1,
                endpoint_policy=RISK_ENDPOINT_POLICY_R1,
                direction=direction,
                risk=value,
                measure_kind=CONTINUOUS_DENSITY,
                log_likelihood=self._interior_log_density(direction, value),
                tensor_contract=self.tensor_contract,
            )
        else:
            raise RuntimeError("ACPOL_R1_RISK_OUT_OF_BOUNDS")
        term.validate()
        return term

    def log_prob(self, direction: str, risk: torch.Tensor) -> torch.Tensor:
        """Return the log likelihood under BC-037's explicit mixed measure.

        Callers that need to distinguish density from point-mass likelihood must
        consume ``likelihood_term`` rather than infer the reference measure.
        """
        return self.likelihood_term(direction, risk).log_likelihood


def make_conditional_bounded_risk_r1(
    *,
    short_location: torch.Tensor,
    short_log_scale: torch.Tensor,
    long_location: torch.Tensor,
    long_log_scale: torch.Tensor,
    tensor_contract: PolicyTensorContractR1,
) -> ConditionalBoundedRiskR1:
    distribution = ConditionalBoundedRiskR1(
        schema_version=CONDITIONAL_RISK_SCHEMA_R1,
        short_location=short_location,
        short_log_scale=short_log_scale,
        long_location=long_location,
        long_log_scale=long_log_scale,
        tensor_contract=tensor_contract,
    )
    distribution.validate()
    return distribution


class ActorPolicyR1(ABC, Generic[ObservationT, SamplingRngT, LogProbT]):
    """Minimal stochastic Actor API for Round 2.

    ``sample`` owns stochastic behavior collection and receives an explicit RNG.
    ``log_prob`` scores a supplied action and must not consume sampling RNG.
    ``deterministic_action`` is a separate evaluation path and must not consume
    sampling RNG. BC-035 through BC-037 define direction, conditional-risk, and
    endpoint component math; joint action likelihood remains BC-038 work.
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
