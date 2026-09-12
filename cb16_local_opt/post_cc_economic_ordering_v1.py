from __future__ import annotations

from dataclasses import dataclass
import math

ECONOMIC_ORDERING_CONTRACT_ID = "CB16_R11_POST_CC_ECONOMIC_ORDERING_V1"
LEFT_HIGHER = "LEFT_HIGHER"
RIGHT_HIGHER = "RIGHT_HIGHER"
TIE = "TIE"


@dataclass(frozen=True)
class EconomicPolicyMeasureV1:
    policy_object_type: str
    policy_identity: str
    cohort_id: str
    common_horizon_id: str
    capital_denominator_id: str
    result_scope: str
    mean_arithmetic_return: float

    def validate(self) -> "EconomicPolicyMeasureV1":
        for name in (
            "policy_object_type",
            "policy_identity",
            "cohort_id",
            "common_horizon_id",
            "capital_denominator_id",
            "result_scope",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if not math.isfinite(float(self.mean_arithmetic_return)):
            raise ValueError("mean_arithmetic_return must be finite")
        return self

    @property
    def comparison_contract(self) -> tuple[str, str, str, str]:
        return (
            self.cohort_id,
            self.common_horizon_id,
            self.capital_denominator_id,
            self.result_scope,
        )


@dataclass(frozen=True)
class ModelOrderingResultV1:
    contract_id: str
    left_policy_identity: str
    right_policy_identity: str
    left_mean_arithmetic_return: float
    right_mean_arithmetic_return: float
    arithmetic_return_delta_left_minus_right: float
    status: str
    tie_atol: float

    def validate(self) -> "ModelOrderingResultV1":
        if self.contract_id != ECONOMIC_ORDERING_CONTRACT_ID:
            raise ValueError("unexpected ordering contract")
        if self.status not in {LEFT_HIGHER, RIGHT_HIGHER, TIE}:
            raise ValueError("invalid ordering status")
        if self.tie_atol < 0 or not math.isfinite(float(self.tie_atol)):
            raise ValueError("tie_atol must be finite and >= 0")
        return self


def compare_policy_measures(
    left: EconomicPolicyMeasureV1,
    right: EconomicPolicyMeasureV1,
    *,
    tie_atol: float = 0.0,
) -> ModelOrderingResultV1:
    left.validate()
    right.validate()
    if not math.isfinite(float(tie_atol)) or tie_atol < 0:
        raise ValueError("tie_atol must be finite and >= 0")
    if left.comparison_contract != right.comparison_contract:
        raise ValueError(
            "CONTRACT_MISMATCH: cohort/horizon/capital denominator/result scope differ"
        )
    delta = float(left.mean_arithmetic_return) - float(right.mean_arithmetic_return)
    if abs(delta) <= tie_atol:
        status = TIE
    elif delta > 0:
        status = LEFT_HIGHER
    else:
        status = RIGHT_HIGHER
    return ModelOrderingResultV1(
        ECONOMIC_ORDERING_CONTRACT_ID,
        left.policy_identity,
        right.policy_identity,
        float(left.mean_arithmetic_return),
        float(right.mean_arithmetic_return),
        delta,
        status,
        float(tie_atol),
    ).validate()
