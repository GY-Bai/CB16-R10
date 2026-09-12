from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .cc_experience_wire_r0 import CCEconomicResultV1
from .post_cc_baseline_components_v1 import BaselineComponentsV1, from_deltas
from .post_cc_economic_ordering_v1 import EconomicPolicyMeasureV1

W05_ADAPTER_CONTRACT_ID = "CB16_R11_POST_CC_W05_ADAPTER_V1"


@dataclass(frozen=True)
class AdaptedEconomicResultV1:
    contract_id: str
    evaluation_id: str
    ordering_measure: EconomicPolicyMeasureV1
    baseline_components: BaselineComponentsV1
    failure_counts: Mapping[str, int]
    tail_diagnostics: Mapping[str, float]

    def validate(self) -> "AdaptedEconomicResultV1":
        if self.contract_id != W05_ADAPTER_CONTRACT_ID:
            raise ValueError("unexpected adapter contract")
        self.ordering_measure.validate()
        self.baseline_components.validate()
        if any(int(v) < 0 for v in self.failure_counts.values()):
            raise ValueError("failure counts must be non-negative")
        return self


def adapt_w05(result: CCEconomicResultV1) -> AdaptedEconomicResultV1:
    result.validate()
    ordering = EconomicPolicyMeasureV1(
        result.policy_object_type,
        result.policy_identity,
        result.cohort_id,
        result.common_horizon_id,
        result.capital_denominator_id,
        result.result_scope,
        float(result.mean_arithmetic_return),
    ).validate()
    components = from_deltas(
        buy_hold_delta=float(result.buy_hold_delta),
        flat_delta=float(result.flat_delta),
    )
    return AdaptedEconomicResultV1(
        W05_ADAPTER_CONTRACT_ID,
        result.evaluation_id,
        ordering,
        components,
        dict(result.failure_counts),
        dict(result.tail_diagnostics),
    ).validate()
