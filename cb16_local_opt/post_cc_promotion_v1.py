from __future__ import annotations

from dataclasses import dataclass

from .post_cc_baseline_components_v1 import BaselineComponentsV1, PASS as COMPONENT_PASS
from .post_cc_economic_ordering_v1 import (
    ModelOrderingResultV1,
    LEFT_HIGHER,
    RIGHT_HIGHER,
    TIE,
)

PROMOTION_CONTRACT_ID = "CB16_R11_POST_CC_PROMOTION_V1"
REPORT_ONLY_NO_PROMOTION = "REPORT_ONLY_NO_PROMOTION"
REQUIRE_BOTH_BASELINE_COMPONENTS = "REQUIRE_BOTH_BASELINE_COMPONENTS"
REQUIRE_ANY_BASELINE_COMPONENT = "REQUIRE_ANY_BASELINE_COMPONENT"
MODEL_ORDERING_ONLY = "MODEL_ORDERING_ONLY"
CANDIDATE_OVER_INCUMBENT = "CANDIDATE_OVER_INCUMBENT"
PROMOTE = "PROMOTE"
DO_NOT_PROMOTE = "DO_NOT_PROMOTE"
EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"


@dataclass(frozen=True)
class PromotionResultV1:
    contract_id: str
    promotion_rule_id: str
    candidate_policy_identity: str
    status: str
    reasons: tuple[str, ...]

    def validate(self) -> "PromotionResultV1":
        if self.contract_id != PROMOTION_CONTRACT_ID:
            raise ValueError("unexpected promotion contract")
        if self.status not in {PROMOTE, DO_NOT_PROMOTE, EVIDENCE_INSUFFICIENT}:
            raise ValueError("invalid promotion status")
        if not self.candidate_policy_identity:
            raise ValueError("candidate_policy_identity required")
        if not self.reasons:
            raise ValueError("promotion reasons required")
        return self


def _candidate_is_higher(
    ordering: ModelOrderingResultV1,
    candidate_policy_identity: str,
) -> bool:
    ordering.validate()
    if candidate_policy_identity == ordering.left_policy_identity:
        return ordering.status == LEFT_HIGHER
    if candidate_policy_identity == ordering.right_policy_identity:
        return ordering.status == RIGHT_HIGHER
    raise ValueError("candidate is not part of ordering result")


def assess_promotion(
    *,
    promotion_rule_id: str,
    candidate_policy_identity: str,
    ordering: ModelOrderingResultV1 | None = None,
    components: BaselineComponentsV1 | None = None,
    improvement_evidence_pass: bool | None = None,
) -> PromotionResultV1:
    if promotion_rule_id == REPORT_ONLY_NO_PROMOTION:
        return PromotionResultV1(
            PROMOTION_CONTRACT_ID,
            promotion_rule_id,
            candidate_policy_identity,
            DO_NOT_PROMOTE,
            ("rule_is_report_only",),
        ).validate()

    if promotion_rule_id == REQUIRE_BOTH_BASELINE_COMPONENTS:
        if components is None:
            raise ValueError("components required")
        components.validate()
        ok = (
            components.buy_and_hold.status == COMPONENT_PASS
            and components.flat.status == COMPONENT_PASS
        )
        return PromotionResultV1(
            PROMOTION_CONTRACT_ID,
            promotion_rule_id,
            candidate_policy_identity,
            PROMOTE if ok else DO_NOT_PROMOTE,
            ("both_baseline_components_pass" if ok else "both_baseline_components_not_pass",),
        ).validate()

    if promotion_rule_id == REQUIRE_ANY_BASELINE_COMPONENT:
        if components is None:
            raise ValueError("components required")
        components.validate()
        ok = (
            components.buy_and_hold.status == COMPONENT_PASS
            or components.flat.status == COMPONENT_PASS
        )
        return PromotionResultV1(
            PROMOTION_CONTRACT_ID,
            promotion_rule_id,
            candidate_policy_identity,
            PROMOTE if ok else DO_NOT_PROMOTE,
            ("at_least_one_baseline_component_pass" if ok else "no_baseline_component_pass",),
        ).validate()

    if promotion_rule_id in {MODEL_ORDERING_ONLY, CANDIDATE_OVER_INCUMBENT}:
        if ordering is None:
            raise ValueError("ordering required")
        higher = _candidate_is_higher(ordering, candidate_policy_identity)
        if ordering.status == TIE:
            return PromotionResultV1(
                PROMOTION_CONTRACT_ID,
                promotion_rule_id,
                candidate_policy_identity,
                DO_NOT_PROMOTE,
                ("candidate_tied_incumbent",),
            ).validate()
        if not higher:
            return PromotionResultV1(
                PROMOTION_CONTRACT_ID,
                promotion_rule_id,
                candidate_policy_identity,
                DO_NOT_PROMOTE,
                ("candidate_not_higher_than_incumbent",),
            ).validate()
        if promotion_rule_id == CANDIDATE_OVER_INCUMBENT:
            if improvement_evidence_pass is None:
                return PromotionResultV1(
                    PROMOTION_CONTRACT_ID,
                    promotion_rule_id,
                    candidate_policy_identity,
                    EVIDENCE_INSUFFICIENT,
                    ("improvement_evidence_not_supplied",),
                ).validate()
            if not improvement_evidence_pass:
                return PromotionResultV1(
                    PROMOTION_CONTRACT_ID,
                    promotion_rule_id,
                    candidate_policy_identity,
                    EVIDENCE_INSUFFICIENT,
                    ("preregistered_improvement_evidence_failed",),
                ).validate()
        return PromotionResultV1(
            PROMOTION_CONTRACT_ID,
            promotion_rule_id,
            candidate_policy_identity,
            PROMOTE,
            (
                "candidate_higher_under_same_contract",
                "baseline_components_are_diagnostic_only",
            ),
        ).validate()

    raise ValueError(f"unknown promotion_rule_id: {promotion_rule_id}")
