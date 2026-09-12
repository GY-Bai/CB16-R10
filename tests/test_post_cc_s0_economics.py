import pytest

from cb16_local_opt.cc_economic_promotion_r0 import (
    UNRESOLVED_OWNER_DECISION,
    assess as legacy_assess,
)
from cb16_local_opt.cc_experience_wire_r0 import CCEconomicResultV1
from cb16_local_opt.post_cc_baseline_components_v1 import (
    FAIL,
    PASS,
    TIE as COMPONENT_TIE,
    from_deltas,
)
from cb16_local_opt.post_cc_economic_adapter_v1 import adapt_w05
from cb16_local_opt.post_cc_economic_migration_v1 import (
    LEGACY_PROMOTION_CONTRACT_ID,
    SUCCESSOR_PROMOTION_CONTRACT_ID,
    migrate_legacy_result,
    require_post_cc_promotion_authority,
)
from cb16_local_opt.post_cc_economic_ordering_v1 import (
    LEFT_HIGHER,
    RIGHT_HIGHER,
    TIE,
    compare_policy_measures,
)
from cb16_local_opt.post_cc_promotion_v1 import (
    CANDIDATE_OVER_INCUMBENT,
    DO_NOT_PROMOTE,
    EVIDENCE_INSUFFICIENT,
    PROMOTE,
    REQUIRE_ANY_BASELINE_COMPONENT,
    REQUIRE_BOTH_BASELINE_COMPONENTS,
    assess_promotion,
)


def result(policy, mean, bh, flat, cohort="c", horizon="T", capital="cap"):
    return CCEconomicResultV1(
        "e-" + policy,
        "frozen_checkpoint",
        policy,
        cohort,
        horizon,
        capital,
        ({"account_lineage_id": "a", "arithmetic_return": mean},),
        mean,
        bh,
        flat,
        {"failed": 0},
        {"median": mean},
        "SYNTHETIC",
    )


def measure(*args, **kwargs):
    return adapt_w05(result(*args, **kwargs)).ordering_measure


def test_ordering_is_independent_of_baselines():
    a = measure("A", 0.10, -0.10, 0.10)
    b = measure("B", 0.05, -0.15, 0.05)
    assert compare_policy_measures(a, b).status == LEFT_HIGHER
    a2 = measure("A", 0.10, 999, -999)
    b2 = measure("B", 0.05, -999, 999)
    assert compare_policy_measures(a2, b2).status == LEFT_HIGHER


def test_ordering_reverse_tie_and_contract_mismatch():
    assert (
        compare_policy_measures(measure("A", 0.01, 0, 0), measure("B", 0.02, 0, 0)).status
        == RIGHT_HIGHER
    )
    assert (
        compare_policy_measures(measure("A", 0.01, 0, 0), measure("B", 0.01, 0, 0)).status
        == TIE
    )
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        compare_policy_measures(
            measure("A", 0.01, 0, 0, cohort="c1"),
            measure("B", 0.02, 0, 0, cohort="c2"),
        )


def test_all_parallel_baseline_combinations_are_valid():
    combos = [
        (1, 1, PASS, PASS),
        (1, -1, PASS, FAIL),
        (-1, 1, FAIL, PASS),
        (-1, -1, FAIL, FAIL),
    ]
    for bh, flat, bh_status, flat_status in combos:
        components = from_deltas(buy_hold_delta=bh, flat_delta=flat)
        assert (components.buy_and_hold.status, components.flat.status) == (
            bh_status,
            flat_status,
        )
    tie = from_deltas(buy_hold_delta=0, flat_delta=0)
    assert tie.buy_and_hold.status == COMPONENT_TIE
    assert tie.flat.status == COMPONENT_TIE


def test_legacy_disagreement_is_reproducible_but_not_current_owner_question():
    old_result = result("A", 0.1, -0.1, 0.1)
    assert legacy_assess(old_result).status == UNRESOLVED_OWNER_DECISION
    migrated = migrate_legacy_result(old_result)
    assert migrated.legacy_status == UNRESOLVED_OWNER_DECISION
    assert migrated.adapted_result.baseline_components.buy_and_hold.status == FAIL
    assert migrated.adapted_result.baseline_components.flat.status == PASS
    assert migrated.promotion_rule_required is True
    assert migrated.current_owner_uncertainty is False


def test_post_cc_router_rejects_legacy_authority():
    with pytest.raises(ValueError, match="CONTRACT_MISMATCH"):
        require_post_cc_promotion_authority(LEGACY_PROMOTION_CONTRACT_ID)
    assert (
        require_post_cc_promotion_authority(SUCCESSOR_PROMOTION_CONTRACT_ID)
        == SUCCESSOR_PROMOTION_CONTRACT_ID
    )


def test_explicit_promotion_rules_separate_concerns():
    components = from_deltas(buy_hold_delta=-0.1, flat_delta=0.1)
    assert (
        assess_promotion(
            promotion_rule_id=REQUIRE_BOTH_BASELINE_COMPONENTS,
            candidate_policy_identity="A",
            components=components,
        ).status
        == DO_NOT_PROMOTE
    )
    assert (
        assess_promotion(
            promotion_rule_id=REQUIRE_ANY_BASELINE_COMPONENT,
            candidate_policy_identity="A",
            components=components,
        ).status
        == PROMOTE
    )
    ordering = compare_policy_measures(
        measure("A", 0.10, -0.1, 0.1),
        measure("INC", 0.05, 0.2, 0.2),
    )
    assert (
        assess_promotion(
            promotion_rule_id=CANDIDATE_OVER_INCUMBENT,
            candidate_policy_identity="A",
            ordering=ordering,
            improvement_evidence_pass=True,
        ).status
        == PROMOTE
    )
    assert (
        assess_promotion(
            promotion_rule_id=CANDIDATE_OVER_INCUMBENT,
            candidate_policy_identity="A",
            ordering=ordering,
            improvement_evidence_pass=False,
        ).status
        == EVIDENCE_INSUFFICIENT
    )


def test_w05_adapter_preserves_independent_deltas_and_diagnostics():
    old_result = result("A", 0.1, -0.2, 0.3)
    adapted = adapt_w05(old_result)
    assert adapted.ordering_measure.mean_arithmetic_return == 0.1
    assert adapted.baseline_components.buy_and_hold.delta == -0.2
    assert adapted.baseline_components.flat.delta == 0.3
    assert adapted.failure_counts == {"failed": 0}
