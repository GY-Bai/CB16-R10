from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT
from cb16_local_opt.actor_critic_supervisor_r1 import make_supervisor_authority_r1
from cb16_local_opt.execution_feasibility_r0 import MechanicalExecutionAuthorityR0
from cb16_local_opt.execution_observation_r0 import build_execution_observation_r0
from cb16_local_opt.observation_normalization_r0 import (
    make_normalization_context_r0,
    normalize_policy_state_r0,
)
from cb16_local_opt.state_sufficiency_r0 import (
    ALIAS_ACTION_CONSISTENT,
    OBSERVATION_DISTINGUISHABLE,
    REPRESENTATION_GAP,
    assess_state_pair_r0,
    detect_state_aliases_r0,
    make_known_answer_state_case_r0,
    policy_observation_sha256_r0,
)
from cb16_local_opt.target_exposure_r1 import make_target_exposure_authority_r1


def _payload(*, equity_scale: float = 1.0, position_notional_scale: float = 0.0):
    return {
        "market_causal_features": (0.1, -0.2, 0.3),
        "account_scaled_features": (
            ("equity_scale", equity_scale),
            ("position_notional_scale", position_notional_scale),
        ),
        "execution_scaled_features": (
            ("legal_short", True),
            ("legal_flat", True),
            ("legal_long", True),
            ("max_permitted_target_risk", 1.0),
        ),
    }


def _case(case_id: str, state_char: str, action: str, payload=None):
    return make_known_answer_state_case_r0(
        case_id=case_id,
        authoritative_state_sha256=state_char * 64,
        policy_observation_payload=_payload() if payload is None else payload,
        optimal_action_key=action,
    )


def test_deliberate_alias_with_different_known_optimal_actions_is_representation_gap() -> None:
    left = _case("low-margin-state", "a", "FLAT")
    right = _case("high-capacity-state", "b", "LONG:1.0")
    assessment = assess_state_pair_r0(left, right)
    assert assessment.same_policy_observation is True
    assert assessment.same_optimal_action is False
    assert assessment.classification == REPRESENTATION_GAP


def test_same_observation_same_optimal_action_is_not_false_gap() -> None:
    left = _case("state-a", "a", "FLAT")
    right = _case("state-b", "b", "FLAT")
    assessment = assess_state_pair_r0(left, right)
    assert assessment.classification == ALIAS_ACTION_CONSISTENT


def test_observable_causal_difference_is_classified_as_distinguishable() -> None:
    left = _case("state-a", "a", "FLAT", _payload(equity_scale=0.5))
    right = _case("state-b", "b", "LONG:1.0", _payload(equity_scale=1.5))
    assessment = assess_state_pair_r0(left, right)
    assert assessment.same_policy_observation is False
    assert assessment.classification == OBSERVATION_DISTINGUISHABLE


def test_policy_observation_hash_is_canonical_over_mapping_insertion_order() -> None:
    first = {"b": (2.0, 3.0), "a": {"x": 1.0}}
    second = {"a": {"x": 1.0}, "b": (2.0, 3.0)}
    assert policy_observation_sha256_r0(first) == policy_observation_sha256_r0(second)


def test_changed_policy_visible_state_changes_observation_identity() -> None:
    assert policy_observation_sha256_r0(_payload(equity_scale=1.0)) != policy_observation_sha256_r0(
        _payload(equity_scale=0.9)
    )


def test_suite_detector_surfaces_any_representation_gap_without_calling_it_optimization() -> None:
    report = detect_state_aliases_r0(
        (
            _case("a", "a", "FLAT"),
            _case("b", "b", "LONG:1.0"),
            _case("c", "c", "FLAT", _payload(position_notional_scale=0.5)),
        )
    )
    assert report.case_count == 3
    assert report.pair_count == 3
    assert report.has_representation_gap is True
    assert report.representation_gap_count == 1
    assert all("OPTIM" not in item.classification for item in report.assessments)


def test_same_authoritative_state_cannot_have_conflicting_known_answers() -> None:
    left = _case("case-a", "a", "FLAT")
    right = _case("case-b", "a", "LONG:1.0")
    with pytest.raises(RuntimeError, match="KNOWN_ANSWER_CONTRADICTION"):
        assess_state_pair_r0(left, right)
    with pytest.raises(RuntimeError, match="KNOWN_ANSWER_CONTRADICTION"):
        detect_state_aliases_r0((left, right))


def test_duplicate_case_identity_fails_closed() -> None:
    left = _case("same-case", "a", "FLAT")
    right = _case("same-case", "b", "FLAT")
    with pytest.raises(RuntimeError, match="DUPLICATE"):
        detect_state_aliases_r0((left, right))


def test_noncanonical_or_nonfinite_policy_payload_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="NONCANONICAL"):
        policy_observation_sha256_r0({"bad": float("nan")})
    with pytest.raises(RuntimeError, match="NONCANONICAL"):
        policy_observation_sha256_r0({"bad": object()})
    with pytest.raises(RuntimeError, match="PAYLOAD_INVALID"):
        policy_observation_sha256_r0({})


def test_assessment_classification_tamper_fails_closed() -> None:
    assessment = assess_state_pair_r0(
        _case("left", "a", "FLAT"),
        _case("right", "b", "LONG:1.0"),
    )
    with pytest.raises(RuntimeError, match="CLASSIFICATION_INCONSISTENT"):
        replace(assessment, classification=OBSERVATION_DISTINGUISHABLE).validate()


# BC-030: a concrete account-dependent known-answer pair.
# The market observation at decision time is identical.  The fixture's future
# move is used only by the offline known-answer oracle below; it is never part
# of either policy observation.
BC030_MARKET_OBSERVATION = (0.125, -0.25, 0.5)
BC030_CURRENT_PRICE = 10.0
BC030_NEXT_PRICE = 9.95
BC030_FEE_PER_UNIT_TURNOVER = 0.10
BC030_TARGET_QUANTITIES = {
    SHORT: -1.0,
    FLAT: 0.0,
    LONG: 1.0,
}


def _bc030_one_step_equity_delta(*, current_quantity: float, target_quantity: float) -> float:
    market_pnl = target_quantity * (BC030_NEXT_PRICE - BC030_CURRENT_PRICE)
    turnover_cost = abs(target_quantity - current_quantity) * BC030_FEE_PER_UNIT_TURNOVER
    return market_pnl - turnover_cost


def _bc030_known_optimal_action(current_quantity: float) -> tuple[str, dict[str, float]]:
    values = {
        direction: _bc030_one_step_equity_delta(
            current_quantity=current_quantity,
            target_quantity=target_quantity,
        )
        for direction, target_quantity in BC030_TARGET_QUANTITIES.items()
    }
    best = max(values, key=values.__getitem__)
    ordered = sorted(values.values(), reverse=True)
    assert ordered[0] > ordered[1], "fixture must have a unique analytic optimum"
    return best, values


def _bc030_policy_case(*, case_id: str, current_quantity: float):
    cost_basis = 0.0 if current_quantity == 0.0 else BC030_CURRENT_PRICE
    margin_collateral = 0.0 if current_quantity == 0.0 else 10.0
    account_state = make_account_economics_state_r0(
        account_id=case_id,
        cash=100.0,
        position_quantity=current_quantity,
        position_cost_basis=cost_basis,
        mark_price=BC030_CURRENT_PRICE,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=margin_collateral,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    account = project_account_observation_r0(account_state)
    account_hash = account.source_restore_state_sha256

    supervisor = make_supervisor_authority_r1(
        authority_id=f"{case_id}-supervisor",
        account_id=case_id,
        account_state_sha256=account_hash,
        terminated=False,
        truncated=False,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=1.0,
    )
    target = make_target_exposure_authority_r1(
        authority_id=f"{case_id}-target",
        account_id=case_id,
        account_state_sha256=account_hash,
        legal_envelope_id="bc030-envelope",
        equity=account.equity,
        current_price=BC030_CURRENT_PRICE,
        margin_capacity=100.0 - margin_collateral,
        max_gross_leverage=2.0,
        initial_margin_rate=0.5,
        declared_max_legal_notional=100.0,
        lot_step_size=1.0,
        lot_min_qty=1.0,
        lot_max_qty=10.0,
        min_notional=10.0,
    )
    mechanical = MechanicalExecutionAuthorityR0(
        authority_id=f"{case_id}-mechanical",
        current_quantity=current_quantity,
        current_price=BC030_CURRENT_PRICE,
        available_margin_for_new_exposure=100.0 - margin_collateral,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.1,
        maintenance_collateral=100.0,
        lot_min_qty=1.0,
        lot_max_qty=10.0,
        min_notional=10.0,
    )
    execution = build_execution_observation_r0(
        supervisor_authority=supervisor,
        target_exposure_authority=target,
        mechanical_authority=mechanical,
    )
    normalized = normalize_policy_state_r0(
        market_causal_features=BC030_MARKET_OBSERVATION,
        account_observation=account,
        execution_observation=execution,
        context=make_normalization_context_r0(
            context_id=f"{case_id}-scale",
            money_scale=100.0,
        ),
    )
    optimal_action, values = _bc030_known_optimal_action(current_quantity)
    case = make_known_answer_state_case_r0(
        case_id=case_id,
        authoritative_state_sha256=account_hash,
        policy_observation_payload=normalized.model_payload(),
        optimal_action_key=optimal_action,
    )
    return case, normalized.model_payload(), values


def test_account_state_changes_known_optimal_action_under_same_market_observation() -> None:
    flat_case, flat_payload, flat_values = _bc030_policy_case(
        case_id="bc030-flat-account",
        current_quantity=0.0,
    )
    long_case, long_payload, long_values = _bc030_policy_case(
        case_id="bc030-long-account",
        current_quantity=1.0,
    )

    assert flat_payload["market_causal_features"] == long_payload["market_causal_features"]
    assert flat_case.optimal_action_key == FLAT
    assert long_case.optimal_action_key == LONG
    assert flat_values == pytest.approx({SHORT: -0.05, FLAT: 0.0, LONG: -0.15})
    assert long_values == pytest.approx({SHORT: -0.15, FLAT: -0.10, LONG: -0.05})

    assessment = assess_state_pair_r0(flat_case, long_case)
    assert assessment.same_optimal_action is False
    assert assessment.classification == OBSERVATION_DISTINGUISHABLE
    assert flat_case.policy_observation_sha256 != long_case.policy_observation_sha256


def test_erasing_account_state_turns_same_known_answer_pair_into_representation_gap() -> None:
    flat_case, flat_payload, _ = _bc030_policy_case(
        case_id="bc030-flat-account",
        current_quantity=0.0,
    )
    long_case, long_payload, _ = _bc030_policy_case(
        case_id="bc030-long-account",
        current_quantity=1.0,
    )
    assert flat_payload["market_causal_features"] == long_payload["market_causal_features"]

    market_only_payload = {
        "market_causal_features": flat_payload["market_causal_features"],
    }
    flat_market_only = make_known_answer_state_case_r0(
        case_id="bc030-flat-market-only",
        authoritative_state_sha256=flat_case.authoritative_state_sha256,
        policy_observation_payload=market_only_payload,
        optimal_action_key=flat_case.optimal_action_key,
    )
    long_market_only = make_known_answer_state_case_r0(
        case_id="bc030-long-market-only",
        authoritative_state_sha256=long_case.authoritative_state_sha256,
        policy_observation_payload=market_only_payload,
        optimal_action_key=long_case.optimal_action_key,
    )
    assessment = assess_state_pair_r0(flat_market_only, long_market_only)
    assert assessment.classification == REPRESENTATION_GAP
