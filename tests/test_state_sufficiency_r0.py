from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.state_sufficiency_r0 import (
    ALIAS_ACTION_CONSISTENT,
    OBSERVATION_DISTINGUISHABLE,
    REPRESENTATION_GAP,
    assess_state_pair_r0,
    detect_state_aliases_r0,
    make_known_answer_state_case_r0,
    policy_observation_sha256_r0,
)


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
