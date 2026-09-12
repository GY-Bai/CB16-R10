from __future__ import annotations

import math

import pytest

from cb16_local_opt.action_contract_r0 import (
    CLOSE_LONG,
    CLOSE_SHORT,
    DECREASE_LONG,
    DECREASE_SHORT,
    FLAT,
    INCREASE_LONG,
    INCREASE_SHORT,
    LONG,
    OPEN_LONG,
    OPEN_SHORT,
    REVERSE_LONG_TO_SHORT,
    REVERSE_SHORT_TO_LONG,
    SAME_TARGET_NOOP,
    SHORT,
    TARGET_POSITION_SEMANTICS_R0,
    TARGET_RISK_SEMANTICS_R0,
    TargetPositionActionR0,
    classify_target_state_transition_r0,
    make_target_position_action_r0,
)
from cb16_local_opt.actor_critic_contract_r0 import ACTION_VERSION_R0


@pytest.mark.parametrize(
    ("direction", "risk"),
    [
        (FLAT, 0.0),
        (SHORT, 0.0),
        (SHORT, 0.25),
        (SHORT, 1.0),
        (LONG, 0.0),
        (LONG, 0.25),
        (LONG, 1.0),
    ],
)
def test_valid_target_position_actions_round_trip_canonically(
    direction: str,
    risk: float,
) -> None:
    action = make_target_position_action_r0(
        action_id=f"A:{direction}:{risk}",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=direction,
        requested_target_risk=risk,
    )

    encoded = action.to_json()
    decoded = TargetPositionActionR0.from_json(encoded)

    assert decoded == action
    assert decoded.to_json() == encoded
    assert decoded.schema_version == ACTION_VERSION_R0
    assert decoded.action_semantics == TARGET_POSITION_SEMANTICS_R0
    assert decoded.requested_target_risk_semantics == TARGET_RISK_SEMANTICS_R0
    assert decoded.target_direction == direction
    assert decoded.requested_target_risk == risk


def test_action_is_explicit_target_state_not_entry_signal() -> None:
    action = make_target_position_action_r0(
        action_id="A:target-state",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=LONG,
        requested_target_risk=0.4,
    )
    payload = action.to_payload()
    assert payload["action_semantics"] == "TARGET_POSITION_STATE"

    payload["action_semantics"] = "ONE_TIME_ENTRY_SIGNAL"
    with pytest.raises(RuntimeError, match="ACACT_SEMANTICS_MISMATCH"):
        TargetPositionActionR0.from_payload(payload)


def test_requested_target_risk_is_exposure_request_not_confidence() -> None:
    action = make_target_position_action_r0(
        action_id="A:risk-semantics",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=LONG,
        requested_target_risk=0.7,
    )
    payload = action.to_payload()
    assert payload["requested_target_risk_semantics"] == "TARGET_EXPOSURE_REQUEST"

    contradictory = dict(payload)
    contradictory["requested_target_risk_semantics"] = "CONFIDENCE"
    with pytest.raises(RuntimeError, match="ACACT_TARGET_RISK_SEMANTICS_MISMATCH"):
        TargetPositionActionR0.from_payload(contradictory)

    fabricated = dict(payload)
    fabricated["confidence"] = 0.7
    with pytest.raises(RuntimeError, match="ACACT_FIELDS_UNKNOWN:confidence"):
        TargetPositionActionR0.from_payload(fabricated)


@pytest.mark.parametrize("risk", [0.000001, 0.25, 1.0])
def test_flat_requires_zero_target_risk(risk: float) -> None:
    with pytest.raises(RuntimeError, match="ACACT_FLAT_TARGET_RISK_MUST_BE_ZERO"):
        make_target_position_action_r0(
            action_id="A:flat-risk",
            policy_id="policy-r0",
            policy_version="policy-v1",
            target_direction=FLAT,
            requested_target_risk=risk,
        )


def test_equivalent_zero_risk_payloads_have_identical_canonical_encoding() -> None:
    base = {
        "action_id": "A:canonical-zero",
        "schema_version": ACTION_VERSION_R0,
        "action_semantics": TARGET_POSITION_SEMANTICS_R0,
        "policy_id": "policy-r0",
        "policy_version": "policy-v1",
        "target_direction": FLAT,
        "requested_target_risk_semantics": TARGET_RISK_SEMANTICS_R0,
    }
    encoded = {
        TargetPositionActionR0.from_payload(
            {**base, "requested_target_risk": value}
        ).to_json()
        for value in (0, 0.0, -0.0)
    }
    assert len(encoded) == 1
    assert '"requested_target_risk":0.0' in next(iter(encoded))


@pytest.mark.parametrize("direction", ["BUY", "SELL", "long", "", "NEUTRAL"])
def test_invalid_direction_fails_closed(direction: str) -> None:
    with pytest.raises(RuntimeError, match="ACACT_TARGET_DIRECTION_INVALID"):
        make_target_position_action_r0(
            action_id="A:bad-direction",
            policy_id="policy-r0",
            policy_version="policy-v1",
            target_direction=direction,
            requested_target_risk=0.5,
        )


@pytest.mark.parametrize(
    "risk",
    [-0.000001, 1.000001, math.inf, -math.inf, math.nan, True, "0.5"],
)
def test_invalid_requested_target_risk_fails_closed(risk: object) -> None:
    with pytest.raises(RuntimeError, match="ACACT_TARGET_RISK"):
        make_target_position_action_r0(
            action_id="A:bad-risk",
            policy_id="policy-r0",
            policy_version="policy-v1",
            target_direction=LONG,
            requested_target_risk=risk,  # type: ignore[arg-type]
        )


def test_action_and_policy_identity_fields_are_required() -> None:
    valid = make_target_position_action_r0(
        action_id="A:identity",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=SHORT,
        requested_target_risk=0.2,
    ).to_payload()

    for field_name in ("action_id", "policy_id", "policy_version"):
        payload = dict(valid)
        payload[field_name] = ""
        with pytest.raises(RuntimeError):
            TargetPositionActionR0.from_payload(payload)


def test_missing_unknown_and_wrong_schema_fail_closed() -> None:
    valid = make_target_position_action_r0(
        action_id="A:schema",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=FLAT,
        requested_target_risk=0.0,
    ).to_payload()

    missing = dict(valid)
    missing.pop("target_direction")
    with pytest.raises(RuntimeError, match="ACACT_FIELDS_MISSING:target_direction"):
        TargetPositionActionR0.from_payload(missing)

    unknown = dict(valid)
    unknown["legacy_risk"] = 0.99
    with pytest.raises(RuntimeError, match="ACACT_FIELDS_UNKNOWN:legacy_risk"):
        TargetPositionActionR0.from_payload(unknown)

    wrong_schema = dict(valid)
    wrong_schema["schema_version"] = "CB16_R11_TARGET_POSITION_ACTION_V2_R0"
    with pytest.raises(RuntimeError, match="ACACT_SCHEMA_VERSION_MISMATCH"):
        TargetPositionActionR0.from_payload(wrong_schema)


def test_json_must_decode_to_exact_action_object() -> None:
    with pytest.raises(RuntimeError, match="ACACT_JSON_INVALID"):
        TargetPositionActionR0.from_json("{")
    with pytest.raises(RuntimeError, match="ACACT_JSON_OBJECT_REQUIRED"):
        TargetPositionActionR0.from_json("[]")


def _transition(
    source_direction: str,
    source_risk: float,
    target_direction: str,
    target_risk: float,
):
    action = make_target_position_action_r0(
        action_id=f"T:{source_direction}:{source_risk}->{target_direction}:{target_risk}",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=target_direction,
        requested_target_risk=target_risk,
    )
    return classify_target_state_transition_r0(
        source_direction=source_direction,
        source_target_risk=source_risk,
        action=action,
    )


@pytest.mark.parametrize(
    ("source_direction", "source_risk", "target_direction", "target_risk", "kind"),
    [
        (FLAT, 0.0, FLAT, 0.0, SAME_TARGET_NOOP),
        (FLAT, 0.0, LONG, 0.4, OPEN_LONG),
        (FLAT, 0.0, SHORT, 0.4, OPEN_SHORT),
        (LONG, 0.4, LONG, 0.4, SAME_TARGET_NOOP),
        (LONG, 0.4, LONG, 0.8, INCREASE_LONG),
        (LONG, 0.8, LONG, 0.4, DECREASE_LONG),
        (LONG, 0.4, FLAT, 0.0, CLOSE_LONG),
        (LONG, 0.4, SHORT, 0.3, REVERSE_LONG_TO_SHORT),
        (SHORT, 0.4, SHORT, 0.4, SAME_TARGET_NOOP),
        (SHORT, 0.4, SHORT, 0.8, INCREASE_SHORT),
        (SHORT, 0.8, SHORT, 0.4, DECREASE_SHORT),
        (SHORT, 0.4, FLAT, 0.0, CLOSE_SHORT),
        (SHORT, 0.4, LONG, 0.3, REVERSE_SHORT_TO_LONG),
    ],
)
def test_full_target_state_transition_matrix(
    source_direction: str,
    source_risk: float,
    target_direction: str,
    target_risk: float,
    kind: str,
) -> None:
    transition = _transition(
        source_direction,
        source_risk,
        target_direction,
        target_risk,
    )
    assert transition.transition_kind == kind
    assert transition.target_direction == target_direction
    assert transition.requested_target_risk == target_risk
    assert transition.is_noop is (kind == SAME_TARGET_NOOP)
    assert transition.is_reversal is (
        kind in (REVERSE_LONG_TO_SHORT, REVERSE_SHORT_TO_LONG)
    )


def test_reversal_encoding_is_unambiguous_before_physics() -> None:
    long_to_short = _transition(LONG, 0.6, SHORT, 0.2)
    short_to_long = _transition(SHORT, 0.6, LONG, 0.2)

    assert long_to_short.transition_kind == REVERSE_LONG_TO_SHORT
    assert long_to_short.source_direction == LONG
    assert long_to_short.target_direction == SHORT
    assert short_to_long.transition_kind == REVERSE_SHORT_TO_LONG
    assert short_to_long.source_direction == SHORT
    assert short_to_long.target_direction == LONG
    assert long_to_short.transition_kind != short_to_long.transition_kind


def test_noncanonical_source_state_fails_closed_before_transition() -> None:
    action = make_target_position_action_r0(
        action_id="T:bad-source",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=LONG,
        requested_target_risk=0.5,
    )
    with pytest.raises(RuntimeError, match="ACACT_FLAT_TARGET_RISK_MUST_BE_ZERO"):
        classify_target_state_transition_r0(
            source_direction=FLAT,
            source_target_risk=0.1,
            action=action,
        )
