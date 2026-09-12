from __future__ import annotations

import math

import pytest

from cb16_local_opt.action_contract_r0 import (
    FLAT,
    LONG,
    SHORT,
    TARGET_POSITION_SEMANTICS_R0,
    TargetPositionActionR0,
    make_target_position_action_r0,
)
from cb16_local_opt.actor_critic_contract_r0 import ACTION_VERSION_R0


@pytest.mark.parametrize("direction", [SHORT, FLAT, LONG])
@pytest.mark.parametrize("risk", [0.0, 0.25, 1.0])
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
    unknown["confidence"] = 0.99
    with pytest.raises(RuntimeError, match="ACACT_FIELDS_UNKNOWN:confidence"):
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
