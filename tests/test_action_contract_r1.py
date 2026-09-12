from __future__ import annotations

import pytest

from cb16_local_opt.action_contract_r1 import (
    FLAT,
    LONG,
    SAME_NOMINAL_REQUEST,
    SHORT,
    TargetRequestRelationR1,
    classify_target_request_relation_r1,
    make_target_position_action_r1,
)
from cb16_local_opt.actor_critic_contract_r1 import ACTION_VERSION_R1


def _action(direction: str = LONG, risk: float = 0.5):
    return make_target_position_action_r1(
        action_id="a-1",
        policy_id="p",
        policy_version="v1",
        target_direction=direction,
        requested_target_risk=risk,
    )


def test_r1_action_uses_round2_semantic_identity() -> None:
    action = _action()
    assert action.schema_version == ACTION_VERSION_R1
    assert action.requested_target_risk_semantics == "TARGET_EXPOSURE_REQUEST"
    assert action.to_payload()["requested_target_risk"] == 0.5


def test_same_direction_and_risk_is_same_nominal_request_not_execution_noop() -> None:
    relation = classify_target_request_relation_r1(
        source_direction=LONG,
        source_requested_risk=0.5,
        action=_action(LONG, 0.5),
    )
    assert relation.relation_kind == SAME_NOMINAL_REQUEST
    assert relation.requires_target_recompute is True
    assert "NOOP" not in relation.relation_kind


def test_every_relation_requires_fresh_target_recompute() -> None:
    fixtures = [
        (FLAT, 0.0, LONG, 0.5),
        (LONG, 0.5, LONG, 0.75),
        (LONG, 0.5, LONG, 0.25),
        (LONG, 0.5, FLAT, 0.0),
        (LONG, 0.5, SHORT, 0.5),
        (SHORT, 0.5, LONG, 0.5),
        (SHORT, 0.5, SHORT, 0.5),
    ]
    for source_direction, source_risk, target_direction, target_risk in fixtures:
        relation = classify_target_request_relation_r1(
            source_direction=source_direction,
            source_requested_risk=source_risk,
            action=_action(target_direction, target_risk),
        )
        assert relation.requires_target_recompute is True


def test_same_nominal_long_half_risk_does_not_bind_quantity() -> None:
    relation = classify_target_request_relation_r1(
        source_direction=LONG,
        source_requested_risk=0.5,
        action=_action(LONG, 0.5),
    )
    payload = relation.__dict__
    assert "target_quantity" not in payload
    assert "current_quantity" not in payload
    assert relation.requires_target_recompute is True


def test_relation_cannot_claim_target_recompute_is_unnecessary() -> None:
    bad = TargetRequestRelationR1(
        action_id="a-1",
        source_direction=LONG,
        source_requested_risk=0.5,
        target_direction=LONG,
        requested_target_risk=0.5,
        relation_kind=SAME_NOMINAL_REQUEST,
        requires_target_recompute=False,
    )
    with pytest.raises(RuntimeError, match="TARGET_RECOMPUTE_REQUIRED"):
        bad.validate()


def test_flat_risk_remains_canonical_zero() -> None:
    with pytest.raises(RuntimeError, match="FLAT_TARGET_RISK_MUST_BE_ZERO"):
        _action(FLAT, 0.1)
