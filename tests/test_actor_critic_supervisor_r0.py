from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from cb16_local_opt.action_contract_r0 import (
    FLAT,
    LONG,
    SHORT,
    make_target_position_action_r0,
)
from cb16_local_opt.actor_critic_supervisor_r0 import (
    ACCEPT,
    CLAMP,
    REJECT,
    SupervisorAuthorityStateR0,
    make_supervisor_authority_state_r0,
    supervise_target_action_r0,
)


def _action(direction: str = LONG, risk: float = 0.4):
    return make_target_position_action_r0(
        action_id=f"action:{direction}:{risk}",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=direction,
        requested_target_risk=risk,
    )


def _authority(**overrides):
    kwargs = dict(
        authority_id="hard-authority-001",
        account_id="account-001",
        current_direction=FLAT,
        current_target_risk=0.0,
        terminated=False,
        truncated=False,
        margin_available_for_new_exposure=True,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=1.0,
    )
    kwargs.update(overrides)
    return make_supervisor_authority_state_r0(**kwargs)


def test_identical_state_action_authority_produces_identical_permission() -> None:
    action = _action(LONG, 0.4)
    authority = _authority(max_permitted_target_risk=0.7)

    a = supervise_target_action_r0(action, authority)
    b = supervise_target_action_r0(action, authority)

    assert a == b
    assert a.to_payload() == b.to_payload()
    assert a.permission_sha256 == b.permission_sha256
    assert a.authority_sha256 == authority.semantic_sha256
    assert a.nominal_action_id == action.action_id


def test_accept_within_all_hard_authority() -> None:
    result = supervise_target_action_r0(
        _action(SHORT, 0.35),
        _authority(max_permitted_target_risk=0.6),
    )
    assert result.outcome == ACCEPT
    assert result.reason_code == "WITHIN_ALL_HARD_AUTHORITY"
    assert result.permitted_target_direction == SHORT
    assert result.permitted_target_risk == 0.35


def test_clamp_cannot_exceed_external_target_risk_cap() -> None:
    result = supervise_target_action_r0(
        _action(LONG, 0.8),
        _authority(max_permitted_target_risk=0.25),
    )
    assert result.outcome == CLAMP
    assert result.reason_code == "TARGET_RISK_CLAMPED_TO_HARD_AUTHORITY"
    assert result.permitted_target_direction == LONG
    assert result.permitted_target_risk == 0.25


def test_flat_target_is_accepted_without_margin_requirement_when_already_flat() -> None:
    result = supervise_target_action_r0(
        _action(FLAT, 0.0),
        _authority(margin_available_for_new_exposure=False),
    )
    assert result.outcome == ACCEPT
    assert result.reason_code == "FLAT_TARGET_WITHIN_HARD_AUTHORITY"
    assert result.permitted_target_direction == FLAT
    assert result.permitted_target_risk == 0.0


@pytest.mark.parametrize(
    ("authority_overrides", "reason"),
    [
        ({"terminated": True}, "TERMINATED_ACCOUNT"),
        ({"truncated": True}, "TRUNCATED_ACCOUNT"),
        ({"legal_target_directions": (FLAT, SHORT)}, "TARGET_DIRECTION_NOT_LEGAL"),
        (
            {"margin_available_for_new_exposure": False},
            "MARGIN_UNAVAILABLE_FOR_NEW_EXPOSURE",
        ),
        ({"max_permitted_target_risk": 0.0}, "TARGET_RISK_AUTHORITY_EXHAUSTED"),
    ],
)
def test_hard_authority_rejects_actor_request_without_bypass(
    authority_overrides: dict[str, object],
    reason: str,
) -> None:
    result = supervise_target_action_r0(
        _action(LONG, 0.7),
        _authority(**authority_overrides),
    )
    assert result.outcome == REJECT
    assert result.reason_code == reason
    assert result.permitted_target_direction == FLAT
    assert result.permitted_target_risk == 0.0


def test_rejection_precedence_is_stable_and_deterministic() -> None:
    result = supervise_target_action_r0(
        _action(LONG, 0.8),
        _authority(
            terminated=True,
            truncated=True,
            margin_available_for_new_exposure=False,
            legal_target_directions=(),
            max_permitted_target_risk=0.0,
        ),
    )
    assert result.outcome == REJECT
    assert result.reason_code == "TERMINATED_ACCOUNT"


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_held_position_reduce_is_authorized_without_new_margin(direction: str) -> None:
    result = supervise_target_action_r0(
        _action(direction, 0.2),
        _authority(
            current_direction=direction,
            current_target_risk=0.6,
            margin_available_for_new_exposure=False,
        ),
    )
    assert result.outcome == ACCEPT
    assert result.reason_code == "HELD_POSITION_REDUCE_AUTHORIZED"
    assert result.permitted_target_direction == direction
    assert result.permitted_target_risk == 0.2


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_held_position_close_is_authorized_without_new_margin(direction: str) -> None:
    result = supervise_target_action_r0(
        _action(FLAT, 0.0),
        _authority(
            current_direction=direction,
            current_target_risk=0.6,
            margin_available_for_new_exposure=False,
        ),
    )
    assert result.outcome == ACCEPT
    assert result.reason_code == "HELD_POSITION_CLOSE_AUTHORIZED"
    assert result.permitted_target_direction == FLAT
    assert result.permitted_target_risk == 0.0
    assert "FORCED_NOOP" not in str(result.to_payload())
    assert "POSITION_ALREADY_OPEN" not in str(result.to_payload())


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_same_held_target_is_not_blocked_by_margin(direction: str) -> None:
    result = supervise_target_action_r0(
        _action(direction, 0.4),
        _authority(
            current_direction=direction,
            current_target_risk=0.4,
            margin_available_for_new_exposure=False,
        ),
    )
    assert result.outcome == ACCEPT
    assert result.reason_code == "HELD_POSITION_SAME_TARGET_AUTHORIZED"
    assert result.permitted_target_risk == 0.4


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_held_exposure_increase_requires_new_margin(direction: str) -> None:
    rejected = supervise_target_action_r0(
        _action(direction, 0.7),
        _authority(
            current_direction=direction,
            current_target_risk=0.4,
            margin_available_for_new_exposure=False,
        ),
    )
    assert rejected.outcome == REJECT
    assert rejected.reason_code == "MARGIN_UNAVAILABLE_FOR_EXPOSURE_INCREASE"
    assert rejected.permitted_target_risk == 0.4

    accepted = supervise_target_action_r0(
        _action(direction, 0.7),
        _authority(
            current_direction=direction,
            current_target_risk=0.4,
            margin_available_for_new_exposure=True,
        ),
    )
    assert accepted.outcome == ACCEPT
    assert accepted.reason_code == "HELD_POSITION_INCREASE_AUTHORIZED"
    assert accepted.permitted_target_risk == 0.7


@pytest.mark.parametrize("direction", [LONG, SHORT])
def test_hard_cap_can_clamp_held_request_to_lower_exposure(direction: str) -> None:
    result = supervise_target_action_r0(
        _action(direction, 0.8),
        _authority(
            current_direction=direction,
            current_target_risk=0.5,
            margin_available_for_new_exposure=False,
            max_permitted_target_risk=0.3,
        ),
    )
    assert result.outcome == CLAMP
    assert result.reason_code == "HELD_POSITION_TARGET_RISK_CLAMPED_TO_HARD_AUTHORITY"
    assert result.permitted_target_direction == direction
    assert result.permitted_target_risk == 0.3


@pytest.mark.parametrize(
    ("current_direction", "target_direction"),
    [(LONG, SHORT), (SHORT, LONG)],
)
def test_reversal_remains_fail_closed_until_ac012(
    current_direction: str,
    target_direction: str,
) -> None:
    result = supervise_target_action_r0(
        _action(target_direction, 0.3),
        _authority(
            current_direction=current_direction,
            current_target_risk=0.5,
        ),
    )
    assert result.outcome == REJECT
    assert result.reason_code == "REVERSAL_NOT_AUTHORIZED_R0"
    assert result.permitted_target_direction == current_direction
    assert result.permitted_target_risk == 0.5


def test_authority_direction_set_is_canonicalized() -> None:
    a = _authority(legal_target_directions=(LONG, SHORT, FLAT))
    b = _authority(legal_target_directions=(SHORT, FLAT, LONG))
    assert a.legal_target_directions == (SHORT, FLAT, LONG)
    assert a == b
    assert a.semantic_sha256 == b.semantic_sha256


def test_authority_and_permission_records_are_immutable() -> None:
    authority = _authority()
    permission = supervise_target_action_r0(_action(), authority)

    with pytest.raises(FrozenInstanceError):
        authority.account_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        permission.outcome = REJECT  # type: ignore[misc]


def test_direct_noncanonical_authority_fails_closed() -> None:
    authority = _authority()
    noncanonical = SupervisorAuthorityStateR0(
        schema_version=authority.schema_version,
        permission_execution_version=authority.permission_execution_version,
        authority_id=authority.authority_id,
        account_id=authority.account_id,
        current_direction=authority.current_direction,
        current_target_risk=authority.current_target_risk,
        terminated=authority.terminated,
        truncated=authority.truncated,
        margin_available_for_new_exposure=authority.margin_available_for_new_exposure,
        legal_target_directions=(LONG, SHORT, FLAT),
        max_permitted_target_risk=authority.max_permitted_target_risk,
    )
    with pytest.raises(RuntimeError, match="ACSUP_LEGAL_DIRECTIONS_NONCANONICAL"):
        noncanonical.validate()


def test_permission_tampering_changes_or_invalidates_identity() -> None:
    permission = supervise_target_action_r0(_action(LONG, 0.4), _authority())
    original_hash = permission.permission_sha256

    modified = replace(permission, reason_code="DIFFERENT_AUDIT_REASON")
    modified.validate()
    assert modified.permission_sha256 != original_hash

    with pytest.raises(RuntimeError, match="ACSUP_OUTCOME_INVALID"):
        replace(permission, outcome="FORCED_NOOP").validate()


def test_requested_target_risk_is_not_confidence_or_permission_authority() -> None:
    action = _action(LONG, 0.9)
    result = supervise_target_action_r0(
        action,
        _authority(max_permitted_target_risk=0.2),
    )
    assert action.requested_target_risk == 0.9
    assert result.outcome == CLAMP
    assert result.permitted_target_risk == 0.2
    serialized = repr(result.to_payload()).lower()
    assert "confidence" not in serialized
