from __future__ import annotations

from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT, make_target_position_action_r1
from cb16_local_opt.actor_critic_supervisor_r1 import (
    ACCEPT,
    CLAMP,
    REJECT,
    make_supervisor_authority_r1,
    supervise_target_action_r1,
)


def _action(direction=LONG, risk=0.5):
    return make_target_position_action_r1(
        action_id="a",
        policy_id="p",
        policy_version="v1",
        target_direction=direction,
        requested_target_risk=(0.0 if direction == FLAT else risk),
    )


def _authority(**overrides):
    kwargs = dict(
        authority_id="sup",
        account_id="acct",
        account_state_sha256="a" * 64,
        terminated=False,
        truncated=False,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=1.0,
    )
    kwargs.update(overrides)
    return make_supervisor_authority_r1(**kwargs)


def test_equal_risk_is_legal_nominal_request_not_same_exposure() -> None:
    result = supervise_target_action_r1(_action(LONG, 0.5), _authority())
    assert result.outcome == ACCEPT
    assert result.permitted_target_risk == 0.5
    assert result.reason_codes == ("LEGAL_NOMINAL_REQUEST",)
    assert all("SAME_EXPOSURE" not in code for code in result.reason_codes)


def test_supervisor_has_no_cached_target_risk_or_margin_availability_field() -> None:
    fields = _authority().__dict__
    assert "current_target_risk" not in fields
    assert "margin_available_for_new_exposure" not in fields
    assert "current_quantity" not in fields


def test_reduced_quantity_cannot_be_blocked_here_for_lack_of_new_margin() -> None:
    result = supervise_target_action_r1(_action(LONG, 0.5), _authority())
    assert result.outcome == ACCEPT
    assert all("MARGIN" not in code for code in result.reason_codes)


def test_close_request_is_reachable_without_entry_margin_authority() -> None:
    result = supervise_target_action_r1(_action(FLAT, 0.0), _authority())
    assert result.outcome == ACCEPT
    assert result.permitted_target_direction == FLAT
    assert result.permitted_target_risk == 0.0
    assert all("MARGIN" not in code for code in result.reason_codes)


def test_hard_risk_cap_clamps_without_inventing_quantity_semantics() -> None:
    result = supervise_target_action_r1(_action(LONG, 0.8), _authority(max_permitted_target_risk=0.3))
    assert result.outcome == CLAMP
    assert result.requested_target_risk == 0.8
    assert result.permitted_target_risk == 0.3
    assert result.reason_codes == ("HARD_RISK_CAP",)


def test_illegal_direction_rejects_and_preserves_requested_action() -> None:
    result = supervise_target_action_r1(_action(SHORT, 0.5), _authority(legal_target_directions=(FLAT, LONG)))
    assert result.outcome == REJECT
    assert result.requested_target_direction == SHORT
    assert result.requested_target_risk == 0.5
    assert result.reason_codes == ("TARGET_DIRECTION_ILLEGAL",)


def test_terminated_or_truncated_account_rejects() -> None:
    assert supervise_target_action_r1(_action(), _authority(terminated=True)).outcome == REJECT
    assert supervise_target_action_r1(_action(), _authority(truncated=True)).outcome == REJECT


def test_permission_binds_authoritative_account_state() -> None:
    result = supervise_target_action_r1(_action(), _authority(account_state_sha256="b" * 64))
    assert result.account_state_sha256 == "b" * 64
    assert len(result.permission_sha256) == 64
