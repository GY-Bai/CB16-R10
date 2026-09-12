from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from cb16_local_opt.action_contract_r0 import FLAT, LONG, SHORT, make_target_position_action_r0
from cb16_local_opt.actor_critic_supervisor_r0 import (
    ACCEPT,
    CLAMP,
    REJECT,
    make_supervisor_authority_state_r0,
    supervise_target_action_r0,
)
from cb16_local_opt.target_exposure_r0 import (
    ACTIVE,
    BELOW_MINIMUM,
    FLAT_ZERO,
    QUANTIZED_TO_ZERO,
    TARGET_RISK_SOURCE_R0,
    ZERO_RISK,
    TargetExposureAuthorityR0,
    make_target_exposure_authority_r0,
    map_permission_to_target_exposure_r0,
)


def _permission(
    direction: str = LONG,
    risk: float = 0.5,
    *,
    supervisor_cap: float = 1.0,
    current_direction: str = FLAT,
    current_risk: float = 0.0,
    margin_available: bool = True,
):
    action = make_target_position_action_r0(
        action_id=f"sizing:{direction}:{risk}",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=direction,
        requested_target_risk=risk,
    )
    authority = make_supervisor_authority_state_r0(
        authority_id="supervisor-authority",
        account_id="account-001",
        current_direction=current_direction,
        current_target_risk=current_risk,
        terminated=False,
        truncated=False,
        margin_available_for_new_exposure=margin_available,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=supervisor_cap,
    )
    return action, supervise_target_action_r0(action, authority)


def _authority(**overrides):
    kwargs = dict(
        authority_id="target-exposure-authority",
        account_id="account-001",
        equity=100_000.0,
        current_price=100.0,
        margin_capacity=100_000.0,
        max_gross_leverage=3.0,
        initial_margin_rate=None,
        declared_max_legal_notional=None,
        lot_step_size=None,
        lot_min_qty=None,
        lot_max_qty=None,
        min_notional=None,
    )
    kwargs.update(overrides)
    return make_target_exposure_authority_r0(**kwargs)


def test_linear_risk_fraction_maps_to_legal_notional_envelope() -> None:
    _, permission = _permission(LONG, 0.5)
    result = map_permission_to_target_exposure_r0(permission, _authority())

    assert permission.outcome == ACCEPT
    assert result.target_risk_source == TARGET_RISK_SOURCE_R0
    assert result.leverage_notional_cap == 300_000.0
    assert result.margin_notional_cap == pytest.approx(300_000.0)
    assert result.legal_notional_cap == pytest.approx(300_000.0)
    assert result.pre_filter_target_notional == pytest.approx(150_000.0)
    assert result.target_quantity == pytest.approx(1500.0)
    assert result.target_notional == pytest.approx(150_000.0)
    assert result.status == ACTIVE


def test_short_uses_same_unsigned_envelope_with_negative_quantity() -> None:
    _, permission = _permission(SHORT, 0.25)
    result = map_permission_to_target_exposure_r0(permission, _authority())
    assert result.target_quantity == pytest.approx(-750.0)
    assert result.target_notional == pytest.approx(75_000.0)
    assert result.status == ACTIVE


def test_declared_and_margin_caps_are_mechanical_minimums() -> None:
    _, permission = _permission(LONG, 1.0)

    declared = map_permission_to_target_exposure_r0(
        permission,
        _authority(declared_max_legal_notional=120_000.0),
    )
    assert declared.legal_notional_cap == 120_000.0
    assert declared.target_notional == 120_000.0

    margin = map_permission_to_target_exposure_r0(
        permission,
        _authority(
            margin_capacity=20_000.0,
            initial_margin_rate=0.2,
        ),
    )
    assert margin.margin_notional_cap == 100_000.0
    assert margin.legal_notional_cap == 100_000.0
    assert margin.target_notional == 100_000.0


def test_quantity_cap_is_part_of_legal_envelope() -> None:
    _, permission = _permission(LONG, 1.0)
    result = map_permission_to_target_exposure_r0(
        permission,
        _authority(lot_max_qty=400.0),
    )
    assert result.quantity_notional_cap == 40_000.0
    assert result.legal_notional_cap == 40_000.0
    assert result.target_quantity == 400.0


def test_lot_step_quantization_is_toward_zero_only() -> None:
    _, long_permission = _permission(LONG, 1.0)
    long_result = map_permission_to_target_exposure_r0(
        long_permission,
        _authority(
            equity=100.0,
            margin_capacity=100.0,
            max_gross_leverage=1.0,
            initial_margin_rate=1.0,
            current_price=30.0,
            lot_step_size=0.3,
        ),
    )
    assert long_result.pre_filter_target_notional == 100.0
    assert long_result.target_quantity == pytest.approx(3.3)
    assert long_result.target_notional == pytest.approx(99.0)
    assert long_result.target_notional <= long_result.pre_filter_target_notional

    _, short_permission = _permission(SHORT, 1.0)
    short_result = map_permission_to_target_exposure_r0(
        short_permission,
        _authority(
            equity=100.0,
            margin_capacity=100.0,
            max_gross_leverage=1.0,
            initial_margin_rate=1.0,
            current_price=30.0,
            lot_step_size=0.3,
        ),
    )
    assert short_result.target_quantity == pytest.approx(-3.3)
    assert short_result.target_notional == pytest.approx(99.0)


def test_minimum_quantity_and_notional_zero_out_without_rounding_up() -> None:
    _, permission = _permission(LONG, 0.1)
    below_qty = map_permission_to_target_exposure_r0(
        permission,
        _authority(
            equity=100.0,
            margin_capacity=100.0,
            max_gross_leverage=1.0,
            initial_margin_rate=1.0,
            current_price=100.0,
            lot_min_qty=0.2,
        ),
    )
    assert below_qty.status == BELOW_MINIMUM
    assert below_qty.target_quantity == 0.0
    assert below_qty.target_notional == 0.0

    below_notional = map_permission_to_target_exposure_r0(
        permission,
        _authority(
            equity=100.0,
            margin_capacity=100.0,
            max_gross_leverage=1.0,
            initial_margin_rate=1.0,
            current_price=100.0,
            min_notional=20.0,
        ),
    )
    assert below_notional.status == BELOW_MINIMUM
    assert below_notional.target_quantity == 0.0


def test_step_can_quantize_positive_request_to_zero() -> None:
    _, permission = _permission(LONG, 0.01)
    result = map_permission_to_target_exposure_r0(
        permission,
        _authority(
            equity=100.0,
            margin_capacity=100.0,
            max_gross_leverage=1.0,
            initial_margin_rate=1.0,
            current_price=100.0,
            lot_step_size=2.0,
        ),
    )
    assert result.status == QUANTIZED_TO_ZERO
    assert result.target_quantity == 0.0


def test_flat_and_nonflat_zero_risk_have_explicit_zero_states() -> None:
    _, flat_permission = _permission(FLAT, 0.0)
    flat = map_permission_to_target_exposure_r0(flat_permission, _authority())
    assert flat.status == FLAT_ZERO
    assert flat.target_quantity == 0.0

    _, zero_permission = _permission(LONG, 0.0)
    zero = map_permission_to_target_exposure_r0(zero_permission, _authority())
    assert zero.status == ZERO_RISK
    assert zero.target_quantity == 0.0


def test_supervisor_clamp_not_nominal_actor_request_controls_sizing() -> None:
    action, permission = _permission(LONG, 0.8, supervisor_cap=0.25)
    assert action.requested_target_risk == 0.8
    assert permission.outcome == CLAMP
    assert permission.permitted_target_risk == 0.25

    result = map_permission_to_target_exposure_r0(permission, _authority())
    assert result.target_risk == 0.25
    assert result.target_notional == pytest.approx(75_000.0)


def test_rejected_permission_is_never_sized() -> None:
    _, permission = _permission(
        LONG,
        0.5,
        margin_available=False,
    )
    assert permission.outcome == REJECT
    with pytest.raises(RuntimeError, match="ACTGT_REJECTED_PERMISSION_NOT_SIZABLE"):
        map_permission_to_target_exposure_r0(permission, _authority())


def test_same_permission_and_authority_are_bitwise_deterministic_at_payload_level() -> None:
    _, permission = _permission(SHORT, 0.37)
    authority = _authority(
        declared_max_legal_notional=123_456.0,
        lot_step_size=0.01,
    )
    a = map_permission_to_target_exposure_r0(permission, authority)
    b = map_permission_to_target_exposure_r0(permission, authority)
    assert a == b
    assert a.to_payload() == b.to_payload()
    assert a.semantic_sha256 == b.semantic_sha256
    assert a.permission_sha256 == permission.permission_sha256
    assert a.authority_sha256 == authority.semantic_sha256


def test_sizing_payload_contains_no_strategy_heuristic_inputs() -> None:
    _, permission = _permission(LONG, 0.4)
    result = map_permission_to_target_exposure_r0(permission, _authority())
    serialized = repr(result.to_payload()).lower()
    for forbidden in (
        "atr",
        "stop_distance",
        "confidence",
        "expected_return",
        "market_regime",
        "risk_fraction_per_trade",
    ):
        assert forbidden not in serialized


def test_authority_defaults_initial_margin_rate_to_inverse_max_leverage() -> None:
    authority = _authority(max_gross_leverage=4.0, initial_margin_rate=None)
    assert authority.initial_margin_rate == pytest.approx(0.25)


def test_authority_is_immutable_and_invalid_contracts_fail_closed() -> None:
    authority = _authority()
    with pytest.raises(FrozenInstanceError):
        authority.equity = 1.0  # type: ignore[misc]

    with pytest.raises(RuntimeError, match="ACTGT_LOT_RANGE_INVALID"):
        _authority(lot_min_qty=2.0, lot_max_qty=1.0)
    with pytest.raises(RuntimeError, match="ACTGT_PRICE_INVALID"):
        _authority(current_price=0.0)
    with pytest.raises(RuntimeError, match="ACTGT_INITIAL_MARGIN_RATE_INVALID"):
        _authority(initial_margin_rate=1.1)

    malformed = replace(
        authority,
        schema_version="UNKNOWN_TARGET_EXPOSURE_SCHEMA",
    )
    with pytest.raises(RuntimeError, match="ACTGT_AUTHORITY_SCHEMA_MISMATCH"):
        malformed.validate()


def test_zero_equity_or_margin_capacity_yields_zero_legal_envelope() -> None:
    _, permission = _permission(LONG, 0.5)
    zero_equity = map_permission_to_target_exposure_r0(
        permission,
        _authority(equity=0.0),
    )
    assert zero_equity.legal_notional_cap == 0.0
    assert zero_equity.target_quantity == 0.0

    zero_margin = map_permission_to_target_exposure_r0(
        permission,
        _authority(margin_capacity=0.0),
    )
    assert zero_margin.legal_notional_cap == 0.0
    assert zero_margin.target_quantity == 0.0
