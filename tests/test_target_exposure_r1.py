from __future__ import annotations

import math

from cb16_local_opt.action_contract_r1 import LONG, make_target_position_action_r1
from cb16_local_opt.target_exposure_r1 import (
    TARGET_RISK_SOURCE_R1,
    make_target_exposure_authority_r1,
    map_action_to_target_exposure_r1,
)


def _action(risk: float = 0.5):
    return make_target_position_action_r1(
        action_id="a",
        policy_id="p",
        policy_version="v1",
        target_direction=LONG,
        requested_target_risk=risk,
    )


def _authority(**overrides):
    kwargs = dict(
        authority_id="auth",
        account_id="acct",
        account_state_sha256="a" * 64,
        legal_envelope_id="env-1",
        equity=100.0,
        current_price=10.0,
        margin_capacity=100.0,
        max_gross_leverage=2.0,
        initial_margin_rate=0.5,
        declared_max_legal_notional=None,
        lot_step_size=None,
        lot_min_qty=None,
        lot_max_qty=None,
        min_notional=None,
    )
    kwargs.update(overrides)
    return make_target_exposure_authority_r1(**kwargs)


def _frozen_expected_qty(authority, risk: float) -> float:
    caps = [
        authority.equity * authority.max_gross_leverage,
        authority.margin_capacity / authority.initial_margin_rate,
    ]
    if authority.declared_max_legal_notional is not None:
        caps.append(authority.declared_max_legal_notional)
    if authority.lot_max_qty is not None:
        caps.append(authority.lot_max_qty * authority.current_price)
    legal_cap = min(caps)
    raw = risk * legal_cap / authority.current_price
    if authority.lot_step_size is not None:
        units = math.floor(abs(raw) / authority.lot_step_size + 1e-12)
        raw = units * authority.lot_step_size
    notional = raw * authority.current_price
    if authority.lot_min_qty is not None and raw < authority.lot_min_qty - 1e-12:
        return 0.0
    if authority.min_notional is not None and notional < authority.min_notional - 1e-9:
        return 0.0
    return raw


def test_baseline_matches_frozen_linear_sizing_math() -> None:
    authority = _authority()
    result = map_action_to_target_exposure_r1(_action(), authority)
    assert result.target_quantity == _frozen_expected_qty(authority, 0.5) == 10.0
    assert result.target_notional == 100.0


def test_changed_equity_recomputes_quantity() -> None:
    high = _authority(equity=100.0, margin_capacity=1000.0)
    low = _authority(equity=50.0, margin_capacity=1000.0, account_state_sha256="b" * 64)
    a = map_action_to_target_exposure_r1(_action(), high)
    b = map_action_to_target_exposure_r1(_action(), low)
    assert a.target_quantity == 10.0
    assert b.target_quantity == 5.0
    assert a.target_quantity != b.target_quantity


def test_changed_price_recomputes_quantity_without_changing_target_notional() -> None:
    p10 = _authority(current_price=10.0)
    p20 = _authority(current_price=20.0, account_state_sha256="b" * 64)
    a = map_action_to_target_exposure_r1(_action(), p10)
    b = map_action_to_target_exposure_r1(_action(), p20)
    assert a.target_notional == b.target_notional == 100.0
    assert a.target_quantity == 10.0
    assert b.target_quantity == 5.0


def test_changed_declared_legal_cap_recomputes_quantity() -> None:
    uncapped = _authority()
    capped = _authority(declared_max_legal_notional=80.0, legal_envelope_id="env-2")
    a = map_action_to_target_exposure_r1(_action(), uncapped)
    b = map_action_to_target_exposure_r1(_action(), capped)
    assert a.target_quantity == 10.0
    assert b.target_quantity == 4.0


def test_lot_max_and_step_follow_frozen_mapping() -> None:
    authority = _authority(lot_max_qty=6.0, lot_step_size=0.5)
    result = map_action_to_target_exposure_r1(_action(), authority)
    assert result.target_quantity == _frozen_expected_qty(authority, 0.5) == 3.0
    assert result.target_notional == 30.0


def test_minimum_filters_match_frozen_mapping() -> None:
    by_qty = _authority(declared_max_legal_notional=20.0, lot_min_qty=2.0)
    by_notional = _authority(declared_max_legal_notional=20.0, min_notional=15.0)
    assert map_action_to_target_exposure_r1(_action(), by_qty).target_quantity == 0.0
    assert map_action_to_target_exposure_r1(_action(), by_notional).target_quantity == 0.0


def test_result_binds_current_account_and_legal_envelope() -> None:
    authority = _authority(account_state_sha256="c" * 64, legal_envelope_id="env-current")
    result = map_action_to_target_exposure_r1(_action(), authority)
    assert result.account_state_sha256 == "c" * 64
    assert result.legal_envelope_id == "env-current"
    assert result.authority_sha256 == authority.semantic_sha256


def test_risk_is_exposure_request_not_confidence_or_cached_quantity() -> None:
    result = map_action_to_target_exposure_r1(_action(0.25), _authority())
    assert result.target_risk_source == TARGET_RISK_SOURCE_R1
    assert "CONFIDENCE" not in result.target_risk_source
    assert result.target_quantity == 5.0


def test_same_nominal_action_maps_differently_under_new_authority() -> None:
    action = _action(0.5)
    first = map_action_to_target_exposure_r1(action, _authority(current_price=10.0))
    second = map_action_to_target_exposure_r1(
        action,
        _authority(current_price=25.0, equity=60.0, margin_capacity=1000.0, account_state_sha256="d" * 64),
    )
    assert first.action_sha256 == second.action_sha256
    assert first.target_quantity != second.target_quantity
    assert second.target_quantity == _frozen_expected_qty(_authority(current_price=25.0, equity=60.0, margin_capacity=1000.0, account_state_sha256="d" * 64), 0.5)
