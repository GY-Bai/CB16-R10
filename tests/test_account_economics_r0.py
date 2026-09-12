from __future__ import annotations

from cb16_local_opt.account_economics_r0 import (
    ACCOUNT_OBSERVATION_RELATION_R0,
    FIELD_AUTHORITY_R0,
    NEGATIVE_CASH_SETTLEMENT_VERSION_R0,
    calculate_equity_r0,
    make_account_economics_state_r0,
    settle_negative_cash_to_liability_r0,
)


def _state(**overrides):
    kwargs = dict(
        account_id="acct",
        cash=80.0,
        position_quantity=2.0,
        position_cost_basis=10.0,
        mark_price=12.0,
        realized_pnl_cumulative=5.0,
        fees_cumulative=2.0,
        funding_cumulative=-1.0,
        margin_collateral=20.0,
        liabilities=3.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    kwargs.update(overrides)
    return make_account_economics_state_r0(**kwargs)


def test_authoritative_equity_uses_stock_fields_once() -> None:
    state = _state()
    assert state.unrealized_pnl == 4.0
    assert state.equity == 101.0
    assert state.equity == calculate_equity_r0(cash=80.0, margin_collateral=20.0, unrealized_pnl=4.0, liabilities=3.0)


def test_realized_fees_funding_and_external_flows_are_audit_flows_not_double_counted() -> None:
    base = _state(realized_pnl_cumulative=0.0, fees_cumulative=0.0, funding_cumulative=0.0, external_capital_flows_cumulative=0.0)
    audit_changed = _state(realized_pnl_cumulative=999.0, fees_cumulative=99.0, funding_cumulative=-88.0, external_capital_flows_cumulative=123.0)
    assert base.equity == audit_changed.equity


def test_every_required_economic_field_has_one_declared_authority_meaning() -> None:
    required = {"cash", "position_quantity", "position_cost_basis", "mark_price", "realized_pnl_cumulative", "unrealized_pnl", "fees_cumulative", "funding_cumulative", "margin_collateral", "liabilities", "external_capital_flows_cumulative", "equity", "economic_responsibility_open"}
    assert set(FIELD_AUTHORITY_R0) == required


def test_short_unrealized_pnl_has_correct_sign() -> None:
    assert _state(position_quantity=-2.0, position_cost_basis=10.0, mark_price=8.0).unrealized_pnl == 4.0
    assert _state(position_quantity=-2.0, position_cost_basis=10.0, mark_price=12.0).unrealized_pnl == -4.0


def test_flat_account_has_zero_cost_basis_and_unrealized_pnl() -> None:
    assert _state(position_quantity=0.0, position_cost_basis=0.0).unrealized_pnl == 0.0


def test_account6_is_explicitly_not_the_authoritative_ledger() -> None:
    assert ACCOUNT_OBSERVATION_RELATION_R0 == "ACCOUNT6_IS_PROJECTION_NOT_LEDGER"


def test_negative_equity_is_not_silently_clamped() -> None:
    insolvent = _state(
        cash=-40.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        margin_collateral=0.0,
        liabilities=15.0,
        economic_responsibility_open=True,
    )
    assert insolvent.cash == -40.0
    assert insolvent.equity == -55.0
    assert insolvent.equity < 0.0


def test_explicit_cash_settlement_moves_shortfall_to_liability_without_destroying_loss() -> None:
    insolvent = _state(
        cash=-40.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        margin_collateral=0.0,
        liabilities=15.0,
        economic_responsibility_open=True,
    )
    settled, receipt = settle_negative_cash_to_liability_r0(insolvent)
    assert receipt.settlement_version == NEGATIVE_CASH_SETTLEMENT_VERSION_R0
    assert settled.cash == 0.0
    assert settled.liabilities == 55.0
    assert settled.equity == insolvent.equity == -55.0
    assert receipt.equity_before == receipt.equity_after == -55.0


def test_negative_equity_remains_negative_while_responsibility_is_open() -> None:
    state = _state(
        cash=0.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        margin_collateral=0.0,
        liabilities=25.0,
        economic_responsibility_open=True,
    )
    assert state.equity == -25.0
    assert state.economic_responsibility_open is True
