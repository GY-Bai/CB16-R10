from __future__ import annotations

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_terminal_r0 import (
    make_account_terminal_state_r0,
)


def _economics(**overrides):
    kwargs = dict(
        account_id="acct",
        cash=0.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=10.0,
        realized_pnl_cumulative=-100.0,
        fees_cumulative=5.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=20.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    kwargs.update(overrides)
    return make_account_economics_state_r0(**kwargs)


def test_liquidation_does_not_imply_absorbing_when_debt_remains() -> None:
    state = make_account_terminal_state_r0(
        economics=_economics(liabilities=20.0),
        liquidation_event_occurred=True,
        trading_disabled=True,
        economic_responsibility_ended=False,
    )
    assert state.liquidation_event_occurred is True
    assert state.trading_disabled is True
    assert state.pending_debt == 20.0
    assert state.is_absorbing is False
    assert state.bootstrap_must_be_zero is False


def test_pending_fees_or_settlement_keep_state_nonabsorbing() -> None:
    econ = _economics(liabilities=0.0)
    fees = make_account_terminal_state_r0(
        economics=econ,
        liquidation_event_occurred=True,
        trading_disabled=True,
        pending_fees=2.0,
        economic_responsibility_ended=False,
    )
    settlement = make_account_terminal_state_r0(
        economics=econ,
        liquidation_event_occurred=True,
        trading_disabled=True,
        pending_settlement=3.0,
        economic_responsibility_ended=False,
    )
    assert fees.is_absorbing is False
    assert settlement.is_absorbing is False
    assert fees.bootstrap_must_be_zero is False
    assert settlement.bootstrap_must_be_zero is False


def test_true_responsibility_end_is_absorbing_only_when_nothing_remains() -> None:
    econ = _economics(liabilities=0.0, economic_responsibility_open=False)
    state = make_account_terminal_state_r0(
        economics=econ,
        liquidation_event_occurred=True,
        trading_disabled=True,
        economic_responsibility_ended=True,
    )
    assert state.has_unresolved_economic_responsibility is False
    assert state.is_absorbing is True
    assert state.bootstrap_must_be_zero is True


def test_cannot_declare_responsibility_end_with_debt() -> None:
    econ = _economics(liabilities=10.0, economic_responsibility_open=False)
    with pytest.raises(RuntimeError, match="RESPONSIBILITY_END_WITH_PENDING_ECONOMICS"):
        make_account_terminal_state_r0(
            economics=econ,
            liquidation_event_occurred=True,
            trading_disabled=True,
            economic_responsibility_ended=True,
        )


def test_trading_disabled_is_distinct_from_economic_absorption() -> None:
    state = make_account_terminal_state_r0(
        economics=_economics(liabilities=0.0),
        liquidation_event_occurred=False,
        trading_disabled=True,
        pending_settlement=1.0,
        economic_responsibility_ended=False,
    )
    assert state.trading_disabled is True
    assert state.is_absorbing is False
