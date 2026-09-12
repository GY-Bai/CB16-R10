from __future__ import annotations

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.capital_flow_r0 import (
    DEPOSIT,
    INITIAL_FUNDING,
    NEW_ACCOUNT_FUNDING,
    TRANSFER,
    WITHDRAWAL,
    apply_capital_flow_r0,
    assert_no_implicit_recapitalization_r0,
    make_capital_flow_event_r0,
)


def _state(account_id="acct", **overrides):
    kwargs = dict(
        account_id=account_id,
        cash=0.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=10.0,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    kwargs.update(overrides)
    return make_account_economics_state_r0(**kwargs)


def _flow(kind, **overrides):
    kwargs = dict(
        flow_id="f-1",
        flow_kind=kind,
        amount=100.0,
        capital_source="OWNER_CAPITAL",
        authorization_id="auth-1",
        source_account_id=None,
        destination_account_id="acct",
    )
    kwargs.update(overrides)
    return make_capital_flow_event_r0(**kwargs)


def test_initial_funding_changes_cash_and_capital_flow_ledger_not_pnl() -> None:
    before = _state()
    after = apply_capital_flow_r0(before, _flow(INITIAL_FUNDING))
    assert after.cash == 100.0
    assert after.external_capital_flows_cumulative == 100.0
    assert after.realized_pnl_cumulative == before.realized_pnl_cumulative == 0.0
    assert after.fees_cumulative == before.fees_cumulative
    assert after.funding_cumulative == before.funding_cumulative


def test_deposit_is_explicit_external_capital_not_trading_return() -> None:
    before = _state(cash=-20.0, liabilities=20.0)
    after = apply_capital_flow_r0(before, _flow(DEPOSIT, amount=50.0))
    assert after.cash == 30.0
    assert after.external_capital_flows_cumulative == 50.0
    assert after.realized_pnl_cumulative == 0.0
    assert after.liabilities == 20.0


def test_withdrawal_records_negative_external_flow() -> None:
    before = _state(cash=100.0, external_capital_flows_cumulative=100.0)
    event = _flow(WITHDRAWAL, amount=25.0, source_account_id="acct", destination_account_id=None)
    after = apply_capital_flow_r0(before, event)
    assert after.cash == 75.0
    assert after.external_capital_flows_cumulative == 75.0


def test_transfer_updates_each_endpoint_with_opposite_ledger_sign() -> None:
    event = _flow(TRANSFER, amount=30.0, source_account_id="a", destination_account_id="b")
    source = apply_capital_flow_r0(_state("a", cash=50.0), event)
    destination = apply_capital_flow_r0(_state("b"), event)
    assert source.cash == 20.0
    assert source.external_capital_flows_cumulative == -30.0
    assert destination.cash == 30.0
    assert destination.external_capital_flows_cumulative == 30.0


def test_new_account_funding_requires_pristine_account() -> None:
    event = _flow(NEW_ACCOUNT_FUNDING)
    funded = apply_capital_flow_r0(_state(), event)
    assert funded.cash == 100.0
    with pytest.raises(RuntimeError, match="FUNDING_REQUIRES_PRISTINE_ACCOUNT"):
        apply_capital_flow_r0(_state(cash=-1.0), event)


def test_initial_funding_cannot_be_used_to_recapitalize_failed_existing_ledger() -> None:
    failed = _state(cash=-100.0, liabilities=100.0)
    with pytest.raises(RuntimeError, match="FUNDING_REQUIRES_PRISTINE_ACCOUNT"):
        apply_capital_flow_r0(failed, _flow(INITIAL_FUNDING))


def test_undeclared_capital_flow_fails_attribution_gate() -> None:
    before = _state()
    after = _state(cash=100.0, external_capital_flows_cumulative=100.0)
    with pytest.raises(RuntimeError, match="UNDECLARED_CAPITAL_FLOW"):
        assert_no_implicit_recapitalization_r0(before, after, declared_capital_flow_delta=0.0)
    assert_no_implicit_recapitalization_r0(before, after, declared_capital_flow_delta=100.0)
