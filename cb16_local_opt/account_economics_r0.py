from __future__ import annotations

from dataclasses import dataclass
import math

ACCOUNT_ECONOMICS_SCHEMA_R0 = "CB16_R11_BC_ACCOUNT_ECONOMICS_V1_R0"
ACCOUNT_OBSERVATION_RELATION_R0 = "ACCOUNT6_IS_PROJECTION_NOT_LEDGER"

FIELD_AUTHORITY_R0 = {
    "cash": "AUTHORITATIVE_LEDGER_STOCK",
    "position_quantity": "AUTHORITATIVE_LEDGER_STOCK",
    "position_cost_basis": "AUTHORITATIVE_LEDGER_STOCK",
    "mark_price": "MARKET_EXECUTION_INPUT",
    "realized_pnl_cumulative": "AUTHORITATIVE_AUDIT_FLOW_ALREADY_SETTLED_IN_CASH",
    "unrealized_pnl": "DERIVED_FROM_POSITION_COST_BASIS_AND_MARK",
    "fees_cumulative": "AUTHORITATIVE_AUDIT_FLOW_ALREADY_SETTLED_IN_CASH",
    "funding_cumulative": "AUTHORITATIVE_AUDIT_FLOW_ALREADY_SETTLED_IN_CASH",
    "margin_collateral": "AUTHORITATIVE_LEDGER_STOCK",
    "liabilities": "AUTHORITATIVE_LEDGER_STOCK",
    "external_capital_flows_cumulative": "AUTHORITATIVE_AUDIT_FLOW_ALREADY_SETTLED_IN_CASH",
    "equity": "DERIVED_AUTHORITATIVE_ECONOMIC_VALUE",
    "economic_responsibility_open": "AUTHORITATIVE_TERMINAL_RESPONSIBILITY_STATE",
}


def _finite(value: object, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(code)
    return 0.0 if out == 0.0 else out


def calculate_unrealized_pnl_r0(position_quantity: float, position_cost_basis: float, mark_price: float) -> float:
    q = _finite(position_quantity, "ACECO_R0_POSITION_INVALID")
    basis = _finite(position_cost_basis, "ACECO_R0_COST_BASIS_INVALID")
    mark = _finite(mark_price, "ACECO_R0_MARK_INVALID")
    if mark <= 0.0:
        raise RuntimeError("ACECO_R0_MARK_INVALID")
    if q == 0.0:
        return 0.0
    if basis <= 0.0:
        raise RuntimeError("ACECO_R0_COST_BASIS_INVALID")
    return q * (mark - basis)


def calculate_equity_r0(*, cash: float, margin_collateral: float, unrealized_pnl: float, liabilities: float) -> float:
    """Stocks-only equity; cumulative audit flows are already settled in cash."""
    return (
        _finite(cash, "ACECO_R0_CASH_INVALID")
        + _finite(margin_collateral, "ACECO_R0_COLLATERAL_INVALID")
        + _finite(unrealized_pnl, "ACECO_R0_UNREALIZED_INVALID")
        - _finite(liabilities, "ACECO_R0_LIABILITY_INVALID")
    )


@dataclass(frozen=True)
class AccountEconomicsStateR0:
    schema_version: str
    account_id: str
    cash: float
    position_quantity: float
    position_cost_basis: float
    mark_price: float
    realized_pnl_cumulative: float
    fees_cumulative: float
    funding_cumulative: float
    margin_collateral: float
    liabilities: float
    external_capital_flows_cumulative: float
    economic_responsibility_open: bool

    @property
    def unrealized_pnl(self) -> float:
        return calculate_unrealized_pnl_r0(
            self.position_quantity,
            self.position_cost_basis,
            self.mark_price,
        )

    @property
    def equity(self) -> float:
        return calculate_equity_r0(
            cash=self.cash,
            margin_collateral=self.margin_collateral,
            unrealized_pnl=self.unrealized_pnl,
            liabilities=self.liabilities,
        )

    def validate(self) -> None:
        if self.schema_version != ACCOUNT_ECONOMICS_SCHEMA_R0:
            raise RuntimeError("ACECO_R0_SCHEMA_MISMATCH")
        if not self.account_id:
            raise RuntimeError("ACECO_R0_ACCOUNT_ID_INVALID")
        for value, code in (
            (self.cash, "ACECO_R0_CASH_INVALID"),
            (self.position_quantity, "ACECO_R0_POSITION_INVALID"),
            (self.realized_pnl_cumulative, "ACECO_R0_REALIZED_INVALID"),
            (self.fees_cumulative, "ACECO_R0_FEES_INVALID"),
            (self.funding_cumulative, "ACECO_R0_FUNDING_INVALID"),
            (self.margin_collateral, "ACECO_R0_COLLATERAL_INVALID"),
            (self.liabilities, "ACECO_R0_LIABILITY_INVALID"),
            (self.external_capital_flows_cumulative, "ACECO_R0_EXTERNAL_FLOW_INVALID"),
        ):
            _finite(value, code)
        if self.margin_collateral < 0.0:
            raise RuntimeError("ACECO_R0_COLLATERAL_INVALID")
        if self.liabilities < 0.0:
            raise RuntimeError("ACECO_R0_LIABILITY_INVALID")
        if self.fees_cumulative < 0.0:
            raise RuntimeError("ACECO_R0_FEES_INVALID")
        if self.position_quantity == 0.0:
            if self.position_cost_basis != 0.0:
                raise RuntimeError("ACECO_R0_FLAT_COST_BASIS_MUST_BE_ZERO")
        elif self.position_cost_basis <= 0.0:
            raise RuntimeError("ACECO_R0_COST_BASIS_INVALID")
        _finite(self.unrealized_pnl, "ACECO_R0_UNREALIZED_INVALID")
        _finite(self.equity, "ACECO_R0_EQUITY_INVALID")
        if not isinstance(self.economic_responsibility_open, bool):
            raise RuntimeError("ACECO_R0_RESPONSIBILITY_FLAG_INVALID")


def make_account_economics_state_r0(**kwargs: object) -> AccountEconomicsStateR0:
    state = AccountEconomicsStateR0(schema_version=ACCOUNT_ECONOMICS_SCHEMA_R0, **kwargs)
    state.validate()
    return state
