from __future__ import annotations

from dataclasses import dataclass

from .account_economics_r0 import AccountEconomicsStateR0

ACCOUNT_TERMINAL_SCHEMA_R0 = "CB16_R11_BC_ACCOUNT_TERMINAL_V1_R0"


@dataclass(frozen=True)
class AccountTerminalStateR0:
    schema_version: str
    account_id: str
    liquidation_event_occurred: bool
    trading_disabled: bool
    pending_debt: float
    pending_fees: float
    pending_settlement: float
    economic_responsibility_ended: bool

    def validate(self) -> None:
        if self.schema_version != ACCOUNT_TERMINAL_SCHEMA_R0:
            raise RuntimeError("ACTER_R0_SCHEMA_MISMATCH")
        if not self.account_id:
            raise RuntimeError("ACTER_R0_ACCOUNT_ID_INVALID")
        for value in (self.pending_debt, self.pending_fees, self.pending_settlement):
            if value < 0.0:
                raise RuntimeError("ACTER_R0_PENDING_VALUE_INVALID")
        if self.economic_responsibility_ended and self.has_unresolved_economic_responsibility:
            raise RuntimeError("ACTER_R0_RESPONSIBILITY_END_WITH_PENDING_ECONOMICS")

    @property
    def has_unresolved_economic_responsibility(self) -> bool:
        return (
            self.pending_debt > 0.0
            or self.pending_fees > 0.0
            or self.pending_settlement > 0.0
        )

    @property
    def is_absorbing(self) -> bool:
        return self.economic_responsibility_ended and not self.has_unresolved_economic_responsibility

    @property
    def bootstrap_must_be_zero(self) -> bool:
        return self.is_absorbing


def make_account_terminal_state_r0(
    *,
    economics: AccountEconomicsStateR0,
    liquidation_event_occurred: bool,
    trading_disabled: bool,
    pending_fees: float = 0.0,
    pending_settlement: float = 0.0,
    economic_responsibility_ended: bool = False,
) -> AccountTerminalStateR0:
    economics.validate()
    state = AccountTerminalStateR0(
        schema_version=ACCOUNT_TERMINAL_SCHEMA_R0,
        account_id=economics.account_id,
        liquidation_event_occurred=bool(liquidation_event_occurred),
        trading_disabled=bool(trading_disabled),
        pending_debt=float(economics.liabilities),
        pending_fees=float(pending_fees),
        pending_settlement=float(pending_settlement),
        economic_responsibility_ended=bool(economic_responsibility_ended),
    )
    state.validate()
    if economics.economic_responsibility_open == state.economic_responsibility_ended:
        raise RuntimeError("ACTER_R0_ECONOMIC_RESPONSIBILITY_STATE_MISMATCH")
    return state
