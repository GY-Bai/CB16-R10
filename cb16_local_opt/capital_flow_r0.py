from __future__ import annotations

from dataclasses import dataclass, replace

from .account_economics_r0 import AccountEconomicsStateR0

CAPITAL_FLOW_SCHEMA_R0 = "CB16_R11_BC_CAPITAL_FLOW_V1_R0"
INITIAL_FUNDING = "INITIAL_FUNDING"
DEPOSIT = "DEPOSIT"
WITHDRAWAL = "WITHDRAWAL"
TRANSFER = "TRANSFER"
NEW_ACCOUNT_FUNDING = "NEW_ACCOUNT_FUNDING"
CAPITAL_FLOW_KINDS_R0 = (
    INITIAL_FUNDING,
    DEPOSIT,
    WITHDRAWAL,
    TRANSFER,
    NEW_ACCOUNT_FUNDING,
)


@dataclass(frozen=True)
class CapitalFlowEventR0:
    schema_version: str
    flow_id: str
    flow_kind: str
    amount: float
    capital_source: str
    authorization_id: str
    source_account_id: str | None
    destination_account_id: str | None

    def validate(self) -> None:
        if self.schema_version != CAPITAL_FLOW_SCHEMA_R0:
            raise RuntimeError("ACCAP_R0_SCHEMA_MISMATCH")
        if not self.flow_id or not self.capital_source or not self.authorization_id:
            raise RuntimeError("ACCAP_R0_IDENTITY_INVALID")
        if self.flow_kind not in CAPITAL_FLOW_KINDS_R0:
            raise RuntimeError("ACCAP_R0_KIND_INVALID")
        if self.amount <= 0.0:
            raise RuntimeError("ACCAP_R0_AMOUNT_INVALID")
        if self.flow_kind in (INITIAL_FUNDING, DEPOSIT, NEW_ACCOUNT_FUNDING):
            if not self.destination_account_id or self.source_account_id is not None:
                raise RuntimeError("ACCAP_R0_DESTINATION_FLOW_INVALID")
        elif self.flow_kind == WITHDRAWAL:
            if not self.source_account_id or self.destination_account_id is not None:
                raise RuntimeError("ACCAP_R0_WITHDRAWAL_INVALID")
        elif self.flow_kind == TRANSFER:
            if not self.source_account_id or not self.destination_account_id:
                raise RuntimeError("ACCAP_R0_TRANSFER_ENDPOINT_INVALID")
            if self.source_account_id == self.destination_account_id:
                raise RuntimeError("ACCAP_R0_SELF_TRANSFER_FORBIDDEN")


def make_capital_flow_event_r0(*, flow_id: str, flow_kind: str, amount: float, capital_source: str, authorization_id: str, source_account_id: str | None = None, destination_account_id: str | None = None) -> CapitalFlowEventR0:
    event = CapitalFlowEventR0(
        schema_version=CAPITAL_FLOW_SCHEMA_R0,
        flow_id=flow_id,
        flow_kind=flow_kind,
        amount=float(amount),
        capital_source=capital_source,
        authorization_id=authorization_id,
        source_account_id=source_account_id,
        destination_account_id=destination_account_id,
    )
    event.validate()
    return event


def _signed_delta_for_account(account_id: str, event: CapitalFlowEventR0) -> float:
    event.validate()
    if event.flow_kind in (INITIAL_FUNDING, DEPOSIT, NEW_ACCOUNT_FUNDING):
        if event.destination_account_id != account_id:
            raise RuntimeError("ACCAP_R0_ACCOUNT_NOT_FLOW_ENDPOINT")
        return event.amount
    if event.flow_kind == WITHDRAWAL:
        if event.source_account_id != account_id:
            raise RuntimeError("ACCAP_R0_ACCOUNT_NOT_FLOW_ENDPOINT")
        return -event.amount
    if event.source_account_id == account_id:
        return -event.amount
    if event.destination_account_id == account_id:
        return event.amount
    raise RuntimeError("ACCAP_R0_ACCOUNT_NOT_FLOW_ENDPOINT")


def apply_capital_flow_r0(state: AccountEconomicsStateR0, event: CapitalFlowEventR0) -> AccountEconomicsStateR0:
    state.validate()
    delta = _signed_delta_for_account(state.account_id, event)
    if event.flow_kind in (INITIAL_FUNDING, NEW_ACCOUNT_FUNDING):
        if state.cash != 0.0 or state.margin_collateral != 0.0 or state.position_quantity != 0.0 or state.external_capital_flows_cumulative != 0.0:
            raise RuntimeError("ACCAP_R0_FUNDING_REQUIRES_PRISTINE_ACCOUNT")
    out = replace(
        state,
        cash=state.cash + delta,
        external_capital_flows_cumulative=state.external_capital_flows_cumulative + delta,
    )
    out.validate()
    return out


def assert_no_implicit_recapitalization_r0(before: AccountEconomicsStateR0, after: AccountEconomicsStateR0, declared_capital_flow_delta: float) -> None:
    before.validate()
    after.validate()
    if before.account_id != after.account_id:
        raise RuntimeError("ACCAP_R0_ACCOUNT_ID_CHANGED")
    observed = after.external_capital_flows_cumulative - before.external_capital_flows_cumulative
    if abs(observed - declared_capital_flow_delta) > 1e-12:
        raise RuntimeError("ACCAP_R0_UNDECLARED_CAPITAL_FLOW")
