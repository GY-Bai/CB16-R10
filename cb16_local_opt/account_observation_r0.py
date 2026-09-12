from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .account_economics_r0 import AccountEconomicsStateR0

ACCOUNT_OBSERVATION_SCHEMA_R0 = "CB16_R11_BC_ACCOUNT_POLICY_OBSERVATION_V1_R0"
ACCOUNT_POLICY_VISIBLE_FIELDS_R0 = (
    "cash",
    "position_quantity",
    "position_cost_basis",
    "unrealized_pnl",
    "equity",
    "margin_collateral",
    "liabilities",
    "economic_responsibility_open",
)
ACCOUNT_POLICY_OMITTED_LEDGER_FIELDS_R0 = (
    "schema_version",
    "account_id",
    "mark_price",
    "realized_pnl_cumulative",
    "fees_cumulative",
    "funding_cumulative",
    "external_capital_flows_cumulative",
)


def account_restore_state_sha256_r0(state: AccountEconomicsStateR0) -> str:
    state.validate()
    payload = dict(state.__dict__)
    payload["unrealized_pnl"] = state.unrealized_pnl
    payload["equity"] = state.equity
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class AccountPolicyObservationR0:
    schema_version: str
    source_restore_state_sha256: str
    cash: float
    position_quantity: float
    position_cost_basis: float
    unrealized_pnl: float
    equity: float
    margin_collateral: float
    liabilities: float
    economic_responsibility_open: bool
    omitted_ledger_fields: tuple[str, ...]

    def validate(self) -> None:
        if self.schema_version != ACCOUNT_OBSERVATION_SCHEMA_R0:
            raise RuntimeError("ACOBS_R0_SCHEMA_MISMATCH")
        if len(self.source_restore_state_sha256) != 64:
            raise RuntimeError("ACOBS_R0_SOURCE_HASH_INVALID")
        if tuple(self.omitted_ledger_fields) != ACCOUNT_POLICY_OMITTED_LEDGER_FIELDS_R0:
            raise RuntimeError("ACOBS_R0_OMITTED_FIELDS_MISMATCH")
        if self.position_quantity == 0.0 and self.position_cost_basis != 0.0:
            raise RuntimeError("ACOBS_R0_FLAT_COST_BASIS_INVALID")

    def model_payload(self) -> dict[str, object]:
        self.validate()
        return {field: getattr(self, field) for field in ACCOUNT_POLICY_VISIBLE_FIELDS_R0}


def project_account_observation_r0(state: AccountEconomicsStateR0) -> AccountPolicyObservationR0:
    state.validate()
    observation = AccountPolicyObservationR0(
        schema_version=ACCOUNT_OBSERVATION_SCHEMA_R0,
        source_restore_state_sha256=account_restore_state_sha256_r0(state),
        cash=state.cash,
        position_quantity=state.position_quantity,
        position_cost_basis=state.position_cost_basis,
        unrealized_pnl=state.unrealized_pnl,
        equity=state.equity,
        margin_collateral=state.margin_collateral,
        liabilities=state.liabilities,
        economic_responsibility_open=state.economic_responsibility_open,
        omitted_ledger_fields=ACCOUNT_POLICY_OMITTED_LEDGER_FIELDS_R0,
    )
    observation.validate()
    return observation


def restore_account_from_policy_observation_r0(_: AccountPolicyObservationR0) -> AccountEconomicsStateR0:
    raise RuntimeError("ACOBS_R0_LOSSY_POLICY_PROJECTION_NOT_RESTORE_AUTHORITY")
