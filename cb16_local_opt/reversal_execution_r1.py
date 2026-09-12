from __future__ import annotations

from dataclasses import dataclass

from .action_contract_r1 import FLAT, LONG, SHORT, TargetPositionActionR1
from .execution_feasibility_r0 import (
    FEASIBLE,
    MechanicalExecutionAuthorityR0,
    evaluate_execution_feasibility_r0,
)
from .target_exposure_r1 import (
    TargetExposureResultR1,
    make_target_exposure_authority_r1,
    map_action_to_target_exposure_r1,
)

REVERSAL_SCHEMA_R1 = "CB16_R11_BC_TWO_PHASE_REVERSAL_V1_R1"
REVERSAL_EXECUTED = "REVERSAL_EXECUTED"
REVERSAL_CLOSED_OPEN_REJECTED = "REVERSAL_CLOSED_OPEN_REJECTED"


@dataclass(frozen=True)
class IntermediateFlatAccountR1:
    account_id: str
    account_state_sha256: str
    source_direction: str
    close_quantity: float
    close_fee: float
    close_realized_pnl: float
    equity_after_close: float
    margin_capacity_after_close: float
    execution_price: float

    def validate(self) -> None:
        if not self.account_id or len(self.account_state_sha256) != 64:
            raise RuntimeError("ACREV_R1_INTERMEDIATE_ACCOUNT_INVALID")
        if self.source_direction not in (LONG, SHORT):
            raise RuntimeError("ACREV_R1_SOURCE_DIRECTION_INVALID")
        if self.close_quantity <= 0.0:
            raise RuntimeError("ACREV_R1_CLOSE_QUANTITY_INVALID")
        if self.close_fee < 0.0:
            raise RuntimeError("ACREV_R1_CLOSE_FEE_INVALID")
        if self.execution_price <= 0.0:
            raise RuntimeError("ACREV_R1_PRICE_INVALID")
        if self.margin_capacity_after_close < 0.0:
            raise RuntimeError("ACREV_R1_MARGIN_CAPACITY_INVALID")


@dataclass(frozen=True)
class ReversalExecutionResultR1:
    schema_version: str
    status: str
    close_leg: IntermediateFlatAccountR1
    open_target: TargetExposureResultR1
    open_feasibility_status: str
    open_reason_codes: tuple[str, ...]
    final_direction: str
    final_target_quantity: float
    close_costs_preserved: bool


def execute_two_phase_reversal_r1(
    *,
    intermediate_flat: IntermediateFlatAccountR1,
    opposite_action: TargetPositionActionR1,
    legal_envelope_id: str,
    max_gross_leverage: float,
    initial_margin_rate: float | None,
    available_margin_for_new_exposure: float,
    maintenance_margin_rate: float,
    maintenance_collateral: float,
    declared_max_legal_notional: float | None = None,
) -> ReversalExecutionResultR1:
    """Coordinate only after the first close has produced an authoritative flat account."""
    intermediate_flat.validate()
    opposite_action.validate()
    expected = SHORT if intermediate_flat.source_direction == LONG else LONG
    if opposite_action.target_direction != expected:
        raise RuntimeError("ACREV_R1_OPPOSITE_DIRECTION_REQUIRED")

    sizing = make_target_exposure_authority_r1(
        authority_id="REVERSAL_INTERMEDIATE_FLAT_SIZING",
        account_id=intermediate_flat.account_id,
        account_state_sha256=intermediate_flat.account_state_sha256,
        legal_envelope_id=legal_envelope_id,
        equity=intermediate_flat.equity_after_close,
        current_price=intermediate_flat.execution_price,
        margin_capacity=intermediate_flat.margin_capacity_after_close,
        max_gross_leverage=max_gross_leverage,
        initial_margin_rate=initial_margin_rate,
        declared_max_legal_notional=declared_max_legal_notional,
    )
    target = map_action_to_target_exposure_r1(opposite_action, sizing)
    feasibility = evaluate_execution_feasibility_r0(
        target,
        MechanicalExecutionAuthorityR0(
            authority_id="REVERSAL_SECOND_LEG_MECHANICAL",
            current_quantity=0.0,
            current_price=intermediate_flat.execution_price,
            available_margin_for_new_exposure=available_margin_for_new_exposure,
            initial_margin_rate=sizing.initial_margin_rate,
            maintenance_margin_rate=maintenance_margin_rate,
            maintenance_collateral=maintenance_collateral,
        ),
    )

    if feasibility.status != FEASIBLE:
        return ReversalExecutionResultR1(
            schema_version=REVERSAL_SCHEMA_R1,
            status=REVERSAL_CLOSED_OPEN_REJECTED,
            close_leg=intermediate_flat,
            open_target=target,
            open_feasibility_status=feasibility.status,
            open_reason_codes=feasibility.reason_codes,
            final_direction=FLAT,
            final_target_quantity=0.0,
            close_costs_preserved=True,
        )

    return ReversalExecutionResultR1(
        schema_version=REVERSAL_SCHEMA_R1,
        status=REVERSAL_EXECUTED,
        close_leg=intermediate_flat,
        open_target=target,
        open_feasibility_status=feasibility.status,
        open_reason_codes=feasibility.reason_codes,
        final_direction=opposite_action.target_direction,
        final_target_quantity=target.target_quantity,
        close_costs_preserved=True,
    )
