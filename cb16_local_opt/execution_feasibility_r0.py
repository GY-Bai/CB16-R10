from __future__ import annotations

from dataclasses import dataclass
import math

from .target_exposure_r1 import TargetExposureResultR1

FEASIBILITY_SCHEMA_R0 = "CB16_R11_BC_EXECUTION_FEASIBILITY_V1_R0"
FEASIBLE = "FEASIBLE"
REJECT_QUANTITY = "REJECT_QUANTITY"
REJECT_NOTIONAL = "REJECT_NOTIONAL"
REJECT_MARGIN = "REJECT_MARGIN"
REJECT_MAINTENANCE = "REJECT_MAINTENANCE"


def _finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("ACFEAS_R0_NUMERIC_INVALID")
    return value


@dataclass(frozen=True)
class MechanicalExecutionAuthorityR0:
    authority_id: str
    current_quantity: float
    current_price: float
    available_margin_for_new_exposure: float
    initial_margin_rate: float
    maintenance_margin_rate: float
    maintenance_collateral: float
    lot_min_qty: float | None = None
    lot_max_qty: float | None = None
    min_notional: float | None = None

    def validate(self) -> None:
        if not self.authority_id:
            raise RuntimeError("ACFEAS_R0_AUTHORITY_ID_INVALID")
        _finite(self.current_quantity)
        if _finite(self.current_price) <= 0.0:
            raise RuntimeError("ACFEAS_R0_PRICE_INVALID")
        if _finite(self.available_margin_for_new_exposure) < 0.0:
            raise RuntimeError("ACFEAS_R0_MARGIN_AVAILABLE_INVALID")
        if not 0.0 < _finite(self.initial_margin_rate) <= 1.0:
            raise RuntimeError("ACFEAS_R0_INITIAL_MARGIN_RATE_INVALID")
        if not 0.0 <= _finite(self.maintenance_margin_rate) <= 1.0:
            raise RuntimeError("ACFEAS_R0_MAINTENANCE_RATE_INVALID")
        if _finite(self.maintenance_collateral) < 0.0:
            raise RuntimeError("ACFEAS_R0_MAINTENANCE_COLLATERAL_INVALID")


@dataclass(frozen=True)
class ExecutionFeasibilityResultR0:
    schema_version: str
    authority_id: str
    current_quantity: float
    target_quantity: float
    delta_quantity: float
    added_quantity: float
    added_notional: float
    added_margin_required: float
    target_notional: float
    maintenance_margin_required: float
    status: str
    reason_codes: tuple[str, ...]

    @property
    def is_reduction_or_close(self) -> bool:
        return abs(self.target_quantity) <= abs(self.current_quantity) and self.current_quantity * self.target_quantity >= 0.0


def evaluate_execution_feasibility_r0(target: TargetExposureResultR1, authority: MechanicalExecutionAuthorityR0) -> ExecutionFeasibilityResultR0:
    authority.validate()
    current = float(authority.current_quantity)
    target_qty = float(target.target_quantity)
    delta = target_qty - current
    price = float(authority.current_price)

    same_side = current == 0.0 or target_qty == 0.0 or current * target_qty > 0.0
    added_quantity = max(0.0, abs(target_qty) - abs(current)) if same_side else abs(target_qty)
    added_notional = added_quantity * price
    added_margin = added_notional * authority.initial_margin_rate
    target_notional = abs(target_qty) * price
    maintenance_required = target_notional * authority.maintenance_margin_rate

    reasons: list[str] = []
    status = FEASIBLE
    if authority.lot_max_qty is not None and abs(target_qty) > authority.lot_max_qty + 1e-12:
        status, reasons = REJECT_QUANTITY, ["LOT_MAX_QTY"]
    elif authority.lot_min_qty is not None and target_qty != 0.0 and abs(target_qty) < authority.lot_min_qty - 1e-12:
        status, reasons = REJECT_QUANTITY, ["LOT_MIN_QTY"]
    elif authority.min_notional is not None and target_qty != 0.0 and target_notional < authority.min_notional - 1e-9:
        status, reasons = REJECT_NOTIONAL, ["MIN_NOTIONAL"]
    elif added_margin > authority.available_margin_for_new_exposure + 1e-9:
        status, reasons = REJECT_MARGIN, ["INSUFFICIENT_NEW_EXPOSURE_MARGIN"]
    elif target_qty != 0.0 and authority.maintenance_collateral <= maintenance_required:
        status, reasons = REJECT_MAINTENANCE, ["MAINTENANCE_MARGIN_VIOLATION"]

    return ExecutionFeasibilityResultR0(
        schema_version=FEASIBILITY_SCHEMA_R0,
        authority_id=authority.authority_id,
        current_quantity=current,
        target_quantity=target_qty,
        delta_quantity=delta,
        added_quantity=added_quantity,
        added_notional=added_notional,
        added_margin_required=added_margin,
        target_notional=target_notional,
        maintenance_margin_required=maintenance_required,
        status=status,
        reason_codes=tuple(reasons),
    )
