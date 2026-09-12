from __future__ import annotations

from dataclasses import dataclass
import math

from .action_contract_r1 import FLAT, LONG, SHORT
from .actor_critic_supervisor_r1 import SupervisorAuthorityStateR1
from .execution_feasibility_r0 import MechanicalExecutionAuthorityR0
from .target_exposure_r1 import TargetExposureAuthorityR1


EXECUTION_OBSERVATION_SCHEMA_R0 = "CB16_R11_BC_EXECUTION_OBSERVATION_V1_R0"
EXECUTION_OBSERVATION_MODEL_FIELDS_R0 = (
    "terminated",
    "truncated",
    "legal_short",
    "legal_flat",
    "legal_long",
    "max_permitted_target_risk",
    "equity",
    "current_quantity",
    "current_price",
    "margin_capacity",
    "available_margin_for_new_exposure",
    "max_gross_leverage",
    "initial_margin_rate",
    "maintenance_margin_rate",
    "maintenance_collateral",
    "declared_max_legal_notional",
    "lot_step_size",
    "lot_min_qty",
    "lot_max_qty",
    "min_notional",
)
EXECUTION_OBSERVATION_FORBIDDEN_FIELDS_R0 = (
    "action",
    "target_quantity",
    "delta_quantity",
    "feasibility_status",
    "feasibility_reason",
    "expected_return",
    "expected_profit",
    "recommended_direction",
    "recommended_risk",
    "stop_loss",
    "take_profit",
    "future_execution",
)


def _finite(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(code)
    return 0.0 if out == 0.0 else out


def _optional_finite(value: object, *, code: str) -> float | None:
    if value is None:
        return None
    return _finite(value, code=code)


def _same_optional(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return float(left) == float(right)


@dataclass(frozen=True)
class ExecutionObservationR0:
    """Contemporaneously known mechanical legality/resources for the policy.

    This object intentionally contains no action-specific target, feasibility
    result, execution result, profitability estimate, or strategy recommendation.
    Those are downstream consequences of a nominal action, not current-state
    policy inputs.
    """

    schema_version: str
    supervisor_authority_sha256: str
    target_exposure_authority_sha256: str
    mechanical_authority_id: str
    account_state_sha256: str
    terminated: bool
    truncated: bool
    legal_short: bool
    legal_flat: bool
    legal_long: bool
    max_permitted_target_risk: float
    equity: float
    current_quantity: float
    current_price: float
    margin_capacity: float
    available_margin_for_new_exposure: float
    max_gross_leverage: float
    initial_margin_rate: float
    maintenance_margin_rate: float
    maintenance_collateral: float
    declared_max_legal_notional: float | None
    lot_step_size: float | None
    lot_min_qty: float | None
    lot_max_qty: float | None
    min_notional: float | None

    def validate(self) -> None:
        if self.schema_version != EXECUTION_OBSERVATION_SCHEMA_R0:
            raise RuntimeError("ACEXOBS_R0_SCHEMA_MISMATCH")
        for value, code in (
            (self.supervisor_authority_sha256, "ACEXOBS_R0_SUPERVISOR_HASH_INVALID"),
            (self.target_exposure_authority_sha256, "ACEXOBS_R0_TARGET_HASH_INVALID"),
            (self.account_state_sha256, "ACEXOBS_R0_ACCOUNT_HASH_INVALID"),
        ):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError(code)
        if not self.mechanical_authority_id:
            raise RuntimeError("ACEXOBS_R0_MECHANICAL_AUTHORITY_ID_INVALID")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise RuntimeError("ACEXOBS_R0_BOUNDARY_FLAG_INVALID")
        if not all(isinstance(v, bool) for v in (self.legal_short, self.legal_flat, self.legal_long)):
            raise RuntimeError("ACEXOBS_R0_LEGAL_DIRECTION_FLAG_INVALID")
        risk = _finite(self.max_permitted_target_risk, code="ACEXOBS_R0_RISK_INVALID")
        if not 0.0 <= risk <= 1.0:
            raise RuntimeError("ACEXOBS_R0_RISK_INVALID")
        for value, code in (
            (self.equity, "ACEXOBS_R0_EQUITY_INVALID"),
            (self.current_quantity, "ACEXOBS_R0_QUANTITY_INVALID"),
            (self.current_price, "ACEXOBS_R0_PRICE_INVALID"),
            (self.margin_capacity, "ACEXOBS_R0_MARGIN_CAPACITY_INVALID"),
            (self.available_margin_for_new_exposure, "ACEXOBS_R0_AVAILABLE_MARGIN_INVALID"),
            (self.max_gross_leverage, "ACEXOBS_R0_LEVERAGE_INVALID"),
            (self.initial_margin_rate, "ACEXOBS_R0_INITIAL_MARGIN_INVALID"),
            (self.maintenance_margin_rate, "ACEXOBS_R0_MAINTENANCE_RATE_INVALID"),
            (self.maintenance_collateral, "ACEXOBS_R0_MAINTENANCE_COLLATERAL_INVALID"),
        ):
            _finite(value, code=code)
        if self.current_price <= 0.0 or self.max_gross_leverage <= 0.0:
            raise RuntimeError("ACEXOBS_R0_POSITIVE_RESOURCE_REQUIRED")
        if self.equity < 0.0 or self.margin_capacity < 0.0 or self.available_margin_for_new_exposure < 0.0 or self.maintenance_collateral < 0.0:
            raise RuntimeError("ACEXOBS_R0_NONNEGATIVE_RESOURCE_REQUIRED")
        if not 0.0 < self.initial_margin_rate <= 1.0 or not 0.0 <= self.maintenance_margin_rate <= 1.0:
            raise RuntimeError("ACEXOBS_R0_MARGIN_RATE_INVALID")
        for name in (
            "declared_max_legal_notional",
            "lot_step_size",
            "lot_min_qty",
            "lot_max_qty",
            "min_notional",
        ):
            value = _optional_finite(getattr(self, name), code=f"ACEXOBS_R0_{name.upper()}_INVALID")
            if value is not None and value < 0.0:
                raise RuntimeError(f"ACEXOBS_R0_{name.upper()}_INVALID")

    def model_payload(self) -> dict[str, object]:
        self.validate()
        return {field: getattr(self, field) for field in EXECUTION_OBSERVATION_MODEL_FIELDS_R0}


def build_execution_observation_r0(
    *,
    supervisor_authority: SupervisorAuthorityStateR1,
    target_exposure_authority: TargetExposureAuthorityR1,
    mechanical_authority: MechanicalExecutionAuthorityR0,
) -> ExecutionObservationR0:
    supervisor_authority.validate()
    target_exposure_authority.validate()
    mechanical_authority.validate()

    if supervisor_authority.account_id != target_exposure_authority.account_id:
        raise RuntimeError("ACEXOBS_R0_ACCOUNT_ID_MISMATCH")
    if supervisor_authority.account_state_sha256 != target_exposure_authority.account_state_sha256:
        raise RuntimeError("ACEXOBS_R0_ACCOUNT_STATE_MISMATCH")
    if float(target_exposure_authority.current_price) != float(mechanical_authority.current_price):
        raise RuntimeError("ACEXOBS_R0_PRICE_AUTHORITY_MISMATCH")
    if float(target_exposure_authority.initial_margin_rate) != float(mechanical_authority.initial_margin_rate):
        raise RuntimeError("ACEXOBS_R0_INITIAL_MARGIN_AUTHORITY_MISMATCH")
    for name in ("lot_min_qty", "lot_max_qty", "min_notional"):
        if not _same_optional(
            getattr(target_exposure_authority, name),
            getattr(mechanical_authority, name),
        ):
            raise RuntimeError(f"ACEXOBS_R0_{name.upper()}_AUTHORITY_MISMATCH")

    legal = set(supervisor_authority.legal_target_directions)
    observation = ExecutionObservationR0(
        schema_version=EXECUTION_OBSERVATION_SCHEMA_R0,
        supervisor_authority_sha256=supervisor_authority.semantic_sha256,
        target_exposure_authority_sha256=target_exposure_authority.semantic_sha256,
        mechanical_authority_id=mechanical_authority.authority_id,
        account_state_sha256=supervisor_authority.account_state_sha256,
        terminated=supervisor_authority.terminated,
        truncated=supervisor_authority.truncated,
        legal_short=SHORT in legal,
        legal_flat=FLAT in legal,
        legal_long=LONG in legal,
        max_permitted_target_risk=float(supervisor_authority.max_permitted_target_risk),
        equity=float(target_exposure_authority.equity),
        current_quantity=float(mechanical_authority.current_quantity),
        current_price=float(mechanical_authority.current_price),
        margin_capacity=float(target_exposure_authority.margin_capacity),
        available_margin_for_new_exposure=float(mechanical_authority.available_margin_for_new_exposure),
        max_gross_leverage=float(target_exposure_authority.max_gross_leverage),
        initial_margin_rate=float(mechanical_authority.initial_margin_rate),
        maintenance_margin_rate=float(mechanical_authority.maintenance_margin_rate),
        maintenance_collateral=float(mechanical_authority.maintenance_collateral),
        declared_max_legal_notional=target_exposure_authority.declared_max_legal_notional,
        lot_step_size=target_exposure_authority.lot_step_size,
        lot_min_qty=target_exposure_authority.lot_min_qty,
        lot_max_qty=target_exposure_authority.lot_max_qty,
        min_notional=target_exposure_authority.min_notional,
    )
    observation.validate()
    return observation
