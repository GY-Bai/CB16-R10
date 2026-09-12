from __future__ import annotations

"""Mechanical Supervisor-permission -> target quantity mapping for Actor-Critic R0.

The target-risk scalar is a linear fraction of the externally declared legal
notional envelope.  This module deliberately contains no ATR, stop-distance,
expected-return, confidence, market-regime, or profitability heuristic.

AC-013 sizes a *target state*.  AC-014 owns conversion from that target state to
an executable delta and authoritative Physics accounting.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional

from .action_contract_r0 import FLAT, LONG, SHORT, TARGET_DIRECTIONS_R0
from .actor_critic_supervisor_r0 import (
    REJECT,
    SupervisorPermissionResultR0,
)


TARGET_EXPOSURE_AUTHORITY_SCHEMA_R0 = "CB16_R11_AC_TARGET_EXPOSURE_AUTHORITY_V1_R0"
TARGET_EXPOSURE_RESULT_SCHEMA_R0 = "CB16_R11_AC_TARGET_EXPOSURE_RESULT_V1_R0"
TARGET_RISK_SOURCE_R0 = "SUPERVISOR_PERMITTED_TARGET_RISK_LINEAR_ENVELOPE_FRACTION"

FLAT_ZERO = "FLAT_ZERO"
ZERO_RISK = "ZERO_RISK"
ACTIVE = "ACTIVE"
QUANTIZED_TO_ZERO = "QUANTIZED_TO_ZERO"
BELOW_MINIMUM = "BELOW_MINIMUM"
TARGET_EXPOSURE_STATUSES_R0 = (
    FLAT_ZERO,
    ZERO_RISK,
    ACTIVE,
    QUANTIZED_TO_ZERO,
    BELOW_MINIMUM,
)


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _finite_nonnegative(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RuntimeError(code)
    return 0.0 if number == 0.0 else number


def _finite_positive(value: object, *, code: str) -> float:
    number = _finite_nonnegative(value, code=code)
    if number <= 0.0:
        raise RuntimeError(code)
    return number


def _optional_positive(value: object, *, code: str) -> Optional[float]:
    if value is None:
        return None
    return _finite_positive(value, code=code)


def _optional_nonnegative(value: object, *, code: str) -> Optional[float]:
    if value is None:
        return None
    return _finite_nonnegative(value, code=code)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_payload(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _direction_sign(direction: str) -> int:
    if direction == LONG:
        return 1
    if direction == SHORT:
        return -1
    if direction == FLAT:
        return 0
    raise RuntimeError("ACTGT_DIRECTION_INVALID")


def _quantize_toward_zero(quantity: float, step: Optional[float]) -> float:
    if step is None:
        return 0.0 if quantity == 0.0 else quantity
    units = math.floor(abs(quantity) / step + 1e-12)
    if units == 0:
        return 0.0
    return math.copysign(units * step, quantity)


@dataclass(frozen=True)
class TargetExposureAuthorityR0:
    """Mechanical account/exchange envelope visible to the sizing map."""

    schema_version: str
    authority_id: str
    account_id: str
    equity: float
    current_price: float
    margin_capacity: float
    max_gross_leverage: float
    initial_margin_rate: float
    declared_max_legal_notional: Optional[float]
    lot_step_size: Optional[float]
    lot_min_qty: Optional[float]
    lot_max_qty: Optional[float]
    min_notional: Optional[float]

    def validate(self) -> None:
        if self.schema_version != TARGET_EXPOSURE_AUTHORITY_SCHEMA_R0:
            raise RuntimeError("ACTGT_AUTHORITY_SCHEMA_MISMATCH")
        _require_nonempty_string(self.authority_id, code="ACTGT_AUTHORITY_ID_INVALID")
        _require_nonempty_string(self.account_id, code="ACTGT_ACCOUNT_ID_INVALID")
        _finite_nonnegative(self.equity, code="ACTGT_EQUITY_INVALID")
        _finite_positive(self.current_price, code="ACTGT_PRICE_INVALID")
        _finite_nonnegative(self.margin_capacity, code="ACTGT_MARGIN_CAPACITY_INVALID")
        _finite_positive(self.max_gross_leverage, code="ACTGT_MAX_GROSS_LEVERAGE_INVALID")
        margin_rate = _finite_positive(
            self.initial_margin_rate,
            code="ACTGT_INITIAL_MARGIN_RATE_INVALID",
        )
        if margin_rate > 1.0:
            raise RuntimeError("ACTGT_INITIAL_MARGIN_RATE_INVALID")
        _optional_nonnegative(
            self.declared_max_legal_notional,
            code="ACTGT_DECLARED_MAX_NOTIONAL_INVALID",
        )
        _optional_positive(self.lot_step_size, code="ACTGT_LOT_STEP_INVALID")
        _optional_positive(self.lot_min_qty, code="ACTGT_LOT_MIN_INVALID")
        _optional_positive(self.lot_max_qty, code="ACTGT_LOT_MAX_INVALID")
        _optional_nonnegative(self.min_notional, code="ACTGT_MIN_NOTIONAL_INVALID")
        if (
            self.lot_min_qty is not None
            and self.lot_max_qty is not None
            and self.lot_min_qty > self.lot_max_qty
        ):
            raise RuntimeError("ACTGT_LOT_RANGE_INVALID")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "account_id": self.account_id,
            "equity": float(self.equity),
            "current_price": float(self.current_price),
            "margin_capacity": float(self.margin_capacity),
            "max_gross_leverage": float(self.max_gross_leverage),
            "initial_margin_rate": float(self.initial_margin_rate),
            "declared_max_legal_notional": self.declared_max_legal_notional,
            "lot_step_size": self.lot_step_size,
            "lot_min_qty": self.lot_min_qty,
            "lot_max_qty": self.lot_max_qty,
            "min_notional": self.min_notional,
        }

    @property
    def semantic_sha256(self) -> str:
        return _sha256_payload(self.to_payload())


@dataclass(frozen=True)
class TargetExposureResultR0:
    schema_version: str
    permission_sha256: str
    authority_sha256: str
    target_risk_source: str
    target_direction: str
    target_risk: float
    leverage_notional_cap: float
    margin_notional_cap: float
    declared_notional_cap: Optional[float]
    quantity_notional_cap: Optional[float]
    legal_notional_cap: float
    pre_filter_target_notional: float
    target_quantity: float
    target_notional: float
    status: str

    def validate(self) -> None:
        if self.schema_version != TARGET_EXPOSURE_RESULT_SCHEMA_R0:
            raise RuntimeError("ACTGT_RESULT_SCHEMA_MISMATCH")
        for value, code in (
            (self.permission_sha256, "ACTGT_PERMISSION_HASH_INVALID"),
            (self.authority_sha256, "ACTGT_AUTHORITY_HASH_INVALID"),
        ):
            text = _require_nonempty_string(value, code=code)
            if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
                raise RuntimeError(code)
        if self.target_risk_source != TARGET_RISK_SOURCE_R0:
            raise RuntimeError("ACTGT_RISK_SOURCE_MISMATCH")
        if self.target_direction not in TARGET_DIRECTIONS_R0:
            raise RuntimeError("ACTGT_DIRECTION_INVALID")
        risk = _finite_nonnegative(self.target_risk, code="ACTGT_TARGET_RISK_INVALID")
        if risk > 1.0:
            raise RuntimeError("ACTGT_TARGET_RISK_INVALID")
        if self.target_direction == FLAT and risk != 0.0:
            raise RuntimeError("ACTGT_FLAT_RISK_MUST_BE_ZERO")
        for value, code in (
            (self.leverage_notional_cap, "ACTGT_LEVERAGE_CAP_INVALID"),
            (self.margin_notional_cap, "ACTGT_MARGIN_CAP_INVALID"),
            (self.legal_notional_cap, "ACTGT_LEGAL_CAP_INVALID"),
            (self.pre_filter_target_notional, "ACTGT_PREFILTER_NOTIONAL_INVALID"),
            (self.target_notional, "ACTGT_TARGET_NOTIONAL_INVALID"),
        ):
            _finite_nonnegative(value, code=code)
        _optional_nonnegative(
            self.declared_notional_cap,
            code="ACTGT_DECLARED_CAP_INVALID",
        )
        _optional_nonnegative(
            self.quantity_notional_cap,
            code="ACTGT_QUANTITY_CAP_INVALID",
        )
        if not math.isfinite(float(self.target_quantity)):
            raise RuntimeError("ACTGT_TARGET_QUANTITY_INVALID")
        sign = _direction_sign(self.target_direction)
        if sign == 0 and self.target_quantity != 0.0:
            raise RuntimeError("ACTGT_FLAT_QUANTITY_MUST_BE_ZERO")
        if sign > 0 and self.target_quantity < 0.0:
            raise RuntimeError("ACTGT_QUANTITY_SIGN_MISMATCH")
        if sign < 0 and self.target_quantity > 0.0:
            raise RuntimeError("ACTGT_QUANTITY_SIGN_MISMATCH")
        if self.status not in TARGET_EXPOSURE_STATUSES_R0:
            raise RuntimeError("ACTGT_STATUS_INVALID")
        if self.target_notional > self.legal_notional_cap + 1e-9:
            raise RuntimeError("ACTGT_TARGET_EXCEEDS_LEGAL_CAP")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "permission_sha256": self.permission_sha256,
            "authority_sha256": self.authority_sha256,
            "target_risk_source": self.target_risk_source,
            "target_direction": self.target_direction,
            "target_risk": float(self.target_risk),
            "leverage_notional_cap": float(self.leverage_notional_cap),
            "margin_notional_cap": float(self.margin_notional_cap),
            "declared_notional_cap": self.declared_notional_cap,
            "quantity_notional_cap": self.quantity_notional_cap,
            "legal_notional_cap": float(self.legal_notional_cap),
            "pre_filter_target_notional": float(self.pre_filter_target_notional),
            "target_quantity": float(self.target_quantity),
            "target_notional": float(self.target_notional),
            "status": self.status,
        }

    @property
    def semantic_sha256(self) -> str:
        return _sha256_payload(self.to_payload())


def make_target_exposure_authority_r0(
    *,
    authority_id: str,
    account_id: str,
    equity: float,
    current_price: float,
    margin_capacity: float,
    max_gross_leverage: float,
    initial_margin_rate: Optional[float],
    declared_max_legal_notional: Optional[float] = None,
    lot_step_size: Optional[float] = None,
    lot_min_qty: Optional[float] = None,
    lot_max_qty: Optional[float] = None,
    min_notional: Optional[float] = None,
) -> TargetExposureAuthorityR0:
    leverage = _finite_positive(
        max_gross_leverage,
        code="ACTGT_MAX_GROSS_LEVERAGE_INVALID",
    )
    effective_margin_rate = (
        1.0 / leverage
        if initial_margin_rate is None
        else _finite_positive(initial_margin_rate, code="ACTGT_INITIAL_MARGIN_RATE_INVALID")
    )
    authority = TargetExposureAuthorityR0(
        schema_version=TARGET_EXPOSURE_AUTHORITY_SCHEMA_R0,
        authority_id=authority_id,
        account_id=account_id,
        equity=_finite_nonnegative(equity, code="ACTGT_EQUITY_INVALID"),
        current_price=_finite_positive(current_price, code="ACTGT_PRICE_INVALID"),
        margin_capacity=_finite_nonnegative(
            margin_capacity,
            code="ACTGT_MARGIN_CAPACITY_INVALID",
        ),
        max_gross_leverage=leverage,
        initial_margin_rate=effective_margin_rate,
        declared_max_legal_notional=_optional_nonnegative(
            declared_max_legal_notional,
            code="ACTGT_DECLARED_MAX_NOTIONAL_INVALID",
        ),
        lot_step_size=_optional_positive(lot_step_size, code="ACTGT_LOT_STEP_INVALID"),
        lot_min_qty=_optional_positive(lot_min_qty, code="ACTGT_LOT_MIN_INVALID"),
        lot_max_qty=_optional_positive(lot_max_qty, code="ACTGT_LOT_MAX_INVALID"),
        min_notional=_optional_nonnegative(min_notional, code="ACTGT_MIN_NOTIONAL_INVALID"),
    )
    authority.validate()
    return authority


def map_permission_to_target_exposure_r0(
    permission: SupervisorPermissionResultR0,
    authority: TargetExposureAuthorityR0,
) -> TargetExposureResultR0:
    """Map an authorized target-risk fraction to one deterministic target quantity."""

    if not isinstance(permission, SupervisorPermissionResultR0):
        raise RuntimeError("ACTGT_PERMISSION_TYPE_INVALID")
    permission.validate()
    authority.validate()
    if permission.outcome == REJECT:
        raise RuntimeError("ACTGT_REJECTED_PERMISSION_NOT_SIZABLE")

    direction = permission.permitted_target_direction
    risk = float(permission.permitted_target_risk)
    sign = _direction_sign(direction)

    leverage_cap = authority.equity * authority.max_gross_leverage
    margin_cap = authority.margin_capacity / authority.initial_margin_rate
    caps = [leverage_cap, margin_cap]
    declared_cap = authority.declared_max_legal_notional
    if declared_cap is not None:
        caps.append(declared_cap)
    quantity_cap = None
    if authority.lot_max_qty is not None:
        quantity_cap = authority.lot_max_qty * authority.current_price
        caps.append(quantity_cap)
    legal_cap = min(caps)

    pre_filter_notional = risk * legal_cap
    raw_quantity = 0.0 if sign == 0 else sign * pre_filter_notional / authority.current_price
    quantized_quantity = _quantize_toward_zero(raw_quantity, authority.lot_step_size)
    target_notional = abs(quantized_quantity) * authority.current_price

    if direction == FLAT:
        status = FLAT_ZERO
        quantized_quantity = 0.0
        target_notional = 0.0
    elif risk == 0.0:
        status = ZERO_RISK
        quantized_quantity = 0.0
        target_notional = 0.0
    elif quantized_quantity == 0.0:
        status = QUANTIZED_TO_ZERO
    elif (
        authority.lot_min_qty is not None
        and abs(quantized_quantity) < authority.lot_min_qty - 1e-12
    ) or (
        authority.min_notional is not None
        and target_notional < authority.min_notional - 1e-9
    ):
        status = BELOW_MINIMUM
        quantized_quantity = 0.0
        target_notional = 0.0
    else:
        status = ACTIVE

    result = TargetExposureResultR0(
        schema_version=TARGET_EXPOSURE_RESULT_SCHEMA_R0,
        permission_sha256=permission.permission_sha256,
        authority_sha256=authority.semantic_sha256,
        target_risk_source=TARGET_RISK_SOURCE_R0,
        target_direction=direction,
        target_risk=risk,
        leverage_notional_cap=leverage_cap,
        margin_notional_cap=margin_cap,
        declared_notional_cap=declared_cap,
        quantity_notional_cap=quantity_cap,
        legal_notional_cap=legal_cap,
        pre_filter_target_notional=pre_filter_notional,
        target_quantity=quantized_quantity,
        target_notional=target_notional,
        status=status,
    )
    result.validate()
    return result
