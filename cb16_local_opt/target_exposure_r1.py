from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from .action_contract_r1 import FLAT, LONG, SHORT, TargetPositionActionR1

TARGET_EXPOSURE_AUTHORITY_SCHEMA_R1 = "CB16_R11_BC_TARGET_EXPOSURE_AUTHORITY_V1_R1"
TARGET_EXPOSURE_RESULT_SCHEMA_R1 = "CB16_R11_BC_TARGET_EXPOSURE_RESULT_V1_R1"
TARGET_RISK_SOURCE_R1 = "ACTION_REQUESTED_TARGET_RISK_LINEAR_LEGAL_ENVELOPE_FRACTION"


def _positive(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("ACTGT_R1_POSITIVE_REQUIRED")
    return value


def _nonnegative(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError("ACTGT_R1_NONNEGATIVE_REQUIRED")
    return value


@dataclass(frozen=True)
class TargetExposureAuthorityR1:
    authority_id: str
    account_id: str
    account_state_sha256: str
    legal_envelope_id: str
    equity: float
    current_price: float
    margin_capacity: float
    max_gross_leverage: float
    initial_margin_rate: float
    declared_max_legal_notional: float | None = None

    def validate(self) -> None:
        if not self.authority_id or not self.account_id or not self.legal_envelope_id:
            raise RuntimeError("ACTGT_R1_AUTHORITY_IDENTITY_INVALID")
        if len(self.account_state_sha256) != 64:
            raise RuntimeError("ACTGT_R1_ACCOUNT_STATE_HASH_INVALID")
        _nonnegative(self.equity)
        _positive(self.current_price)
        _nonnegative(self.margin_capacity)
        _positive(self.max_gross_leverage)
        rate = _positive(self.initial_margin_rate)
        if rate > 1.0:
            raise RuntimeError("ACTGT_R1_MARGIN_RATE_INVALID")
        if self.declared_max_legal_notional is not None:
            _nonnegative(self.declared_max_legal_notional)

    @property
    def semantic_sha256(self) -> str:
        payload = "|".join(str(x) for x in self.__dict__.values())
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TargetExposureResultR1:
    schema_version: str
    action_sha256: str
    authority_sha256: str
    account_state_sha256: str
    legal_envelope_id: str
    target_risk_source: str
    target_direction: str
    target_risk: float
    legal_notional_cap: float
    target_quantity: float
    target_notional: float


def make_target_exposure_authority_r1(*, authority_id: str, account_id: str, account_state_sha256: str, legal_envelope_id: str, equity: float, current_price: float, margin_capacity: float, max_gross_leverage: float, initial_margin_rate: float | None, declared_max_legal_notional: float | None = None) -> TargetExposureAuthorityR1:
    leverage = _positive(max_gross_leverage)
    authority = TargetExposureAuthorityR1(
        authority_id=authority_id,
        account_id=account_id,
        account_state_sha256=account_state_sha256,
        legal_envelope_id=legal_envelope_id,
        equity=_nonnegative(equity),
        current_price=_positive(current_price),
        margin_capacity=_nonnegative(margin_capacity),
        max_gross_leverage=leverage,
        initial_margin_rate=(1.0 / leverage if initial_margin_rate is None else _positive(initial_margin_rate)),
        declared_max_legal_notional=declared_max_legal_notional,
    )
    authority.validate()
    return authority


def map_action_to_target_exposure_r1(action: TargetPositionActionR1, authority: TargetExposureAuthorityR1) -> TargetExposureResultR1:
    action.validate()
    authority.validate()
    caps = [
        authority.equity * authority.max_gross_leverage,
        authority.margin_capacity / authority.initial_margin_rate,
    ]
    if authority.declared_max_legal_notional is not None:
        caps.append(authority.declared_max_legal_notional)
    legal_cap = min(caps)
    sign = 1 if action.target_direction == LONG else -1 if action.target_direction == SHORT else 0
    risk = float(action.requested_target_risk)
    target_notional = risk * legal_cap if sign else 0.0
    target_quantity = sign * target_notional / authority.current_price if sign else 0.0
    return TargetExposureResultR1(
        schema_version=TARGET_EXPOSURE_RESULT_SCHEMA_R1,
        action_sha256=hashlib.sha256(action.to_json().encode()).hexdigest(),
        authority_sha256=authority.semantic_sha256,
        account_state_sha256=authority.account_state_sha256,
        legal_envelope_id=authority.legal_envelope_id,
        target_risk_source=TARGET_RISK_SOURCE_R1,
        target_direction=action.target_direction,
        target_risk=risk,
        legal_notional_cap=legal_cap,
        target_quantity=target_quantity,
        target_notional=target_notional,
    )
