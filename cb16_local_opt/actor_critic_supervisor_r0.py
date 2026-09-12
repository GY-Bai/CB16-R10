from __future__ import annotations

"""Deterministic hard-authority Supervisor contract for Actor-Critic R0.

AC-010 freezes the permission contract without quantity sizing or Physics.
AC-011 additionally authorizes mechanically legal held-position same-direction
resize and close operations without inheriting the legacy POSITION_ALREADY_OPEN
forced no-op. Reversal remains fail-closed until AC-012.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

from .action_contract_r0 import (
    FLAT,
    LONG,
    SHORT,
    TARGET_DIRECTIONS_R0,
    TargetPositionActionR0,
)
from .actor_critic_contract_r0 import PERMISSION_EXECUTION_VERSION_R0


ACCEPT = "ACCEPT"
CLAMP = "CLAMP"
REJECT = "REJECT"
PERMISSION_OUTCOMES_R0 = (ACCEPT, CLAMP, REJECT)
SUPERVISOR_AUTHORITY_SCHEMA_R0 = "CB16_R11_AC_SUPERVISOR_AUTHORITY_V1_R0"
SUPERVISOR_PERMISSION_SCHEMA_R0 = "CB16_R11_AC_SUPERVISOR_PERMISSION_V1_R0"


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _canonical_risk(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    risk = float(value)
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise RuntimeError(code)
    return 0.0 if risk == 0.0 else risk


def _validate_direction_risk(direction: object, risk: object, *, prefix: str) -> tuple[str, float]:
    if not isinstance(direction, str) or direction not in TARGET_DIRECTIONS_R0:
        raise RuntimeError(f"{prefix}_DIRECTION_INVALID")
    canonical_risk = _canonical_risk(risk, code=f"{prefix}_RISK_INVALID")
    if direction == FLAT and canonical_risk != 0.0:
        raise RuntimeError(f"{prefix}_FLAT_RISK_MUST_BE_ZERO")
    return direction, canonical_risk


def _canonical_direction_subset(values: Iterable[object]) -> tuple[str, ...]:
    raw = tuple(values)
    if any(not isinstance(value, str) or value not in TARGET_DIRECTIONS_R0 for value in raw):
        raise RuntimeError("ACSUP_LEGAL_DIRECTION_INVALID")
    if len(set(raw)) != len(raw):
        raise RuntimeError("ACSUP_LEGAL_DIRECTION_DUPLICATE")
    present = set(raw)
    return tuple(direction for direction in TARGET_DIRECTIONS_R0 if direction in present)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupervisorAuthorityStateR0:
    """Causal projection of external legality/risk authority for one account state."""

    schema_version: str
    permission_execution_version: str
    authority_id: str
    account_id: str
    current_direction: str
    current_target_risk: float
    terminated: bool
    truncated: bool
    margin_available_for_new_exposure: bool
    legal_target_directions: tuple[str, ...]
    max_permitted_target_risk: float

    def validate(self) -> None:
        if self.schema_version != SUPERVISOR_AUTHORITY_SCHEMA_R0:
            raise RuntimeError("ACSUP_AUTHORITY_SCHEMA_MISMATCH")
        if self.permission_execution_version != PERMISSION_EXECUTION_VERSION_R0:
            raise RuntimeError("ACSUP_PERMISSION_EXECUTION_VERSION_MISMATCH")
        _require_nonempty_string(self.authority_id, code="ACSUP_AUTHORITY_ID_INVALID")
        _require_nonempty_string(self.account_id, code="ACSUP_ACCOUNT_ID_INVALID")
        _validate_direction_risk(
            self.current_direction,
            self.current_target_risk,
            prefix="ACSUP_CURRENT_TARGET",
        )
        if not isinstance(self.terminated, bool):
            raise RuntimeError("ACSUP_TERMINATED_FLAG_INVALID")
        if not isinstance(self.truncated, bool):
            raise RuntimeError("ACSUP_TRUNCATED_FLAG_INVALID")
        if not isinstance(self.margin_available_for_new_exposure, bool):
            raise RuntimeError("ACSUP_MARGIN_FLAG_INVALID")
        canonical = _canonical_direction_subset(self.legal_target_directions)
        if canonical != self.legal_target_directions:
            raise RuntimeError("ACSUP_LEGAL_DIRECTIONS_NONCANONICAL")
        _canonical_risk(
            self.max_permitted_target_risk,
            code="ACSUP_MAX_TARGET_RISK_INVALID",
        )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "permission_execution_version": self.permission_execution_version,
            "authority_id": self.authority_id,
            "account_id": self.account_id,
            "current_direction": self.current_direction,
            "current_target_risk": _canonical_risk(
                self.current_target_risk,
                code="ACSUP_CURRENT_TARGET_RISK_INVALID",
            ),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "margin_available_for_new_exposure": self.margin_available_for_new_exposure,
            "legal_target_directions": list(self.legal_target_directions),
            "max_permitted_target_risk": _canonical_risk(
                self.max_permitted_target_risk,
                code="ACSUP_MAX_TARGET_RISK_INVALID",
            ),
        }

    @property
    def semantic_sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_payload()))


@dataclass(frozen=True)
class SupervisorPermissionResultR0:
    """Auditable permission output; REJECT means no policy-driven state change."""

    schema_version: str
    permission_execution_version: str
    nominal_action_id: str
    nominal_action_sha256: str
    authority_sha256: str
    outcome: str
    reason_code: str
    permitted_target_direction: str
    permitted_target_risk: float

    def validate(self) -> None:
        if self.schema_version != SUPERVISOR_PERMISSION_SCHEMA_R0:
            raise RuntimeError("ACSUP_PERMISSION_SCHEMA_MISMATCH")
        if self.permission_execution_version != PERMISSION_EXECUTION_VERSION_R0:
            raise RuntimeError("ACSUP_PERMISSION_EXECUTION_VERSION_MISMATCH")
        _require_nonempty_string(self.nominal_action_id, code="ACSUP_ACTION_ID_INVALID")
        for value, code in (
            (self.nominal_action_sha256, "ACSUP_ACTION_HASH_INVALID"),
            (self.authority_sha256, "ACSUP_AUTHORITY_HASH_INVALID"),
        ):
            text = _require_nonempty_string(value, code=code)
            if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
                raise RuntimeError(code)
        if self.outcome not in PERMISSION_OUTCOMES_R0:
            raise RuntimeError("ACSUP_OUTCOME_INVALID")
        _require_nonempty_string(self.reason_code, code="ACSUP_REASON_CODE_INVALID")
        _validate_direction_risk(
            self.permitted_target_direction,
            self.permitted_target_risk,
            prefix="ACSUP_PERMITTED_TARGET",
        )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "permission_execution_version": self.permission_execution_version,
            "nominal_action_id": self.nominal_action_id,
            "nominal_action_sha256": self.nominal_action_sha256,
            "authority_sha256": self.authority_sha256,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "permitted_target_direction": self.permitted_target_direction,
            "permitted_target_risk": _canonical_risk(
                self.permitted_target_risk,
                code="ACSUP_PERMITTED_TARGET_RISK_INVALID",
            ),
        }

    @property
    def permission_sha256(self) -> str:
        return _sha256_text(_canonical_json(self.to_payload()))


def make_supervisor_authority_state_r0(
    *,
    authority_id: str,
    account_id: str,
    current_direction: str,
    current_target_risk: float,
    terminated: bool,
    truncated: bool,
    margin_available_for_new_exposure: bool,
    legal_target_directions: Iterable[str],
    max_permitted_target_risk: float,
) -> SupervisorAuthorityStateR0:
    authority = SupervisorAuthorityStateR0(
        schema_version=SUPERVISOR_AUTHORITY_SCHEMA_R0,
        permission_execution_version=PERMISSION_EXECUTION_VERSION_R0,
        authority_id=authority_id,
        account_id=account_id,
        current_direction=current_direction,
        current_target_risk=_canonical_risk(
            current_target_risk,
            code="ACSUP_CURRENT_TARGET_RISK_INVALID",
        ),
        terminated=terminated,
        truncated=truncated,
        margin_available_for_new_exposure=margin_available_for_new_exposure,
        legal_target_directions=_canonical_direction_subset(legal_target_directions),
        max_permitted_target_risk=_canonical_risk(
            max_permitted_target_risk,
            code="ACSUP_MAX_TARGET_RISK_INVALID",
        ),
    )
    authority.validate()
    return authority


def _action_sha256(action: TargetPositionActionR0) -> str:
    action.validate()
    return _sha256_text(action.to_json())


def _result(
    *,
    action: TargetPositionActionR0,
    authority: SupervisorAuthorityStateR0,
    outcome: str,
    reason_code: str,
    permitted_target_direction: str,
    permitted_target_risk: float,
) -> SupervisorPermissionResultR0:
    result = SupervisorPermissionResultR0(
        schema_version=SUPERVISOR_PERMISSION_SCHEMA_R0,
        permission_execution_version=PERMISSION_EXECUTION_VERSION_R0,
        nominal_action_id=action.action_id,
        nominal_action_sha256=_action_sha256(action),
        authority_sha256=authority.semantic_sha256,
        outcome=outcome,
        reason_code=reason_code,
        permitted_target_direction=permitted_target_direction,
        permitted_target_risk=_canonical_risk(
            permitted_target_risk,
            code="ACSUP_PERMITTED_TARGET_RISK_INVALID",
        ),
    )
    result.validate()
    return result


def supervise_target_action_r0(
    action: TargetPositionActionR0,
    authority: SupervisorAuthorityStateR0,
) -> SupervisorPermissionResultR0:
    """Apply hard authority only, with deterministic fail-closed precedence."""

    action.validate()
    authority.validate()
    current_direction, current_risk = _validate_direction_risk(
        authority.current_direction,
        authority.current_target_risk,
        prefix="ACSUP_CURRENT_TARGET",
    )

    if authority.terminated:
        return _result(
            action=action,
            authority=authority,
            outcome=REJECT,
            reason_code="TERMINATED_ACCOUNT",
            permitted_target_direction=current_direction,
            permitted_target_risk=current_risk,
        )
    if authority.truncated:
        return _result(
            action=action,
            authority=authority,
            outcome=REJECT,
            reason_code="TRUNCATED_ACCOUNT",
            permitted_target_direction=current_direction,
            permitted_target_risk=current_risk,
        )
    if action.target_direction not in authority.legal_target_directions:
        return _result(
            action=action,
            authority=authority,
            outcome=REJECT,
            reason_code="TARGET_DIRECTION_NOT_LEGAL",
            permitted_target_direction=current_direction,
            permitted_target_risk=current_risk,
        )

    requested = _canonical_risk(
        action.requested_target_risk,
        code="ACSUP_REQUESTED_TARGET_RISK_INVALID",
    )
    cap = _canonical_risk(
        authority.max_permitted_target_risk,
        code="ACSUP_MAX_TARGET_RISK_INVALID",
    )

    if current_direction != FLAT:
        if action.target_direction not in (current_direction, FLAT):
            return _result(
                action=action,
                authority=authority,
                outcome=REJECT,
                reason_code="REVERSAL_NOT_AUTHORIZED_R0",
                permitted_target_direction=current_direction,
                permitted_target_risk=current_risk,
            )

        if action.target_direction == FLAT:
            return _result(
                action=action,
                authority=authority,
                outcome=ACCEPT,
                reason_code="HELD_POSITION_CLOSE_AUTHORIZED",
                permitted_target_direction=FLAT,
                permitted_target_risk=0.0,
            )

        permitted_risk = min(requested, cap)
        is_exposure_increase = permitted_risk > current_risk
        if is_exposure_increase and not authority.margin_available_for_new_exposure:
            return _result(
                action=action,
                authority=authority,
                outcome=REJECT,
                reason_code="MARGIN_UNAVAILABLE_FOR_EXPOSURE_INCREASE",
                permitted_target_direction=current_direction,
                permitted_target_risk=current_risk,
            )

        if permitted_risk < requested:
            return _result(
                action=action,
                authority=authority,
                outcome=CLAMP,
                reason_code="HELD_POSITION_TARGET_RISK_CLAMPED_TO_HARD_AUTHORITY",
                permitted_target_direction=current_direction,
                permitted_target_risk=permitted_risk,
            )

        if requested < current_risk:
            reason = "HELD_POSITION_REDUCE_AUTHORIZED"
        elif requested == current_risk:
            reason = "HELD_POSITION_SAME_TARGET_AUTHORIZED"
        else:
            reason = "HELD_POSITION_INCREASE_AUTHORIZED"
        return _result(
            action=action,
            authority=authority,
            outcome=ACCEPT,
            reason_code=reason,
            permitted_target_direction=current_direction,
            permitted_target_risk=requested,
        )

    if action.target_direction == FLAT:
        return _result(
            action=action,
            authority=authority,
            outcome=ACCEPT,
            reason_code="FLAT_TARGET_WITHIN_HARD_AUTHORITY",
            permitted_target_direction=FLAT,
            permitted_target_risk=0.0,
        )

    if not authority.margin_available_for_new_exposure:
        return _result(
            action=action,
            authority=authority,
            outcome=REJECT,
            reason_code="MARGIN_UNAVAILABLE_FOR_NEW_EXPOSURE",
            permitted_target_direction=FLAT,
            permitted_target_risk=0.0,
        )

    if cap == 0.0:
        return _result(
            action=action,
            authority=authority,
            outcome=REJECT,
            reason_code="TARGET_RISK_AUTHORITY_EXHAUSTED",
            permitted_target_direction=FLAT,
            permitted_target_risk=0.0,
        )
    if requested > cap:
        return _result(
            action=action,
            authority=authority,
            outcome=CLAMP,
            reason_code="TARGET_RISK_CLAMPED_TO_HARD_AUTHORITY",
            permitted_target_direction=action.target_direction,
            permitted_target_risk=cap,
        )
    return _result(
        action=action,
        authority=authority,
        outcome=ACCEPT,
        reason_code="WITHIN_ALL_HARD_AUTHORITY",
        permitted_target_direction=action.target_direction,
        permitted_target_risk=requested,
    )
