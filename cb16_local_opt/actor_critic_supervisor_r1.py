from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .action_contract_r1 import FLAT, LONG, SHORT, TargetPositionActionR1
from .actor_critic_contract_r1 import PERMISSION_EXECUTION_VERSION_R1

ACCEPT = "ACCEPT"
CLAMP = "CLAMP"
REJECT = "REJECT"
SUPERVISOR_AUTHORITY_SCHEMA_R1 = "CB16_R11_BC_SUPERVISOR_AUTHORITY_V1_R1"
SUPERVISOR_PERMISSION_SCHEMA_R1 = "CB16_R11_BC_SUPERVISOR_PERMISSION_V1_R1"
LEGAL_DIRECTIONS = (SHORT, FLAT, LONG)


def _risk(value: object) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise RuntimeError("ACSUP_R1_RISK_INVALID")
    return 0.0 if value == 0.0 else value


def _hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class SupervisorAuthorityStateR1:
    schema_version: str
    permission_execution_version: str
    authority_id: str
    account_id: str
    account_state_sha256: str
    terminated: bool
    truncated: bool
    legal_target_directions: tuple[str, ...]
    max_permitted_target_risk: float

    def validate(self) -> None:
        if self.schema_version != SUPERVISOR_AUTHORITY_SCHEMA_R1:
            raise RuntimeError("ACSUP_R1_AUTHORITY_SCHEMA_MISMATCH")
        if self.permission_execution_version != PERMISSION_EXECUTION_VERSION_R1:
            raise RuntimeError("ACSUP_R1_PERMISSION_VERSION_MISMATCH")
        if not self.authority_id or not self.account_id:
            raise RuntimeError("ACSUP_R1_AUTHORITY_IDENTITY_INVALID")
        if len(self.account_state_sha256) != 64:
            raise RuntimeError("ACSUP_R1_ACCOUNT_STATE_HASH_INVALID")
        if any(d not in LEGAL_DIRECTIONS for d in self.legal_target_directions):
            raise RuntimeError("ACSUP_R1_LEGAL_DIRECTION_INVALID")
        if len(set(self.legal_target_directions)) != len(self.legal_target_directions):
            raise RuntimeError("ACSUP_R1_LEGAL_DIRECTION_DUPLICATE")
        _risk(self.max_permitted_target_risk)

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _hash(dict(self.__dict__))


@dataclass(frozen=True)
class SupervisorPermissionResultR1:
    schema_version: str
    authority_sha256: str
    account_state_sha256: str
    action_sha256: str
    outcome: str
    reason_codes: tuple[str, ...]
    requested_target_direction: str
    requested_target_risk: float
    permitted_target_direction: str
    permitted_target_risk: float

    @property
    def permission_sha256(self) -> str:
        return _hash(dict(self.__dict__))


def make_supervisor_authority_r1(*, authority_id: str, account_id: str, account_state_sha256: str, terminated: bool, truncated: bool, legal_target_directions: tuple[str, ...], max_permitted_target_risk: float) -> SupervisorAuthorityStateR1:
    authority = SupervisorAuthorityStateR1(
        schema_version=SUPERVISOR_AUTHORITY_SCHEMA_R1,
        permission_execution_version=PERMISSION_EXECUTION_VERSION_R1,
        authority_id=authority_id,
        account_id=account_id,
        account_state_sha256=account_state_sha256,
        terminated=bool(terminated),
        truncated=bool(truncated),
        legal_target_directions=tuple(legal_target_directions),
        max_permitted_target_risk=_risk(max_permitted_target_risk),
    )
    authority.validate()
    return authority


def supervise_target_action_r1(action: TargetPositionActionR1, authority: SupervisorAuthorityStateR1) -> SupervisorPermissionResultR1:
    action.validate()
    authority.validate()
    requested_direction = action.target_direction
    requested_risk = float(action.requested_target_risk)
    reasons: list[str] = []

    if authority.terminated:
        outcome, permitted_direction, permitted_risk = REJECT, FLAT, 0.0
        reasons.append("ACCOUNT_TERMINATED")
    elif authority.truncated:
        outcome, permitted_direction, permitted_risk = REJECT, FLAT, 0.0
        reasons.append("ACCOUNT_TRUNCATED")
    elif requested_direction not in authority.legal_target_directions:
        outcome, permitted_direction, permitted_risk = REJECT, FLAT, 0.0
        reasons.append("TARGET_DIRECTION_ILLEGAL")
    elif requested_risk > authority.max_permitted_target_risk:
        outcome = CLAMP
        permitted_direction = requested_direction
        permitted_risk = authority.max_permitted_target_risk
        reasons.append("HARD_RISK_CAP")
    else:
        outcome, permitted_direction, permitted_risk = ACCEPT, requested_direction, requested_risk
        reasons.append("LEGAL_NOMINAL_REQUEST")

    return SupervisorPermissionResultR1(
        schema_version=SUPERVISOR_PERMISSION_SCHEMA_R1,
        authority_sha256=authority.semantic_sha256,
        account_state_sha256=authority.account_state_sha256,
        action_sha256=hashlib.sha256(action.to_json().encode()).hexdigest(),
        outcome=outcome,
        reason_codes=tuple(reasons),
        requested_target_direction=requested_direction,
        requested_target_risk=requested_risk,
        permitted_target_direction=permitted_direction,
        permitted_target_risk=permitted_risk,
    )
