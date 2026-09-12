from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping

from .actor_critic_contract_r1 import ACTION_VERSION_R1


SHORT = "SHORT"
FLAT = "FLAT"
LONG = "LONG"
TARGET_DIRECTIONS_R1 = (SHORT, FLAT, LONG)
TARGET_POSITION_SEMANTICS_R1 = "TARGET_POSITION_STATE"
TARGET_RISK_SEMANTICS_R1 = "TARGET_EXPOSURE_REQUEST"

SAME_NOMINAL_REQUEST = "SAME_NOMINAL_REQUEST"
OPEN_LONG = "OPEN_LONG"
OPEN_SHORT = "OPEN_SHORT"
INCREASE_LONG_REQUEST = "INCREASE_LONG_REQUEST"
DECREASE_LONG_REQUEST = "DECREASE_LONG_REQUEST"
INCREASE_SHORT_REQUEST = "INCREASE_SHORT_REQUEST"
DECREASE_SHORT_REQUEST = "DECREASE_SHORT_REQUEST"
CLOSE_LONG_REQUEST = "CLOSE_LONG_REQUEST"
CLOSE_SHORT_REQUEST = "CLOSE_SHORT_REQUEST"
REVERSE_LONG_TO_SHORT_REQUEST = "REVERSE_LONG_TO_SHORT_REQUEST"
REVERSE_SHORT_TO_LONG_REQUEST = "REVERSE_SHORT_TO_LONG_REQUEST"
TARGET_REQUEST_RELATIONS_R1 = (
    SAME_NOMINAL_REQUEST,
    OPEN_LONG,
    OPEN_SHORT,
    INCREASE_LONG_REQUEST,
    DECREASE_LONG_REQUEST,
    INCREASE_SHORT_REQUEST,
    DECREASE_SHORT_REQUEST,
    CLOSE_LONG_REQUEST,
    CLOSE_SHORT_REQUEST,
    REVERSE_LONG_TO_SHORT_REQUEST,
    REVERSE_SHORT_TO_LONG_REQUEST,
)

ACTION_FIELDS_R1 = (
    "action_id",
    "schema_version",
    "action_semantics",
    "policy_id",
    "policy_version",
    "target_direction",
    "requested_target_risk",
    "requested_target_risk_semantics",
)


def _nonempty(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _risk(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("ACACT_R1_TARGET_RISK_TYPE_INVALID")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise RuntimeError("ACACT_R1_TARGET_RISK_OUT_OF_RANGE")
    return 0.0 if out == 0.0 else out


def _direction_risk(direction: object, risk: object) -> tuple[str, float]:
    d = _nonempty(direction, "ACACT_R1_TARGET_DIRECTION_INVALID")
    if d not in TARGET_DIRECTIONS_R1:
        raise RuntimeError("ACACT_R1_TARGET_DIRECTION_INVALID")
    r = _risk(risk)
    if d == FLAT and r != 0.0:
        raise RuntimeError("ACACT_R1_FLAT_TARGET_RISK_MUST_BE_ZERO")
    return d, r


@dataclass(frozen=True)
class TargetPositionActionR1:
    action_id: str
    schema_version: str
    action_semantics: str
    policy_id: str
    policy_version: str
    target_direction: str
    requested_target_risk: float
    requested_target_risk_semantics: str

    def validate(self) -> None:
        _nonempty(self.action_id, "ACACT_R1_ACTION_ID_INVALID")
        _nonempty(self.policy_id, "ACACT_R1_POLICY_ID_INVALID")
        _nonempty(self.policy_version, "ACACT_R1_POLICY_VERSION_INVALID")
        if self.schema_version != ACTION_VERSION_R1:
            raise RuntimeError("ACACT_R1_SCHEMA_VERSION_MISMATCH")
        if self.action_semantics != TARGET_POSITION_SEMANTICS_R1:
            raise RuntimeError("ACACT_R1_SEMANTICS_MISMATCH")
        if self.requested_target_risk_semantics != TARGET_RISK_SEMANTICS_R1:
            raise RuntimeError("ACACT_R1_TARGET_RISK_SEMANTICS_MISMATCH")
        _direction_risk(self.target_direction, self.requested_target_risk)

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "schema_version": self.schema_version,
            "action_semantics": self.action_semantics,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "target_direction": self.target_direction,
            "requested_target_risk": _risk(self.requested_target_risk),
            "requested_target_risk_semantics": self.requested_target_risk_semantics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TargetPositionActionR1":
        required = set(ACTION_FIELDS_R1)
        actual = set(payload)
        missing = sorted(required - actual)
        unknown = sorted(str(k) for k in actual if k not in required)
        if missing:
            raise RuntimeError(f"ACACT_R1_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACACT_R1_FIELDS_UNKNOWN:{','.join(unknown)}")
        out = cls(
            action_id=_nonempty(payload["action_id"], "ACACT_R1_ACTION_ID_INVALID"),
            schema_version=_nonempty(payload["schema_version"], "ACACT_R1_SCHEMA_VERSION_INVALID"),
            action_semantics=_nonempty(payload["action_semantics"], "ACACT_R1_SEMANTICS_INVALID"),
            policy_id=_nonempty(payload["policy_id"], "ACACT_R1_POLICY_ID_INVALID"),
            policy_version=_nonempty(payload["policy_version"], "ACACT_R1_POLICY_VERSION_INVALID"),
            target_direction=_nonempty(payload["target_direction"], "ACACT_R1_TARGET_DIRECTION_INVALID"),
            requested_target_risk=_risk(payload["requested_target_risk"]),
            requested_target_risk_semantics=_nonempty(payload["requested_target_risk_semantics"], "ACACT_R1_TARGET_RISK_SEMANTICS_INVALID"),
        )
        out.validate()
        return out


@dataclass(frozen=True)
class TargetRequestRelationR1:
    action_id: str
    source_direction: str
    source_requested_risk: float
    target_direction: str
    requested_target_risk: float
    relation_kind: str
    requires_target_recompute: bool

    def validate(self) -> None:
        _nonempty(self.action_id, "ACACT_R1_ACTION_ID_INVALID")
        _direction_risk(self.source_direction, self.source_requested_risk)
        _direction_risk(self.target_direction, self.requested_target_risk)
        if self.relation_kind not in TARGET_REQUEST_RELATIONS_R1:
            raise RuntimeError("ACACT_R1_RELATION_KIND_INVALID")
        if self.requires_target_recompute is not True:
            raise RuntimeError("ACACT_R1_TARGET_RECOMPUTE_REQUIRED")


def make_target_position_action_r1(*, action_id: str, policy_id: str, policy_version: str, target_direction: str, requested_target_risk: float) -> TargetPositionActionR1:
    action = TargetPositionActionR1(
        action_id=action_id,
        schema_version=ACTION_VERSION_R1,
        action_semantics=TARGET_POSITION_SEMANTICS_R1,
        policy_id=policy_id,
        policy_version=policy_version,
        target_direction=target_direction,
        requested_target_risk=_risk(requested_target_risk),
        requested_target_risk_semantics=TARGET_RISK_SEMANTICS_R1,
    )
    action.validate()
    return action


def classify_target_request_relation_r1(*, source_direction: str, source_requested_risk: float, action: TargetPositionActionR1) -> TargetRequestRelationR1:
    source_direction, source_risk = _direction_risk(source_direction, source_requested_risk)
    action.validate()
    target_direction = action.target_direction
    target_risk = _risk(action.requested_target_risk)

    if source_direction == target_direction:
        if source_risk == target_risk:
            kind = SAME_NOMINAL_REQUEST
        elif source_direction == LONG:
            kind = INCREASE_LONG_REQUEST if target_risk > source_risk else DECREASE_LONG_REQUEST
        elif source_direction == SHORT:
            kind = INCREASE_SHORT_REQUEST if target_risk > source_risk else DECREASE_SHORT_REQUEST
        else:
            raise RuntimeError("ACACT_R1_FLAT_RELATION_NONCANONICAL")
    elif source_direction == FLAT:
        kind = OPEN_LONG if target_direction == LONG else OPEN_SHORT
    elif target_direction == FLAT:
        kind = CLOSE_LONG_REQUEST if source_direction == LONG else CLOSE_SHORT_REQUEST
    elif source_direction == LONG and target_direction == SHORT:
        kind = REVERSE_LONG_TO_SHORT_REQUEST
    elif source_direction == SHORT and target_direction == LONG:
        kind = REVERSE_SHORT_TO_LONG_REQUEST
    else:
        raise RuntimeError("ACACT_R1_RELATION_UNREPRESENTABLE")

    relation = TargetRequestRelationR1(
        action_id=action.action_id,
        source_direction=source_direction,
        source_requested_risk=source_risk,
        target_direction=target_direction,
        requested_target_risk=target_risk,
        relation_kind=kind,
        requires_target_recompute=True,
    )
    relation.validate()
    return relation
