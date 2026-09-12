from __future__ import annotations

"""Canonical target-state action contract for the R11 Actor-Critic lane.

AC-005 introduces the datatype.  AC-006 freezes canonical encoding, FLAT risk
semantics, and the fact that ``requested_target_risk`` is a target-exposure
request rather than confidence.  AC-007 owns the transition matrix and AC-009
owns behavior-policy probability/log-probability binding.
"""

from dataclasses import dataclass
import json
import math
from typing import Mapping

from .actor_critic_contract_r0 import ACTION_VERSION_R0


SHORT = "SHORT"
FLAT = "FLAT"
LONG = "LONG"
TARGET_DIRECTIONS_R0 = (SHORT, FLAT, LONG)
TARGET_POSITION_SEMANTICS_R0 = "TARGET_POSITION_STATE"
TARGET_RISK_SEMANTICS_R0 = "TARGET_EXPOSURE_REQUEST"
ACTION_FIELDS_R0 = (
    "action_id",
    "schema_version",
    "action_semantics",
    "policy_id",
    "policy_version",
    "target_direction",
    "requested_target_risk",
    "requested_target_risk_semantics",
)


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _canonical_risk(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("ACACT_TARGET_RISK_TYPE_INVALID")
    risk = float(value)
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise RuntimeError("ACACT_TARGET_RISK_OUT_OF_RANGE")
    # JSON distinguishes -0.0 from 0.0 even though the target exposure does not.
    # Normalize the zero representation before hashing/receipt use downstream.
    return 0.0 if risk == 0.0 else risk


@dataclass(frozen=True)
class TargetPositionActionR0:
    """Nominal target position requested by one identified policy action."""

    action_id: str
    schema_version: str
    action_semantics: str
    policy_id: str
    policy_version: str
    target_direction: str
    requested_target_risk: float
    requested_target_risk_semantics: str

    def validate(self) -> None:
        _require_nonempty_string(self.action_id, code="ACACT_ACTION_ID_INVALID")
        _require_nonempty_string(self.policy_id, code="ACACT_POLICY_ID_INVALID")
        _require_nonempty_string(self.policy_version, code="ACACT_POLICY_VERSION_INVALID")
        if self.schema_version != ACTION_VERSION_R0:
            raise RuntimeError("ACACT_SCHEMA_VERSION_MISMATCH")
        if self.action_semantics != TARGET_POSITION_SEMANTICS_R0:
            raise RuntimeError("ACACT_SEMANTICS_MISMATCH")
        if self.requested_target_risk_semantics != TARGET_RISK_SEMANTICS_R0:
            raise RuntimeError("ACACT_TARGET_RISK_SEMANTICS_MISMATCH")
        if self.target_direction not in TARGET_DIRECTIONS_R0:
            raise RuntimeError("ACACT_TARGET_DIRECTION_INVALID")
        risk = _canonical_risk(self.requested_target_risk)
        if self.target_direction == FLAT and risk != 0.0:
            raise RuntimeError("ACACT_FLAT_TARGET_RISK_MUST_BE_ZERO")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "action_id": self.action_id,
            "schema_version": self.schema_version,
            "action_semantics": self.action_semantics,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "target_direction": self.target_direction,
            "requested_target_risk": _canonical_risk(self.requested_target_risk),
            "requested_target_risk_semantics": self.requested_target_risk_semantics,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "TargetPositionActionR0":
        required = set(ACTION_FIELDS_R0)
        actual = set(payload.keys())
        missing = sorted(required - actual)
        unknown = sorted(str(key) for key in actual if key not in required)
        if missing:
            raise RuntimeError(f"ACACT_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACACT_FIELDS_UNKNOWN:{','.join(unknown)}")

        action = cls(
            action_id=_require_nonempty_string(
                payload["action_id"], code="ACACT_ACTION_ID_INVALID"
            ),
            schema_version=_require_nonempty_string(
                payload["schema_version"], code="ACACT_SCHEMA_VERSION_INVALID"
            ),
            action_semantics=_require_nonempty_string(
                payload["action_semantics"], code="ACACT_SEMANTICS_INVALID"
            ),
            policy_id=_require_nonempty_string(
                payload["policy_id"], code="ACACT_POLICY_ID_INVALID"
            ),
            policy_version=_require_nonempty_string(
                payload["policy_version"], code="ACACT_POLICY_VERSION_INVALID"
            ),
            target_direction=_require_nonempty_string(
                payload["target_direction"], code="ACACT_TARGET_DIRECTION_INVALID"
            ),
            requested_target_risk=_canonical_risk(payload["requested_target_risk"]),
            requested_target_risk_semantics=_require_nonempty_string(
                payload["requested_target_risk_semantics"],
                code="ACACT_TARGET_RISK_SEMANTICS_INVALID",
            ),
        )
        action.validate()
        return action

    @classmethod
    def from_json(cls, encoded: str) -> "TargetPositionActionR0":
        try:
            payload = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("ACACT_JSON_INVALID") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("ACACT_JSON_OBJECT_REQUIRED")
        return cls.from_payload(payload)


def make_target_position_action_r0(
    *,
    action_id: str,
    policy_id: str,
    policy_version: str,
    target_direction: str,
    requested_target_risk: float,
) -> TargetPositionActionR0:
    action = TargetPositionActionR0(
        action_id=action_id,
        schema_version=ACTION_VERSION_R0,
        action_semantics=TARGET_POSITION_SEMANTICS_R0,
        policy_id=policy_id,
        policy_version=policy_version,
        target_direction=target_direction,
        requested_target_risk=_canonical_risk(requested_target_risk),
        requested_target_risk_semantics=TARGET_RISK_SEMANTICS_R0,
    )
    action.validate()
    return action
