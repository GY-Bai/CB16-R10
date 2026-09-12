from __future__ import annotations

"""Canonical target-state action contract for the R11 Actor-Critic lane.

AC-005 introduces the datatype. AC-006 freezes canonical encoding, FLAT risk
semantics, and the fact that ``requested_target_risk`` is a target-exposure
request rather than confidence. AC-007 adds the complete nominal target-state
transition matrix. AC-009 binds each sampled action to one immutable behavior
policy identity without fabricating any probability or log-probability.
"""

from dataclasses import dataclass
import json
import math
import re
from typing import Mapping

from .actor_critic_contract_r0 import (
    ACTION_VERSION_R0,
    ACTOR_DISTRIBUTION_VERSION_R0,
)


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

SAME_TARGET_NOOP = "SAME_TARGET_NOOP"
OPEN_LONG = "OPEN_LONG"
OPEN_SHORT = "OPEN_SHORT"
INCREASE_LONG = "INCREASE_LONG"
DECREASE_LONG = "DECREASE_LONG"
INCREASE_SHORT = "INCREASE_SHORT"
DECREASE_SHORT = "DECREASE_SHORT"
CLOSE_LONG = "CLOSE_LONG"
CLOSE_SHORT = "CLOSE_SHORT"
REVERSE_LONG_TO_SHORT = "REVERSE_LONG_TO_SHORT"
REVERSE_SHORT_TO_LONG = "REVERSE_SHORT_TO_LONG"
TARGET_TRANSITION_KINDS_R0 = (
    SAME_TARGET_NOOP,
    OPEN_LONG,
    OPEN_SHORT,
    INCREASE_LONG,
    DECREASE_LONG,
    INCREASE_SHORT,
    DECREASE_SHORT,
    CLOSE_LONG,
    CLOSE_SHORT,
    REVERSE_LONG_TO_SHORT,
    REVERSE_SHORT_TO_LONG,
)


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _require_sha256_hex(value: object, *, code: str) -> str:
    text = _require_nonempty_string(value, code=code)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise RuntimeError(code)
    return text


def _canonical_risk(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("ACACT_TARGET_RISK_TYPE_INVALID")
    risk = float(value)
    if not math.isfinite(risk) or not 0.0 <= risk <= 1.0:
        raise RuntimeError("ACACT_TARGET_RISK_OUT_OF_RANGE")
    # JSON distinguishes -0.0 from 0.0 even though target exposure does not.
    return 0.0 if risk == 0.0 else risk


def _validate_direction_risk_pair(direction: object, risk_value: object) -> tuple[str, float]:
    direction_text = _require_nonempty_string(
        direction,
        code="ACACT_TARGET_DIRECTION_INVALID",
    )
    if direction_text not in TARGET_DIRECTIONS_R0:
        raise RuntimeError("ACACT_TARGET_DIRECTION_INVALID")
    risk = _canonical_risk(risk_value)
    if direction_text == FLAT and risk != 0.0:
        raise RuntimeError("ACACT_FLAT_TARGET_RISK_MUST_BE_ZERO")
    return direction_text, risk


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
        _validate_direction_risk_pair(
            self.target_direction,
            self.requested_target_risk,
        )

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


@dataclass(frozen=True)
class TargetStateTransitionR0:
    """Nominal state transition before Supervisor permission and Physics."""

    action_id: str
    source_direction: str
    source_target_risk: float
    target_direction: str
    requested_target_risk: float
    transition_kind: str

    def validate(self) -> None:
        _require_nonempty_string(self.action_id, code="ACACT_ACTION_ID_INVALID")
        _validate_direction_risk_pair(self.source_direction, self.source_target_risk)
        _validate_direction_risk_pair(self.target_direction, self.requested_target_risk)
        if self.transition_kind not in TARGET_TRANSITION_KINDS_R0:
            raise RuntimeError("ACACT_TRANSITION_KIND_INVALID")

    @property
    def is_noop(self) -> bool:
        return self.transition_kind == SAME_TARGET_NOOP

    @property
    def is_reversal(self) -> bool:
        return self.transition_kind in (
            REVERSE_LONG_TO_SHORT,
            REVERSE_SHORT_TO_LONG,
        )


@dataclass(frozen=True)
class ActionBehaviorPolicyBindingR0:
    """Immutable provenance linking one sampled action to one behavior policy."""

    action_id: str
    behavior_policy_id: str
    behavior_policy_version: str
    behavior_policy_hash: str
    actor_distribution_version: str

    def validate(self) -> None:
        _require_nonempty_string(self.action_id, code="ACACT_BINDING_ACTION_ID_INVALID")
        _require_nonempty_string(
            self.behavior_policy_id,
            code="ACACT_BINDING_POLICY_ID_INVALID",
        )
        _require_nonempty_string(
            self.behavior_policy_version,
            code="ACACT_BINDING_POLICY_VERSION_INVALID",
        )
        _require_sha256_hex(
            self.behavior_policy_hash,
            code="ACACT_BINDING_POLICY_HASH_INVALID",
        )
        if self.actor_distribution_version != ACTOR_DISTRIBUTION_VERSION_R0:
            raise RuntimeError("ACACT_BINDING_DISTRIBUTION_VERSION_MISMATCH")

    def to_payload(self) -> dict[str, str]:
        self.validate()
        return {
            "action_id": self.action_id,
            "behavior_policy_id": self.behavior_policy_id,
            "behavior_policy_version": self.behavior_policy_version,
            "behavior_policy_hash": self.behavior_policy_hash,
            "actor_distribution_version": self.actor_distribution_version,
        }


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


def classify_target_state_transition_r0(
    *,
    source_direction: str,
    source_target_risk: float,
    action: TargetPositionActionR0,
) -> TargetStateTransitionR0:
    """Classify every legal nominal target-state change without execution policy."""

    source_direction, source_risk = _validate_direction_risk_pair(
        source_direction,
        source_target_risk,
    )
    action.validate()
    target_direction = action.target_direction
    target_risk = _canonical_risk(action.requested_target_risk)

    if source_direction == target_direction:
        if source_risk == target_risk:
            kind = SAME_TARGET_NOOP
        elif source_direction == LONG:
            kind = INCREASE_LONG if target_risk > source_risk else DECREASE_LONG
        elif source_direction == SHORT:
            kind = INCREASE_SHORT if target_risk > source_risk else DECREASE_SHORT
        else:
            # FLAT has exactly one canonical risk representation: zero.
            raise RuntimeError("ACACT_FLAT_TRANSITION_NONCANONICAL")
    elif source_direction == FLAT:
        kind = OPEN_LONG if target_direction == LONG else OPEN_SHORT
    elif target_direction == FLAT:
        kind = CLOSE_LONG if source_direction == LONG else CLOSE_SHORT
    elif source_direction == LONG and target_direction == SHORT:
        kind = REVERSE_LONG_TO_SHORT
    elif source_direction == SHORT and target_direction == LONG:
        kind = REVERSE_SHORT_TO_LONG
    else:
        raise RuntimeError("ACACT_TRANSITION_UNREPRESENTABLE")

    transition = TargetStateTransitionR0(
        action_id=action.action_id,
        source_direction=source_direction,
        source_target_risk=source_risk,
        target_direction=target_direction,
        requested_target_risk=target_risk,
        transition_kind=kind,
    )
    transition.validate()
    return transition


def bind_action_behavior_policy_r0(
    action: TargetPositionActionR0,
    *,
    behavior_policy_hash: str,
    actor_distribution_version: str = ACTOR_DISTRIBUTION_VERSION_R0,
) -> ActionBehaviorPolicyBindingR0:
    """Bind provenance only; probability/log_mu is intentionally absent here."""

    action.validate()
    binding = ActionBehaviorPolicyBindingR0(
        action_id=action.action_id,
        behavior_policy_id=action.policy_id,
        behavior_policy_version=action.policy_version,
        behavior_policy_hash=_require_sha256_hex(
            behavior_policy_hash,
            code="ACACT_BINDING_POLICY_HASH_INVALID",
        ),
        actor_distribution_version=actor_distribution_version,
    )
    binding.validate()
    return binding


def validate_action_behavior_binding_r0(
    action: TargetPositionActionR0,
    binding: ActionBehaviorPolicyBindingR0,
) -> None:
    action.validate()
    if not isinstance(binding, ActionBehaviorPolicyBindingR0):
        raise RuntimeError("ACACT_BINDING_TYPE_INVALID")
    binding.validate()
    if binding.action_id != action.action_id:
        raise RuntimeError("ACACT_BINDING_ACTION_MISMATCH")
    if binding.behavior_policy_id != action.policy_id:
        raise RuntimeError("ACACT_BINDING_POLICY_ID_MISMATCH")
    if binding.behavior_policy_version != action.policy_version:
        raise RuntimeError("ACACT_BINDING_POLICY_VERSION_MISMATCH")
