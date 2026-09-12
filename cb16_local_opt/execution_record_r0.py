from __future__ import annotations

"""Immutable nominal / permitted / executed record separation for Actor-Critic R0.

AC-008 freezes the separation boundary only. AC-009 adds immutable behavior-
policy provenance for sampled nominal actions without storing or fabricating a
probability. Supervisor permission semantics remain an immutable canonical
payload until AC-010 owns ACCEPT/CLAMP/REJECT. Physics execution details remain
an immutable payload plus actual signed executed quantity until AC-014.
"""

from dataclasses import dataclass
import json
import math
from typing import Mapping

from .action_contract_r0 import (
    ActionBehaviorPolicyBindingR0,
    TargetPositionActionR0,
    validate_action_behavior_binding_r0,
)
from .actor_critic_contract_r0 import PERMISSION_EXECUTION_VERSION_R0


NOMINAL_ACTOR_RECORD_SCHEMA_R0 = "CB16_R11_AC_NOMINAL_ACTOR_RECORD_V1_R0"
BEHAVIOR_BOUND_NOMINAL_RECORD_SCHEMA_R0 = (
    "CB16_R11_AC_BEHAVIOR_BOUND_NOMINAL_RECORD_V1_R0"
)
PERMISSION_RECORD_SCHEMA_R0 = "CB16_R11_AC_PERMISSION_RECORD_V1_R0"
EXECUTION_DELTA_RECORD_SCHEMA_R0 = "CB16_R11_AC_EXECUTION_DELTA_RECORD_V1_R0"
COMBINED_EXECUTION_RECORD_SCHEMA_R0 = "CB16_R11_AC_ACTION_EXECUTION_RECORD_V1_R0"


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _canonical_json_object(payload: Mapping[str, object], *, code: str) -> str:
    if not isinstance(payload, Mapping):
        raise RuntimeError(code)
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(code) from exc


def _require_finite_number(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(code)
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(code)
    return 0.0 if number == 0.0 else number


@dataclass(frozen=True)
class NominalActorActionRecordR0:
    record_id: str
    schema_version: str
    action: TargetPositionActionR0

    def validate(self) -> None:
        _require_nonempty_string(self.record_id, code="ACREC_NOMINAL_RECORD_ID_INVALID")
        if self.schema_version != NOMINAL_ACTOR_RECORD_SCHEMA_R0:
            raise RuntimeError("ACREC_NOMINAL_SCHEMA_MISMATCH")
        if not isinstance(self.action, TargetPositionActionR0):
            raise RuntimeError("ACREC_NOMINAL_ACTION_TYPE_INVALID")
        self.action.validate()


@dataclass(frozen=True)
class BehaviorBoundNominalActorActionRecordR0:
    """Nominal sampled action plus exact immutable behavior-policy provenance."""

    record_id: str
    schema_version: str
    nominal: NominalActorActionRecordR0
    behavior_binding: ActionBehaviorPolicyBindingR0

    def validate(self) -> None:
        _require_nonempty_string(
            self.record_id,
            code="ACREC_BOUND_NOMINAL_RECORD_ID_INVALID",
        )
        if self.schema_version != BEHAVIOR_BOUND_NOMINAL_RECORD_SCHEMA_R0:
            raise RuntimeError("ACREC_BOUND_NOMINAL_SCHEMA_MISMATCH")
        if not isinstance(self.nominal, NominalActorActionRecordR0):
            raise RuntimeError("ACREC_BOUND_NOMINAL_TYPE_INVALID")
        self.nominal.validate()
        validate_action_behavior_binding_r0(
            self.nominal.action,
            self.behavior_binding,
        )

    def to_payload(self) -> dict[str, object]:
        self.validate()
        payload = {
            "record_id": self.record_id,
            "schema_version": self.schema_version,
            "nominal": {
                "record_id": self.nominal.record_id,
                "schema_version": self.nominal.schema_version,
                "action": self.nominal.action.to_payload(),
            },
            "behavior_policy": self.behavior_binding.to_payload(),
        }
        # AC-009 freezes provenance only. log_mu/log_prob enters only after the
        # exact Actor distribution is frozen and auditable in AC-019.
        assert "log_mu" not in payload
        return payload


@dataclass(frozen=True)
class SupervisorPermissionRecordR0:
    record_id: str
    schema_version: str
    permission_execution_version: str
    nominal_action_id: str
    permission_payload_json: str

    def validate(self) -> None:
        _require_nonempty_string(self.record_id, code="ACREC_PERMISSION_RECORD_ID_INVALID")
        _require_nonempty_string(
            self.nominal_action_id,
            code="ACREC_PERMISSION_ACTION_ID_INVALID",
        )
        if self.schema_version != PERMISSION_RECORD_SCHEMA_R0:
            raise RuntimeError("ACREC_PERMISSION_SCHEMA_MISMATCH")
        if self.permission_execution_version != PERMISSION_EXECUTION_VERSION_R0:
            raise RuntimeError("ACREC_PERMISSION_EXECUTION_VERSION_MISMATCH")
        _require_nonempty_string(
            self.permission_payload_json,
            code="ACREC_PERMISSION_PAYLOAD_INVALID",
        )
        try:
            decoded = json.loads(self.permission_payload_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ACREC_PERMISSION_PAYLOAD_INVALID") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("ACREC_PERMISSION_PAYLOAD_INVALID")
        canonical = _canonical_json_object(
            decoded,
            code="ACREC_PERMISSION_PAYLOAD_INVALID",
        )
        if canonical != self.permission_payload_json:
            raise RuntimeError("ACREC_PERMISSION_PAYLOAD_NONCANONICAL")

    def payload(self) -> dict[str, object]:
        self.validate()
        return json.loads(self.permission_payload_json)


@dataclass(frozen=True)
class ExecutedDeltaRecordR0:
    record_id: str
    schema_version: str
    permission_execution_version: str
    nominal_action_id: str
    permission_record_id: str
    executed_delta_quantity: float
    execution_payload_json: str

    def validate(self) -> None:
        _require_nonempty_string(self.record_id, code="ACREC_EXECUTION_RECORD_ID_INVALID")
        _require_nonempty_string(
            self.nominal_action_id,
            code="ACREC_EXECUTION_ACTION_ID_INVALID",
        )
        _require_nonempty_string(
            self.permission_record_id,
            code="ACREC_EXECUTION_PERMISSION_ID_INVALID",
        )
        if self.schema_version != EXECUTION_DELTA_RECORD_SCHEMA_R0:
            raise RuntimeError("ACREC_EXECUTION_SCHEMA_MISMATCH")
        if self.permission_execution_version != PERMISSION_EXECUTION_VERSION_R0:
            raise RuntimeError("ACREC_PERMISSION_EXECUTION_VERSION_MISMATCH")
        _require_finite_number(
            self.executed_delta_quantity,
            code="ACREC_EXECUTED_DELTA_INVALID",
        )
        _require_nonempty_string(
            self.execution_payload_json,
            code="ACREC_EXECUTION_PAYLOAD_INVALID",
        )
        try:
            decoded = json.loads(self.execution_payload_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ACREC_EXECUTION_PAYLOAD_INVALID") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("ACREC_EXECUTION_PAYLOAD_INVALID")
        canonical = _canonical_json_object(
            decoded,
            code="ACREC_EXECUTION_PAYLOAD_INVALID",
        )
        if canonical != self.execution_payload_json:
            raise RuntimeError("ACREC_EXECUTION_PAYLOAD_NONCANONICAL")

    def payload(self) -> dict[str, object]:
        self.validate()
        return json.loads(self.execution_payload_json)


@dataclass(frozen=True)
class ActionExecutionRecordR0:
    record_id: str
    schema_version: str
    nominal: NominalActorActionRecordR0
    permission: SupervisorPermissionRecordR0
    execution: ExecutedDeltaRecordR0

    def validate(self) -> None:
        _require_nonempty_string(self.record_id, code="ACREC_COMBINED_RECORD_ID_INVALID")
        if self.schema_version != COMBINED_EXECUTION_RECORD_SCHEMA_R0:
            raise RuntimeError("ACREC_COMBINED_SCHEMA_MISMATCH")
        self.nominal.validate()
        self.permission.validate()
        self.execution.validate()
        action_id = self.nominal.action.action_id
        if self.permission.nominal_action_id != action_id:
            raise RuntimeError("ACREC_PERMISSION_ACTION_LINK_MISMATCH")
        if self.execution.nominal_action_id != action_id:
            raise RuntimeError("ACREC_EXECUTION_ACTION_LINK_MISMATCH")
        if self.execution.permission_record_id != self.permission.record_id:
            raise RuntimeError("ACREC_EXECUTION_PERMISSION_LINK_MISMATCH")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "record_id": self.record_id,
            "schema_version": self.schema_version,
            "nominal": {
                "record_id": self.nominal.record_id,
                "schema_version": self.nominal.schema_version,
                "action": self.nominal.action.to_payload(),
            },
            "permission": {
                "record_id": self.permission.record_id,
                "schema_version": self.permission.schema_version,
                "permission_execution_version": self.permission.permission_execution_version,
                "nominal_action_id": self.permission.nominal_action_id,
                "payload": self.permission.payload(),
            },
            "execution": {
                "record_id": self.execution.record_id,
                "schema_version": self.execution.schema_version,
                "permission_execution_version": self.execution.permission_execution_version,
                "nominal_action_id": self.execution.nominal_action_id,
                "permission_record_id": self.execution.permission_record_id,
                "executed_delta_quantity": _require_finite_number(
                    self.execution.executed_delta_quantity,
                    code="ACREC_EXECUTED_DELTA_INVALID",
                ),
                "payload": self.execution.payload(),
            },
        }


def make_nominal_actor_record_r0(
    *,
    record_id: str,
    action: TargetPositionActionR0,
) -> NominalActorActionRecordR0:
    record = NominalActorActionRecordR0(
        record_id=record_id,
        schema_version=NOMINAL_ACTOR_RECORD_SCHEMA_R0,
        action=action,
    )
    record.validate()
    return record


def make_behavior_bound_nominal_actor_record_r0(
    *,
    record_id: str,
    nominal: NominalActorActionRecordR0,
    behavior_binding: ActionBehaviorPolicyBindingR0,
) -> BehaviorBoundNominalActorActionRecordR0:
    record = BehaviorBoundNominalActorActionRecordR0(
        record_id=record_id,
        schema_version=BEHAVIOR_BOUND_NOMINAL_RECORD_SCHEMA_R0,
        nominal=nominal,
        behavior_binding=behavior_binding,
    )
    record.validate()
    return record


def make_supervisor_permission_record_r0(
    *,
    record_id: str,
    nominal_action_id: str,
    permission_payload: Mapping[str, object],
) -> SupervisorPermissionRecordR0:
    record = SupervisorPermissionRecordR0(
        record_id=record_id,
        schema_version=PERMISSION_RECORD_SCHEMA_R0,
        permission_execution_version=PERMISSION_EXECUTION_VERSION_R0,
        nominal_action_id=nominal_action_id,
        permission_payload_json=_canonical_json_object(
            permission_payload,
            code="ACREC_PERMISSION_PAYLOAD_INVALID",
        ),
    )
    record.validate()
    return record


def make_executed_delta_record_r0(
    *,
    record_id: str,
    nominal_action_id: str,
    permission_record_id: str,
    executed_delta_quantity: float,
    execution_payload: Mapping[str, object],
) -> ExecutedDeltaRecordR0:
    record = ExecutedDeltaRecordR0(
        record_id=record_id,
        schema_version=EXECUTION_DELTA_RECORD_SCHEMA_R0,
        permission_execution_version=PERMISSION_EXECUTION_VERSION_R0,
        nominal_action_id=nominal_action_id,
        permission_record_id=permission_record_id,
        executed_delta_quantity=_require_finite_number(
            executed_delta_quantity,
            code="ACREC_EXECUTED_DELTA_INVALID",
        ),
        execution_payload_json=_canonical_json_object(
            execution_payload,
            code="ACREC_EXECUTION_PAYLOAD_INVALID",
        ),
    )
    record.validate()
    return record


def make_action_execution_record_r0(
    *,
    record_id: str,
    nominal: NominalActorActionRecordR0,
    permission: SupervisorPermissionRecordR0,
    execution: ExecutedDeltaRecordR0,
) -> ActionExecutionRecordR0:
    record = ActionExecutionRecordR0(
        record_id=record_id,
        schema_version=COMBINED_EXECUTION_RECORD_SCHEMA_R0,
        nominal=nominal,
        permission=permission,
        execution=execution,
    )
    record.validate()
    return record
