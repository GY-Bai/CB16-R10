from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from cb16_local_opt.action_contract_r0 import LONG, make_target_position_action_r0
from cb16_local_opt.execution_record_r0 import (
    ExecutedDeltaRecordR0,
    SupervisorPermissionRecordR0,
    make_action_execution_record_r0,
    make_executed_delta_record_r0,
    make_nominal_actor_record_r0,
    make_supervisor_permission_record_r0,
)


def _records():
    action = make_target_position_action_r0(
        action_id="action-001",
        policy_id="policy-r0",
        policy_version="policy-v1",
        target_direction=LONG,
        requested_target_risk=0.8,
    )
    nominal = make_nominal_actor_record_r0(
        record_id="nominal-001",
        action=action,
    )
    permission = make_supervisor_permission_record_r0(
        record_id="permission-001",
        nominal_action_id=action.action_id,
        permission_payload={
            "outcome": "CLAMP",
            "reason_code": "MAX_EXPOSURE",
            "permitted_target_direction": LONG,
            "permitted_target_risk": 0.4,
        },
    )
    execution = make_executed_delta_record_r0(
        record_id="execution-001",
        nominal_action_id=action.action_id,
        permission_record_id=permission.record_id,
        executed_delta_quantity=0.31,
        execution_payload={
            "execution_status": "EXECUTED",
            "post_position_quantity": 0.31,
            "price": 30000.0,
        },
    )
    return nominal, permission, execution


def test_nominal_permission_and_execution_can_all_differ_without_field_loss() -> None:
    nominal, permission, execution = _records()
    combined = make_action_execution_record_r0(
        record_id="combined-001",
        nominal=nominal,
        permission=permission,
        execution=execution,
    )

    payload = combined.to_payload()
    assert payload["nominal"]["action"]["requested_target_risk"] == 0.8
    assert payload["permission"]["payload"]["outcome"] == "CLAMP"
    assert payload["permission"]["payload"]["permitted_target_risk"] == 0.4
    assert payload["execution"]["executed_delta_quantity"] == 0.31
    assert payload["execution"]["payload"]["post_position_quantity"] == 0.31

    assert set(payload) == {
        "record_id",
        "schema_version",
        "nominal",
        "permission",
        "execution",
    }


def test_records_are_immutable() -> None:
    nominal, permission, execution = _records()

    with pytest.raises(FrozenInstanceError):
        nominal.record_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        permission.nominal_action_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        execution.executed_delta_quantity = 99.0  # type: ignore[misc]


def test_permission_and_execution_payloads_are_canonical_immutable_strings() -> None:
    action_id = "action-001"
    permission_a = make_supervisor_permission_record_r0(
        record_id="permission-001",
        nominal_action_id=action_id,
        permission_payload={"b": 2, "a": 1},
    )
    permission_b = make_supervisor_permission_record_r0(
        record_id="permission-001",
        nominal_action_id=action_id,
        permission_payload={"a": 1, "b": 2},
    )
    assert permission_a.permission_payload_json == permission_b.permission_payload_json
    assert permission_a.payload() == {"a": 1, "b": 2}

    execution_a = make_executed_delta_record_r0(
        record_id="execution-001",
        nominal_action_id=action_id,
        permission_record_id="permission-001",
        executed_delta_quantity=-0.0,
        execution_payload={"z": 3, "a": 1},
    )
    execution_b = make_executed_delta_record_r0(
        record_id="execution-001",
        nominal_action_id=action_id,
        permission_record_id="permission-001",
        executed_delta_quantity=0,
        execution_payload={"a": 1, "z": 3},
    )
    assert execution_a.executed_delta_quantity == 0.0
    assert execution_a.execution_payload_json == execution_b.execution_payload_json


def test_link_mismatches_fail_closed() -> None:
    nominal, permission, execution = _records()

    bad_permission = SupervisorPermissionRecordR0(
        record_id=permission.record_id,
        schema_version=permission.schema_version,
        permission_execution_version=permission.permission_execution_version,
        nominal_action_id="wrong-action",
        permission_payload_json=permission.permission_payload_json,
    )
    with pytest.raises(RuntimeError, match="ACREC_PERMISSION_ACTION_LINK_MISMATCH"):
        make_action_execution_record_r0(
            record_id="combined-bad-permission",
            nominal=nominal,
            permission=bad_permission,
            execution=execution,
        )

    bad_execution = ExecutedDeltaRecordR0(
        record_id=execution.record_id,
        schema_version=execution.schema_version,
        permission_execution_version=execution.permission_execution_version,
        nominal_action_id=execution.nominal_action_id,
        permission_record_id="wrong-permission",
        executed_delta_quantity=execution.executed_delta_quantity,
        execution_payload_json=execution.execution_payload_json,
    )
    with pytest.raises(RuntimeError, match="ACREC_EXECUTION_PERMISSION_LINK_MISMATCH"):
        make_action_execution_record_r0(
            record_id="combined-bad-execution",
            nominal=nominal,
            permission=permission,
            execution=bad_execution,
        )


def test_noncanonical_or_invalid_payloads_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="ACREC_PERMISSION_PAYLOAD_INVALID"):
        make_supervisor_permission_record_r0(
            record_id="permission-bad",
            nominal_action_id="action-001",
            permission_payload={"bad": float("nan")},
        )

    with pytest.raises(RuntimeError, match="ACREC_EXECUTED_DELTA_INVALID"):
        make_executed_delta_record_r0(
            record_id="execution-bad",
            nominal_action_id="action-001",
            permission_record_id="permission-001",
            executed_delta_quantity=float("inf"),
            execution_payload={},
        )
