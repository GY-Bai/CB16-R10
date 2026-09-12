from __future__ import annotations

from dataclasses import fields, replace
import inspect

import pytest

from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT
from cb16_local_opt.actor_critic_supervisor_r1 import make_supervisor_authority_r1
from cb16_local_opt.execution_feasibility_r0 import MechanicalExecutionAuthorityR0
from cb16_local_opt.execution_observation_r0 import (
    EXECUTION_OBSERVATION_FORBIDDEN_FIELDS_R0,
    EXECUTION_OBSERVATION_MODEL_FIELDS_R0,
    ExecutionObservationR0,
    build_execution_observation_r0,
)
from cb16_local_opt.target_exposure_r1 import make_target_exposure_authority_r1


ACCOUNT_HASH = "a" * 64


def _supervisor(*, legal=(SHORT, FLAT, LONG), max_risk=0.8, account_hash=ACCOUNT_HASH):
    return make_supervisor_authority_r1(
        authority_id="supervisor-authority",
        account_id="acct",
        account_state_sha256=account_hash,
        terminated=False,
        truncated=False,
        legal_target_directions=legal,
        max_permitted_target_risk=max_risk,
    )


def _target(*, price=10.0, initial_margin_rate=0.2, lot_min_qty=0.1, lot_max_qty=50.0, min_notional=5.0, account_hash=ACCOUNT_HASH):
    return make_target_exposure_authority_r1(
        authority_id="target-authority",
        account_id="acct",
        account_state_sha256=account_hash,
        legal_envelope_id="envelope",
        equity=100.0,
        current_price=price,
        margin_capacity=60.0,
        max_gross_leverage=5.0,
        initial_margin_rate=initial_margin_rate,
        declared_max_legal_notional=300.0,
        lot_step_size=0.1,
        lot_min_qty=lot_min_qty,
        lot_max_qty=lot_max_qty,
        min_notional=min_notional,
    )


def _mechanical(*, price=10.0, initial_margin_rate=0.2, available_margin=25.0, lot_min_qty=0.1, lot_max_qty=50.0, min_notional=5.0):
    return MechanicalExecutionAuthorityR0(
        authority_id="mechanical-authority",
        current_quantity=2.0,
        current_price=price,
        available_margin_for_new_exposure=available_margin,
        initial_margin_rate=initial_margin_rate,
        maintenance_margin_rate=0.1,
        maintenance_collateral=30.0,
        lot_min_qty=lot_min_qty,
        lot_max_qty=lot_max_qty,
        min_notional=min_notional,
    )


def _build(**kwargs):
    return build_execution_observation_r0(
        supervisor_authority=kwargs.get("supervisor", _supervisor()),
        target_exposure_authority=kwargs.get("target", _target()),
        mechanical_authority=kwargs.get("mechanical", _mechanical()),
    )


def test_model_payload_is_exact_mechanical_pre_action_whitelist() -> None:
    observation = _build()
    assert tuple(observation.model_payload()) == EXECUTION_OBSERVATION_MODEL_FIELDS_R0
    assert tuple(field.name for field in fields(ExecutionObservationR0))[:5] == (
        "schema_version",
        "supervisor_authority_sha256",
        "target_exposure_authority_sha256",
        "mechanical_authority_id",
        "account_state_sha256",
    )
    assert set(EXECUTION_OBSERVATION_FORBIDDEN_FIELDS_R0).isdisjoint(observation.model_payload())


def test_policy_can_distinguish_current_legal_directions_without_profitability_advice() -> None:
    all_legal = _build(supervisor=_supervisor(legal=(SHORT, FLAT, LONG))).model_payload()
    flat_only = _build(supervisor=_supervisor(legal=(FLAT,))).model_payload()

    assert all_legal["legal_short"] is True
    assert all_legal["legal_long"] is True
    assert flat_only["legal_short"] is False
    assert flat_only["legal_flat"] is True
    assert flat_only["legal_long"] is False
    assert all_legal != flat_only
    assert not any("profit" in key or "recommend" in key for key in all_legal)


def test_current_mechanical_resources_are_visible_to_policy_projection() -> None:
    rich = _build(mechanical=_mechanical(available_margin=25.0)).model_payload()
    constrained = _build(mechanical=_mechanical(available_margin=0.0)).model_payload()
    assert rich["available_margin_for_new_exposure"] == 25.0
    assert constrained["available_margin_for_new_exposure"] == 0.0
    assert rich != constrained


def test_projection_builder_accepts_only_pre_action_authorities() -> None:
    parameters = tuple(inspect.signature(build_execution_observation_r0).parameters)
    assert parameters == (
        "supervisor_authority",
        "target_exposure_authority",
        "mechanical_authority",
    )
    assert "target" not in parameters
    assert "action" not in parameters
    assert "feasibility" not in parameters
    assert "execution" not in parameters


def test_action_specific_and_strategy_fields_are_absent_from_schema() -> None:
    schema_fields = {field.name for field in fields(ExecutionObservationR0)}
    assert set(EXECUTION_OBSERVATION_FORBIDDEN_FIELDS_R0).isdisjoint(schema_fields)
    assert "target_quantity" not in schema_fields
    assert "delta_quantity" not in schema_fields
    assert "feasibility_status" not in schema_fields
    assert "expected_return" not in schema_fields
    assert "stop_loss" not in schema_fields
    assert "take_profit" not in schema_fields


def test_duplicate_authority_facts_must_agree_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="PRICE_AUTHORITY_MISMATCH"):
        _build(target=_target(price=10.0), mechanical=_mechanical(price=11.0))
    with pytest.raises(RuntimeError, match="INITIAL_MARGIN_AUTHORITY_MISMATCH"):
        _build(target=_target(initial_margin_rate=0.2), mechanical=_mechanical(initial_margin_rate=0.25))
    with pytest.raises(RuntimeError, match="LOT_MIN_QTY_AUTHORITY_MISMATCH"):
        _build(target=_target(lot_min_qty=0.1), mechanical=_mechanical(lot_min_qty=0.2))
    with pytest.raises(RuntimeError, match="LOT_MAX_QTY_AUTHORITY_MISMATCH"):
        _build(target=_target(lot_max_qty=50.0), mechanical=_mechanical(lot_max_qty=40.0))
    with pytest.raises(RuntimeError, match="MIN_NOTIONAL_AUTHORITY_MISMATCH"):
        _build(target=_target(min_notional=5.0), mechanical=_mechanical(min_notional=10.0))


def test_account_identity_and_snapshot_must_agree_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="ACCOUNT_STATE_MISMATCH"):
        _build(supervisor=_supervisor(account_hash="b" * 64), target=_target(account_hash="c" * 64))

    wrong_account = replace(_supervisor(), account_id="other-account")
    wrong_account.validate()
    with pytest.raises(RuntimeError, match="ACCOUNT_ID_MISMATCH"):
        _build(supervisor=wrong_account)


def test_authority_metadata_is_not_part_of_numeric_policy_payload() -> None:
    payload = _build().model_payload()
    assert "schema_version" not in payload
    assert "supervisor_authority_sha256" not in payload
    assert "target_exposure_authority_sha256" not in payload
    assert "mechanical_authority_id" not in payload
    assert "account_state_sha256" not in payload
