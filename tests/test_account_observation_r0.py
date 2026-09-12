from __future__ import annotations

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import (
    ACCOUNT_POLICY_OMITTED_LEDGER_FIELDS_R0,
    ACCOUNT_POLICY_VISIBLE_FIELDS_R0,
    project_account_observation_r0,
    restore_account_from_policy_observation_r0,
)


def _state(**overrides):
    kwargs = dict(
        account_id="acct",
        cash=80.0,
        position_quantity=2.0,
        position_cost_basis=10.0,
        mark_price=12.0,
        realized_pnl_cumulative=5.0,
        fees_cumulative=2.0,
        funding_cumulative=-1.0,
        margin_collateral=20.0,
        liabilities=3.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    kwargs.update(overrides)
    return make_account_economics_state_r0(**kwargs)


def test_policy_projection_has_exact_declared_model_whitelist() -> None:
    obs = project_account_observation_r0(_state())
    assert tuple(obs.model_payload()) == ACCOUNT_POLICY_VISIBLE_FIELDS_R0
    assert set(obs.model_payload()) == set(ACCOUNT_POLICY_VISIBLE_FIELDS_R0)
    assert tuple(obs.omitted_ledger_fields) == ACCOUNT_POLICY_OMITTED_LEDGER_FIELDS_R0


def test_account_identity_and_audit_flows_do_not_enter_model_payload() -> None:
    payload = project_account_observation_r0(_state()).model_payload()
    forbidden = {
        "account_id",
        "realized_pnl_cumulative",
        "fees_cumulative",
        "funding_cumulative",
        "external_capital_flows_cumulative",
        "source_restore_state_sha256",
        "mark_price",
    }
    assert forbidden.isdisjoint(payload)


def test_hidden_audit_truth_change_does_not_leak_through_projection() -> None:
    first = project_account_observation_r0(_state())
    poisoned = project_account_observation_r0(
        _state(
            realized_pnl_cumulative=9999.0,
            fees_cumulative=777.0,
            funding_cumulative=-555.0,
            external_capital_flows_cumulative=12345.0,
        )
    )
    assert first.model_payload() == poisoned.model_payload()
    assert first.source_restore_state_sha256 != poisoned.source_restore_state_sha256


def test_causal_economic_state_change_is_visible() -> None:
    first = project_account_observation_r0(_state(cash=80.0))
    second = project_account_observation_r0(_state(cash=70.0))
    assert first.model_payload() != second.model_payload()
    assert first.equity != second.equity


def test_restore_from_policy_projection_is_forbidden() -> None:
    obs = project_account_observation_r0(_state())
    with pytest.raises(RuntimeError, match="LOSSY_POLICY_PROJECTION_NOT_RESTORE_AUTHORITY"):
        restore_account_from_policy_observation_r0(obs)


def test_projection_version_and_restore_hash_are_metadata_not_model_features() -> None:
    obs = project_account_observation_r0(_state())
    payload = obs.model_payload()
    assert "schema_version" not in payload
    assert "source_restore_state_sha256" not in payload
    assert len(obs.source_restore_state_sha256) == 64
