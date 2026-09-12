from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.actor_observation_r0 import ActorObservationR0
from cb16_local_opt.critic_observation_r0 import (
    CRITIC_FORBIDDEN_FIELDS_R0,
    CRITIC_TAU_SEMANTICS_R0,
    CriticObservationR0,
    build_critic_observation_r0,
)


def _account_obs(*, cash: float = 80.0, mark_price: float = 12.0):
    state = make_account_economics_state_r0(
        account_id="acct",
        cash=cash,
        position_quantity=2.0,
        position_cost_basis=10.0,
        mark_price=mark_price,
        realized_pnl_cumulative=5.0,
        fees_cumulative=2.0,
        funding_cumulative=0.0,
        margin_collateral=20.0,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    return project_account_observation_r0(state)


def _build(
    *,
    market=(1.0, 2.0, 3.0),
    execution=(1.0, 0.5),
    tau: float = 72.0,
    account=None,
):
    return build_critic_observation_r0(
        market_sensory_version="MARKET_CAUSAL_V1",
        market_causal_features=market,
        account_observation=_account_obs() if account is None else account,
        legal_execution_version="LEGAL_EXEC_CAUSAL_V1",
        legal_execution_causal_features=execution,
        tau=tau,
    )


def test_critic_can_receive_tau_while_actor_schema_cannot() -> None:
    critic_fields = set(CriticObservationR0.__dataclass_fields__)
    actor_fields = set(ActorObservationR0.__dataclass_fields__)
    assert "tau" in critic_fields
    assert "tau" not in actor_fields
    obs = _build(tau=48.0)
    assert obs.tau == 48.0
    assert obs.tau_semantics == CRITIC_TAU_SEMANTICS_R0
    assert obs.model_payload()["tau"] == 48.0


def test_known_objective_remainder_changes_critic_input_at_same_causal_state() -> None:
    short = _build(tau=12.0)
    long = _build(tau=120.0)
    short_payload = short.model_payload()
    long_payload = long.model_payload()
    assert short_payload["tau"] != long_payload["tau"]
    assert short_payload["market_causal_features"] == long_payload["market_causal_features"]
    assert short_payload["account_causal_features"] == long_payload["account_causal_features"]
    assert short_payload["legal_execution_causal_features"] == long_payload["legal_execution_causal_features"]


def test_future_market_poison_does_not_enter_current_critic_observation() -> None:
    causal_prefix = (1.0, 2.0, 3.0)
    future_a = (999.0, 999.0)
    future_b = (-999.0, -999.0)
    first = _build(market=causal_prefix)
    second = _build(market=causal_prefix)
    assert future_a != future_b
    assert first.model_payload() == second.model_payload()


def test_critic_schema_has_no_future_outcome_or_teacher_fields() -> None:
    fields = set(CriticObservationR0.__dataclass_fields__)
    assert set(CRITIC_FORBIDDEN_FIELDS_R0).isdisjoint(fields)


def test_current_causal_market_account_and_execution_state_are_visible() -> None:
    baseline = _build().model_payload()
    changed_market = _build(market=(1.0, 2.0, 4.0)).model_payload()
    changed_account = _build(account=_account_obs(cash=70.0)).model_payload()
    changed_execution = _build(execution=(0.0, 0.2)).model_payload()
    assert baseline != changed_market
    assert baseline != changed_account
    assert baseline != changed_execution


def test_tau_must_be_finite_nonnegative_numeric() -> None:
    assert _build(tau=0).tau == 0.0
    for bad in (-1, float("inf"), float("nan"), True, "72"):
        with pytest.raises(RuntimeError):
            _build(tau=bad)  # type: ignore[arg-type]


def test_tau_semantics_tamper_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="TAU_SEMANTICS_MISMATCH"):
        replace(_build(), tau_semantics="FUTURE_ORACLE").validate()


def test_optional_memory_disabled_mode_is_valid() -> None:
    obs = _build()
    assert obs.policy_memory_version is None
    assert obs.policy_memory_causal_features == ()


def test_schema_versions_are_metadata_not_model_numeric_features() -> None:
    payload = _build().model_payload()
    assert "market_sensory_version" not in payload
    assert "account_projection_version" not in payload
    assert "legal_execution_version" not in payload
    assert "tau_semantics" not in payload
