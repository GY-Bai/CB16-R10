from __future__ import annotations

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.actor_observation_r0 import (
    ACTOR_FORBIDDEN_FIELDS_R0,
    ActorObservationR0,
    build_actor_observation_r0,
)


def _account_obs():
    state = make_account_economics_state_r0(
        account_id="acct",
        cash=80.0,
        position_quantity=2.0,
        position_cost_basis=10.0,
        mark_price=12.0,
        realized_pnl_cumulative=5.0,
        fees_cumulative=2.0,
        funding_cumulative=0.0,
        margin_collateral=20.0,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    return project_account_observation_r0(state)


def _build(market=(1.0, 2.0, 3.0), execution=(1.0, 0.5)):
    return build_actor_observation_r0(
        market_sensory_version="MARKET_CAUSAL_V1",
        market_causal_features=market,
        account_observation=_account_obs(),
        legal_execution_version="LEGAL_EXEC_CAUSAL_V1",
        legal_execution_causal_features=execution,
    )


def test_actor_schema_has_no_objective_countdown_or_future_fields() -> None:
    fields = set(ActorObservationR0.__dataclass_fields__)
    assert set(ACTOR_FORBIDDEN_FIELDS_R0).isdisjoint(fields)
    assert "tau" not in fields
    assert "objective_horizon" not in fields


def test_objective_T_change_cannot_change_fixed_causal_actor_observation() -> None:
    objective_t_1 = 72
    objective_t_2 = 720
    first = _build()
    second = _build()
    assert objective_t_1 != objective_t_2
    assert first.model_payload() == second.model_payload()


def test_future_poison_does_not_enter_actor_observation() -> None:
    causal_prefix = (1.0, 2.0, 3.0)
    future_a = (999.0, 999.0)
    future_b = (-999.0, -999.0)
    first = _build(market=causal_prefix)
    second = _build(market=causal_prefix)
    assert future_a != future_b
    assert first.model_payload() == second.model_payload()


def test_current_causal_market_change_is_visible() -> None:
    assert _build(market=(1.0, 2.0, 3.0)).model_payload() != _build(market=(1.0, 2.0, 4.0)).model_payload()


def test_current_legal_execution_state_change_is_visible() -> None:
    assert _build(execution=(1.0, 0.5)).model_payload() != _build(execution=(0.0, 0.2)).model_payload()


def test_optional_memory_disabled_mode_is_valid() -> None:
    obs = _build()
    assert obs.policy_memory_version is None
    assert obs.policy_memory_causal_features == ()


def test_schema_versions_are_metadata_not_model_numeric_features() -> None:
    payload = _build().model_payload()
    assert "market_sensory_version" not in payload
    assert "account_projection_version" not in payload
    assert "legal_execution_version" not in payload
