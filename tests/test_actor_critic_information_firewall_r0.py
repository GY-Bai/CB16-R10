from __future__ import annotations

from dataclasses import fields

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import (
    ACCOUNT_POLICY_VISIBLE_FIELDS_R0,
    project_account_observation_r0,
)
from cb16_local_opt.actor_observation_r0 import (
    ACTOR_FORBIDDEN_FIELDS_R0,
    ActorObservationR0,
    build_actor_observation_r0,
)
from cb16_local_opt.critic_observation_r0 import (
    CRITIC_FORBIDDEN_FIELDS_R0,
    CriticObservationR0,
    build_critic_observation_r0,
)


ACTOR_SCHEMA_FIELDS_R0 = (
    "schema_version",
    "market_sensory_version",
    "market_causal_features",
    "account_projection_version",
    "account_causal_features",
    "legal_execution_version",
    "legal_execution_causal_features",
    "policy_memory_version",
    "policy_memory_causal_features",
)
CRITIC_SCHEMA_FIELDS_R0 = ACTOR_SCHEMA_FIELDS_R0 + (
    "tau_semantics",
    "tau",
)
ACTOR_MODEL_FIELDS_R0 = (
    "market_causal_features",
    "account_causal_features",
    "legal_execution_causal_features",
    "policy_memory_causal_features",
)
CRITIC_MODEL_FIELDS_R0 = ACTOR_MODEL_FIELDS_R0 + ("tau",)

FORBIDDEN_ONLINE_NAMES_R0 = frozenset(
    {
        "objective_horizon",
        "objective_end_time",
        "future_market",
        "future_return",
        "future_outcome",
        "teacher_target",
        "teacher_value",
        "realized_future_return",
        "realized_winner",
    }
)


def _account_observation():
    state = make_account_economics_state_r0(
        account_id="acct-firewall",
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


def _actor(*, market=(1.0, 2.0, 3.0)):
    return build_actor_observation_r0(
        market_sensory_version="MARKET_CAUSAL_V1",
        market_causal_features=market,
        account_observation=_account_observation(),
        legal_execution_version="LEGAL_EXEC_CAUSAL_V1",
        legal_execution_causal_features=(1.0, 0.5),
    )


def _critic(*, market=(1.0, 2.0, 3.0), tau=72.0):
    return build_critic_observation_r0(
        market_sensory_version="MARKET_CAUSAL_V1",
        market_causal_features=market,
        account_observation=_account_observation(),
        legal_execution_version="LEGAL_EXEC_CAUSAL_V1",
        legal_execution_causal_features=(1.0, 0.5),
        tau=tau,
    )


def _field_names(dataclass_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def _account_feature_names(payload: dict[str, object]) -> tuple[str, ...]:
    raw = payload["account_causal_features"]
    assert isinstance(raw, tuple)
    return tuple(key for key, _ in raw)


def test_actor_and_critic_schema_whitelists_are_exact() -> None:
    assert _field_names(ActorObservationR0) == ACTOR_SCHEMA_FIELDS_R0
    assert _field_names(CriticObservationR0) == CRITIC_SCHEMA_FIELDS_R0
    assert tuple(_actor().model_payload()) == ACTOR_MODEL_FIELDS_R0
    assert tuple(_critic().model_payload()) == CRITIC_MODEL_FIELDS_R0


def test_tau_firewall_is_one_way_critic_only() -> None:
    actor_fields = set(_field_names(ActorObservationR0))
    critic_fields = set(_field_names(CriticObservationR0))
    actor_payload = _actor().model_payload()
    critic_payload = _critic(tau=48.0).model_payload()

    assert "tau" not in actor_fields
    assert "tau" not in actor_payload
    assert "tau" in critic_fields
    assert critic_payload["tau"] == 48.0


def test_no_future_teacher_or_objective_field_enters_online_schema_or_payload() -> None:
    actor_fields = set(_field_names(ActorObservationR0))
    critic_fields = set(_field_names(CriticObservationR0))
    actor_payload = _actor().model_payload()
    critic_payload = _critic().model_payload()

    assert FORBIDDEN_ONLINE_NAMES_R0.isdisjoint(actor_fields)
    assert FORBIDDEN_ONLINE_NAMES_R0.isdisjoint(critic_fields)
    assert FORBIDDEN_ONLINE_NAMES_R0.isdisjoint(actor_payload)
    assert FORBIDDEN_ONLINE_NAMES_R0.isdisjoint(critic_payload)
    assert set(ACTOR_FORBIDDEN_FIELDS_R0).isdisjoint(actor_fields)
    assert set(CRITIC_FORBIDDEN_FIELDS_R0).isdisjoint(critic_fields)


def test_account_projection_inside_actor_and_critic_is_exactly_policy_whitelist() -> None:
    expected = tuple(ACCOUNT_POLICY_VISIBLE_FIELDS_R0)
    assert _account_feature_names(_actor().model_payload()) == expected
    assert _account_feature_names(_critic().model_payload()) == expected
    assert FORBIDDEN_ONLINE_NAMES_R0.isdisjoint(expected)


def test_future_market_and_outcome_poison_cannot_change_fixed_current_inputs() -> None:
    causal_market = (1.0, 2.0, 3.0)
    future_market_a = (999.0, 999.0)
    future_market_b = (-999.0, -999.0)
    future_outcome_a = {"future_return": 100.0, "teacher_target": "LONG"}
    future_outcome_b = {"future_return": -100.0, "teacher_target": "SHORT"}

    actor_a = _actor(market=causal_market).model_payload()
    actor_b = _actor(market=causal_market).model_payload()
    critic_a = _critic(market=causal_market, tau=72.0).model_payload()
    critic_b = _critic(market=causal_market, tau=72.0).model_payload()

    assert future_market_a != future_market_b
    assert future_outcome_a != future_outcome_b
    assert actor_a == actor_b
    assert critic_a == critic_b


def test_actor_objective_horizon_poison_is_blocked_but_critic_known_tau_is_visible() -> None:
    objective_t_a = 72.0
    objective_t_b = 720.0

    actor_a = _actor().model_payload()
    actor_b = _actor().model_payload()
    critic_a = _critic(tau=objective_t_a).model_payload()
    critic_b = _critic(tau=objective_t_b).model_payload()

    assert actor_a == actor_b
    assert critic_a != critic_b
    assert critic_a["tau"] == objective_t_a
    assert critic_b["tau"] == objective_t_b
