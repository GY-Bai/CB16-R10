from __future__ import annotations

from dataclasses import replace
import inspect

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.actor_observation_r0 import build_actor_observation_r0
from cb16_local_opt.policy_memory_r0 import (
    POLICY_MEMORY_DEFAULT_MODE_R0,
    POLICY_MEMORY_DISABLED,
    POLICY_MEMORY_ENABLED,
    CausalMemoryUpdateR0,
    PolicyMemoryActivationEvidenceR0,
    PolicyMemoryConfigR0,
    advance_policy_memory_r0,
    disabled_policy_memory_config_r0,
    enabled_policy_memory_config_r0,
    initial_policy_memory_state_r0,
    policy_memory_model_features_r0,
    restore_policy_memory_state_r0,
    seal_policy_memory_state_r0,
)


def _account_observation():
    state = make_account_economics_state_r0(
        account_id="acct-memory",
        cash=100.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=10.0,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0,
        economic_responsibility_open=True,
    )
    return project_account_observation_r0(state)


def _evidence(*, gap_count: int = 1, causal_history_required: bool = True):
    return PolicyMemoryActivationEvidenceR0(
        schema_version="CB16_R11_BC_POLICY_MEMORY_ACTIVATION_EVIDENCE_V1_R0",
        evidence_id="state-sufficiency-history-gap-r0",
        state_sufficiency_report_sha256="a" * 64,
        representation_gap_count=gap_count,
        causal_history_required=causal_history_required,
        rationale_sha256="b" * 64,
    )


def _enabled_config():
    return enabled_policy_memory_config_r0(
        memory_version="TEST_CAUSAL_MEMORY_V1",
        activation_evidence=_evidence(),
    )


def _update(index: int, features=(0.25, -0.5)):
    return CausalMemoryUpdateR0(
        logical_account_id="acct-memory",
        decision_index=index,
        source_policy_observation_sha256=(f"{index + 1:x}" * 64)[:64],
        causal_prefix_sha256=(f"{index + 9:x}" * 64)[:64],
        memory_features=features,
    )


def test_policy_memory_is_disabled_by_default_and_fully_valid() -> None:
    config = disabled_policy_memory_config_r0()
    assert config.mode == POLICY_MEMORY_DEFAULT_MODE_R0 == POLICY_MEMORY_DISABLED
    assert config.activation_evidence is None
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    assert state.last_decision_index == -1
    assert state.memory_features == ()
    assert policy_memory_model_features_r0(config=config, state=state) == ()


def test_disabled_mode_does_not_consume_or_mutate_hidden_state() -> None:
    config = disabled_policy_memory_config_r0()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    advanced = advance_policy_memory_r0(config=config, state=state, update=_update(0, (9.0, 8.0)))
    assert advanced == state
    assert policy_memory_model_features_r0(config=config, state=advanced) == ()


def test_enabled_mode_requires_representation_gap_and_causal_history_evidence() -> None:
    with pytest.raises(RuntimeError, match="REPRESENTATION_GAP_REQUIRED"):
        enabled_policy_memory_config_r0(
            memory_version="MEMORY_V1",
            activation_evidence=_evidence(gap_count=0),
        )
    with pytest.raises(RuntimeError, match="CAUSAL_HISTORY_EVIDENCE_REQUIRED"):
        enabled_policy_memory_config_r0(
            memory_version="MEMORY_V1",
            activation_evidence=_evidence(causal_history_required=False),
        )
    direct = PolicyMemoryConfigR0(
        schema_version="CB16_R11_BC_POLICY_MEMORY_V1_R0",
        mode=POLICY_MEMORY_ENABLED,
        memory_version="MEMORY_V1",
        activation_evidence=None,
    )
    with pytest.raises(RuntimeError, match="ENABLED_WITHOUT_EVIDENCE"):
        direct.validate()


def test_disabled_config_cannot_smuggle_activation_evidence() -> None:
    config = replace(disabled_policy_memory_config_r0(), activation_evidence=_evidence())
    with pytest.raises(RuntimeError, match="DISABLED_HAS_ACTIVATION_EVIDENCE"):
        config.validate()


def test_enabled_memory_advances_only_in_exact_causal_decision_order() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    first = advance_policy_memory_r0(config=config, state=state, update=_update(0, (1.0, 2.0)))
    second = advance_policy_memory_r0(config=config, state=first, update=_update(1, (3.0, 4.0)))
    assert first.last_decision_index == 0
    assert second.last_decision_index == 1
    assert policy_memory_model_features_r0(config=config, state=second) == (3.0, 4.0)

    with pytest.raises(RuntimeError, match="ORDER_VIOLATION"):
        advance_policy_memory_r0(config=config, state=second, update=_update(3))
    with pytest.raises(RuntimeError, match="ORDER_VIOLATION"):
        advance_policy_memory_r0(config=config, state=second, update=_update(1))


def test_memory_state_is_bound_to_one_logical_account_and_config_identity() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    wrong_account_update = replace(_update(0), logical_account_id="other-account")
    with pytest.raises(RuntimeError, match="ACCOUNT_ID_MISMATCH"):
        advance_policy_memory_r0(config=config, state=state, update=wrong_account_update)

    different_config = enabled_policy_memory_config_r0(
        memory_version="DIFFERENT_MEMORY_V2",
        activation_evidence=_evidence(),
    )
    with pytest.raises(RuntimeError, match="CONFIG_IDENTITY_MISMATCH"):
        policy_memory_model_features_r0(config=different_config, state=state)


def test_pause_restart_seal_restore_is_exact_and_continuable() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    state = advance_policy_memory_r0(config=config, state=state, update=_update(0, (1.25, -2.5)))
    state = advance_policy_memory_r0(config=config, state=state, update=_update(1, (3.5, 4.75)))

    restored = restore_policy_memory_state_r0(seal_policy_memory_state_r0(state))
    assert restored == state

    uninterrupted = advance_policy_memory_r0(config=config, state=state, update=_update(2, (7.0, 8.0)))
    resumed = advance_policy_memory_r0(config=config, state=restored, update=_update(2, (7.0, 8.0)))
    assert resumed == uninterrupted


def test_seal_tamper_fails_closed() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    state = advance_policy_memory_r0(config=config, state=state, update=_update(0))
    seal = seal_policy_memory_state_r0(state)
    tampered = dict(seal)
    tampered_state = dict(tampered["state"])
    tampered_state["last_decision_index"] = 99
    tampered["state"] = tampered_state
    with pytest.raises(RuntimeError, match="SEAL_TAMPERED"):
        restore_policy_memory_state_r0(tampered)


def test_nonfinite_memory_features_fail_closed() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    with pytest.raises(RuntimeError, match="FEATURE_NONFINITE"):
        advance_policy_memory_r0(
            config=config,
            state=state,
            update=_update(0, (float("nan"),)),
        )


def test_policy_memory_update_interface_has_no_future_or_teacher_argument() -> None:
    params = tuple(inspect.signature(advance_policy_memory_r0).parameters)
    assert params == ("config", "state", "update")
    update_fields = set(CausalMemoryUpdateR0.__dataclass_fields__)
    forbidden = {
        "future_market",
        "future_return",
        "future_outcome",
        "teacher_target",
        "teacher_value",
        "objective_end_time",
    }
    assert forbidden.isdisjoint(update_fields)


def test_disabled_baseline_integrates_with_actor_as_empty_memory() -> None:
    config = disabled_policy_memory_config_r0()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    observation = build_actor_observation_r0(
        market_sensory_version="MARKET_V1",
        market_causal_features=(0.1, 0.2),
        account_observation=_account_observation(),
        legal_execution_version="LEGAL_V1",
        legal_execution_causal_features=(1.0, 0.5),
        policy_memory_version=None,
        policy_memory_causal_features=policy_memory_model_features_r0(config=config, state=state),
    )
    assert observation.policy_memory_version is None
    assert observation.policy_memory_causal_features == ()


def test_enabled_memory_can_be_wired_only_with_declared_version_and_causal_features() -> None:
    config = _enabled_config()
    state = initial_policy_memory_state_r0(config=config, logical_account_id="acct-memory")
    state = advance_policy_memory_r0(config=config, state=state, update=_update(0, (0.75, -0.25)))
    observation = build_actor_observation_r0(
        market_sensory_version="MARKET_V1",
        market_causal_features=(0.1, 0.2),
        account_observation=_account_observation(),
        legal_execution_version="LEGAL_V1",
        legal_execution_causal_features=(1.0, 0.5),
        policy_memory_version=config.memory_version,
        policy_memory_causal_features=policy_memory_model_features_r0(config=config, state=state),
    )
    assert observation.policy_memory_version == "TEST_CAUSAL_MEMORY_V1"
    assert observation.policy_memory_causal_features == (0.75, -0.25)
