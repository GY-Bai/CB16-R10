from __future__ import annotations

import json
from pathlib import Path

import pytest

from cb16_local_opt.actor_critic_environment_profile_r1 import (
    ActorCriticEnvironmentProfileR1,
    ENVIRONMENT_PROFILE_VERSION_R1,
    policy_neutral_environment_profile_r1,
)


ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_PATH = ROOT / "authority/rearchitecture_r11/CB16_R11_BC_EXECUTION_RULE_TAXONOMY_V1.json"


def _taxonomy() -> dict:
    return json.loads(TAXONOMY_PATH.read_text())


def test_default_round2_profile_is_policy_neutral() -> None:
    profile = policy_neutral_environment_profile_r1()
    profile.validate(_taxonomy())
    assert profile.profile_version == ENVIRONMENT_PROFILE_VERSION_R1
    for rule_id in ("stop_loss", "take_profit", "max_hold", "cooldown", "finalize_behavior"):
        assert profile.is_enabled(rule_id) is False
    for rule_id in ("fee", "slippage", "funding", "margin", "liquidation", "leverage_cap", "risk_limits"):
        assert profile.is_enabled(rule_id) is True


def test_strategy_preference_cannot_be_silently_enabled() -> None:
    base = policy_neutral_environment_profile_r1()
    enabled = tuple(sorted(set(base.enabled_rules) | {"stop_loss"}))
    disabled = tuple(sorted(set(base.disabled_rules) - {"stop_loss"}))
    bad = ActorCriticEnvironmentProfileR1(
        profile_version=base.profile_version,
        taxonomy_schema=base.taxonomy_schema,
        enabled_rules=enabled,
        disabled_rules=disabled,
        explicit_strategy_authorizations=(),
    )
    with pytest.raises(RuntimeError, match="STRATEGY_PREFERENCE_SILENTLY_ENABLED"):
        bad.validate(_taxonomy())


def test_historical_finalize_can_never_be_round2_execution_authority() -> None:
    base = policy_neutral_environment_profile_r1()
    enabled = tuple(sorted(set(base.enabled_rules) | {"finalize_behavior"}))
    disabled = tuple(sorted(set(base.disabled_rules) - {"finalize_behavior"}))
    bad = ActorCriticEnvironmentProfileR1(
        profile_version=base.profile_version,
        taxonomy_schema=base.taxonomy_schema,
        enabled_rules=enabled,
        disabled_rules=disabled,
        explicit_strategy_authorizations=(),
    )
    with pytest.raises(RuntimeError, match="HISTORICAL_ONLY_RULE_ENABLED"):
        bad.validate(_taxonomy())


def test_mechanical_rule_cannot_disappear_from_profile() -> None:
    base = policy_neutral_environment_profile_r1()
    enabled = tuple(rule for rule in base.enabled_rules if rule != "fee")
    disabled = tuple(sorted(set(base.disabled_rules) | {"fee"}))
    bad = ActorCriticEnvironmentProfileR1(
        profile_version=base.profile_version,
        taxonomy_schema=base.taxonomy_schema,
        enabled_rules=enabled,
        disabled_rules=disabled,
        explicit_strategy_authorizations=(),
    )
    with pytest.raises(RuntimeError, match="MECHANICAL_RULE_DISABLED"):
        bad.validate(_taxonomy())


def test_profile_must_cover_taxonomy_exactly_once() -> None:
    base = policy_neutral_environment_profile_r1()
    bad = ActorCriticEnvironmentProfileR1(
        profile_version=base.profile_version,
        taxonomy_schema=base.taxonomy_schema,
        enabled_rules=base.enabled_rules,
        disabled_rules=tuple(rule for rule in base.disabled_rules if rule != "cooldown"),
        explicit_strategy_authorizations=(),
    )
    with pytest.raises(RuntimeError, match="RULE_COVERAGE_INCOMPLETE"):
        bad.validate(_taxonomy())
