from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.account_economics_r0 import make_account_economics_state_r0
from cb16_local_opt.account_observation_r0 import project_account_observation_r0
from cb16_local_opt.action_contract_r1 import FLAT, LONG, SHORT
from cb16_local_opt.actor_critic_supervisor_r1 import make_supervisor_authority_r1
from cb16_local_opt.execution_feasibility_r0 import MechanicalExecutionAuthorityR0
from cb16_local_opt.execution_observation_r0 import build_execution_observation_r0
from cb16_local_opt.observation_normalization_r0 import (
    ACCOUNT_SCALED_FIELDS_R0,
    EXECUTION_SCALED_FIELDS_R0,
    MONEY_SCALE_SEMANTICS_R0,
    OBSERVATION_NORMALIZER_SPEC_R0,
    make_normalization_context_r0,
    normalize_policy_state_r0,
)
from cb16_local_opt.target_exposure_r1 import make_target_exposure_authority_r1


def _inputs(*, money_unit=1.0, asset_price_unit=1.0, scale_quantity_constraints=True):
    m = float(money_unit)
    p = float(asset_price_unit)
    quantity = 2.0 / p
    price = 12.0 * m * p
    cost_basis = 10.0 * m * p

    account_state = make_account_economics_state_r0(
        account_id="acct",
        cash=80.0 * m,
        position_quantity=quantity,
        position_cost_basis=cost_basis,
        mark_price=price,
        realized_pnl_cumulative=5.0 * m,
        fees_cumulative=2.0 * m,
        funding_cumulative=0.0,
        margin_collateral=20.0 * m,
        liabilities=0.0,
        external_capital_flows_cumulative=100.0 * m,
        economic_responsibility_open=True,
    )
    account = project_account_observation_r0(account_state)
    account_hash = account.source_restore_state_sha256

    qty_scale = p if scale_quantity_constraints else 1.0
    lot_step = 0.1 / qty_scale
    lot_min = 0.1 / qty_scale
    lot_max = 50.0 / qty_scale

    supervisor = make_supervisor_authority_r1(
        authority_id="supervisor",
        account_id="acct",
        account_state_sha256=account_hash,
        terminated=False,
        truncated=False,
        legal_target_directions=(SHORT, FLAT, LONG),
        max_permitted_target_risk=0.8,
    )
    target = make_target_exposure_authority_r1(
        authority_id="target",
        account_id="acct",
        account_state_sha256=account_hash,
        legal_envelope_id="envelope",
        equity=account.equity,
        current_price=price,
        margin_capacity=60.0 * m,
        max_gross_leverage=5.0,
        initial_margin_rate=0.2,
        declared_max_legal_notional=300.0 * m,
        lot_step_size=lot_step,
        lot_min_qty=lot_min,
        lot_max_qty=lot_max,
        min_notional=5.0 * m,
    )
    mechanical = MechanicalExecutionAuthorityR0(
        authority_id="mechanical",
        current_quantity=quantity,
        current_price=price,
        available_margin_for_new_exposure=25.0 * m,
        initial_margin_rate=0.2,
        maintenance_margin_rate=0.1,
        maintenance_collateral=30.0 * m,
        lot_min_qty=lot_min,
        lot_max_qty=lot_max,
        min_notional=5.0 * m,
    )
    execution = build_execution_observation_r0(
        supervisor_authority=supervisor,
        target_exposure_authority=target,
        mechanical_authority=mechanical,
    )
    context = make_normalization_context_r0(
        context_id=f"account-scale-{m:g}",
        money_scale=100.0 * m,
    )
    return account, execution, context


def _normalized(*, money_unit=1.0, asset_price_unit=1.0, scale_quantity_constraints=True):
    account, execution, context = _inputs(
        money_unit=money_unit,
        asset_price_unit=asset_price_unit,
        scale_quantity_constraints=scale_quantity_constraints,
    )
    return normalize_policy_state_r0(
        market_causal_features=(0.25, -0.5, 1.5),
        account_observation=account,
        execution_observation=execution,
        context=context,
    ), context


def _assert_feature_pairs_close(left, right) -> None:
    assert tuple(name for name, _ in left) == tuple(name for name, _ in right)
    for (left_name, left_value), (right_name, right_value) in zip(left, right):
        assert left_name == right_name
        if isinstance(left_value, bool) or left_value is None:
            assert left_value == right_value
        else:
            assert float(left_value) == pytest.approx(float(right_value), rel=0.0, abs=1e-12)


def _assert_model_payload_close(left, right) -> None:
    assert left["market_causal_features"] == pytest.approx(right["market_causal_features"], rel=0.0, abs=1e-12)
    _assert_feature_pairs_close(left["account_scaled_features"], right["account_scaled_features"])
    _assert_feature_pairs_close(left["execution_scaled_features"], right["execution_scaled_features"])


def test_normalizer_identity_is_versioned_stable_and_not_reward_eref() -> None:
    spec = OBSERVATION_NORMALIZER_SPEC_R0
    spec.validate()
    assert spec.money_scale_semantics == MONEY_SCALE_SEMANTICS_R0
    assert "E_REF" in MONEY_SCALE_SEMANTICS_R0
    assert spec.semantic_sha256 == OBSERVATION_NORMALIZER_SPEC_R0.semantic_sha256
    with pytest.raises(RuntimeError, match="VERSION_MISMATCH"):
        replace(spec, normalizer_version="OTHER").validate()


def test_normalized_payload_has_exact_named_field_order_and_excludes_identity_metadata() -> None:
    normalized, _ = _normalized()
    payload = normalized.model_payload()
    assert tuple(name for name, _ in payload["account_scaled_features"]) == ACCOUNT_SCALED_FIELDS_R0
    assert tuple(name for name, _ in payload["execution_scaled_features"]) == EXECUTION_SCALED_FIELDS_R0
    assert "normalizer_sha256" not in payload
    assert "normalization_context_sha256" not in payload
    assert all("asset" not in name and "symbol" not in name for name in payload)


def test_currency_unit_rescaling_with_matching_context_is_model_equivalent() -> None:
    base, base_context = _normalized(money_unit=1.0)
    scaled, scaled_context = _normalized(money_unit=10.0)
    _assert_model_payload_close(base.model_payload(), scaled.model_payload())
    assert base.normalizer_sha256 == scaled.normalizer_sha256
    assert base_context.context_sha256 != scaled_context.context_sha256
    assert base.normalization_context_sha256 != scaled.normalization_context_sha256


def test_asset_price_quantity_redenomination_is_equivalent_when_mechanics_scale() -> None:
    base, _ = _normalized(asset_price_unit=1.0, scale_quantity_constraints=True)
    redenominated, _ = _normalized(asset_price_unit=2.0, scale_quantity_constraints=True)
    _assert_model_payload_close(base.model_payload(), redenominated.model_payload())


def test_non_equivariant_exchange_quantity_constraint_remains_observable() -> None:
    base, _ = _normalized(asset_price_unit=1.0, scale_quantity_constraints=True)
    changed_mechanics, _ = _normalized(asset_price_unit=2.0, scale_quantity_constraints=False)
    base_execution = dict(base.execution_scaled_features)
    changed_execution = dict(changed_mechanics.execution_scaled_features)
    assert changed_execution["lot_min_notional_scale"] != pytest.approx(base_execution["lot_min_notional_scale"])
    assert changed_mechanics.model_payload() != base.model_payload()


def test_dimensionless_risk_leverage_and_margin_rates_are_preserved() -> None:
    normalized, _ = _normalized()
    execution = dict(normalized.execution_scaled_features)
    assert execution["max_permitted_target_risk"] == 0.8
    assert execution["max_gross_leverage"] == 5.0
    assert execution["initial_margin_rate"] == 0.2
    assert execution["maintenance_margin_rate"] == 0.1


def test_absolute_price_and_quantity_identity_are_not_direct_model_features() -> None:
    normalized, _ = _normalized()
    account = dict(normalized.account_scaled_features)
    execution = dict(normalized.execution_scaled_features)
    assert "position_quantity" not in account
    assert "position_cost_basis" not in account
    assert "current_price" not in execution
    assert "lot_min_qty" not in execution
    assert "lot_max_qty" not in execution
    assert "position_notional_scale" in account
    assert "cost_basis_price_ratio" in account
    assert "lot_min_notional_scale" in execution


def test_position_snapshot_mismatch_fails_closed() -> None:
    account, execution, context = _inputs()
    mismatched = replace(execution, current_quantity=execution.current_quantity + 1.0)
    mismatched.validate()
    with pytest.raises(RuntimeError, match="POSITION_AUTHORITY_MISMATCH"):
        normalize_policy_state_r0(
            market_causal_features=(1.0,),
            account_observation=account,
            execution_observation=mismatched,
            context=context,
        )


def test_normalization_context_requires_positive_finite_scale() -> None:
    for bad in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(RuntimeError):
            make_normalization_context_r0(context_id="bad", money_scale=bad)  # type: ignore[arg-type]
