"""Bounded synthetic S0-v2 foundation fixtures.

These production fixtures construct the canonical CC account/runtime path for
the durable qualification runner.  They are synthetic-only and never access
FINAL or fresh market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from cb16_local_opt.account_economics_r0 import AccountEconomicsStateR0, make_account_economics_state_r0
from cb16_local_opt.actor_critic_physics_adapter_r1 import Round2MechanicalExecutionConfigR1
from cb16_local_opt.cc_clock_r0 import CCFourClockR0
from cb16_local_opt.cc_environment_advance_r0 import CCEnvironmentIntervalR0
from cb16_local_opt.cc_policy_brain_r0 import BrainBindings, CCCentralBrain
from cb16_local_opt.cc_policy_distribution_r0 import NominalAction, sample_nominal
from cb16_local_opt.cc_policy_reward_r0 import arithmetic_equity_reward
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
from cb16_local_opt.cc_runtime_account_loop_r0 import (
    CCContinuousAccountRuntimeR0,
    execute_via_frozen_r1_r0,
)
from cb16_local_opt.cc_runtime_boundary_r0 import OBJECTIVE_HORIZON_REACHED
from cb16_local_opt.cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from cb16_local_opt.post_cc_durable_collection_v1 import (
    DurableObservationCollectorV1,
    canonical_brain_vectors_from_account_v1,
    environment_time_text_v1,
)
from cb16_local_opt.post_cc_joint_batch_v1 import (
    MATERIALIZER_CONTRACT_ID,
    DurableBatchProvenanceV1,
    JointActionBatchV1,
    build_joint_batch_v1,
)
from cb16_local_opt.post_cc_joint_replay_v1 import DurableJointReplaySampleV1
from cb16_local_opt.post_cc_observation_contract_v1 import PostCCObservationFactV1
from cb16_local_opt.post_cc_observation_fact_v1 import (
    CANONICAL_NORMALIZER_IDENTITY_V1,
    build_canonical_observation_fact_v1,
)
from cb16_local_opt.post_cc_observation_store_v1 import ObservationStoreV1
from cb16_local_opt.post_cc_replay_materializer_v1 import MaterializedReplayV1, ReplayMaterializerV1

SCIENCE_SEMANTIC_VERSION = "CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1"
DEMO_LINEAGE_ID = "cc-s0v2-account-0001"
DEMO_POLICY_GENERATION = "0"
DEMO_POLICY_ID = "cc-s0v2-policy-g0"
DEMO_POLICY_SHA256 = "0" * 63 + "1"
DEMO_REWARD_REFERENCE_EQUITY = 1000.0
DEMO_PRICES = (102.0, 98.0, 105.0, 97.0, 110.0, 90.0)
DEMO_MARKET_SOURCE_IDENTITY = "CC_INTEGRATION_SYNTHETIC_MARKET_V1"
DEMO_MARKET_SOURCE_VERSION = "V1"


def make_bindings() -> BrainBindings:
    return BrainBindings("1" * 64, "2" * 64, "3" * 64, "4" * 64)


def make_brain(*, seed: int = 1701, direction_bias: Sequence[float] = (-0.25, -1.0, 0.35)) -> CCCentralBrain:
    torch.manual_seed(seed)
    brain = CCCentralBrain(2, 3, 2, 8, make_bindings())
    brain.assert_gradient_ownership()
    with torch.no_grad():
        brain.direction_head.bias.copy_(torch.tensor(list(direction_bias), dtype=brain.direction_head.bias.dtype))
        brain.risk_loc_head.bias.copy_(torch.tensor([-0.4, 0.2, 0.4], dtype=brain.risk_loc_head.bias.dtype))
    return brain


def make_account(
    *,
    account_id: str = "cc-s0v2-ledger-0001",
    cash: float = 1000.0,
    position_quantity: float = 0.0,
    position_cost_basis: float = 0.0,
    mark_price: float = 100.0,
    economic_responsibility_open: bool = True,
) -> AccountEconomicsStateR0:
    account = make_account_economics_state_r0(
        account_id=account_id,
        cash=cash,
        position_quantity=position_quantity,
        position_cost_basis=position_cost_basis,
        mark_price=mark_price,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=economic_responsibility_open,
    )
    account.validate()
    return account


def make_executor():
    config = Round2MechanicalExecutionConfigR1(
        fee_rate=0.0005,
        slippage_bps=1.0,
        initial_margin_rate=0.1,
        maintenance_margin_rate=0.05,
        max_gross_leverage=2.0,
    )
    return partial(execute_via_frozen_r1_r0, config=config)


def make_runtime(
    *,
    account: AccountEconomicsStateR0,
    account_lineage_id: str = DEMO_LINEAGE_ID,
    policy_generation: str = DEMO_POLICY_GENERATION,
    policy_id: str = DEMO_POLICY_ID,
    policy_sha256: str = DEMO_POLICY_SHA256,
) -> CCContinuousAccountRuntimeR0:
    return CCContinuousAccountRuntimeR0(
        account_lineage_id=account_lineage_id,
        account=account,
        clocks=CCFourClockR0(0, 0, 0, "CC_S0V2_SYNTHETIC_HORIZON"),
        schedule=CCDecisionScheduleR0(1),
        executor=make_executor(),
        policy_generation=policy_generation,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
    )


def make_policy_rng(*, policy_id: str = DEMO_POLICY_ID, account_lineage_id: str = DEMO_LINEAGE_ID, seed: int = 20260912) -> PolicyRNG:
    return PolicyRNG(policy_id, account_lineage_id, seed)


def make_interval(mark_price_after: float, *, boundary_type: str = "CONTINUE") -> CCEnvironmentIntervalR0:
    interval = CCEnvironmentIntervalR0(mark_price_after, boundary_type=boundary_type)
    interval.validate()
    return interval


@dataclass
class CollectedSequenceV1:
    root: Path
    sequence_id: str
    transition_ids: tuple[str, ...]
    decisions: tuple[Any, ...]
    transition_records: tuple[Any, ...]
    account_after_collection: AccountEconomicsStateR0
    account_lineage_id: str
    policy_generation: str
    policy_id: str
    policy_sha256: str
    behavior_brain: CCCentralBrain
    final_boundary_type: str
    runtime: CCContinuousAccountRuntimeR0


def collect_durable_sequence_v1(
    root: str | Path | None = None,
    *,
    sequence_id: str = "cc-s0v2-sequence-0001",
    step_count: int = 6,
    final_boundary_type: str = OBJECTIVE_HORIZON_REACHED,
    account: AccountEconomicsStateR0 | None = None,
    brain: CCCentralBrain | None = None,
    policy_rng: PolicyRNG | None = None,
    account_lineage_id: str = DEMO_LINEAGE_ID,
    policy_generation: str = DEMO_POLICY_GENERATION,
    policy_id: str = DEMO_POLICY_ID,
    policy_sha256: str = DEMO_POLICY_SHA256,
) -> CollectedSequenceV1:
    """Collect one bounded durable decision sequence with no live-only truth."""
    if root is None:
        root = tempfile.mkdtemp(prefix="cc-s0v2-collection-")
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    if account is None:
        account = make_account()
    if brain is None:
        brain = make_brain()
    if policy_rng is None:
        policy_rng = make_policy_rng(policy_id=policy_id, account_lineage_id=account_lineage_id)
    runtime = make_runtime(
        account=account,
        account_lineage_id=account_lineage_id,
        policy_generation=policy_generation,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
    )
    collector = DurableObservationCollectorV1(
        str(root_path),
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
        market_source_version=DEMO_MARKET_SOURCE_VERSION,
    )
    decisions: list[Any] = []
    transition_records: list[Any] = []

    def callback(account_state, clocks):
        market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
        logits, risk_loc, risk_log_scale = brain(
            torch.tensor(market, dtype=torch.float32),
            torch.tensor(account_values, dtype=torch.float32),
            torch.tensor(execution_values, dtype=torch.float32),
        )
        nominal = sample_nominal(logits, risk_loc, risk_log_scale, policy_rng)
        decision, _fact = collector.capture_policy_decision(
            account_state,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=policy_generation,
            policy_id=policy_id,
            policy_sha256=policy_sha256,
            nominal=nominal,
        )
        decisions.append(decision)
        return decision

    prices = DEMO_PRICES[: int(step_count)]
    for index, price in enumerate(prices):
        boundary = final_boundary_type if index == len(prices) - 1 else "CONTINUE"
        equity_before = runtime.account.equity
        transition = runtime.step(
            make_interval(price, boundary_type=boundary),
            callback,
            expected_predecessor_token=runtime.predecessor_token,
        )
        equity_after = runtime.account.equity
        record = collector.persist_transition(
            sequence_id=sequence_id,
            transition=transition,
            decision=decisions[-1],
            reward=arithmetic_equity_reward(equity_before, equity_after, DEMO_REWARD_REFERENCE_EQUITY),
            discount=0.99,
        )
        transition_records.append(record)
    collector.finalize_sequence(
        sequence_id=sequence_id,
        market_lineage_id=DEMO_MARKET_SOURCE_IDENTITY,
        source_classification="CC_STOCHASTIC_TRAJECTORY",
        transition_ids=tuple(record.transition_id for record in transition_records),
        chunk_boundary_type=final_boundary_type,
        bootstrap_state_ref_or_null=None,
    )
    collector.close()
    return CollectedSequenceV1(
        root=root_path,
        sequence_id=sequence_id,
        transition_ids=tuple(record.transition_id for record in transition_records),
        decisions=tuple(decisions),
        transition_records=tuple(transition_records),
        account_after_collection=runtime.account,
        account_lineage_id=account_lineage_id,
        policy_generation=policy_generation,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        behavior_brain=brain,
        final_boundary_type=final_boundary_type,
        runtime=runtime,
    )


def materialize_v1(
    root: str | Path,
    sequence_id: str,
    *,
    target_policy_identity: str = "cc-s0v2-target-policy-g0",
    restart_verified: bool = True,
    expected_materialization_id: str | None = None,
) -> MaterializedReplayV1:
    materializer = ReplayMaterializerV1.from_durable_state_v1(root)
    return materializer.materialize_sequence(
        sequence_id,
        target_policy_identity=target_policy_identity,
        restart_verified=restart_verified,
        expected_materialization_id=expected_materialization_id,
    )


def make_canonical_observation_v1(
    *,
    account_lineage_id: str = DEMO_LINEAGE_ID,
    decision_index: int = 0,
    environment_time: int = 0,
    market_values: Sequence[float] = (0.02, 1.0),
    account_values: Sequence[float] = (1.0, 0.0, 0.0),
    execution_values: Sequence[float] = (0.0, 0.0),
) -> PostCCObservationFactV1:
    time_text = environment_time_text_v1(environment_time)
    return build_canonical_observation_fact_v1(
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_values=tuple(float(value) for value in market_values),
        account_values=tuple(float(value) for value in account_values),
        execution_values=tuple(float(value) for value in execution_values),
        market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
        market_source_version=DEMO_MARKET_SOURCE_VERSION,
        market_visible_through_time=time_text,
        account_lineage_id=account_lineage_id,
        decision_index=int(decision_index),
        environment_time=time_text,
        normalizer_identity=CANONICAL_NORMALIZER_IDENTITY_V1,
    )


def make_joint_sample_v1(
    *,
    root: str | Path,
    sequence_id: str = "cc-s0v2-unit-sequence",
    transition_id: str = "cc-s0v2-unit-transition-0",
    account_lineage_id: str = DEMO_LINEAGE_ID,
    decision_index: int = 0,
    environment_time: int = 0,
    nominal_direction: str = "LONG",
    nominal_target_risk: float = 0.4,
    behavior_log_mu: float = -1.25,
    reward: float = 0.01,
    discount: float = 0.99,
    boundary_type: str = OBJECTIVE_HORIZON_REACHED,
    bootstrap_state_ref_or_null: str | None = None,
    sampling_probability_or_weight: float = 1.0,
    target_policy_identity: str = "cc-s0v2-target-policy-g0",
    consequence_context: Mapping[str, Any] | None = None,
) -> DurableJointReplaySampleV1:
    store = ObservationStoreV1(root)
    fact = make_canonical_observation_v1(
        account_lineage_id=account_lineage_id,
        decision_index=decision_index,
        environment_time=environment_time,
    )
    receipt = store.put(fact)
    return DurableJointReplaySampleV1(
        sample_id="a" * 64,
        sequence_id=sequence_id,
        transition_id=transition_id,
        account_lineage_id=account_lineage_id,
        decision_index=int(decision_index),
        environment_time=environment_time_text_v1(environment_time),
        observation=fact,
        observation_logical_id=receipt.logical_id,
        observation_content_sha256=receipt.content_sha256,
        transition_record_sha256="b" * 64,
        policy_decision_ref="c" * 64,
        nominal_direction=nominal_direction,
        nominal_target_risk=float(nominal_target_risk),
        risk_measure_kind="point_mass" if nominal_direction == "FLAT" else "continuous_density",
        behavior_log_mu=float(behavior_log_mu),
        behavior_policy_generation="0",
        behavior_policy_id=DEMO_POLICY_ID,
        behavior_policy_sha256=DEMO_POLICY_SHA256,
        log_mu_source="DECISION_TIME_PERSISTED",
        reward=float(reward),
        discount=float(discount),
        boundary_type=boundary_type,
        bootstrap_state_ref_or_null=bootstrap_state_ref_or_null,
        sampling_probability_or_weight=float(sampling_probability_or_weight),
        target_policy_identity=target_policy_identity,
        source_fact_hashes=("b" * 64, receipt.content_sha256, "c" * 64, DEMO_POLICY_SHA256),
        consequence_context=consequence_context,
    ).validate()


def make_batch_v1(
    samples: Sequence[DurableJointReplaySampleV1],
    *,
    bootstrap_observations_by_sequence: Mapping[str, PostCCObservationFactV1] | None = None,
    target_policy_identity: str = "cc-s0v2-target-policy-g0",
    restart_verified: bool = True,
    materialization_id: str = "d" * 64,
    manifest_sha256: str = "e" * 64,
) -> JointActionBatchV1:
    provenance = DurableBatchProvenanceV1(
        materializer_contract_id=MATERIALIZER_CONTRACT_ID,
        materialization_id=materialization_id,
        materialization_manifest_sha256=manifest_sha256,
        source_store_root_identity="f" * 64,
        restart_verified=restart_verified,
        durable_replay_only=True,
        collector_private_records_used=False,
    )
    return build_joint_batch_v1(
        samples=tuple(samples),
        provenance=provenance,
        target_policy_identity=target_policy_identity,
        bootstrap_observations_by_sequence=bootstrap_observations_by_sequence,
    )
