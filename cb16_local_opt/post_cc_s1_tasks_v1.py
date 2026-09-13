"""CB16 R11 post-CC S1 R1 task environments, oracles and controls.

This module owns the five frozen S1 known-answer synthetic environments, their
finite oracle comparators, the negative-control transforms and the frozen
policy-evaluation population.  It is synthetic-only: no FINAL, no fresh data
and no historical market corpus.

The module deliberately keeps every scientific constant explicit and named so
that ``post_cc_s1_execution_manifest_v1`` can hash the same values that the
training/qualification runtime consumes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence

import torch

from .account_economics_r0 import AccountEconomicsStateR0, make_account_economics_state_r0
from .actor_critic_physics_adapter_r1 import Round2MechanicalExecutionConfigR1
from .cc_clock_r0 import CCFourClockR0
from .cc_environment_advance_r0 import CCEnvironmentIntervalR0, advance_environment_r0
from .cc_policy_brain_r0 import BrainBindings, CCCentralBrain
from .cc_policy_distribution_r0 import (
    DIRECTION_ORDER,
    INDEX as DIRECTION_INDEX,
    NominalAction,
    joint_log_prob,
    sample_nominal,
)
from .cc_policy_rng_r0 import PolicyRNG
from .cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0, execute_via_frozen_r1_r0
from .cc_runtime_boundary_r0 import COMPUTE_CHUNK, CONTINUE, OBJECTIVE_HORIZON_REACHED
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .cc_runtime_wire_r0 import CCPolicyDecisionV1
from .post_cc_durable_collection_v1 import (
    DurableObservationCollectorV1,
    canonical_brain_vectors_from_account_v1,
    environment_time_text_v1,
)
from .post_cc_observation_fact_v1 import CANONICAL_NORMALIZER_IDENTITY_V1, build_canonical_observation_fact_v1
from .post_cc_observation_store_v1 import ObservationStoreV1

SCIENCE_SEMANTIC_VERSION = "CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1"
S1_MARKET_SOURCE_IDENTITY = "CC_S1_SYNTHETIC_MARKET_V1"
S1_MARKET_SOURCE_VERSION = "V1"

STRICT_TOLERANCE_V1 = 1e-9
ANALYTIC_VTRACE_TOLERANCE_V1 = 1e-6
GAP_REDUCTION_THRESHOLD_V1 = 0.50
MINIMUM_PASSING_SEEDS_V1 = 4
MAXIMUM_FALSE_POSITIVE_CONTROL_SEEDS_V1 = 1
FROZEN_SEEDS_V1 = (1701, 1702, 1703, 1704, 1705)
MODEL_INITIALIZATION_SEED_V1 = 1701
ACTOR_LEARNING_RATE_V1 = 0.01
CRITIC_LEARNING_RATE_V1 = 0.01
MAXIMUM_POLICY_DECISIONS_V1 = 100000
QUALIFICATION_POLICY_DECISIONS_V1 = 16384
QUALIFICATION_COLLECTION_UNIT_V1 = 128
QUALIFICATION_REPLAY_BATCH_V1 = 128
QUALIFICATION_EVALUATION_ACTIONS_V1 = 2048
ABA_PHASE_DECISIONS_V1 = (("A", 4096), ("B", 4096), ("A", 4096))
DEFAULT_ARITHMETIC_DISCOUNT_V1 = 1.0
REWARD_REFERENCE_EQUITY_V1 = 1000.0
GENESIS_CAPITAL_V1 = 1000.0
GENESIS_MARK_V1 = 100.0

ACTION_RISK_VALUES_V1 = (0.1, 0.3, 0.5, 0.7, 0.9)
ORACLE_DIRECTIONS_V1 = ("FLAT", "LONG", "SHORT")

S1_BRAIN_BINDINGS = BrainBindings("a" * 64, "b" * 64, "c" * 64, "d" * 64)

TASK_ACCOUNT_DEPENDENT_ACTION = "ACCOUNT_DEPENDENT_ACTION"
TASK_DELAYED_CONSEQUENCE_CREDIT = "DELAYED_CONSEQUENCE_CREDIT"
TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION = "HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION"
TASK_OFF_POLICY_VTRACE_CORRECTION = "OFF_POLICY_VTRACE_CORRECTION"
TASK_ABA_RETENTION = "A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION"

FROZEN_TASK_IDS_V1 = (
    TASK_ACCOUNT_DEPENDENT_ACTION,
    TASK_DELAYED_CONSEQUENCE_CREDIT,
    TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION,
    TASK_OFF_POLICY_VTRACE_CORRECTION,
    TASK_ABA_RETENTION,
)

CONTROL_NO_SIGNAL = "NO_SIGNAL_OR_ZERO_REWARD_CONTROL"
CONTROL_SHUFFLED_CREDIT = "SHUFFLED_CREDIT_CONTROL"
CONTROL_RANDOM_IMPOSSIBLE = "RANDOM_OR_IMPOSSIBLE_LABEL_CONTROL"
CONTROL_ACCOUNT_ABLATION = "ACCOUNT_INPUT_ABLATION"
CONTROL_OBJECTIVE_FIREWALL = "OBJECTIVE_FIREWALL"
CONTROL_FABRICATED_LOG_MU_REJECTION = "FABRICATED_LOG_MU_REJECTION"

BEHAVIOR_MODE_CHILD_SWITCH_V1 = "CHILD_SWITCH_AFTER_COMMITTED_UPDATE"
BEHAVIOR_MODE_FIXED_DISTINCT_V1 = "FIXED_DISTINCT_BEHAVIOR_CHECKPOINT"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_sha256_v1(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def derived_stream_seed_v1(*parts: object) -> int:
    """Stable 63-bit seed material for one dedicated RNG stream."""
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True)
class BranchOutcomeV1:
    """One exact terminal environment branch."""

    probability: float
    terminal_mark: float


@dataclass(frozen=True)
class EpisodeDynamicsV1:
    """Declared intra-episode market dynamics for one context."""

    decision_mark: float
    stage1_mark: float
    stage2_branches: tuple[BranchOutcomeV1, ...]
    stage2_is_no_decision_advance: bool


@dataclass(frozen=True)
class ContextSpecV1:
    context_id: str
    account_kind: str
    setup_direction: str | None
    setup_risk: float | None
    oracle_direction_expected: str
    dynamics: EpisodeDynamicsV1


@dataclass(frozen=True)
class TaskSpecV1:
    task_id: str
    environment_version: str
    observation_semantics: str
    action_semantics: str
    reward_semantics: str
    horizon_semantics: str
    known_answer: str
    contexts: tuple[ContextSpecV1, ...]
    context_schedule: str
    phase_context_ids: tuple[tuple[str, int], ...]
    training_decisions: int
    gap_reduction_threshold: float
    behavior_mode: str
    credit_stage2_no_decision: bool
    fee_rate: float
    slippage_bps: float
    initial_margin_rate: float
    maintenance_margin_rate: float
    max_gross_leverage: float
    reward_reference_equity: float
    declared_controls: tuple[str, ...]
    higher_ev_family_directions: tuple[str, ...]
    higher_ev_family_risk_low: float
    higher_ev_family_risk_high: float
    handcrafted_regime_activation: bool = False
    objective_orientation: str = "COMPLETE_SAMPLE_ARITHMETIC_EQUITY_DELTA"

    def payload(self) -> Mapping[str, Any]:
        body = asdict(self)
        return {
            "schema": "CB16_R11_POST_CC_S1_TASK_SPEC_V1",
            **body,
        }

    @property
    def spec_hash(self) -> str:
        return stable_sha256_v1(self.payload())


def _branch(probability: float, terminal_mark: float) -> BranchOutcomeV1:
    return BranchOutcomeV1(probability=float(probability), terminal_mark=float(terminal_mark))


def _context(
    *,
    context_id: str,
    account_kind: str,
    oracle_direction: str,
    dynamics: EpisodeDynamicsV1,
    setup_direction: str | None = None,
    setup_risk: float | None = None,
) -> ContextSpecV1:
    return ContextSpecV1(
        context_id=context_id,
        account_kind=account_kind,
        setup_direction=setup_direction,
        setup_risk=setup_risk,
        oracle_direction_expected=oracle_direction,
        dynamics=dynamics,
    )


def build_task_specs_v1() -> dict[str, TaskSpecV1]:
    """Build the five frozen positive task specs (no result-dependent values)."""

    account_dependent = TaskSpecV1(
        task_id=TASK_ACCOUNT_DEPENDENT_ACTION,
        environment_version="POST_CC_S1_ACCOUNT_DEPENDENT_ACTION_V1",
        observation_semantics="Same market observation paired with distinct reachable account states requiring distinct optimal joint actions.",
        action_semantics="Canonical direction plus conditional target risk.",
        reward_semantics="Finite-horizon arithmetic equity delta under declared initial-capital denominator.",
        horizon_semantics="One scored policy decision followed by a flat-price terminal mark; no cross-episode account stitching.",
        known_answer="Optimal action depends on Account input even when Market is identical.",
        contexts=(
            _context(
                context_id="HELD_LONG",
                account_kind="HELD_LONG_SETUP",
                setup_direction="LONG",
                setup_risk=0.5,
                oracle_direction="LONG",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=100.0,
                    stage1_mark=100.0,
                    stage2_branches=(_branch(1.0, 100.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
            _context(
                context_id="HELD_SHORT",
                account_kind="HELD_SHORT_SETUP",
                setup_direction="SHORT",
                setup_risk=0.5,
                oracle_direction="SHORT",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=100.0,
                    stage1_mark=100.0,
                    stage2_branches=(_branch(1.0, 100.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
        ),
        context_schedule="UNIFORM_CONTEXT_RNG",
        phase_context_ids=(),
        training_decisions=QUALIFICATION_POLICY_DECISIONS_V1,
        gap_reduction_threshold=GAP_REDUCTION_THRESHOLD_V1,
        behavior_mode=BEHAVIOR_MODE_CHILD_SWITCH_V1,
        credit_stage2_no_decision=False,
        fee_rate=0.001,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.25,
        max_gross_leverage=2.0,
        reward_reference_equity=REWARD_REFERENCE_EQUITY_V1,
        declared_controls=(CONTROL_ACCOUNT_ABLATION, CONTROL_NO_SIGNAL),
        higher_ev_family_directions=(),
        higher_ev_family_risk_low=0.0,
        higher_ev_family_risk_high=0.0,
    )

    delayed = TaskSpecV1(
        task_id=TASK_DELAYED_CONSEQUENCE_CREDIT,
        environment_version="POST_CC_S1_DELAYED_CONSEQUENCE_CREDIT_V1",
        observation_semantics="Early decision has no direct label; the consequence arrives after a later no-decision account transition and a durable chunk boundary.",
        action_semantics="Canonical joint action.",
        reward_semantics="Arithmetic account consequence retained on the same logical account lineage.",
        horizon_semantics="One early decision, a compute chunk boundary, a later no-decision advance and terminal liquidation.",
        known_answer="The early action maximizing complete-horizon arithmetic return.",
        contexts=(
            _context(
                context_id="UP_SIGNAL",
                account_kind="GENESIS",
                oracle_direction="LONG",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=101.0,
                    stage1_mark=101.0,
                    stage2_branches=(_branch(1.0, 111.0),),
                    stage2_is_no_decision_advance=True,
                ),
            ),
            _context(
                context_id="DOWN_SIGNAL",
                account_kind="GENESIS",
                oracle_direction="SHORT",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=99.0,
                    stage1_mark=99.0,
                    stage2_branches=(_branch(1.0, 89.0),),
                    stage2_is_no_decision_advance=True,
                ),
            ),
        ),
        context_schedule="UNIFORM_CONTEXT_RNG",
        phase_context_ids=(),
        training_decisions=QUALIFICATION_POLICY_DECISIONS_V1,
        gap_reduction_threshold=GAP_REDUCTION_THRESHOLD_V1,
        behavior_mode=BEHAVIOR_MODE_CHILD_SWITCH_V1,
        credit_stage2_no_decision=True,
        fee_rate=0.0,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.25,
        max_gross_leverage=2.0,
        reward_reference_equity=REWARD_REFERENCE_EQUITY_V1,
        declared_controls=(CONTROL_SHUFFLED_CREDIT, CONTROL_NO_SIGNAL),
        higher_ev_family_directions=(),
        higher_ev_family_risk_low=0.0,
        higher_ev_family_risk_high=0.0,
    )

    high_bankruptcy = TaskSpecV1(
        task_id=TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION,
        environment_version="POST_CC_S1_HIGH_BANKRUPTCY_HIGHER_EXPECTATION_V1",
        observation_semantics="Synthetic outcomes expose no privileged winner label to the learner.",
        action_semantics="At least two reachable canonical actions differ in full-sample arithmetic expectation and bankruptcy frequency.",
        reward_semantics="Complete-sample arithmetic expected return; all failures remain in the denominator.",
        horizon_semantics="One scored decision, two exact terminal branches and mechanical liquidation.",
        known_answer="The higher complete-sample arithmetic expectation action wins even with higher bankruptcy frequency.",
        contexts=(
            _context(
                context_id="TWO_BRANCH_TERMINAL",
                account_kind="GENESIS",
                oracle_direction="LONG",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=100.0,
                    stage1_mark=100.0,
                    stage2_branches=(_branch(0.80, 120.0), _branch(0.20, 40.0)),
                    stage2_is_no_decision_advance=False,
                ),
            ),
        ),
        context_schedule="SINGLE_CONTEXT",
        phase_context_ids=(),
        training_decisions=QUALIFICATION_POLICY_DECISIONS_V1,
        gap_reduction_threshold=GAP_REDUCTION_THRESHOLD_V1,
        behavior_mode=BEHAVIOR_MODE_CHILD_SWITCH_V1,
        credit_stage2_no_decision=False,
        fee_rate=0.0,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.25,
        max_gross_leverage=2.0,
        reward_reference_equity=REWARD_REFERENCE_EQUITY_V1,
        declared_controls=(CONTROL_OBJECTIVE_FIREWALL, CONTROL_RANDOM_IMPOSSIBLE),
        higher_ev_family_directions=("LONG",),
        higher_ev_family_risk_low=0.85,
        higher_ev_family_risk_high=0.95,
    )

    off_policy = TaskSpecV1(
        task_id=TASK_OFF_POLICY_VTRACE_CORRECTION,
        environment_version="POST_CC_S1_OFF_POLICY_VTRACE_V1",
        observation_semantics="Behavior-policy trajectories are replayed under a deliberately different target policy.",
        action_semantics="Persisted nominal joint behavior action; executed action appears only as consequence context.",
        reward_semantics="Task-specific arithmetic reward frozen before collection.",
        horizon_semantics="One decision with explicit terminal boundary and persisted true behavior log_mu.",
        known_answer="Known target/behavior log-probability ratios plus learnable policy improvement.",
        contexts=(
            _context(
                context_id="OFF_POLICY_UP",
                account_kind="GENESIS",
                oracle_direction="LONG",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=102.0,
                    stage1_mark=102.0,
                    stage2_branches=(_branch(1.0, 112.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
            _context(
                context_id="OFF_POLICY_DOWN",
                account_kind="GENESIS",
                oracle_direction="SHORT",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=98.0,
                    stage1_mark=98.0,
                    stage2_branches=(_branch(1.0, 88.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
        ),
        context_schedule="UNIFORM_CONTEXT_RNG",
        phase_context_ids=(),
        training_decisions=QUALIFICATION_POLICY_DECISIONS_V1,
        gap_reduction_threshold=GAP_REDUCTION_THRESHOLD_V1,
        behavior_mode=BEHAVIOR_MODE_FIXED_DISTINCT_V1,
        credit_stage2_no_decision=False,
        fee_rate=0.0,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.25,
        max_gross_leverage=2.0,
        reward_reference_equity=REWARD_REFERENCE_EQUITY_V1,
        declared_controls=(CONTROL_FABRICATED_LOG_MU_REJECTION, CONTROL_SHUFFLED_CREDIT),
        higher_ev_family_directions=(),
        higher_ev_family_risk_low=0.0,
        higher_ev_family_risk_high=0.0,
    )

    aba = TaskSpecV1(
        task_id=TASK_ABA_RETENTION,
        environment_version="POST_CC_S1_ABA_RETENTION_V1",
        observation_semantics="Recurring synthetic A/B/A conditions distinguishable only through causal market observation; no phase label or detector input.",
        action_semantics="Canonical joint action learned through weights and retained replay.",
        reward_semantics="Arithmetic task return under the active synthetic condition.",
        horizon_semantics="A1 4096, B 4096 and returning A2 4096 policy decisions within the frozen maximum.",
        known_answer="Old A experience remains available to weights and replay so returning A meets the frozen retention-or-relearning criterion.",
        contexts=(
            _context(
                context_id="A",
                account_kind="GENESIS",
                oracle_direction="LONG",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=101.0,
                    stage1_mark=101.0,
                    stage2_branches=(_branch(1.0, 111.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
            _context(
                context_id="B",
                account_kind="GENESIS",
                oracle_direction="SHORT",
                dynamics=EpisodeDynamicsV1(
                    decision_mark=99.0,
                    stage1_mark=99.0,
                    stage2_branches=(_branch(1.0, 89.0),),
                    stage2_is_no_decision_advance=False,
                ),
            ),
        ),
        context_schedule="PHASES",
        phase_context_ids=ABA_PHASE_DECISIONS_V1,
        training_decisions=sum(count for _, count in ABA_PHASE_DECISIONS_V1),
        gap_reduction_threshold=GAP_REDUCTION_THRESHOLD_V1,
        behavior_mode=BEHAVIOR_MODE_CHILD_SWITCH_V1,
        credit_stage2_no_decision=False,
        fee_rate=0.0,
        slippage_bps=0.0,
        initial_margin_rate=0.5,
        maintenance_margin_rate=0.25,
        max_gross_leverage=2.0,
        reward_reference_equity=REWARD_REFERENCE_EQUITY_V1,
        declared_controls=(CONTROL_NO_SIGNAL, CONTROL_RANDOM_IMPOSSIBLE),
        higher_ev_family_directions=(),
        higher_ev_family_risk_low=0.0,
        higher_ev_family_risk_high=0.0,
    )

    return {
        account_dependent.task_id: account_dependent,
        delayed.task_id: delayed,
        high_bankruptcy.task_id: high_bankruptcy,
        off_policy.task_id: off_policy,
        aba.task_id: aba,
    }


def finite_oracle_actions_v1() -> tuple[tuple[str, float], ...]:
    actions: list[tuple[str, float]] = [("FLAT", 0.0)]
    for risk in ACTION_RISK_VALUES_V1:
        actions.append(("LONG", risk))
        actions.append(("SHORT", risk))
    return tuple(actions)


# ---------------------------------------------------------------------------
# Canonical account / execution / runtime mechanics
# ---------------------------------------------------------------------------


def make_genesis_account_v1(
    *,
    account_id: str,
    capital: float = GENESIS_CAPITAL_V1,
    mark_price: float = GENESIS_MARK_V1,
) -> AccountEconomicsStateR0:
    account = make_account_economics_state_r0(
        account_id=account_id,
        cash=float(capital),
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=float(mark_price),
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    account.validate()
    return account


def make_execution_config_v1(spec: TaskSpecV1) -> Round2MechanicalExecutionConfigR1:
    config = Round2MechanicalExecutionConfigR1(
        fee_rate=float(spec.fee_rate),
        slippage_bps=float(spec.slippage_bps),
        initial_margin_rate=float(spec.initial_margin_rate),
        maintenance_margin_rate=float(spec.maintenance_margin_rate),
        max_gross_leverage=float(spec.max_gross_leverage),
    )
    config.validate()
    return config


def make_executor_v1(spec: TaskSpecV1) -> Callable[[AccountEconomicsStateR0, CCPolicyDecisionV1], Any]:
    return partial(execute_via_frozen_r1_r0, config=make_execution_config_v1(spec))


def make_runtime_v1(
    *,
    spec: TaskSpecV1,
    account: AccountEconomicsStateR0,
    account_lineage_id: str,
    policy_generation: str,
    policy_id: str,
    policy_sha256: str,
    schedule_every: int,
) -> CCContinuousAccountRuntimeR0:
    runtime = CCContinuousAccountRuntimeR0(
        account_lineage_id=account_lineage_id,
        account=account,
        clocks=CCFourClockR0(0, 0, 0, f"CC_S1_{spec.task_id}"),
        schedule=CCDecisionScheduleR0(int(schedule_every)),
        executor=make_executor_v1(spec),
        policy_generation=str(policy_generation),
        policy_id=policy_id,
        policy_sha256=policy_sha256,
    )
    return runtime


def make_s1_brain_v1() -> CCCentralBrain:
    torch.manual_seed(MODEL_INITIALIZATION_SEED_V1)
    brain = CCCentralBrain(2, 3, 2, 8, S1_BRAIN_BINDINGS)
    brain.assert_gradient_ownership()
    return brain


def make_s1_critic_v1() -> Any:
    from .cc_critic_value_r0 import SeparateCritic

    return SeparateCritic(7, 8)


def make_analysis_decision_v1(
    *,
    account: AccountEconomicsStateR0,
    direction: str,
    risk: float,
    decision_index: int = 0,
    environment_time: int = 0,
) -> CCPolicyDecisionV1:
    """Evaluation-only W-01 decision; never persisted into training replay."""
    kind = "point_mass" if direction == "FLAT" else "continuous_density"
    decision = CCPolicyDecisionV1(
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        account_lineage_id="cc-s1-analysis-lineage",
        decision_index=int(decision_index),
        environment_time=int(environment_time),
        policy_generation="0",
        policy_id="cc-s1-analysis-policy",
        policy_sha256="0" * 64,
        observation_schema="PostCCObservationFactV1",
        observation_hash="0" * 64,
        normalizer_id="POST_CC_S1_SYNTHETIC_NORMALIZER_V1",
        nominal_direction=str(direction),
        nominal_target_risk=float(risk),
        log_mu=0.0,
        risk_measure_kind=kind,
        rng_stream_id="cc-s1-analysis-rng",
        rng_position_or_counter=0,
    )
    decision.validate()
    return decision


def final_equity_for_action_branch_v1(
    *,
    spec: TaskSpecV1,
    account: AccountEconomicsStateR0,
    direction: str,
    risk: float,
    terminal_mark: float,
    stage1_mark: float,
    stage2_is_no_decision_advance: bool,
) -> float:
    """Canonical mechanical consequence of one action under one exact branch."""
    decision = make_analysis_decision_v1(account=account, direction=direction, risk=risk)
    execution = execute_via_frozen_r1_r0(account, decision, config=make_execution_config_v1(spec))
    after_execution = execution.account_after_execution
    if stage2_is_no_decision_advance:
        after_stage1 = advance_environment_r0(
            after_execution,
            CCEnvironmentIntervalR0(float(stage1_mark), boundary_type=COMPUTE_CHUNK),
        )
    else:
        after_stage1 = after_execution
    liquidate = abs(after_stage1.position_quantity) > 0.0
    final = advance_environment_r0(
        after_stage1,
        CCEnvironmentIntervalR0(
            float(terminal_mark),
            force_liquidate=bool(liquidate),
            boundary_type=OBJECTIVE_HORIZON_REACHED,
        ),
    )
    return float(final.equity)


def expected_action_return_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    account: AccountEconomicsStateR0,
    direction: str,
    risk: float,
) -> float:
    reference = float(spec.reward_reference_equity)
    weighted_equity = 0.0
    for branch in context.dynamics.stage2_branches:
        equity = final_equity_for_action_branch_v1(
            spec=spec,
            account=account,
            direction=direction,
            risk=risk,
            terminal_mark=branch.terminal_mark,
            stage1_mark=context.dynamics.stage1_mark,
            stage2_is_no_decision_advance=context.dynamics.stage2_is_no_decision_advance,
        )
        weighted_equity += float(branch.probability) * equity
    return (weighted_equity - reference) / reference


def expected_action_bankruptcy_frequency_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    account: AccountEconomicsStateR0,
    direction: str,
    risk: float,
) -> float:
    frequency = 0.0
    for branch in context.dynamics.stage2_branches:
        equity = final_equity_for_action_branch_v1(
            spec=spec,
            account=account,
            direction=direction,
            risk=risk,
            terminal_mark=branch.terminal_mark,
            stage1_mark=context.dynamics.stage1_mark,
            stage2_is_no_decision_advance=context.dynamics.stage2_is_no_decision_advance,
        )
        if equity <= 0.0:
            frequency += float(branch.probability)
    return frequency


def enumerate_oracle_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    account: AccountEconomicsStateR0,
) -> dict[str, Any]:
    """Independent finite-class oracle enumeration through canonical mechanics."""
    rows: list[dict[str, Any]] = []
    for direction, risk in finite_oracle_actions_v1():
        rows.append(
            {
                "direction": direction,
                "risk": float(risk),
                "expected_arithmetic_return": expected_action_return_v1(
                    spec=spec, context=context, account=account, direction=direction, risk=risk
                ),
                "expected_bankruptcy_frequency": expected_action_bankruptcy_frequency_v1(
                    spec=spec, context=context, account=account, direction=direction, risk=risk
                ),
            }
        )
    best_return = max(row["expected_arithmetic_return"] for row in rows)
    best_rows = [row for row in rows if best_return - row["expected_arithmetic_return"] <= STRICT_TOLERANCE_V1]
    best_directions = tuple(sorted({row["direction"] for row in best_rows}))
    flat_row = next(row for row in rows if row["direction"] == "FLAT" and row["risk"] == 0.0)
    flat_return = float(flat_row["expected_arithmetic_return"])
    flat_bankruptcy = float(flat_row["expected_bankruptcy_frequency"])
    non_flat_higher_return = [
        row
        for row in rows
        if row["direction"] != "FLAT" and row["expected_arithmetic_return"] > flat_return + STRICT_TOLERANCE_V1
    ]
    non_flat_higher_bankruptcy = [
        row for row in non_flat_higher_return if row["expected_bankruptcy_frequency"] > flat_bankruptcy + STRICT_TOLERANCE_V1
    ]
    return {
        "rows": rows,
        "best_return": float(best_return),
        "best_directions": best_directions,
        "expected_direction_present": bool(context.oracle_direction_expected in best_directions),
        "flat_return": flat_return,
        "flat_bankruptcy_frequency": flat_bankruptcy,
        "non_flat_higher_return_count": len(non_flat_higher_return),
        "non_flat_higher_return_and_bankruptcy_count": len(non_flat_higher_bankruptcy),
    }


# ---------------------------------------------------------------------------
# Reachable setup contexts (canonical execution, not forged accounts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupProvenanceV1:
    context_id: str
    setup_direction: str
    setup_risk: float
    setup_root: str
    sequence_id: str
    decision_ref: str
    transition_id: str
    transition_content_sha256: str
    account_truth_sha256: str
    achieved_position_quantity: float
    achieved_equity: float
    achieved_risk_fraction: float

    def payload(self) -> Mapping[str, Any]:
        return {"schema": "CB16_R11_POST_CC_S1_SETUP_PROVENANCE_V1", **asdict(self)}

    @property
    def content_sha256(self) -> str:
        return stable_sha256_v1(self.payload())


def establish_account_context_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    lineage: str,
    setup_root: str,
) -> tuple[AccountEconomicsStateR0, SetupProvenanceV1 | None]:
    """Run the canonical execution path from genesis to the scored account."""
    if context.account_kind == "GENESIS":
        account = make_genesis_account_v1(
            account_id=f"{lineage}-genesis",
            mark_price=float(context.dynamics.decision_mark),
        )
        return account, None
    if context.account_kind not in ("HELD_LONG_SETUP", "HELD_SHORT_SETUP"):
        raise ValueError(f"UNKNOWN_ACCOUNT_KIND:{context.account_kind}")
    setup_direction = str(context.setup_direction)
    setup_risk = float(context.setup_risk)
    policy_id = f"cc-s1-setup-{context.context_id.lower()}"
    policy_sha256 = stable_sha256_v1({"setup": context.context_id, "direction": setup_direction, "risk": setup_risk})
    sequence_id = f"{lineage}-setup-{context.context_id.lower()}"
    genesis = make_genesis_account_v1(
        account_id=f"{lineage}-genesis",
        mark_price=float(context.dynamics.decision_mark),
    )
    runtime = make_runtime_v1(
        spec=spec,
        account=genesis,
        account_lineage_id=lineage,
        policy_generation="0",
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        schedule_every=1,
    )
    collector = DurableObservationCollectorV1(
        setup_root,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_source_identity=S1_MARKET_SOURCE_IDENTITY,
        market_source_version=S1_MARKET_SOURCE_VERSION,
    )
    setup_rng = PolicyRNG(policy_id, lineage, derived_stream_seed_v1(spec.task_id, context.context_id, "setup"))
    stream_id, counter = setup_rng.provenance()
    nominal = NominalAction(
        direction=setup_direction,
        target_risk=setup_risk,
        log_prob=float(joint_log_prob(
            torch.zeros(3, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), -0.5, dtype=torch.float32),
            setup_direction,
            setup_risk,
        ).item()),
        risk_measure_kind="point_mass" if setup_direction == "FLAT" else "continuous_density",
        rng_stream_id=stream_id,
        rng_counter=counter,
    )
    box: dict[str, Any] = {}

    def callback(account_state, clocks):
        decision, _fact = collector.capture_policy_decision(
            account_state,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation="0",
            policy_id=policy_id,
            policy_sha256=policy_sha256,
            nominal=nominal,
        )
        box["decision"] = decision
        return decision

    equity_before = runtime.account.equity
    transition = runtime.step(
        CCEnvironmentIntervalR0(context.dynamics.decision_mark, boundary_type=CONTINUE),
        callback,
        expected_predecessor_token=runtime.predecessor_token,
    )
    record = collector.persist_transition(
        sequence_id=sequence_id,
        transition=transition,
        decision=box["decision"],
        reward=(runtime.account.equity - equity_before) / float(spec.reward_reference_equity),
        discount=DEFAULT_ARITHMETIC_DISCOUNT_V1,
    )
    collector.finalize_sequence(
        sequence_id=sequence_id,
        market_lineage_id=S1_MARKET_SOURCE_IDENTITY,
        source_classification="S1_SETUP_PROVENANCE_ONLY",
        transition_ids=(record.transition_id,),
        chunk_boundary_type=CONTINUE,
        bootstrap_state_ref_or_null=None,
    )
    collector.close()
    account = runtime.account
    mark = float(account.mark_price)
    position = float(account.position_quantity)
    notional = abs(position) * mark
    risk_fraction = notional / max(float(spec.reward_reference_equity) * float(spec.max_gross_leverage), 1e-12)
    provenance = SetupProvenanceV1(
        context_id=context.context_id,
        setup_direction=setup_direction,
        setup_risk=setup_risk,
        setup_root=str(setup_root),
        sequence_id=sequence_id,
        decision_ref=box["decision"].ref,
        transition_id=record.transition_id,
        transition_content_sha256=record.content_sha256,
        account_truth_sha256=stable_sha256_v1(asdict(account)),
        achieved_position_quantity=position,
        achieved_equity=float(account.equity),
        achieved_risk_fraction=float(risk_fraction),
    )
    return account, provenance


# ---------------------------------------------------------------------------
# Frozen policy evaluation population
# ---------------------------------------------------------------------------


def policy_inputs_v1(
    account: AccountEconomicsStateR0,
    *,
    zero_account_inputs: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account)
    if zero_account_inputs:
        account_values = tuple(0.0 for _ in account_values)
        execution_values = tuple(0.0 for _ in execution_values)
    return (
        torch.tensor(market, dtype=torch.float32),
        torch.tensor(account_values, dtype=torch.float32),
        torch.tensor(execution_values, dtype=torch.float32),
    )


def sample_policy_action_v1(
    *,
    actor: CCCentralBrain,
    account: AccountEconomicsStateR0,
    rng: PolicyRNG,
    zero_account_inputs: bool,
) -> NominalAction:
    market, account_t, execution_t = policy_inputs_v1(account, zero_account_inputs=zero_account_inputs)
    with torch.no_grad():
        logits, risk_loc, risk_log_scale = actor(market, account_t, execution_t)
    return sample_nominal(logits, risk_loc, risk_log_scale, rng)


def evaluation_rng_v1(*, task_id: str, seed: int, context_id: str, checkpoint_id: str) -> PolicyRNG:
    policy_id = f"cc-s1-eval-{task_id}-{context_id}-{checkpoint_id}"
    lineage = f"cc-s1-eval-lineage-{seed}"
    stream_seed = derived_stream_seed_v1(task_id, seed, context_id, checkpoint_id, "evaluation")
    return PolicyRNG(policy_id, lineage, stream_seed)


def _phi_v1(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def direction_probabilities_v1(
    *,
    actor: CCCentralBrain,
    account: AccountEconomicsStateR0,
    zero_account_inputs: bool,
) -> dict[str, float]:
    market, account_t, execution_t = policy_inputs_v1(account, zero_account_inputs=zero_account_inputs)
    with torch.no_grad():
        logits, _risk_loc, _risk_log_scale = actor(market, account_t, execution_t)
        probabilities = torch.softmax(logits, dim=-1).tolist()
    return {direction: float(probabilities[DIRECTION_INDEX[direction]]) for direction in DIRECTION_ORDER}


def risk_band_mass_v1(
    *,
    actor: CCCentralBrain,
    account: AccountEconomicsStateR0,
    direction: str,
    risk_low: float,
    risk_high: float,
    zero_account_inputs: bool,
) -> float:
    market, account_t, execution_t = policy_inputs_v1(account, zero_account_inputs=zero_account_inputs)
    with torch.no_grad():
        _logits, risk_loc, risk_log_scale = actor(market, account_t, execution_t)
    loc = float(risk_loc[DIRECTION_INDEX[direction]].item())
    scale = float(torch.exp(risk_log_scale[DIRECTION_INDEX[direction]]).item())
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("BAD_RISK_SCALE")
    logit_low = math.log(risk_low / (1.0 - risk_low))
    logit_high = math.log(risk_high / (1.0 - risk_high))
    return max(0.0, min(1.0, _phi_v1((logit_high - loc) / scale) - _phi_v1((logit_low - loc) / scale)))


def evaluate_context_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    account: AccountEconomicsStateR0,
    actor: CCCentralBrain,
    seed: int,
    checkpoint_id: str,
    population: int,
    zero_account_inputs: bool,
) -> dict[str, Any]:
    rng = evaluation_rng_v1(task_id=spec.task_id, seed=seed, context_id=context.context_id, checkpoint_id=checkpoint_id)
    returns: list[float] = []
    bankruptcy: list[float] = []
    family_mass = 0.0
    for _ in range(int(population)):
        nominal = sample_policy_action_v1(actor=actor, account=account, rng=rng, zero_account_inputs=zero_account_inputs)
        returns.append(
            expected_action_return_v1(
                spec=spec,
                context=context,
                account=account,
                direction=nominal.direction,
                risk=float(nominal.target_risk),
            )
        )
        bankruptcy.append(
            expected_action_bankruptcy_frequency_v1(
                spec=spec,
                context=context,
                account=account,
                direction=nominal.direction,
                risk=float(nominal.target_risk),
            )
        )
    direction_probability = direction_probabilities_v1(
        actor=actor, account=account, zero_account_inputs=zero_account_inputs
    )
    if spec.higher_ev_family_directions:
        for direction in spec.higher_ev_family_directions:
            family_mass += direction_probability[direction] * risk_band_mass_v1(
                actor=actor,
                account=account,
                direction=direction,
                risk_low=float(spec.higher_ev_family_risk_low),
                risk_high=float(spec.higher_ev_family_risk_high),
                zero_account_inputs=zero_account_inputs,
            )
    return {
        "context_id": context.context_id,
        "population": int(population),
        "mean_complete_sample_arithmetic_return": float(sum(returns) / len(returns)),
        "mean_expected_bankruptcy_frequency": float(sum(bankruptcy) / len(bankruptcy)),
        "oracle_direction_probability": float(direction_probability[context.oracle_direction_expected]),
        "direction_probabilities": direction_probability,
        "higher_ev_family_mass": float(family_mass),
    }


def evaluate_task_v1(
    *,
    spec: TaskSpecV1,
    accounts: Mapping[str, AccountEconomicsStateR0],
    actor: CCCentralBrain,
    seed: int,
    checkpoint_id: str,
    population: int,
    zero_account_inputs: bool,
) -> dict[str, Any]:
    contexts: dict[str, Any] = {}
    for context in spec.contexts:
        contexts[context.context_id] = evaluate_context_v1(
            spec=spec,
            context=context,
            account=accounts[context.context_id],
            actor=actor,
            seed=seed,
            checkpoint_id=checkpoint_id,
            population=int(population),
            zero_account_inputs=zero_account_inputs,
        )
    scores = [record["mean_complete_sample_arithmetic_return"] for record in contexts.values()]
    mean_score = float(sum(scores) / len(scores))
    return {
        "checkpoint_id": checkpoint_id,
        "population_per_context": int(population),
        "mean_complete_sample_arithmetic_return": mean_score,
        "contexts": contexts,
    }


# ---------------------------------------------------------------------------
# Durable episode collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeEvidenceV1:
    task_id: str
    context_id: str
    sequence_id: str
    lineage: str
    sequence_content_sha256: str
    decision_ref: str
    transition_id: str
    transition_content_sha256: str
    observation_logical_id: str
    nominal_direction: str
    nominal_target_risk: float
    behavior_log_mu: float
    behavior_policy_generation: str
    behavior_policy_id: str
    behavior_policy_sha256: str
    reward: float
    discount: float
    boundary_type: str
    final_equity: float
    terminal_branch_probability: float
    terminal_branch_mark: float
    credit_view_id: str | None
    raw_advance_ids: tuple[str, ...]
    reward_mode: str
    zero_account_inputs: bool
    policy_generation_after_episode: str
    final_account: AccountEconomicsStateR0

    def payload(self) -> Mapping[str, Any]:
        return {"schema": "CB16_R11_POST_CC_S1_EPISODE_EVIDENCE_V1", **asdict(self)}

    @property
    def content_sha256(self) -> str:
        return stable_sha256_v1(self.payload())


def sample_branch_v1(branches: Sequence[BranchOutcomeV1], rng: random.Random) -> BranchOutcomeV1:
    if not branches:
        raise ValueError("EMPTY_BRANCH_SET")
    total = sum(float(branch.probability) for branch in branches)
    if abs(total - 1.0) > 1e-12:
        raise ValueError("BRANCH_PROBABILITIES_MUST_SUM_TO_ONE")
    draw = rng.random()
    cumulative = 0.0
    for branch in branches:
        cumulative += float(branch.probability)
        if draw <= cumulative:
            return branch
    return branches[-1]


def boundary_observation_ref_v1(
    *,
    root: str,
    lineage: str,
    decision_index: int,
    environment_time: int,
    account: AccountEconomicsStateR0,
) -> str:
    market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account)
    time_text = environment_time_text_v1(int(environment_time))
    fact = build_canonical_observation_fact_v1(
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_values=market,
        account_values=account_values,
        execution_values=execution_values,
        market_source_identity=S1_MARKET_SOURCE_IDENTITY,
        market_source_version=S1_MARKET_SOURCE_VERSION,
        market_visible_through_time=time_text,
        account_lineage_id=lineage,
        decision_index=int(decision_index),
        environment_time=time_text,
        normalizer_identity=CANONICAL_NORMALIZER_IDENTITY_V1,
    )
    store = ObservationStoreV1(root)
    receipt = store.put(fact)
    store.close()
    return receipt.logical_id


def collect_episode_v1(
    *,
    spec: TaskSpecV1,
    context: ContextSpecV1,
    account: AccountEconomicsStateR0,
    root: str,
    lineage: str,
    policy_generation: str,
    policy_id: str,
    policy_sha256: str,
    actor: CCCentralBrain,
    action_rng: PolicyRNG,
    env_rng: random.Random,
    sequence_id: str,
    zero_account_inputs: bool = False,
    reward_mode: str = "RAW",
    branch_override: BranchOutcomeV1 | None = None,
) -> EpisodeEvidenceV1:
    """Collect one fully durable episode through the canonical runtime path."""
    if reward_mode not in ("RAW", "ZERO"):
        raise ValueError("UNKNOWN_REWARD_MODE")
    schedule_every = 2 if spec.credit_stage2_no_decision else 1
    runtime = make_runtime_v1(
        spec=spec,
        account=account,
        account_lineage_id=lineage,
        policy_generation=policy_generation,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        schedule_every=schedule_every,
    )
    collector = DurableObservationCollectorV1(
        root,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_source_identity=S1_MARKET_SOURCE_IDENTITY,
        market_source_version=S1_MARKET_SOURCE_VERSION,
    )
    box: dict[str, Any] = {}
    equity_before = float(runtime.account.equity)

    def callback(account_state, clocks):
        nominal = sample_policy_action_v1(
            actor=actor,
            account=account_state,
            rng=action_rng,
            zero_account_inputs=zero_account_inputs,
        )
        decision, fact = collector.capture_policy_decision(
            account_state,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=policy_generation,
            policy_id=policy_id,
            policy_sha256=policy_sha256,
            nominal=nominal,
        )
        box["decision"] = decision
        box["observation_logical_id"] = fact.observation_hash
        return decision

    if spec.credit_stage2_no_decision:
        stage1_interval = CCEnvironmentIntervalR0(context.dynamics.stage1_mark, boundary_type=COMPUTE_CHUNK)
    else:
        stage1_branch = branch_override or sample_branch_v1(context.dynamics.stage2_branches, env_rng)
        box["terminal_branch"] = stage1_branch
        stage1_interval = CCEnvironmentIntervalR0(
            stage1_branch.terminal_mark,
            force_liquidate=True,
            boundary_type=OBJECTIVE_HORIZON_REACHED,
        )
    transition = runtime.step(
        stage1_interval,
        callback,
        expected_predecessor_token=runtime.predecessor_token,
    )
    reward = 0.0 if reward_mode == "ZERO" else (float(runtime.account.equity) - equity_before) / float(spec.reward_reference_equity)
    boundary_observation_ref: str | None = None
    if spec.credit_stage2_no_decision:
        boundary_observation_ref = boundary_observation_ref_v1(
            root=root,
            lineage=lineage,
            decision_index=int(runtime.clocks.policy_decision_index),
            environment_time=int(runtime.clocks.environment_time),
            account=runtime.account,
        )
    record = collector.persist_transition(
        sequence_id=sequence_id,
        transition=transition,
        decision=box["decision"],
        reward=float(reward),
        discount=DEFAULT_ARITHMETIC_DISCOUNT_V1,
        bootstrap_state_ref_or_null=boundary_observation_ref,
    )
    if spec.credit_stage2_no_decision:
        collector.finalize_sequence(
            sequence_id=sequence_id,
            market_lineage_id=S1_MARKET_SOURCE_IDENTITY,
            source_classification="S1_DECISION_INTERVAL_CREDIT",
            transition_ids=(record.transition_id,),
            chunk_boundary_type=COMPUTE_CHUNK,
            bootstrap_state_ref_or_null=boundary_observation_ref,
        )
        collector.close()
        from .cc_account_recovery_r0 import restore_runtime_r0, seal_runtime_r0
        from .post_cc_s1_credit_adapter_v1 import raw_advance_logical_id_v1, record_decision_interval_credit_v1

        runtime_seal = seal_runtime_r0(runtime)
        del runtime
        del collector
        runtime, _token = restore_runtime_r0(runtime_seal, executor=make_executor_v1(spec))
        stage2_branch = branch_override or sample_branch_v1(context.dynamics.stage2_branches, env_rng)
        stage2_transition = runtime.step(
            CCEnvironmentIntervalR0(
                stage2_branch.terminal_mark,
                force_liquidate=True,
                boundary_type=OBJECTIVE_HORIZON_REACHED,
            ),
            None,
            expected_predecessor_token=runtime.predecessor_token,
        )
        if stage2_transition.policy_decision_ref is not None:
            raise RuntimeError("DELAYED_STAGE2_MUST_BE_NO_DECISION_ADVANCE")
        collector_after_restart = DurableObservationCollectorV1(
            root,
            science_semantic_version=SCIENCE_SEMANTIC_VERSION,
            market_source_identity=S1_MARKET_SOURCE_IDENTITY,
            market_source_version=S1_MARKET_SOURCE_VERSION,
        )
        raw_advance_id = raw_advance_logical_id_v1(
            lineage=stage2_transition.account_lineage_id,
            decision_index=int(stage2_transition.decision_index),
            environment_time_before=int(stage2_transition.environment_time_before),
        )
        persisted_raw_content_sha256 = collector_after_restart.persist_no_decision_advance(stage2_transition)
        collector_after_restart.close()
        if not persisted_raw_content_sha256:
            raise RuntimeError("RAW_ADVANCE_PERSIST_FAILED" )
        credit = record_decision_interval_credit_v1(
            root=root,
            sequence_id=sequence_id,
            decision_transition_id=record.transition_id,
            decision_transition_content_sha256=record.content_sha256,
            decision_equity_before=equity_before,
            final_equity=float(runtime.account.equity),
            raw_advance_ids=(raw_advance_id,),
            boundary_type=OBJECTIVE_HORIZON_REACHED,
            mechanical_terminal=False,
            reward_reference_equity=float(spec.reward_reference_equity),
            discount=DEFAULT_ARITHMETIC_DISCOUNT_V1,
            lineage=lineage,
            decision_index=int(record.decision_index),
            source_classification=(
                "S1_DELAYED_CREDIT_RAW"
                if branch_override is None
                else "S1_DELAYED_CREDIT_CONTROL_OVERRIDE"
            ),
        )
        final_account = runtime.account
        return EpisodeEvidenceV1(
            task_id=spec.task_id,
            context_id=context.context_id,
            sequence_id=sequence_id,
            lineage=lineage,
            sequence_content_sha256=stable_sha256_v1(
                {
                    "sequence_id": sequence_id,
                    "decision_transition": record.content_sha256,
                    "raw_advance_ids": (raw_advance_id,),
                    "credit_id": credit.credit_id,
                }
            ),
            decision_ref=box["decision"].ref,
            transition_id=record.transition_id,
            transition_content_sha256=record.content_sha256,
            observation_logical_id=box["observation_logical_id"],
            nominal_direction=box["decision"].nominal_direction,
            nominal_target_risk=float(box["decision"].nominal_target_risk),
            behavior_log_mu=float(box["decision"].log_mu),
            behavior_policy_generation=str(box["decision"].policy_generation),
            behavior_policy_id=box["decision"].policy_id,
            behavior_policy_sha256=box["decision"].policy_sha256,
            reward=float(credit.reward),
            discount=float(credit.discount),
            boundary_type=OBJECTIVE_HORIZON_REACHED,
            final_equity=float(final_account.equity),
            terminal_branch_probability=float(stage2_branch.probability),
            terminal_branch_mark=float(stage2_branch.terminal_mark),
            credit_view_id=credit.credit_id,
            raw_advance_ids=(raw_advance_id,),
            reward_mode=reward_mode,
            zero_account_inputs=bool(zero_account_inputs),
            policy_generation_after_episode=str(runtime.policy_generation),
            final_account=final_account,
        )
    collector.finalize_sequence(
        sequence_id=sequence_id,
        market_lineage_id=S1_MARKET_SOURCE_IDENTITY,
        source_classification="S1_TERMINAL_EPISODE",
        transition_ids=(record.transition_id,),
        chunk_boundary_type=transition.boundary_type,
        bootstrap_state_ref_or_null=None,
    )
    collector.close()
    branch_used = box["terminal_branch"]
    return EpisodeEvidenceV1(
        task_id=spec.task_id,
        context_id=context.context_id,
        sequence_id=sequence_id,
        lineage=lineage,
        sequence_content_sha256=stable_sha256_v1(
            {"sequence_id": sequence_id, "decision_transition": record.content_sha256}
        ),
        decision_ref=box["decision"].ref,
        transition_id=record.transition_id,
        transition_content_sha256=record.content_sha256,
        observation_logical_id=box["observation_logical_id"],
        nominal_direction=box["decision"].nominal_direction,
        nominal_target_risk=float(box["decision"].nominal_target_risk),
        behavior_log_mu=float(box["decision"].log_mu),
        behavior_policy_generation=str(box["decision"].policy_generation),
        behavior_policy_id=box["decision"].policy_id,
        behavior_policy_sha256=box["decision"].policy_sha256,
        reward=float(reward),
        discount=DEFAULT_ARITHMETIC_DISCOUNT_V1,
        boundary_type=transition.boundary_type,
        final_equity=float(runtime.account.equity),
        terminal_branch_probability=float(branch_used.probability),
        terminal_branch_mark=float(branch_used.terminal_mark),
        credit_view_id=None,
        raw_advance_ids=(),
        reward_mode=reward_mode,
        zero_account_inputs=bool(zero_account_inputs),
        policy_generation_after_episode=str(runtime.policy_generation),
        final_account=runtime.account,
    )


def frozen_random_control_contexts_v1(spec: TaskSpecV1) -> tuple[ContextSpecV1, ...]:
    """Deterministic symmetric-random realization for the RANDOM/IMPOSSIBLE control.

    Training uses per-episode random draws; the frozen control evaluation uses
    these declared exact 50/50 branches so the control is not a memorization
    test.  For the sign-symmetric branches the arithmetic expectation of every
    direction is equal in the zero-fee tasks, so no accidental PASS is
    possible without violating the positive oracle.
    """
    contexts = []
    for context in spec.contexts:
        if spec.task_id == TASK_HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION:
            branches = (_branch(0.5, 120.0), _branch(0.5, 80.0))
        else:
            mark = float(context.dynamics.decision_mark)
            branches = (_branch(0.5, mark + 10.0), _branch(0.5, mark - 10.0))
        contexts.append(
            ContextSpecV1(
                context_id=context.context_id,
                account_kind=context.account_kind,
                setup_direction=context.setup_direction,
                setup_risk=context.setup_risk,
                oracle_direction_expected=context.oracle_direction_expected,
                dynamics=EpisodeDynamicsV1(
                    decision_mark=float(context.dynamics.decision_mark),
                    stage1_mark=float(context.dynamics.stage1_mark),
                    stage2_branches=branches,
                    stage2_is_no_decision_advance=bool(context.dynamics.stage2_is_no_decision_advance),
                ),
            )
        )
    return tuple(contexts)
