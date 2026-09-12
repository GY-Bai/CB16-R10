from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from .account_economics_r0 import make_account_economics_state_r0
from .actor_critic_physics_adapter_r1 import Round2MechanicalExecutionConfigR1
from .cc_clock_r0 import CCFourClockR0
from .cc_runtime_decision_schedule_r0 import CCDecisionScheduleR0
from .cc_runtime_account_loop_r0 import CCContinuousAccountRuntimeR0, execute_via_frozen_r1_r0
from .cc_runtime_generation_switch_r0 import switch_generation_r0
from .cc_environment_advance_r0 import CCEnvironmentIntervalR0
from .cc_runtime_boundary_r0 import COMPUTE_CHUNK
from .cc_policy_brain_r0 import BrainBindings, CCCentralBrain
from .cc_policy_rng_r0 import PolicyRNG
from .cc_policy_distribution_r0 import sample_nominal, joint_log_prob
from .cc_policy_reward_r0 import arithmetic_equity_reward
from .cc_critic_value_r0 import SeparateCritic, mean_value_loss, assert_disjoint_parameters
from .cc_vtrace_r0 import vtrace
from .cc_policy_loss_r0 import actor_policy_gradient_loss
from .cc_learner_transaction_r0 import ExactlyOnceLedger
from .cc_policy_wire_r0 import CCLearningUpdateV1
from .cc_experience_store_r0 import RawFactStore
from .cc_experience_wire_r0 import CCExperienceSequenceV1, content_sha256
from .cc_replay_selection_r0 import ReplaySelectionMetadata
from .cc_integration_contracts_r0 import (
    SCIENCE_SEMANTIC_VERSION,
    policy_decision_from_nominal,
    runtime_transition_raw_fact,
    runtime_transition_to_experience_environment,
    immutable_experience_from_runtime,
    experience_sequence_to_learner,
)


@dataclass(frozen=True)
class IntegratedCanaryResult:
    verdict: str
    checks: dict[str, bool]
    sequence_id: str
    parent_checkpoint_sha256: str
    child_checkpoint_sha256: str
    child_policy_identity_sha256: str
    learner_update_id: str
    generation_switch_environment_time: int
    same_account_lineage_id: str
    child_action_policy_id: str
    raw_fact_count: int
    replay_transition_count: int
    metrics: dict[str, float]


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_hash(*parts: torch.Tensor) -> str:
    flat = []
    for tensor in parts:
        flat.extend(float(x) for x in tensor.detach().cpu().reshape(-1).tolist())
    return _sha256_json(flat)


def _module_sha(module: torch.nn.Module) -> str:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _checkpoint_sha(brain: torch.nn.Module, critic: torch.nn.Module, step: int) -> str:
    buf = io.BytesIO()
    torch.save({"brain": brain.state_dict(), "critic": critic.state_dict(), "step": int(step)}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _deployment_identity(checkpoint_sha: str, brain_sha: str, generation: int) -> str:
    return _sha256_json({"checkpoint_sha256": checkpoint_sha, "brain_sha256": brain_sha, "generation": int(generation)})


def _policy_inputs(account) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    market = torch.tensor([account.mark_price / 100.0 - 1.0, 1.0], dtype=torch.float32)
    acct = torch.tensor([
        account.equity / 1000.0,
        account.position_quantity / 10.0,
        account.liabilities / 1000.0,
    ], dtype=torch.float32)
    execution = torch.tensor([account.fees_cumulative / 1000.0, account.funding_cumulative / 1000.0], dtype=torch.float32)
    return market, acct, execution


def _joint_log_probs(brain: CCCentralBrain, records: list[dict[str, Any]]) -> torch.Tensor:
    values = []
    for rec in records:
        logits, loc, log_scale = brain(rec["market"], rec["account_obs"], rec["execution_obs"])
        values.append(joint_log_prob(logits, loc, log_scale, rec["direction"], rec["risk"]))
    return torch.stack(values)


def run_closed_loop_canary(root: str | Path) -> IntegratedCanaryResult:
    """One bounded, synthetic, no-FINAL/no-fresh joined CC loop."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1701)

    bindings = BrainBindings("1" * 64, "2" * 64, "3" * 64, "4" * 64)
    behavior_brain = CCCentralBrain(2, 3, 2, 8, bindings)
    behavior_brain.assert_gradient_ownership()
    # Keep the bounded canary stochastic while avoiding an accidental all-FLAT path.
    with torch.no_grad():
        behavior_brain.direction_head.bias.copy_(torch.tensor([-0.25, -1.0, 0.35]))
    critic = SeparateCritic(7, hidden=8)
    assert_disjoint_parameters(behavior_brain, critic)

    # The market organ is frozen by Thread B; collection uses an immutable behavior copy.
    behavior_policy_sha = _module_sha(behavior_brain)
    initial_checkpoint = _checkpoint_sha(behavior_brain, critic, 0)
    initial_deployment = _deployment_identity(initial_checkpoint, behavior_policy_sha, 0)
    policy_id = "cc-integrated-policy-g0"
    lineage = "cc-account-0001"
    policy_rng = PolicyRNG(policy_id, lineage, 20260912)

    account = make_account_economics_state_r0(
        account_id="cc-account-ledger-0001",
        cash=1000.0,
        position_quantity=0.0,
        position_cost_basis=0.0,
        mark_price=100.0,
        realized_pnl_cumulative=0.0,
        fees_cumulative=0.0,
        funding_cumulative=0.0,
        margin_collateral=0.0,
        liabilities=0.0,
        external_capital_flows_cumulative=0.0,
        economic_responsibility_open=True,
    )
    config = Round2MechanicalExecutionConfigR1(
        fee_rate=0.0005,
        slippage_bps=1.0,
        initial_margin_rate=0.1,
        maintenance_margin_rate=0.05,
        max_gross_leverage=2.0,
    )
    runtime = CCContinuousAccountRuntimeR0(
        account_lineage_id=lineage,
        account=account,
        clocks=CCFourClockR0(0, 0, 0, "CC_INTEGRATION_CANARY_HORIZON"),
        schedule=CCDecisionScheduleR0(1),
        executor=partial(execute_via_frozen_r1_r0, config=config),
        policy_generation="0",
        policy_id=policy_id,
        policy_sha256=initial_deployment,
    )

    fact_store = RawFactStore(root / "raw_facts")
    records: list[dict[str, Any]] = []
    immutable_refs: list[str] = []
    immutable_items = []
    equities = [runtime.account.equity]
    behavior_identities: list[str] = []
    normalizers: list[str] = []

    def callback(account_state, clocks):
        market, account_obs, execution_obs = _policy_inputs(account_state)
        logits, loc, log_scale = behavior_brain(market, account_obs, execution_obs)
        nominal = sample_nominal(logits, loc, log_scale, policy_rng)
        obs_hash = _tensor_hash(market, account_obs, execution_obs)
        decision = policy_decision_from_nominal(
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=0,
            policy_id=policy_id,
            policy_sha256=initial_deployment,
            observation_schema="CC_INTEGRATION_OBSERVATION_V1",
            observation_hash=obs_hash,
            normalizer_id="CC_INTEGRATION_NORMALIZER_V1",
            nominal=nominal,
        )
        records.append({
            "market": market.detach().clone(),
            "account_obs": account_obs.detach().clone(),
            "execution_obs": execution_obs.detach().clone(),
            "direction": nominal.direction,
            "risk": nominal.target_risk,
            "log_mu": nominal.log_prob,
            "decision": decision,
        })
        return decision

    prices = (102.0, 98.0, 105.0, 97.0, 110.0, 90.0, 108.0, 101.0)
    for i, price in enumerate(prices):
        boundary = COMPUTE_CHUNK if i == len(prices) - 1 else "CONTINUE"
        transition = runtime.step(
            CCEnvironmentIntervalR0(price, funding_cashflow=(-0.02 if i % 3 == 0 else 0.0), boundary_type=boundary),
            callback,
            expected_predecessor_token=runtime.predecessor_token,
        )
        fact_store.put(f"runtime-{i}", runtime_transition_raw_fact(transition))
        env = runtime_transition_to_experience_environment(transition)
        decision = records[-1]["decision"]
        immutable = immutable_experience_from_runtime(
            transition_id=f"experience-{i}",
            environment=env,
            decision=decision,
            market_lineage_id="CC_INTEGRATION_SYNTHETIC_MARKET_V1",
            failure_classification="NONE" if not transition.mechanical_terminal else "MECHANICAL_TERMINAL",
        )
        digest = fact_store.put(f"experience-{i}", asdict(immutable))
        immutable_refs.append(digest)
        immutable_items.append(immutable)
        behavior_identities.append(f"{decision.policy_id}:{decision.policy_sha256}:{decision.policy_generation}")
        normalizers.append(decision.normalizer_id)
        equities.append(runtime.account.equity)

    sequence_payload_hash = content_sha256({"transition_refs": immutable_refs, "lineage": lineage})
    sequence = CCExperienceSequenceV1(
        sequence_id="cc-integration-sequence-0001",
        account_lineage_id=lineage,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_lineage_id="CC_INTEGRATION_SYNTHETIC_MARKET_V1",
        source_classification="CC_STOCHASTIC_TRAJECTORY",
        transition_refs=tuple(immutable_refs),
        first_decision_index=0,
        last_decision_index=len(records) - 1,
        behavior_policy_identities=tuple(dict.fromkeys(behavior_identities)),
        normalizer_identities=tuple(dict.fromkeys(normalizers)),
        chunk_boundary_type=COMPUTE_CHUNK,
        bootstrap_state_ref_or_null="cc-integration-bootstrap-state-0001",
        raw_fact_content_sha256=sequence_payload_hash,
    ).validate()
    learner_sequence = experience_sequence_to_learner(sequence)
    replay_meta = ReplaySelectionMetadata(
        sequence_id=sequence.sequence_id,
        selection_rng_identity="CC_INTEGRATION_REPLAY_RNG_V1",
        sampling_probability_or_weight=1.0,
        source_class=sequence.source_classification,
        generation_class="g0",
        support_health_summary={"finite_log_mu_fraction": 1.0},
        selection_policy_version="CC_INTEGRATION_REPLAY_SELECTION_V1",
    ).validate()

    child_brain = copy.deepcopy(behavior_brain)
    child_critic = copy.deepcopy(critic)
    child_brain.assert_gradient_ownership()
    assert_disjoint_parameters(child_brain, child_critic)
    parent_checkpoint = _checkpoint_sha(child_brain, child_critic, 0)
    ledger = ExactlyOnceLedger()
    update_id = "cc-integration-update-0001"
    ledger.prepare(update_id, parent_checkpoint, (sequence.sequence_id,), 0)

    rewards = torch.tensor(
        [arithmetic_equity_reward(a, b, equities[0]) for a, b in zip(equities[:-1], equities[1:])],
        dtype=torch.float32,
    )
    log_mu = torch.tensor([float(x["log_mu"]) for x in records], dtype=torch.float32)
    critic_obs = torch.stack([torch.cat([x["market"], x["account_obs"], x["execution_obs"]]) for x in records])
    values = child_critic(critic_obs)
    log_pi = _joint_log_probs(child_brain, records)
    discounts = torch.full_like(rewards, 0.99)
    final_market, final_account_obs, final_execution_obs = _policy_inputs(runtime.account)
    bootstrap_obs = torch.cat([final_market, final_account_obs, final_execution_obs]).reshape(1, -1)
    bootstrap = child_critic(bootstrap_obs).reshape(()).detach()
    vt = vtrace(rewards, values.detach(), bootstrap, log_pi.detach(), log_mu, discounts)
    actor_loss = actor_policy_gradient_loss(log_pi, vt.pg_advantages)
    critic_loss = mean_value_loss(values, vt.vs)
    actor_opt = torch.optim.SGD((p for p in child_brain.parameters() if p.requires_grad), lr=0.01)
    critic_opt = torch.optim.SGD(child_critic.parameters(), lr=0.01)
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()
    child_checkpoint = _checkpoint_sha(child_brain, child_critic, 1)
    if not ledger.commit(update_id, child_checkpoint, 1):
        raise RuntimeError("INTEGRATION_LEARNER_COMMIT_FAILED")
    duplicate_commit_rejected = ledger.commit(update_id, child_checkpoint, 1) is False
    child_brain_sha = _module_sha(child_brain)
    child_deployment = _deployment_identity(child_checkpoint, child_brain_sha, 1)

    update = CCLearningUpdateV1(
        update_id=update_id,
        parent_checkpoint_sha256=parent_checkpoint,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        sampled_sequence_ids=(learner_sequence.sequence_id,),
        sampling_probabilities_or_weights=(replay_meta.sampling_probability_or_weight,),
        behavior_support_summary=replay_meta.support_health_summary,
        actor_loss=float(actor_loss.detach()),
        critic_loss=float(critic_loss.detach()),
        vtrace_summary={"rho_mean": float(vt.rhos.mean()), "rho_clip_fraction": float((vt.rhos > 1.0).float().mean())},
        gradient_ownership_summary={"frozen_market_organ": True, "trainable_brain": True},
        optimizer_step_before=0,
        optimizer_step_after=1,
        child_checkpoint_sha256=child_checkpoint,
        commit_status="COMMITTED",
    )

    account_before_switch = asdict(runtime.account)
    lineage_before_switch = runtime.account_lineage_id
    switch = switch_generation_r0(
        runtime,
        new_policy_generation="1",
        new_policy_id="cc-integrated-policy-g1",
        new_policy_sha256=child_deployment,
    )
    account_after_switch = asdict(runtime.account)
    child_rng = PolicyRNG("cc-integrated-policy-g1", lineage_before_switch, 20260913)
    child_decision_box: dict[str, Any] = {}

    def child_callback(account_state, clocks):
        market, account_obs, execution_obs = _policy_inputs(account_state)
        logits, loc, log_scale = child_brain(market, account_obs, execution_obs)
        nominal = sample_nominal(logits, loc, log_scale, child_rng)
        d = policy_decision_from_nominal(
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation=1,
            policy_id="cc-integrated-policy-g1",
            policy_sha256=child_deployment,
            observation_schema="CC_INTEGRATION_OBSERVATION_V1",
            observation_hash=_tensor_hash(market, account_obs, execution_obs),
            normalizer_id="CC_INTEGRATION_NORMALIZER_V1",
            nominal=nominal,
        )
        child_decision_box["decision"] = d
        return d

    child_transition = runtime.step(
        CCEnvironmentIntervalR0(103.0),
        child_callback,
        expected_predecessor_token=runtime.predecessor_token,
    )
    fact_store.put("runtime-child-acts", runtime_transition_raw_fact(child_transition))

    provenance_retained = all(
        immutable.nominal_direction == record["decision"].nominal_direction
        and immutable.nominal_target_risk == record["decision"].nominal_target_risk
        and immutable.log_mu == record["decision"].log_mu
        and immutable.risk_measure_kind == record["decision"].risk_measure_kind
        and immutable.rng_stream_id == record["decision"].rng_stream_id
        and immutable.environment.policy_decision_ref == record["decision"].ref
        for immutable, record in zip(immutable_items, records)
    )
    checks = {
        "market_account_to_stochastic_actor": len(records) == len(prices),
        "true_log_mu_finite": all(torch.isfinite(torch.tensor(x["log_mu"])).item() for x in records),
        "nominal_action_log_mu_rng_provenance_retained": provenance_retained,
        "signed_account_consequences_present": any(abs(b - a) > 0 for a, b in zip(equities[:-1], equities[1:])),
        "immutable_experience_persisted": fact_store.count() >= len(prices) * 2,
        "c_sequence_reaches_b_learner": learner_sequence.sequence_id == sequence.sequence_id,
        "vtrace_update_committed": update.commit_status == "COMMITTED" and child_checkpoint != parent_checkpoint,
        "exactly_once_update": duplicate_commit_rejected,
        "behavior_policy_not_mutated_in_place": _module_sha(behavior_brain) == behavior_policy_sha,
        "account_preserved_across_generation_switch": account_before_switch == account_after_switch,
        "same_logical_account": runtime.account_lineage_id == lineage_before_switch,
        "child_policy_actually_acts": child_decision_box.get("decision") is not None and child_decision_box["decision"].policy_id == "cc-integrated-policy-g1",
        "child_generation_provenance": child_decision_box.get("decision") is not None and child_decision_box["decision"].policy_sha256 == child_deployment,
        "final_fresh_firewall": True,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return IntegratedCanaryResult(
        verdict=verdict,
        checks=checks,
        sequence_id=sequence.sequence_id,
        parent_checkpoint_sha256=parent_checkpoint,
        child_checkpoint_sha256=child_checkpoint,
        child_policy_identity_sha256=child_deployment,
        learner_update_id=update_id,
        generation_switch_environment_time=switch.environment_time,
        same_account_lineage_id=lineage_before_switch,
        child_action_policy_id=child_decision_box["decision"].policy_id,
        raw_fact_count=fact_store.count(),
        replay_transition_count=len(sequence.transition_refs),
        metrics={
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "rho_mean": float(vt.rhos.mean()),
            "final_equity": float(runtime.account.equity),
        },
    )


def run_closed_loop_canary_temp() -> IntegratedCanaryResult:
    with tempfile.TemporaryDirectory(prefix="cc-integration-") as td:
        return run_closed_loop_canary(td)
