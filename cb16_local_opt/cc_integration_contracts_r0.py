from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .cc_runtime_wire_r0 import CCPolicyDecisionV1 as RuntimePolicyDecision
from .cc_runtime_wire_r0 import CCEnvironmentTransitionV1 as RuntimeEnvironmentTransition
from .cc_policy_wire_r0 import CCExperienceSequenceV1 as LearnerExperienceSequence
from .cc_experience_wire_r0 import CCEnvironmentTransitionV1 as ExperienceEnvironmentTransition
from .cc_experience_wire_r0 import CCExperienceSequenceV1 as ExperienceSequence
from .cc_experience_transition_r0 import ImmutableExperienceTransitionV1

SCIENCE_SEMANTIC_VERSION = "CB16_R11_CC_SCIENCE_SEMANTIC_V1"
W_SCHEMA_IDENTITIES = {
    "W-01": "CCPolicyDecisionV1",
    "W-02": "CCEnvironmentTransitionV1",
    "W-03": "CCExperienceSequenceV1",
    "W-04": "CCLearningUpdateV1",
    "W-05": "CCEconomicResultV1",
}


def _ordered_time_text(value: int) -> str:
    """Canonical lexical representation for C's string timestamp fixture."""
    value = int(value)
    if value < 0:
        raise ValueError("NEGATIVE_ENVIRONMENT_TIME")
    return f"{value:020d}"


def policy_decision_from_nominal(
    *,
    account_lineage_id: str,
    decision_index: int,
    environment_time: int,
    policy_generation: int,
    policy_id: str,
    policy_sha256: str,
    observation_schema: str,
    observation_hash: str,
    normalizer_id: str,
    nominal: Any,
) -> RuntimePolicyDecision:
    """Bind Thread-B nominal sample to Thread-A W-01 without changing meaning."""
    decision = RuntimePolicyDecision(
        SCIENCE_SEMANTIC_VERSION,
        account_lineage_id,
        int(decision_index),
        int(environment_time),
        str(int(policy_generation)),
        policy_id,
        policy_sha256,
        observation_schema,
        observation_hash,
        normalizer_id,
        str(nominal.direction),
        float(nominal.target_risk),
        float(nominal.log_prob),
        str(nominal.risk_measure_kind),
        str(nominal.rng_stream_id),
        int(nominal.rng_counter),
    )
    decision.validate()
    return decision


def runtime_transition_raw_fact(transition: RuntimeEnvironmentTransition) -> Mapping[str, Any]:
    """Every A transition is durable raw truth, including no-decision mechanical advances."""
    transition.validate()
    payload = asdict(transition)
    payload["schema"] = "CCRuntimeRawFactV1"
    payload["w02_replay_eligible"] = transition.policy_decision_ref is not None
    payload["raw_only_reason"] = None if transition.policy_decision_ref is not None else "NO_POLICY_DECISION_NO_LOG_MU_FABRICATION"
    return payload


def runtime_transition_to_experience_environment(
    transition: RuntimeEnvironmentTransition,
) -> ExperienceEnvironmentTransition:
    """Adapt only decision-bearing A transitions into C's W-02 fixture."""
    transition.validate()
    if transition.policy_decision_ref is None:
        raise ValueError("RAW_ONLY_NO_POLICY_DECISION")
    legs = tuple(
        {
            "leg_index": leg.leg_index,
            "executed_quantity": float(leg.delta_quantity),
            "execution_price": None if leg.fill_price is None else float(leg.fill_price),
            "status": leg.status,
        }
        for leg in transition.execution_legs
    )
    out = ExperienceEnvironmentTransition(
        account_lineage_id=transition.account_lineage_id,
        decision_index=transition.decision_index,
        environment_time_before=_ordered_time_text(transition.environment_time_before),
        environment_time_after=_ordered_time_text(transition.environment_time_after),
        pre_account_truth_hash=transition.pre_account_truth_hash,
        policy_decision_ref=transition.policy_decision_ref,
        permission_status=transition.permission_status,
        permission_reason=transition.permission_reason,
        permitted_target_direction=transition.permitted_target_direction,
        permitted_target_risk=transition.permitted_target_risk,
        target_quantity=transition.target_quantity,
        execution_legs=legs,
        fees=transition.fees,
        funding=transition.funding,
        realized_pnl=transition.realized_pnl,
        unrealized_pnl_delta=transition.unrealized_pnl_delta,
        liability_delta=transition.liability_delta,
        post_account_truth_hash=transition.post_account_truth_hash,
        post_equity=transition.post_equity,
        boundary_type=transition.boundary_type,
        mechanical_terminal=transition.mechanical_terminal,
        external_capital_flow_ref_or_null=transition.external_capital_flow_ref_or_null,
    )
    return out.validate()


def immutable_experience_from_runtime(
    *,
    transition_id: str,
    environment: ExperienceEnvironmentTransition,
    decision: RuntimePolicyDecision,
    market_lineage_id: str,
    source_classification: str = "CC_STOCHASTIC_TRAJECTORY",
    failure_classification: str = "NONE",
) -> ImmutableExperienceTransitionV1:
    decision.validate()
    if environment.policy_decision_ref != decision.ref:
        raise ValueError("POLICY_DECISION_REF_MISMATCH")
    out = ImmutableExperienceTransitionV1(
        transition_id=transition_id,
        environment=environment,
        science_semantic_version=decision.science_semantic_version,
        policy_generation=decision.policy_generation,
        policy_id=decision.policy_id,
        policy_sha256=decision.policy_sha256,
        observation_schema=decision.observation_schema,
        observation_hash=decision.observation_hash,
        normalizer_id=decision.normalizer_id,
        nominal_direction=decision.nominal_direction,
        nominal_target_risk=decision.nominal_target_risk,
        log_mu=decision.log_mu,
        risk_measure_kind=decision.risk_measure_kind,
        rng_stream_id=decision.rng_stream_id,
        rng_position_or_counter=str(decision.rng_position_or_counter),
        market_lineage_id=market_lineage_id,
        source_classification=source_classification,
        failure_classification=failure_classification,
    )
    return out.validate()


def experience_sequence_to_learner(sequence: ExperienceSequence) -> LearnerExperienceSequence:
    """W-03 is semantically identical across C and B; normalize only representation."""
    sequence.validate()
    return LearnerExperienceSequence(
        sequence_id=sequence.sequence_id,
        account_lineage_id=sequence.account_lineage_id,
        science_semantic_version=sequence.science_semantic_version,
        market_lineage_id=sequence.market_lineage_id,
        source_classification=sequence.source_classification,
        transition_refs=sequence.transition_refs,
        first_decision_index=sequence.first_decision_index,
        last_decision_index=sequence.last_decision_index,
        behavior_policy_identities=sequence.behavior_policy_identities,
        normalizer_identities=sequence.normalizer_identities,
        chunk_boundary_type=sequence.chunk_boundary_type,
        bootstrap_state_ref_or_null=sequence.bootstrap_state_ref_or_null,
        raw_fact_content_sha256=sequence.raw_fact_content_sha256,
        extensions={"adapter": "CC_INTEGRATION_R0"},
    )
