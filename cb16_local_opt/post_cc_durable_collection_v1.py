"""S0-v2 durable collection binding.

The collector persists the canonical decision observation before a transition
may link to it.  No-action mechanical advances are stored as raw-only truth and
never receive a fabricated policy decision or behaviour ``log_mu``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .account_economics_r0 import AccountEconomicsStateR0
from .cc_experience_wire_r0 import content_sha256
from .cc_integration_contracts_r0 import policy_decision_from_nominal
from .cc_policy_distribution_r0 import NominalAction
from .cc_runtime_wire_r0 import CCPolicyDecisionV1, CCEnvironmentTransitionV1
from .post_cc_joint_batch_v1 import boundary_requires_bootstrap_v1, boundary_is_terminal_v1
from .post_cc_joint_replay_v1 import (
    DurableSequenceRecordV1,
    DurableTransitionRecordV1,
)
from .post_cc_observation_contract_v1 import PostCCObservationFactV1
from .post_cc_observation_fact_v1 import (
    CANONICAL_NORMALIZER_IDENTITY_V1,
    CANONICAL_OBSERVATION_SCHEMA_V1,
    assert_w01_observation_identity_v1,
    build_canonical_observation_fact_v1,
    canonical_observation_vectors_v1,
)
from .post_cc_observation_store_v1 import ObservationStoreV1, ObservationStoreReceiptV1
from .post_cc_replay_materializer_v1 import ReplayStoreV1


class DurableCollectionError(RuntimeError):
    pass


def environment_time_text_v1(environment_time: int) -> str:
    if isinstance(environment_time, bool) or int(environment_time) < 0:
        raise ValueError("environment_time must be >= 0")
    return f"{int(environment_time):020d}"


def canonical_brain_vectors_from_account_v1(
    account: AccountEconomicsStateR0,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Canonical separated Brain inputs: market / account / execution."""
    account.validate()
    market = (float(account.mark_price) / 100.0 - 1.0, 1.0)
    account_values = (
        float(account.equity) / 1000.0,
        float(account.position_quantity) / 10.0,
        float(account.liabilities) / 1000.0,
    )
    execution = (float(account.fees_cumulative) / 1000.0, float(account.funding_cumulative) / 1000.0)
    return market, account_values, execution


@dataclass(frozen=True)
class PendingObservationV1:
    fact: PostCCObservationFactV1
    receipt: ObservationStoreReceiptV1
    decision_index: int
    environment_time: str


class DurableObservationCollectorV1:
    """Persist decision observations, then link transitions to them."""

    def __init__(
        self,
        root: str,
        *,
        science_semantic_version: str,
        market_source_identity: str,
        market_source_version: str,
        normalizer_identity: str = CANONICAL_NORMALIZER_IDENTITY_V1,
        observation_store: ObservationStoreV1 | None = None,
        replay_store: ReplayStoreV1 | None = None,
    ):
        for name, value in (
            ("science_semantic_version", science_semantic_version),
            ("market_source_identity", market_source_identity),
            ("market_source_version", market_source_version),
            ("normalizer_identity", normalizer_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        self.root = root
        self.science_semantic_version = science_semantic_version
        self.market_source_identity = market_source_identity
        self.market_source_version = market_source_version
        self.normalizer_identity = normalizer_identity
        self.observation_store = observation_store or ObservationStoreV1(root)
        self.replay_store = replay_store or ReplayStoreV1(root)
        self._pending: dict[tuple[str, int, str], PendingObservationV1] = {}

    @staticmethod
    def _pending_key(account_lineage_id: str, decision_index: int, environment_time: str) -> tuple[str, int, str]:
        return (account_lineage_id, int(decision_index), environment_time)

    def capture_policy_decision(
        self,
        account: AccountEconomicsStateR0,
        *,
        account_lineage_id: str,
        decision_index: int,
        environment_time: int,
        policy_generation: str,
        policy_id: str,
        policy_sha256: str,
        nominal: NominalAction,
    ) -> tuple[CCPolicyDecisionV1, PostCCObservationFactV1]:
        """Persist the observation before returning the W-01 decision."""
        if not isinstance(nominal, NominalAction):
            raise TypeError("nominal must be a NominalAction")
        time_text = environment_time_text_v1(environment_time)
        market, account_values, execution = canonical_brain_vectors_from_account_v1(account)
        fact = build_canonical_observation_fact_v1(
            science_semantic_version=self.science_semantic_version,
            market_values=market,
            account_values=account_values,
            execution_values=execution,
            market_source_identity=self.market_source_identity,
            market_source_version=self.market_source_version,
            market_visible_through_time=time_text,
            account_lineage_id=account_lineage_id,
            decision_index=int(decision_index),
            environment_time=time_text,
            normalizer_identity=self.normalizer_identity,
            observation_schema=CANONICAL_OBSERVATION_SCHEMA_V1,
        )
        receipt = self.observation_store.put(fact)
        decision = policy_decision_from_nominal(
            account_lineage_id=account_lineage_id,
            decision_index=int(decision_index),
            environment_time=int(environment_time),
            policy_generation=policy_generation,
            policy_id=policy_id,
            policy_sha256=policy_sha256,
            observation_schema=fact.observation_schema,
            observation_hash=fact.observation_hash,
            normalizer_id=fact.normalizer_identity,
            nominal=nominal,
        )
        assert_w01_observation_identity_v1(decision, fact)
        key = self._pending_key(account_lineage_id, int(decision_index), time_text)
        if key in self._pending:
            raise DurableCollectionError("PENDING_OBSERVATION_ALREADY_CAPTURED")
        self._pending[key] = PendingObservationV1(
            fact=fact,
            receipt=receipt,
            decision_index=int(decision_index),
            environment_time=time_text,
        )
        return decision, fact

    def persist_transition(
        self,
        *,
        sequence_id: str,
        transition: CCEnvironmentTransitionV1,
        decision: CCPolicyDecisionV1,
        reward: float,
        discount: float,
        bootstrap_state_ref_or_null: str | None = None,
        consequence_context: Mapping[str, Any] | None = None,
    ) -> DurableTransitionRecordV1:
        """Link one decision-bearing runtime transition to durable truth."""
        transition.validate()
        decision.validate()
        if transition.policy_decision_ref is None:
            raise DurableCollectionError("MECHANICAL_ADVANCE_MUST_NOT_BE_REPLAY_TRANSITION")
        if transition.policy_decision_ref != decision.ref:
            raise DurableCollectionError("POLICY_DECISION_REF_MISMATCH")
        if transition.account_lineage_id != decision.account_lineage_id:
            raise DurableCollectionError("TRANSITION_DECISION_LINEAGE_MISMATCH")
        expected_time = environment_time_text_v1(transition.environment_time_before)
        if expected_time != environment_time_text_v1(decision.environment_time):
            raise DurableCollectionError("TRANSITION_DECISION_ENVIRONMENT_TIME_MISMATCH")
        key = self._pending_key(decision.account_lineage_id, int(decision.decision_index), expected_time)
        pending = self._pending.pop(key, None)
        if pending is None:
            raise DurableCollectionError("COLLECTION_OBSERVATION_NOT_PERSISTED")
        durable_fact = self.observation_store.get(pending.receipt.logical_id)
        if durable_fact.observation_hash != decision.observation_hash:
            raise DurableCollectionError("DURABLE_OBSERVATION_HASH_MISMATCH")
        assert_w01_observation_identity_v1(decision, durable_fact)
        if bootstrap_state_ref_or_null is not None:
            bootstrap_observation = self.observation_store.get(bootstrap_state_ref_or_null)
            canonical_observation_vectors_v1(bootstrap_observation)
        context = dict(consequence_context or {})
        context.setdefault("permission_status", transition.permission_status)
        context.setdefault("permitted_target_direction", transition.permitted_target_direction)
        context.setdefault("permitted_target_risk", float(transition.permitted_target_risk))
        context.setdefault("target_quantity", float(transition.target_quantity))
        context.setdefault("executed_legs", tuple(dict(leg) if isinstance(leg, Mapping) else leg for leg in transition.execution_legs))
        record = DurableTransitionRecordV1(
            transition_id=f"{sequence_id}::{decision.decision_index}",
            sequence_id=sequence_id,
            account_lineage_id=decision.account_lineage_id,
            decision_index=int(decision.decision_index),
            environment_time_before=expected_time,
            environment_time_after=environment_time_text_v1(transition.environment_time_after),
            pre_account_truth_hash=transition.pre_account_truth_hash,
            post_account_truth_hash=transition.post_account_truth_hash,
            observation_logical_id=pending.receipt.logical_id,
            observation_hash=decision.observation_hash,
            observation_content_sha256=pending.receipt.content_sha256,
            policy_decision_ref=decision.ref,
            science_semantic_version=decision.science_semantic_version,
            policy_generation=str(decision.policy_generation),
            policy_id=decision.policy_id,
            policy_sha256=decision.policy_sha256,
            observation_schema=decision.observation_schema,
            normalizer_id=decision.normalizer_id,
            nominal_direction=decision.nominal_direction,
            nominal_target_risk=float(decision.nominal_target_risk),
            risk_measure_kind=decision.risk_measure_kind,
            behavior_log_mu=float(decision.log_mu),
            rng_stream_id=decision.rng_stream_id,
            rng_position_or_counter=int(decision.rng_position_or_counter),
            decision_environment_time=int(decision.environment_time),
            reward=float(reward),
            discount=float(discount),
            boundary_type=transition.boundary_type,
            mechanical_terminal=bool(transition.mechanical_terminal),
            bootstrap_state_ref_or_null=bootstrap_state_ref_or_null,
            consequence_context=context,
            raw_transition_content_sha256=content_sha256(transition),
        )
        record.validate()
        self.replay_store.put_transition(record)
        return record

    def persist_no_decision_advance(self, transition: CCEnvironmentTransitionV1) -> str:
        """Persist mechanical advance raw-only; never fabricate a decision."""
        transition.validate()
        if transition.policy_decision_ref is not None:
            raise DurableCollectionError("NO_DECISION_ADVANCE_HAS_POLICY_DECISION")
        return self.replay_store.put_raw_advance(transition)

    def finalize_sequence(
        self,
        *,
        sequence_id: str,
        market_lineage_id: str,
        source_classification: str,
        transition_ids: tuple[str, ...],
        chunk_boundary_type: str,
        bootstrap_state_ref_or_null: str | None,
    ) -> DurableSequenceRecordV1:
        if not transition_ids:
            raise DurableCollectionError("SEQUENCE_TRANSITION_IDS_EMPTY")
        transitions = [self.replay_store.get_transition(ref) for ref in transition_ids]
        if any(record.sequence_id != sequence_id for record in transitions):
            raise DurableCollectionError("TRANSITION_SEQUENCE_ID_MISMATCH")
        lineages = {record.account_lineage_id for record in transitions}
        if len(lineages) != 1:
            raise DurableCollectionError("SEQUENCE_CROSS_ACCOUNT_SPLICE")
        indices = [int(record.decision_index) for record in transitions]
        if any(right != left + 1 for left, right in zip(indices, indices[1:])):
            raise DurableCollectionError("SEQUENCE_DECISION_INDEX_DISCONTINUITY")
        sequence_boundary_class = (
            "TERMINAL"
            if boundary_is_terminal_v1(
                transitions[-1].boundary_type, mechanical_terminal=transitions[-1].mechanical_terminal
            )
            else "TRUNCATION"
        )
        if boundary_requires_bootstrap_v1(
            transitions[-1].boundary_type, mechanical_terminal=transitions[-1].mechanical_terminal
        ):
            if not bootstrap_state_ref_or_null:
                raise DurableCollectionError("TRUNCATION_SEQUENCE_REQUIRES_BOOTSTRAP_REF")
            self.observation_store.get(bootstrap_state_ref_or_null)
        else:
            if bootstrap_state_ref_or_null is not None:
                raise DurableCollectionError("TERMINAL_SEQUENCE_MUST_NOT_HAVE_BOOTSTRAP_REF")
        for index, transition in enumerate(transitions):
            if index < len(transitions) - 1:
                if transition.bootstrap_state_ref_or_null is not None:
                    raise DurableCollectionError("NON_FINAL_TRANSITION_HAS_BOOTSTRAP_REF")
            else:
                if transition.bootstrap_state_ref_or_null != bootstrap_state_ref_or_null:
                    raise DurableCollectionError("FINAL_TRANSITION_BOOTSTRAP_REF_MISMATCH")
        if transitions[-1].boundary_type != chunk_boundary_type:
            raise DurableCollectionError("SEQUENCE_BOUNDARY_DOES_NOT_MATCH_FINAL_TRANSITION")
        record = DurableSequenceRecordV1(
            sequence_id=sequence_id,
            account_lineage_id=transitions[0].account_lineage_id,
            science_semantic_version=self.science_semantic_version,
            market_lineage_id=market_lineage_id,
            source_classification=source_classification,
            transition_ids=tuple(transition_ids),
            first_decision_index=int(transitions[0].decision_index),
            last_decision_index=int(transitions[-1].decision_index),
            behavior_policy_identities=tuple(dict.fromkeys(t.policy_id for t in transitions)),
            normalizer_identities=tuple(dict.fromkeys(t.normalizer_id for t in transitions)),
            chunk_boundary_type=chunk_boundary_type,
            bootstrap_state_ref_or_null=bootstrap_state_ref_or_null,
            raw_fact_content_sha256=content_sha256(tuple(t.content_sha256 for t in transitions)),
        )
        record.validate()
        self.replay_store.put_sequence(record)
        return record

    def close(self) -> None:
        self.observation_store.close()
        self.replay_store.close()
