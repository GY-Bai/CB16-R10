"""CB16 R11 S1 durable decision-interval credit adapter.

The S1 delayed-consequence task gives the reward to an early decision through a
later no-decision mechanical advance.  The adapter persists a validated
decision-level view:

    early decision transition (reward 0, COMPUTE_CHUNK truncation)
      -> durable restart
      -> raw no-decision advance chain (never rewritten, no fabricated action)
      -> DecisionIntervalCreditV1 view with the complete arithmetic consequence

The view is content-addressed and re-derived from durable state on every
materialization, so restart cannot lose the delayed credit and no rollout-side
object is ever required as training truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .cc_experience_wire_r0 import content_sha256
from .cc_runtime_boundary_r0 import BOUNDARIES
from .post_cc_joint_batch_v1 import boundary_requires_bootstrap_v1
from .post_cc_joint_replay_v1 import DurableJointReplaySampleV1
from .post_cc_observation_fact_v1 import canonical_observation_vectors_v1, observation_content_sha256_v1
from .post_cc_durable_collection_v1 import environment_time_text_v1
from .post_cc_observation_store_v1 import ImmutableContentStore, ObservationStoreV1
from .post_cc_replay_materializer_v1 import MaterializedReplayV1, ReplayStoreV1

CREDIT_SCHEMA_V1 = "CB16_R11_POST_CC_S1_DECISION_INTERVAL_CREDIT_V1"
CREDIT_NAMESPACE_V1 = "replay_credit_views"
RAW_ADVANCE_NAMESPACE_V1 = "replay_raw_advances"
CREDIT_TOLERANCE_V1 = 1e-9

_FORBIDDEN_CONTEXT_KEYS = frozenset({"nominal_direction", "nominal_target_risk", "behavior_log_mu", "log_mu", "log_pi"})


class CreditAdapterError(RuntimeError):
    pass


class CreditCorruption(CreditAdapterError):
    pass


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CreditAdapterError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise CreditAdapterError(f"{name} must be a finite number")
    return out


def credit_logical_id_v1(sequence_id: str, decision_index: int) -> str:
    return f"credit|{sequence_id}|{int(decision_index)}"


def raw_advance_logical_id_v1(*, lineage: str, decision_index: int, environment_time_before: Any) -> str:
    """Mirror ReplayStoreV1.put_raw_advance logical-id derivation."""
    return "raw|{}|{}|{}".format(str(lineage), int(decision_index), environment_time_before)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class DecisionIntervalCreditV1:
    schema: str
    credit_id: str
    sequence_id: str
    lineage: str
    decision_index: int
    decision_transition_id: str
    decision_transition_content_sha256: str
    boundary_observation_logical_id: str
    boundary_observation_content_sha256: str
    raw_advance_ids: tuple[str, ...]
    raw_advance_content_sha256s: tuple[str, ...]
    environment_time_before: str
    environment_time_after: str
    decision_equity_before: float
    final_equity: float
    reward: float
    discount: float
    boundary_type: str
    mechanical_terminal: bool
    reward_reference_equity: float
    source_classification: str

    def validate(self) -> "DecisionIntervalCreditV1":
        if self.schema != CREDIT_SCHEMA_V1:
            raise CreditCorruption("CREDIT_SCHEMA_MISMATCH")
        for name in (
            "credit_id",
            "sequence_id",
            "lineage",
            "decision_transition_id",
            "decision_transition_content_sha256",
            "boundary_observation_logical_id",
            "boundary_observation_content_sha256",
            "environment_time_before",
            "environment_time_after",
            "source_classification",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CreditCorruption(f"CREDIT_FIELD_EMPTY:{name}")
        if not self.raw_advance_ids or len(self.raw_advance_ids) != len(self.raw_advance_content_sha256s):
            raise CreditCorruption("CREDIT_RAW_ADVANCE_CARDINALITY")
        if isinstance(self.decision_index, bool) or int(self.decision_index) < 0:
            raise CreditCorruption("CREDIT_DECISION_INDEX_INVALID")
        if str(self.environment_time_after) <= str(self.environment_time_before):
            raise CreditCorruption("CREDIT_ENVIRONMENT_TIME_NOT_ADVANCING")
        if self.boundary_type not in BOUNDARIES:
            raise CreditCorruption("CREDIT_BOUNDARY_INVALID")
        if not isinstance(self.mechanical_terminal, bool):
            raise CreditCorruption("CREDIT_TERMINAL_FLAG_INVALID")
        if self.mechanical_terminal and self.boundary_type != "ECONOMIC_TERMINAL":
            raise CreditCorruption("CREDIT_TERMINAL_BOUNDARY_MISMATCH")
        reference = _require_finite("reward_reference_equity", self.reward_reference_equity)
        if reference <= 0.0:
            raise CreditCorruption("CREDIT_REFERENCE_EQUITY_INVALID")
        before = _require_finite("decision_equity_before", self.decision_equity_before)
        after = _require_finite("final_equity", self.final_equity)
        reward = _require_finite("reward", self.reward)
        expected_reward = (after - before) / reference
        if abs(reward - expected_reward) > CREDIT_TOLERANCE_V1:
            raise CreditCorruption("CREDIT_REWARD_DOES_NOT_MATCH_EQUITY_DELTA")
        discount = _require_finite("discount", self.discount)
        if discount < 0.0:
            raise CreditCorruption("CREDIT_DISCOUNT_INVALID")
        if self.credit_id != stable_credit_payload_sha256_v1(self):
            raise CreditCorruption("CREDIT_ID_MISMATCH")
        return self

    def payload(self) -> Mapping[str, Any]:
        return asdict(self)

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)


def stable_credit_payload_sha256_v1(credit: DecisionIntervalCreditV1) -> str:
    payload = asdict(credit)
    payload.pop("credit_id", None)
    return content_sha256(payload)


def _load_raw_advance_v1(root: str, logical_id: str) -> tuple[Mapping[str, Any], str]:
    store = ImmutableContentStore(root, RAW_ADVANCE_NAMESPACE_V1)
    payload = store.get_bytes(logical_id)
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, Mapping):
        raise CreditCorruption("RAW_ADVANCE_BYTES_INVALID")
    return decoded, hashlib.sha256(payload).hexdigest()


def validate_credit_against_durable_v1(root: str, credit: DecisionIntervalCreditV1) -> Mapping[str, Any]:
    """Re-derive the credit from durable truth; fail closed on any gap."""
    credit.validate()
    replay_store = ReplayStoreV1(root)
    transition = replay_store.get_transition(credit.decision_transition_id)
    if transition.content_sha256 != credit.decision_transition_content_sha256:
        raise CreditCorruption("CREDIT_DECISION_TRANSITION_HASH_MISMATCH")
    if transition.sequence_id != credit.sequence_id:
        raise CreditCorruption("CREDIT_SEQUENCE_MISMATCH")
    if transition.account_lineage_id != credit.lineage:
        raise CreditCorruption("CREDIT_LINEAGE_MISMATCH")
    if int(transition.decision_index) != int(credit.decision_index):
        raise CreditCorruption("CREDIT_DECISION_INDEX_MISMATCH")
    if transition.environment_time_before != credit.environment_time_before:
        raise CreditCorruption("CREDIT_TIME_BEFORE_MISMATCH")
    if transition.bootstrap_state_ref_or_null != credit.boundary_observation_logical_id:
        raise CreditCorruption("CREDIT_BOUNDARY_OBSERVATION_REF_MISMATCH")

    sequence = replay_store.get_sequence(credit.sequence_id)
    if sequence.bootstrap_state_ref_or_null != credit.boundary_observation_logical_id:
        raise CreditCorruption("CREDIT_SEQUENCE_BOOTSTRAP_REF_MISMATCH")

    observation_store = ObservationStoreV1(root)
    fact = observation_store.get(credit.boundary_observation_logical_id)
    canonical_observation_vectors_v1(fact)
    if observation_content_sha256_v1(fact) != credit.boundary_observation_content_sha256:
        raise CreditCorruption("CREDIT_BOUNDARY_OBSERVATION_CONTENT_MISMATCH")

    previous_post_hash = transition.post_account_truth_hash
    previous_time = transition.environment_time_after
    final_raw: Mapping[str, Any] | None = None
    for raw_id, expected_raw_sha in zip(credit.raw_advance_ids, credit.raw_advance_content_sha256s):
        raw, actual_raw_sha = _load_raw_advance_v1(root, raw_id)
        if actual_raw_sha != expected_raw_sha:
            raise CreditCorruption("CREDIT_RAW_ADVANCE_HASH_MISMATCH")
        if raw.get("policy_decision_ref") is not None:
            raise CreditCorruption("CREDIT_RAW_ADVANCE_MUST_NOT_HAVE_POLICY_DECISION")
        if raw.get("account_lineage_id") != credit.lineage:
            raise CreditCorruption("CREDIT_RAW_ADVANCE_LINEAGE_MISMATCH")
        if raw.get("pre_account_truth_hash") != previous_post_hash:
            raise CreditCorruption("CREDIT_RAW_ADVANCE_ACCOUNT_TRUTH_DISCONTINUITY")
        raw_time_before_text = environment_time_text_v1(int(raw.get("environment_time_before")))
        if raw_time_before_text != str(previous_time):
            raise CreditCorruption("CREDIT_RAW_ADVANCE_TIME_DISCONTINUITY")
        previous_post_hash = raw.get("post_account_truth_hash")
        previous_time = environment_time_text_v1(int(raw.get("environment_time_after")))
        final_raw = raw
    if final_raw is None:
        raise CreditCorruption("CREDIT_RAW_ADVANCE_MISSING")
    if final_raw.get("boundary_type") != credit.boundary_type:
        raise CreditCorruption("CREDIT_RAW_ADVANCE_BOUNDARY_MISMATCH")
    if bool(final_raw.get("mechanical_terminal")) != bool(credit.mechanical_terminal):
        raise CreditCorruption("CREDIT_RAW_ADVANCE_TERMINAL_FLAG_MISMATCH")
    if str(previous_time) != str(credit.environment_time_after):
        raise CreditCorruption("CREDIT_TIME_AFTER_MISMATCH")
    final_equity = _require_finite("raw_post_equity", final_raw.get("post_equity"))
    if abs(final_equity - float(credit.final_equity)) > CREDIT_TOLERANCE_V1:
        raise CreditCorruption("CREDIT_FINAL_EQUITY_MISMATCH")
    return {
        "decision_transition_content_sha256": transition.content_sha256,
        "raw_advance_count": len(credit.raw_advance_ids),
        "final_equity": final_equity,
    }


def record_decision_interval_credit_v1(
    *,
    root: str,
    sequence_id: str,
    decision_transition_id: str,
    decision_transition_content_sha256: str,
    decision_equity_before: float,
    final_equity: float,
    raw_advance_ids: Sequence[str],
    boundary_type: str,
    mechanical_terminal: bool,
    reward_reference_equity: float,
    discount: float,
    lineage: str,
    decision_index: int,
    source_classification: str,
) -> DecisionIntervalCreditV1:
    replay_store = ReplayStoreV1(root)
    sequence = replay_store.get_sequence(sequence_id)
    boundary_ref = sequence.bootstrap_state_ref_or_null
    if not boundary_ref:
        raise CreditCorruption("CREDIT_REQUIRES_DURABLE_BOUNDARY_OBSERVATION")
    observation_store = ObservationStoreV1(root)
    boundary_fact = observation_store.get(boundary_ref)
    raw_ids = tuple(str(item) for item in raw_advance_ids)
    raw_hashes = tuple(_load_raw_advance_v1(root, item)[1] for item in raw_ids)
    reference = float(reward_reference_equity)
    before = float(decision_equity_before)
    after = float(final_equity)
    draft = DecisionIntervalCreditV1(
        schema=CREDIT_SCHEMA_V1,
        credit_id="",
        sequence_id=str(sequence_id),
        lineage=str(lineage),
        decision_index=int(decision_index),
        decision_transition_id=str(decision_transition_id),
        decision_transition_content_sha256=str(decision_transition_content_sha256),
        boundary_observation_logical_id=str(boundary_ref),
        boundary_observation_content_sha256=observation_content_sha256_v1(boundary_fact),
        raw_advance_ids=raw_ids,
        raw_advance_content_sha256s=raw_hashes,
        environment_time_before=str(replay_store.get_transition(decision_transition_id).environment_time_before),
        environment_time_after=environment_time_text_v1(
            int(_load_raw_advance_v1(root, raw_ids[-1])[0].get("environment_time_after"))
        ),
        decision_equity_before=before,
        final_equity=after,
        reward=(after - before) / reference,
        discount=float(discount),
        boundary_type=str(boundary_type),
        mechanical_terminal=bool(mechanical_terminal),
        reward_reference_equity=reference,
        source_classification=str(source_classification),
    )
    credit = replace(draft, credit_id=stable_credit_payload_sha256_v1(draft))
    credit.validate()
    validate_credit_against_durable_v1(root, credit)
    store = ImmutableContentStore(root, CREDIT_NAMESPACE_V1)
    payload = _canonical_json_bytes(asdict(credit))
    store.put_bytes(credit_logical_id_v1(sequence_id, decision_index), payload, expected_sha256=hashlib.sha256(payload).hexdigest())
    return credit


def load_credit_v1(root: str, sequence_id: str, decision_index: int) -> DecisionIntervalCreditV1 | None:
    store = ImmutableContentStore(root, CREDIT_NAMESPACE_V1)
    logical_id = credit_logical_id_v1(sequence_id, decision_index)
    if not store.has(logical_id):
        return None
    payload = store.get_bytes(logical_id)
    raw = json.loads(payload.decode("utf-8"))
    for name in ("raw_advance_ids", "raw_advance_content_sha256s"):
        raw[name] = tuple(raw.get(name, ()))
    credit = DecisionIntervalCreditV1(**raw)
    credit.validate()
    validate_credit_against_durable_v1(root, credit)
    if hashlib.sha256(payload).hexdigest() != store.content_sha256_for(logical_id):
        raise CreditCorruption("CREDIT_STORE_HASH_MISMATCH")
    return credit


def apply_credit_views_v1(
    root: str,
    materialized: MaterializedReplayV1,
) -> tuple[tuple[DurableJointReplaySampleV1, ...], dict[str, Any]]:
    """Replace credited samples' reward/boundary with the validated view.

    Credited sequences become terminal views, so their truncation bootstrap is
    intentionally dropped: the durable raw chain already contains the complete
    downstream consequence.
    """
    materialized.validate()
    credits_by_sequence: dict[str, DecisionIntervalCreditV1] = {}
    updated: list[DurableJointReplaySampleV1] = []
    bootstrap_map: dict[str, Any] = {}
    for sample in materialized.samples:
        credit = credits_by_sequence.get(sample.sequence_id)
        if credit is None:
            credit = load_credit_v1(root, sample.sequence_id, int(sample.decision_index))
            if credit is not None:
                credits_by_sequence[sample.sequence_id] = credit
        if credit is None:
            updated.append(sample)
            continue
        if sample.transition_record_sha256 != credit.decision_transition_content_sha256:
            raise CreditCorruption("CREDIT_VIEW_BOUND_TO_OTHER_TRANSITION")
        if sample.bootstrap_state_ref_or_null != credit.boundary_observation_logical_id:
            raise CreditCorruption("CREDIT_VIEW_BOOTSTRAP_REF_MISMATCH")
        if boundary_requires_bootstrap_v1(credit.boundary_type, mechanical_terminal=credit.mechanical_terminal):
            raise CreditCorruption("CREDIT_VIEW_MUST_BE_TERMINAL")
        context = dict(sample.consequence_context or {})
        if _FORBIDDEN_CONTEXT_KEYS & set(context):
            raise CreditCorruption("CREDIT_CONTEXT_REPLACES_NOMINAL_ACTION")
        context["credit_view_id"] = credit.credit_id
        context["credit_raw_advance_content_sha256s"] = list(credit.raw_advance_content_sha256s)
        updated.append(
            replace(
                sample,
                reward=float(credit.reward),
                discount=float(credit.discount),
                boundary_type=str(credit.boundary_type),
                mechanical_terminal=bool(credit.mechanical_terminal),
                bootstrap_state_ref_or_null=None,
                source_fact_hashes=tuple(sample.source_fact_hashes)
                + (credit.content_sha256,)
                + tuple(credit.raw_advance_content_sha256s),
                consequence_context=context,
            ).validate()
        )
    for sequence_id, fact in materialized.bootstrap_observations_by_sequence.items():
        if sequence_id in credits_by_sequence:
            continue
        bootstrap_map[sequence_id] = fact
    return tuple(updated), bootstrap_map


def assert_sample_log_mu_bound_to_durable_v1(root: str, sample: DurableJointReplaySampleV1) -> None:
    """Fail closed if a sample's nominal action / true log_mu is not durable-bound."""
    sample.validate()
    record = ReplayStoreV1(root).get_transition(sample.transition_id)
    if record.content_sha256 != sample.transition_record_sha256:
        raise CreditCorruption("SAMPLE_TRANSITION_RECORD_HASH_MISMATCH")
    if float(record.behavior_log_mu) != float(sample.behavior_log_mu):
        raise CreditCorruption("SAMPLE_LOG_MU_NOT_BOUND_TO_DECISION_TIME_PERSISTED_VALUE")
    if record.policy_decision_ref != sample.policy_decision_ref:
        raise CreditCorruption("SAMPLE_DECISION_REF_MISMATCH")
    if record.nominal_direction != sample.nominal_direction:
        raise CreditCorruption("SAMPLE_NOMINAL_DIRECTION_MISMATCH")
    if float(record.nominal_target_risk) != float(sample.nominal_target_risk):
        raise CreditCorruption("SAMPLE_NOMINAL_RISK_MISMATCH")
    return None


def verify_credits_for_sequences_v1(root: str, sequence_ids: Sequence[str]) -> Mapping[str, int]:
    """Restart-safe validation sweep used by the qualification audit."""
    replay_store = ReplayStoreV1(root)
    checked = 0
    for sequence_id in sequence_ids:
        sequence = replay_store.get_sequence(sequence_id)
        for decision_index in range(int(sequence.first_decision_index), int(sequence.last_decision_index) + 1):
            credit = load_credit_v1(root, sequence_id, decision_index)
            if credit is not None:
                checked += 1
    return {"credits_verified": checked}
