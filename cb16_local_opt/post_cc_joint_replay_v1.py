"""S0-v2 durable joint replay records.

These records are the persistent linkage between decision-time observations,
nominal joint actions, persisted true behaviour ``log_mu`` and their economic
consequences.  They reuse the frozen ``PostCCJointReplaySampleV1`` semantics;
the wrapper adds only durable provenance and validation that the frozen
contract itself does not carry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping

from .cc_experience_wire_r0 import content_sha256
from .cc_runtime_boundary_r0 import BOUNDARIES
from .cc_runtime_wire_r0 import CCPolicyDecisionV1, canonical_sha256
from .post_cc_joint_replay_contract_v1 import PostCCJointReplaySampleV1
from .post_cc_observation_contract_v1 import PostCCObservationFactV1
from .post_cc_observation_fact_v1 import (
    encode_observation_fact_v1,
    observation_content_sha256_v1,
    observation_logical_id_from_identity_v1,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
LOCKED_REPLAY_DIRECTIONS = ("SHORT", "FLAT", "LONG")
VALID_RISK_MEASURE_KINDS = ("point_mass", "continuous_density")
LOG_MU_SOURCE_DECISION_TIME_PERSISTED = "DECISION_TIME_PERSISTED"
_CONSEQUENCE_FORBIDDEN_KEYS = frozenset(
    {
        "nominal_direction",
        "nominal_target_risk",
        "behavior_log_mu",
        "log_mu",
        "log_pi",
    }
)


def _require_nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if out == 0.0 else out


def _require_hex64(name: str, value: Any) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64-hex text")
    return value


def _validate_direction_risk(name: str, direction: str, risk: float, risk_measure_kind: str) -> float:
    if direction not in LOCKED_REPLAY_DIRECTIONS:
        raise ValueError(f"{name}: invalid direction")
    value = _require_finite(name, risk)
    if direction == "FLAT":
        if value != 0.0 or risk_measure_kind != "point_mass":
            raise ValueError(f"{name}: FLAT must be exact risk 0 with point_mass")
    else:
        if not 0.0 < value < 1.0 or risk_measure_kind != "continuous_density":
            raise ValueError(f"{name}: non-FLAT risk must be interior continuous density")
    return value


def _validate_consequence_context(context: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if context is None:
        return None
    if not isinstance(context, Mapping):
        raise ValueError("consequence_context must be a mapping or None")
    overlap = _CONSEQUENCE_FORBIDDEN_KEYS & {str(key) for key in context.keys()}
    if overlap:
        raise ValueError("CONSEQUENCE_CONTEXT_MUST_NOT_REPLACE_NOMINAL_ACTION")
    return dict(context)


@dataclass(frozen=True)
class DurableTransitionRecordV1:
    """Persistent W-02/decision joint record for one replay-admissible transition."""

    transition_id: str
    sequence_id: str
    account_lineage_id: str
    decision_index: int
    environment_time_before: str
    environment_time_after: str
    pre_account_truth_hash: str
    post_account_truth_hash: str
    observation_logical_id: str
    observation_hash: str
    observation_content_sha256: str
    policy_decision_ref: str
    science_semantic_version: str
    policy_generation: str
    policy_id: str
    policy_sha256: str
    observation_schema: str
    normalizer_id: str
    nominal_direction: str
    nominal_target_risk: float
    risk_measure_kind: str
    behavior_log_mu: float
    rng_stream_id: str
    rng_position_or_counter: int
    decision_environment_time: int
    reward: float
    discount: float
    boundary_type: str
    mechanical_terminal: bool
    bootstrap_state_ref_or_null: str | None
    consequence_context: Mapping[str, Any] | None
    raw_transition_content_sha256: str

    def validate(self) -> "DurableTransitionRecordV1":
        for name in (
            "transition_id",
            "sequence_id",
            "account_lineage_id",
            "environment_time_before",
            "environment_time_after",
            "observation_logical_id",
            "policy_decision_ref",
            "science_semantic_version",
            "policy_generation",
            "policy_id",
            "observation_schema",
            "normalizer_id",
            "rng_stream_id",
        ):
            _require_nonempty(name, getattr(self, name))
        _require_hex64("observation_hash", self.observation_hash)
        _require_hex64("observation_content_sha256", self.observation_content_sha256)
        _require_hex64("pre_account_truth_hash", self.pre_account_truth_hash)
        _require_hex64("post_account_truth_hash", self.post_account_truth_hash)
        _require_hex64("policy_sha256", self.policy_sha256)
        _require_hex64("policy_decision_ref", self.policy_decision_ref)
        _require_hex64("raw_transition_content_sha256", self.raw_transition_content_sha256)
        if isinstance(self.decision_index, bool) or int(self.decision_index) < 0:
            raise ValueError("decision_index must be >= 0")
        if isinstance(self.decision_environment_time, bool) or int(self.decision_environment_time) < 0:
            raise ValueError("decision_environment_time must be >= 0")
        if str(self.environment_time_after) <= str(self.environment_time_before):
            raise ValueError("ENVIRONMENT_TIME_MUST_ADVANCE")
        if int(self.decision_environment_time) != int(str(self.environment_time_before)):
            raise ValueError("DECISION_ENVIRONMENT_TIME_MISMATCH")
        if isinstance(self.rng_position_or_counter, bool) or int(self.rng_position_or_counter) < 0:
            raise ValueError("rng_position_or_counter must be >= 0")
        _validate_direction_risk("nominal", self.nominal_direction, self.nominal_target_risk, self.risk_measure_kind)
        _require_finite("behavior_log_mu", self.behavior_log_mu)
        _require_finite("reward", self.reward)
        discount = _require_finite("discount", self.discount)
        if discount < 0.0:
            raise ValueError("discount must be >= 0")
        if self.boundary_type not in BOUNDARIES:
            raise ValueError("BOUNDARY_TYPE_INVALID")
        if not isinstance(self.mechanical_terminal, bool):
            raise ValueError("mechanical_terminal must be bool")
        if self.mechanical_terminal and self.boundary_type != "ECONOMIC_TERMINAL":
            raise ValueError("MECHANICAL_TERMINAL_BOUNDARY_MISMATCH")
        if not isinstance(self.mechanical_terminal, bool):
            raise ValueError("mechanical_terminal must be bool")
        if self.bootstrap_state_ref_or_null is not None:
            _require_nonempty("bootstrap_state_ref_or_null", self.bootstrap_state_ref_or_null)
        _validate_consequence_context(self.consequence_context)
        expected_ref = self.expected_decision_ref()
        if expected_ref != self.policy_decision_ref:
            raise ValueError("POLICY_DECISION_REF_MISMATCH")
        return self

    def expected_decision_ref(self) -> str:
        decision = CCPolicyDecisionV1(
            science_semantic_version=self.science_semantic_version,
            account_lineage_id=self.account_lineage_id,
            decision_index=int(self.decision_index),
            environment_time=int(self.decision_environment_time),
            policy_generation=str(self.policy_generation),
            policy_id=self.policy_id,
            policy_sha256=self.policy_sha256,
            observation_schema=self.observation_schema,
            observation_hash=self.observation_hash,
            normalizer_id=self.normalizer_id,
            nominal_direction=self.nominal_direction,
            nominal_target_risk=float(self.nominal_target_risk),
            log_mu=float(self.behavior_log_mu),
            risk_measure_kind=self.risk_measure_kind,
            rng_stream_id=self.rng_stream_id,
            rng_position_or_counter=int(self.rng_position_or_counter),
        )
        decision.validate()
        return decision.ref

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)


@dataclass(frozen=True)
class DurableSequenceRecordV1:
    """Persistent W-03 sequence index over durable transition identities."""

    sequence_id: str
    account_lineage_id: str
    science_semantic_version: str
    market_lineage_id: str
    source_classification: str
    transition_ids: tuple[str, ...]
    first_decision_index: int
    last_decision_index: int
    behavior_policy_identities: tuple[str, ...]
    normalizer_identities: tuple[str, ...]
    chunk_boundary_type: str
    bootstrap_state_ref_or_null: str | None
    raw_fact_content_sha256: str

    def validate(self) -> "DurableSequenceRecordV1":
        for name in (
            "sequence_id",
            "account_lineage_id",
            "science_semantic_version",
            "market_lineage_id",
            "source_classification",
            "chunk_boundary_type",
        ):
            _require_nonempty(name, getattr(self, name))
        if not self.transition_ids:
            raise ValueError("SEQUENCE_TRANSITION_IDS_EMPTY")
        if len(set(self.transition_ids)) != len(self.transition_ids):
            raise ValueError("SEQUENCE_TRANSITION_IDS_DUPLICATE")
        if any(not isinstance(item, str) or not item for item in self.transition_ids):
            raise ValueError("SEQUENCE_TRANSITION_ID_INVALID")
        if isinstance(self.first_decision_index, bool) or int(self.first_decision_index) < 0:
            raise ValueError("first_decision_index must be >= 0")
        if int(self.last_decision_index) < int(self.first_decision_index):
            raise ValueError("last_decision_index must be >= first_decision_index")
        if not self.behavior_policy_identities or not self.normalizer_identities:
            raise ValueError("SEQUENCE_IDENTITIES_REQUIRED")
        if self.chunk_boundary_type not in BOUNDARIES:
            raise ValueError("SEQUENCE_BOUNDARY_TYPE_INVALID")
        if self.bootstrap_state_ref_or_null is not None:
            _require_nonempty("bootstrap_state_ref_or_null", self.bootstrap_state_ref_or_null)
        _require_hex64("raw_fact_content_sha256", self.raw_fact_content_sha256)
        return self

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)


@dataclass(frozen=True)
class DurableJointReplaySampleV1:
    """Materialized sample with durable-only provenance and persisted true log_mu."""

    sample_id: str
    sequence_id: str
    transition_id: str
    account_lineage_id: str
    decision_index: int
    environment_time: str
    observation: PostCCObservationFactV1
    observation_logical_id: str
    observation_content_sha256: str
    transition_record_sha256: str
    policy_decision_ref: str
    nominal_direction: str
    nominal_target_risk: float
    risk_measure_kind: str
    behavior_log_mu: float
    behavior_policy_generation: str
    behavior_policy_id: str
    behavior_policy_sha256: str
    log_mu_source: str
    reward: float
    discount: float
    boundary_type: str
    mechanical_terminal: bool
    bootstrap_state_ref_or_null: str | None
    sampling_probability_or_weight: float
    target_policy_identity: str
    source_fact_hashes: tuple[str, ...]
    consequence_context: Mapping[str, Any] | None = None

    def validate(self) -> "DurableJointReplaySampleV1":
        for name in (
            "sample_id",
            "sequence_id",
            "transition_id",
            "account_lineage_id",
            "environment_time",
            "observation_logical_id",
            "behavior_policy_generation",
            "behavior_policy_id",
            "log_mu_source",
            "boundary_type",
            "target_policy_identity",
        ):
            _require_nonempty(name, getattr(self, name))
        if isinstance(self.decision_index, bool) or int(self.decision_index) < 0:
            raise ValueError("decision_index must be >= 0")
        self.observation.validate()
        if self.observation.account_lineage_id != self.account_lineage_id:
            raise ValueError("ACCOUNT_LINEAGE_MISMATCH")
        if int(self.observation.decision_index) != int(self.decision_index):
            raise ValueError("DECISION_INDEX_MISMATCH")
        if str(self.observation.environment_time) != str(self.environment_time):
            raise ValueError("ENVIRONMENT_TIME_MISMATCH")
        expected_logical_id = observation_logical_id_from_identity_v1(
            account_lineage_id=self.account_lineage_id,
            decision_index=int(self.decision_index),
            environment_time=self.environment_time,
            observation_hash=self.observation.observation_hash,
        )
        if self.observation_logical_id != expected_logical_id:
            raise ValueError("OBSERVATION_LOGICAL_ID_MISMATCH")
        if self.observation_content_sha256 != observation_content_sha256_v1(self.observation):
            raise ValueError("OBSERVATION_CONTENT_HASH_MISMATCH")
        _validate_direction_risk("nominal", self.nominal_direction, self.nominal_target_risk, self.risk_measure_kind)
        _require_finite("behavior_log_mu", self.behavior_log_mu)
        if self.log_mu_source != LOG_MU_SOURCE_DECISION_TIME_PERSISTED:
            raise ValueError("LOG_MU_NOT_DECISION_TIME_PERSISTED")
        if self.boundary_type not in BOUNDARIES:
            raise ValueError("BOUNDARY_TYPE_INVALID")
        if not isinstance(self.mechanical_terminal, bool):
            raise ValueError("mechanical_terminal must be bool")
        if self.mechanical_terminal and self.boundary_type != "ECONOMIC_TERMINAL":
            raise ValueError("MECHANICAL_TERMINAL_BOUNDARY_MISMATCH")
        discount = _require_finite("discount", self.discount)
        if discount < 0.0:
            raise ValueError("discount must be >= 0")
        reward = _require_finite("reward", self.reward)
        weight = _require_finite("sampling_probability_or_weight", self.sampling_probability_or_weight)
        if not 0.0 < weight <= 1.0:
            raise ValueError("sampling_probability_or_weight must be in (0,1]")
        for name in ("transition_record_sha256", "policy_decision_ref", "behavior_policy_sha256"):
            _require_hex64(name, getattr(self, name))
        if not self.source_fact_hashes:
            raise ValueError("SOURCE_FACT_HASHES_REQUIRED")
        for item in self.source_fact_hashes:
            _require_hex64("source_fact_hashes[]", item)
        if self.bootstrap_state_ref_or_null is not None:
            _require_nonempty("bootstrap_state_ref_or_null", self.bootstrap_state_ref_or_null)
        _validate_consequence_context(self.consequence_context)
        return self

    @property
    def sample_content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)

    @property
    def behavior_policy_identity(self) -> str:
        return f"{self.behavior_policy_id}:{self.behavior_policy_sha256}"

    def as_contract(self) -> PostCCJointReplaySampleV1:
        self.validate()
        return PostCCJointReplaySampleV1(
            sequence_id=self.sequence_id,
            transition_ref=self.transition_id,
            account_lineage_id=self.account_lineage_id,
            decision_index=int(self.decision_index),
            environment_time=self.environment_time,
            observation=self.observation,
            nominal_direction=self.nominal_direction,
            nominal_target_risk=float(self.nominal_target_risk),
            risk_measure_kind=self.risk_measure_kind,
            behavior_log_mu=float(self.behavior_log_mu),
            behavior_policy_identity=self.behavior_policy_identity,
            behavior_generation=self.behavior_policy_generation,
            reward=float(self.reward),
            discount=float(self.discount),
            boundary_type=self.boundary_type,
            bootstrap_state_ref_or_null=self.bootstrap_state_ref_or_null,
            sampling_probability_or_weight=float(self.sampling_probability_or_weight),
            target_policy_identity=self.target_policy_identity,
            source_fact_hashes=tuple(self.source_fact_hashes),
            consequence_context=None if self.consequence_context is None else dict(self.consequence_context),
        ).validate()


def materialized_sample_id_v1(*, sequence_id: str, transition_id: str, transition_record_sha256: str) -> str:
    return canonical_sha256(
        {
            "sequence_id": sequence_id,
            "transition_id": transition_id,
            "transition_record_sha256": transition_record_sha256,
        }
    )
