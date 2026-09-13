"""S0-v2 persistent replay materializer.

Materialization reads only durable transition/sequence/observation stores.  It
never accepts collector-private rollout records as training truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cc_experience_wire_r0 import canonical_json_bytes, content_sha256
from .post_cc_joint_batch_v1 import (
    BOUNDARY_CLASS_CONTINUE,
    BOUNDARY_CLASS_TERMINAL,
    MATERIALIZER_CONTRACT_ID,
    DurableBatchProvenanceV1,
    JointActionBatchV1,
    build_joint_batch_v1,
    classify_boundary_v1,
)
from .post_cc_joint_replay_v1 import (
    DurableJointReplaySampleV1,
    DurableSequenceRecordV1,
    DurableTransitionRecordV1,
    LOG_MU_SOURCE_DECISION_TIME_PERSISTED,
    materialized_sample_id_v1,
)
from .post_cc_observation_contract_v1 import PostCCObservationFactV1
from .post_cc_observation_fact_v1 import (
    canonical_observation_vectors_v1,
    observation_content_sha256_v1,
)
from .post_cc_observation_store_v1 import (
    ImmutableContentStore,
    ObservationStoreCorruption,
    ObservationStoreV1,
)


class ReplayStoreError(RuntimeError):
    pass


class ReplaySemanticConflict(ReplayStoreError):
    pass


class ReplayCorruption(ReplayStoreError):
    pass


def _read_replay_bytes(store: ImmutableContentStore, logical_id: str) -> bytes:
    try:
        return store.get_bytes(logical_id)
    except ObservationStoreCorruption as exc:
        raise ReplayCorruption("REPLAY_STORE_CONTENT_CORRUPTION") from exc


def _decode_mapping(payload: bytes, code: str) -> dict[str, Any]:
    try:
        raw = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayCorruption(code) from exc
    if not isinstance(raw, dict):
        raise ReplayCorruption(code)
    return raw


class ReplayStoreV1:
    """Durable transition/sequence/raw-advance store."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self._transitions = ImmutableContentStore(self.root, "replay_transitions")
        self._sequences = ImmutableContentStore(self.root, "replay_sequences")
        self._raw_advances = ImmutableContentStore(self.root, "replay_raw_advances")
        self._manifests = ImmutableContentStore(self.root, "replay_manifests")

    def put_transition(self, record: DurableTransitionRecordV1) -> str:
        record.validate()
        payload = canonical_json_bytes(asdict(record))
        return self._transitions.put_bytes(record.transition_id, payload)

    def get_transition(self, transition_id: str) -> DurableTransitionRecordV1:
        payload = _read_replay_bytes(self._transitions, transition_id)
        raw = _decode_mapping(payload, "TRANSITION_BYTES_INVALID")
        try:
            record = DurableTransitionRecordV1(**raw)
        except TypeError as exc:
            raise ReplayCorruption("TRANSITION_FIELDS_INVALID") from exc
        record.validate()
        if record.content_sha256 != hashlib.sha256(payload).hexdigest():
            raise ReplayCorruption("TRANSITION_CONTENT_HASH_MISMATCH")
        return record

    def put_sequence(self, record: DurableSequenceRecordV1) -> str:
        record.validate()
        payload = canonical_json_bytes(asdict(record))
        return self._sequences.put_bytes(record.sequence_id, payload)

    def get_sequence(self, sequence_id: str) -> DurableSequenceRecordV1:
        payload = _read_replay_bytes(self._sequences, sequence_id)
        raw = _decode_mapping(payload, "SEQUENCE_BYTES_INVALID")
        raw["transition_ids"] = tuple(raw.get("transition_ids", ()))
        raw["behavior_policy_identities"] = tuple(raw.get("behavior_policy_identities", ()))
        raw["normalizer_identities"] = tuple(raw.get("normalizer_identities", ()))
        try:
            record = DurableSequenceRecordV1(**raw)
        except TypeError as exc:
            raise ReplayCorruption("SEQUENCE_FIELDS_INVALID") from exc
        record.validate()
        if record.content_sha256 != hashlib.sha256(payload).hexdigest():
            raise ReplayCorruption("SEQUENCE_CONTENT_HASH_MISMATCH")
        return record

    def put_raw_advance(self, transition: Any) -> str:
        """Persist a no-decision mechanical advance as raw-only truth."""
        payload = asdict(transition)
        if payload.get("policy_decision_ref") is not None:
            raise ValueError("RAW_ADVANCE_MUST_HAVE_NO_POLICY_DECISION")
        logical_id = "raw|{}|{}|{}".format(
            payload.get("account_lineage_id"),
            payload.get("decision_index"),
            payload.get("environment_time_before"),
        )
        return self._raw_advances.put_bytes(logical_id, canonical_json_bytes(payload))

    def get_raw_advance(self, logical_id: str) -> Mapping[str, Any]:
        payload = _read_replay_bytes(self._raw_advances, logical_id)
        return _decode_mapping(payload, "RAW_ADVANCE_BYTES_INVALID")

    def put_manifest(self, manifest: "MaterializationManifestV1") -> str:
        manifest.validate()
        payload = canonical_json_bytes(asdict(manifest))
        return self._manifests.put_bytes(manifest.manifest_id, payload)

    def get_manifest(self, manifest_id: str) -> "MaterializationManifestV1":
        payload = _read_replay_bytes(self._manifests, manifest_id)
        raw = _decode_mapping(payload, "MANIFEST_BYTES_INVALID")
        for name in ("ordered_sample_hashes", "ordered_transition_hashes", "ordered_observation_hashes"):
            raw[name] = tuple(raw.get(name, ()))
        try:
            manifest = MaterializationManifestV1(**raw)
        except TypeError as exc:
            raise ReplayCorruption("MANIFEST_FIELDS_INVALID") from exc
        manifest.validate()
        if manifest.manifest_sha256 != hashlib.sha256(payload).hexdigest():
            raise ReplayCorruption("MANIFEST_HASH_MISMATCH")
        return manifest

    def count_transitions(self) -> int:
        return self._transitions.count()

    def count_sequences(self) -> int:
        return self._sequences.count()

    def count_raw_advances(self) -> int:
        return self._raw_advances.count()

    def verify_all(self) -> dict[str, int]:
        return {
            "transitions": self._transitions.verify_all(),
            "sequences": self._sequences.verify_all(),
            "raw_advances": self._raw_advances.verify_all(),
            "manifests": self._manifests.verify_all(),
        }

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class MaterializationManifestV1:
    manifest_id: str
    sequence_id: str
    materializer_contract_id: str
    target_policy_identity: str
    source_store_root_identity: str
    ordered_sample_hashes: tuple[str, ...]
    ordered_transition_hashes: tuple[str, ...]
    ordered_observation_hashes: tuple[str, ...]
    bootstrap_state_ref_or_null: str | None
    bootstrap_observation_content_sha256: str | None
    restart_verified: bool

    def validate(self) -> "MaterializationManifestV1":
        for name in (
            "manifest_id",
            "sequence_id",
            "materializer_contract_id",
            "target_policy_identity",
            "source_store_root_identity",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if self.materializer_contract_id != MATERIALIZER_CONTRACT_ID:
            raise ValueError("MATERIALIZER_CONTRACT_ID_MISMATCH")
        if not self.ordered_sample_hashes or len(self.ordered_sample_hashes) != len(self.ordered_transition_hashes):
            raise ValueError("MANIFEST_ORDERED_HASHES_INVALID")
        if len(self.ordered_observation_hashes) != len(self.ordered_sample_hashes):
            raise ValueError("MANIFEST_OBSERVATION_HASH_CARDINALITY")
        if len(set(self.ordered_sample_hashes)) != len(self.ordered_sample_hashes):
            raise ValueError("MANIFEST_SAMPLE_HASH_DUPLICATE")
        if self.bootstrap_state_ref_or_null is None and self.bootstrap_observation_content_sha256 is not None:
            raise ValueError("MANIFEST_BOOTSTRAP_INCONSISTENT")
        if self.bootstrap_state_ref_or_null is not None and self.bootstrap_observation_content_sha256 is None:
            raise ValueError("MANIFEST_BOOTSTRAP_INCONSISTENT")
        return self

    @property
    def manifest_sha256(self) -> str:
        self.validate()
        return content_sha256(self)

    @property
    def stable_materialization_id(self) -> str:
        """Identity that is unchanged by the restart-verification flag."""
        self.validate()
        return self.manifest_id


@dataclass(frozen=True)
class MaterializedReplayV1:
    manifest: MaterializationManifestV1
    samples: tuple[DurableJointReplaySampleV1, ...]
    bootstrap_observations_by_sequence: Mapping[str, PostCCObservationFactV1]

    def validate(self) -> "MaterializedReplayV1":
        self.manifest.validate()
        if len(self.samples) != len(self.manifest.ordered_sample_hashes):
            raise ValueError("MATERIALIZED_SAMPLE_CARDINALITY")
        if tuple(sample.sample_content_sha256 for sample in self.samples) != tuple(self.manifest.ordered_sample_hashes):
            raise ValueError("MATERIALIZED_SAMPLE_HASH_MISMATCH")
        expected_sequences = {sample.sequence_id for sample in self.samples}
        if set(self.bootstrap_observations_by_sequence) - expected_sequences:
            raise ValueError("MATERIALIZED_BOOTSTRAP_FOREIGN_SEQUENCE")
        return self

    def to_batch(self) -> JointActionBatchV1:
        self.validate()
        provenance = DurableBatchProvenanceV1(
            materializer_contract_id=MATERIALIZER_CONTRACT_ID,
            materialization_id=self.manifest.manifest_id,
            materialization_manifest_sha256=self.manifest.manifest_sha256,
            source_store_root_identity=self.manifest.source_store_root_identity,
            restart_verified=bool(self.manifest.restart_verified),
            durable_replay_only=True,
            collector_private_records_used=False,
        )
        return build_joint_batch_v1(
            samples=self.samples,
            provenance=provenance,
            target_policy_identity=self.manifest.target_policy_identity,
            bootstrap_observations_by_sequence=self.bootstrap_observations_by_sequence,
        )


def _store_root_identity(root: Path) -> str:
    return hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()


class ReplayMaterializerV1:
    """Rebuild ordered joint replay samples solely from durable state."""

    def __init__(
        self,
        root: str | Path,
        *,
        observation_store: ObservationStoreV1 | None = None,
        replay_store: ReplayStoreV1 | None = None,
    ):
        self.root = Path(root).resolve()
        self.observation_store = observation_store or ObservationStoreV1(self.root)
        self.replay_store = replay_store or ReplayStoreV1(self.root)
        self.source_store_root_identity = _store_root_identity(self.root)

    @classmethod
    def from_durable_state_v1(cls, root: str | Path) -> "ReplayMaterializerV1":
        """Fresh context constructor used by the restart sentinel."""
        return cls(root)

    def _materialize_sample(
        self,
        *,
        sequence: DurableSequenceRecordV1,
        transition_record: DurableTransitionRecordV1,
        previous: DurableTransitionRecordV1 | None,
        bootstrap_state_ref_or_null: str | None,
        sampling_probability_or_weight: float,
        target_policy_identity: str,
    ) -> DurableJointReplaySampleV1:
        if transition_record.sequence_id != sequence.sequence_id:
            raise ReplayCorruption("TRANSITION_SEQUENCE_MISMATCH")
        if transition_record.account_lineage_id != sequence.account_lineage_id:
            raise ReplayCorruption("TRANSITION_LINEAGE_MISMATCH")
        if previous is not None:
            if previous.environment_time_after != transition_record.environment_time_before:
                raise ReplayCorruption("TRANSITION_ENVIRONMENT_TIME_DISCONTINUITY")
            if previous.post_account_truth_hash != transition_record.pre_account_truth_hash:
                raise ReplayCorruption("TRANSITION_ACCOUNT_TRUTH_DISCONTINUITY")
            if int(transition_record.decision_index) != int(previous.decision_index) + 1:
                raise ReplayCorruption("TRANSITION_DECISION_INDEX_DISCONTINUITY")
        fact = self.observation_store.get(transition_record.observation_logical_id)
        if fact.observation_hash != transition_record.observation_hash:
            raise ReplayCorruption("TRANSITION_OBSERVATION_HASH_MISMATCH")
        if fact.observation_schema != transition_record.observation_schema:
            raise ReplayCorruption("TRANSITION_OBSERVATION_SCHEMA_MISMATCH")
        if observation_content_sha256_v1(fact) != transition_record.observation_content_sha256:
            raise ReplayCorruption("TRANSITION_OBSERVATION_CONTENT_HASH_MISMATCH")
        try:
            canonical_observation_vectors_v1(fact)
        except ValueError as exc:
            raise ReplayCorruption("TRANSITION_OBSERVATION_NOT_CANONICAL") from exc
        sample = DurableJointReplaySampleV1(
            sample_id=materialized_sample_id_v1(
                sequence_id=sequence.sequence_id,
                transition_id=transition_record.transition_id,
                transition_record_sha256=transition_record.content_sha256,
            ),
            sequence_id=sequence.sequence_id,
            transition_id=transition_record.transition_id,
            account_lineage_id=transition_record.account_lineage_id,
            decision_index=int(transition_record.decision_index),
            environment_time=transition_record.environment_time_before,
            observation=fact,
            observation_logical_id=transition_record.observation_logical_id,
            observation_content_sha256=transition_record.observation_content_sha256,
            transition_record_sha256=transition_record.content_sha256,
            policy_decision_ref=transition_record.policy_decision_ref,
            nominal_direction=transition_record.nominal_direction,
            nominal_target_risk=float(transition_record.nominal_target_risk),
            risk_measure_kind=transition_record.risk_measure_kind,
            behavior_log_mu=float(transition_record.behavior_log_mu),
            behavior_policy_generation=transition_record.policy_generation,
            behavior_policy_id=transition_record.policy_id,
            behavior_policy_sha256=transition_record.policy_sha256,
            log_mu_source=LOG_MU_SOURCE_DECISION_TIME_PERSISTED,
            reward=float(transition_record.reward),
            discount=float(transition_record.discount),
            boundary_type=transition_record.boundary_type,
            bootstrap_state_ref_or_null=bootstrap_state_ref_or_null,
            sampling_probability_or_weight=float(sampling_probability_or_weight),
            target_policy_identity=target_policy_identity,
            source_fact_hashes=(
                transition_record.content_sha256,
                transition_record.observation_content_sha256,
                transition_record.policy_decision_ref,
                transition_record.policy_sha256,
            ),
            consequence_context=None
            if transition_record.consequence_context is None
            else dict(transition_record.consequence_context),
        )
        return sample.validate()

    def materialize_sequence(
        self,
        sequence_id: str,
        *,
        sampling_probability_or_weight: float = 1.0,
        target_policy_identity: str,
        restart_verified: bool,
        expected_materialization_id: str | None = None,
    ) -> MaterializedReplayV1:
        sequence = self.replay_store.get_sequence(sequence_id)
        transitions = [self.replay_store.get_transition(ref) for ref in sequence.transition_ids]
        if not transitions:
            raise ReplayCorruption("EMPTY_DURABLE_SEQUENCE")
        if int(transitions[0].decision_index) != int(sequence.first_decision_index):
            raise ReplayCorruption("SEQUENCE_FIRST_DECISION_INDEX_MISMATCH")
        if int(transitions[-1].decision_index) != int(sequence.last_decision_index):
            raise ReplayCorruption("SEQUENCE_LAST_DECISION_INDEX_MISMATCH")
        boundary_class = classify_boundary_v1(
            transitions[-1].boundary_type, mechanical_terminal=transitions[-1].mechanical_terminal
        )
        if boundary_class == BOUNDARY_CLASS_CONTINUE:
            raise ReplayCorruption("SEQUENCE_FINAL_BOUNDARY_MUST_BE_TERMINAL_OR_TRUNCATION")
        bootstrap_observation: PostCCObservationFactV1 | None = None
        bootstrap_ref: str | None = None
        if boundary_class == BOUNDARY_CLASS_TERMINAL:
            if sequence.bootstrap_state_ref_or_null is not None:
                raise ReplayCorruption("TERMINAL_SEQUENCE_HAS_BOOTSTRAP_REF")
            if transitions[-1].bootstrap_state_ref_or_null is not None:
                raise ReplayCorruption("TERMINAL_TRANSITION_HAS_BOOTSTRAP_REF")
        else:
            bootstrap_ref = sequence.bootstrap_state_ref_or_null
            if not bootstrap_ref:
                raise ReplayCorruption("TRUNCATION_SEQUENCE_BOOTSTRAP_REF_MISSING")
            try:
                bootstrap_observation = self.observation_store.get(bootstrap_ref)
            except KeyError as exc:
                raise ReplayCorruption("BOOTSTRAP_OBSERVATION_NOT_DURABLE") from exc
            canonical_observation_vectors_v1(bootstrap_observation)
        samples: list[DurableJointReplaySampleV1] = []
        for index, transition_record in enumerate(transitions):
            sample_bootstrap_ref = bootstrap_ref if index == len(transitions) - 1 else None
            samples.append(
                self._materialize_sample(
                    sequence=sequence,
                    transition_record=transition_record,
                    previous=None if index == 0 else transitions[index - 1],
                    bootstrap_state_ref_or_null=sample_bootstrap_ref,
                    sampling_probability_or_weight=sampling_probability_or_weight,
                    target_policy_identity=target_policy_identity,
                )
            )
        sample_hashes = tuple(sample.sample_content_sha256 for sample in samples)
        transition_hashes = tuple(record.content_sha256 for record in transitions)
        observation_hashes = tuple(record.observation_content_sha256 for record in transitions)
        manifest_id = hashlib.sha256(
            canonical_json_bytes(
                {
                    "sequence_id": sequence.sequence_id,
                    "materializer_contract_id": MATERIALIZER_CONTRACT_ID,
                    "sample_hashes": sample_hashes,
                    "transition_hashes": transition_hashes,
                    "observation_hashes": observation_hashes,
                    "target_policy_identity": target_policy_identity,
                    "source_store_root_identity": self.source_store_root_identity,
                    "bootstrap_state_ref_or_null": bootstrap_ref,
                    "bootstrap_observation_content_sha256": (
                        None
                        if bootstrap_observation is None
                        else observation_content_sha256_v1(bootstrap_observation)
                    ),
                }
            )
        ).hexdigest()
        manifest = MaterializationManifestV1(
            manifest_id=manifest_id,
            sequence_id=sequence.sequence_id,
            materializer_contract_id=MATERIALIZER_CONTRACT_ID,
            target_policy_identity=target_policy_identity,
            source_store_root_identity=self.source_store_root_identity,
            ordered_sample_hashes=sample_hashes,
            ordered_transition_hashes=transition_hashes,
            ordered_observation_hashes=observation_hashes,
            bootstrap_state_ref_or_null=bootstrap_ref,
            bootstrap_observation_content_sha256=(
                None if bootstrap_observation is None else observation_content_sha256_v1(bootstrap_observation)
            ),
            restart_verified=bool(restart_verified),
        ).validate()
        if expected_materialization_id is not None and manifest.stable_materialization_id != expected_materialization_id:
            raise ReplayCorruption("RESTART_MANIFEST_MISMATCH")
        bootstrap_map: dict[str, PostCCObservationFactV1] = {}
        if bootstrap_observation is not None:
            bootstrap_map[sequence.sequence_id] = bootstrap_observation
        return MaterializedReplayV1(
            manifest=manifest,
            samples=tuple(samples),
            bootstrap_observations_by_sequence=bootstrap_map,
        ).validate()
