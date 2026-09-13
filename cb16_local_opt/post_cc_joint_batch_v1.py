"""S0-v2 canonical joint-action batch.

The batch is built only from durable replay materialization provenance.  It
exposes the nominal direction + conditional target risk, persisted true
``log_mu`` and the boundary/bootstrap data needed by the separate Critic and
V-trace path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch

from .cc_policy_distribution_r0 import INDEX as DIRECTION_INDEX
from .post_cc_joint_replay_v1 import DurableJointReplaySampleV1
from .post_cc_observation_contract_v1 import PostCCObservationFactV1
from .post_cc_observation_fact_v1 import (
    CANONICAL_ACCOUNT_DIM_V1,
    CANONICAL_EXECUTION_DIM_V1,
    CANONICAL_MARKET_DIM_V1,
    canonical_observation_vectors_v1,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MATERIALIZER_CONTRACT_ID = "CB16_R11_S0V2_REPLAY_MATERIALIZER_V1"

# Boundary classes are intentionally explicit.  A single generic ``done`` flag
# is forbidden: terminal responsibility, task horizon, compute truncation and
# replay/chunk truncation all have different bootstrap meanings.
BOUNDARY_CLASS_CONTINUE = "CONTINUE"
BOUNDARY_CLASS_TERMINAL = "TERMINAL"
BOUNDARY_CLASS_TRUNCATION = "TRUNCATION"

_COMPUTE_TRUNCATION_BOUNDARIES = frozenset({"COMPUTE_CHUNK", "PAUSE", "PROCESS_FAILURE"})
_DATASET_TRUNCATION_BOUNDARIES = frozenset({"DATA_END_TRUNCATION"})
_PENDING_SETTLEMENT_BOUNDARIES = frozenset({"TRADING_DISABLED_PENDING_SETTLEMENT"})


def boundary_semantics_v1(boundary_type: str, *, mechanical_terminal: bool = False) -> str:
    """Explicit, non-collapsed boundary semantics used by bootstrap/masks."""
    if not isinstance(boundary_type, str) or not boundary_type:
        raise ValueError("BOUNDARY_TYPE_INVALID")
    if mechanical_terminal and boundary_type != "ECONOMIC_TERMINAL":
        raise ValueError("MECHANICAL_TERMINAL_BOUNDARY_MISMATCH")
    if boundary_type == "CONTINUE":
        return "CONTINUE"
    if boundary_type == "ECONOMIC_TERMINAL":
        return "MECHANICAL_TERMINAL" if mechanical_terminal else "ECONOMIC_TERMINAL"
    if boundary_type == "OBJECTIVE_HORIZON_REACHED":
        return "TASK_HORIZON"
    if boundary_type in _COMPUTE_TRUNCATION_BOUNDARIES:
        return "COMPUTE_TRUNCATION"
    if boundary_type in _DATASET_TRUNCATION_BOUNDARIES:
        return "DATASET_TRUNCATION"
    if boundary_type in _PENDING_SETTLEMENT_BOUNDARIES:
        return "PENDING_SETTLEMENT"
    raise ValueError(f"UNKNOWN_BOUNDARY_TYPE:{boundary_type}")


def classify_boundary_v1(boundary_type: str, *, mechanical_terminal: bool = False) -> str:
    """Map boundary semantics to the bootstrap class."""
    semantics = boundary_semantics_v1(boundary_type, mechanical_terminal=mechanical_terminal)
    if semantics == "CONTINUE":
        return BOUNDARY_CLASS_CONTINUE
    if semantics in {"ECONOMIC_TERMINAL", "MECHANICAL_TERMINAL", "TASK_HORIZON"}:
        return BOUNDARY_CLASS_TERMINAL
    return BOUNDARY_CLASS_TRUNCATION


def boundary_requires_bootstrap_v1(boundary_type: str, *, mechanical_terminal: bool = False) -> bool:
    return classify_boundary_v1(boundary_type, mechanical_terminal=mechanical_terminal) == BOUNDARY_CLASS_TRUNCATION


def boundary_is_terminal_v1(boundary_type: str, *, mechanical_terminal: bool = False) -> bool:
    return classify_boundary_v1(boundary_type, mechanical_terminal=mechanical_terminal) == BOUNDARY_CLASS_TERMINAL


@dataclass(frozen=True)
class DurableBatchProvenanceV1:
    materializer_contract_id: str
    materialization_id: str
    materialization_manifest_sha256: str
    source_store_root_identity: str
    restart_verified: bool
    durable_replay_only: bool
    collector_private_records_used: bool

    def validate(self) -> "DurableBatchProvenanceV1":
        if self.materializer_contract_id != MATERIALIZER_CONTRACT_ID:
            raise ValueError("MATERIALIZER_CONTRACT_ID_MISMATCH")
        for name in ("materialization_id", "source_store_root_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if not isinstance(self.materialization_manifest_sha256, str) or HEX64.fullmatch(
            self.materialization_manifest_sha256
        ) is None:
            raise ValueError("materialization_manifest_sha256 must be 64-hex text")
        if self.durable_replay_only is not True:
            raise ValueError("DURABLE_REPLAY_ONLY_REQUIRED")
        if self.collector_private_records_used is not False:
            raise ValueError("COLLECTOR_PRIVATE_RECORDS_FORBIDDEN")
        if self.restart_verified is not True:
            raise ValueError("RESTART_VERIFIED_REQUIRED")
        return self


def observation_to_tensors_v1(fact: PostCCObservationFactV1, *, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    market, account, execution = canonical_observation_vectors_v1(fact)
    return (
        torch.tensor(market, dtype=dtype),
        torch.tensor(account, dtype=dtype),
        torch.tensor(execution, dtype=dtype),
    )


@dataclass(frozen=True)
class JointActionBatchV1:
    samples: tuple[DurableJointReplaySampleV1, ...]
    provenance: DurableBatchProvenanceV1
    sequence_ids: tuple[str, ...]
    sequence_offsets: tuple[tuple[int, int], ...]
    market_observations: torch.Tensor
    account_observations: torch.Tensor
    execution_observations: torch.Tensor
    critic_observations: torch.Tensor
    direction_indices: torch.Tensor
    target_risks: torch.Tensor
    risk_point_mass_mask: torch.Tensor
    behavior_log_mu: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    replay_weights: torch.Tensor
    boundary_types: tuple[str, ...]
    mechanical_terminals: tuple[bool, ...]
    bootstrap_state_refs: tuple[str | None, ...]
    observation_hashes: tuple[str, ...]
    behavior_policy_identities: tuple[str, ...]
    source_fact_hashes: tuple[tuple[str, ...], ...]
    bootstrap_observations: torch.Tensor | None
    bootstrap_sequence_indices: tuple[int, ...]
    target_policy_identity: str

    def validate(self) -> "JointActionBatchV1":
        self.provenance.validate()
        if not isinstance(self.target_policy_identity, str) or not self.target_policy_identity.strip():
            raise ValueError("target_policy_identity must be non-empty text")
        count = len(self.samples)
        if count == 0:
            raise ValueError("EMPTY_BATCH")
        if any(not isinstance(sample, DurableJointReplaySampleV1) for sample in self.samples):
            raise ValueError("BATCH_REQUIRES_DURABLE_REPLAY_SAMPLES")
        for sample in self.samples:
            sample.validate()
        for name in (
            "boundary_types",
            "mechanical_terminals",
            "bootstrap_state_refs",
            "observation_hashes",
            "behavior_policy_identities",
            "source_fact_hashes",
        ):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} length mismatch")
        if len(self.sequence_ids) == 0:
            raise ValueError("sequence_ids must be non-empty")
        if len(self.sequence_offsets) != len(self.sequence_ids):
            raise ValueError("sequence offset count mismatch")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence_ids duplicate")
        self._validate_sequence_offsets()
        self._validate_tensor_shapes()
        self._validate_observation_provenance()
        self._validate_time_order()
        self._validate_boundaries()
        self._validate_bootstrap_observations()
        return self

    def _validate_sequence_offsets(self) -> None:
        seen_sequences: list[str] = []
        expected_start = 0
        for offset_index, (start, end) in enumerate(self.sequence_offsets):
            if start != expected_start or end <= start or end > len(self.samples):
                raise ValueError("sequence offsets must be contiguous and cover the batch")
            if offset_index >= len(self.sequence_ids):
                raise ValueError("sequence offset beyond sequence id list")
            sequence_id = self.sequence_ids[offset_index]
            if any(self.samples[i].sequence_id != sequence_id for i in range(start, end)):
                raise ValueError("sequence offset block contains foreign sequence")
            if sequence_id in seen_sequences:
                raise ValueError("SEQUENCE_BLOCK_NOT_CONTIGUOUS")
            seen_sequences.append(sequence_id)
            expected_start = end
        if expected_start != len(self.samples):
            raise ValueError("sequence offsets must cover the batch")

    def _validate_tensor_shapes(self) -> None:
        count = len(self.samples)
        expected = {
            "market_observations": (count, CANONICAL_MARKET_DIM_V1),
            "account_observations": (count, CANONICAL_ACCOUNT_DIM_V1),
            "execution_observations": (count, CANONICAL_EXECUTION_DIM_V1),
            "critic_observations": (count, CANONICAL_MARKET_DIM_V1 + CANONICAL_ACCOUNT_DIM_V1 + CANONICAL_EXECUTION_DIM_V1),
            "direction_indices": (count,),
            "target_risks": (count,),
            "risk_point_mass_mask": (count,),
            "behavior_log_mu": (count,),
            "rewards": (count,),
            "discounts": (count,),
            "replay_weights": (count,),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{name} must be a tensor")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} shape mismatch")
        if self.direction_indices.dtype != torch.long:
            raise ValueError("direction_indices must be torch.long")
        if self.risk_point_mass_mask.dtype != torch.bool:
            raise ValueError("risk_point_mass_mask must be torch.bool")
        for name in ("market_observations", "account_observations", "execution_observations", "critic_observations", "target_risks", "behavior_log_mu", "rewards", "discounts", "replay_weights"):
            if not torch.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")

    def _validate_observation_provenance(self) -> None:
        for index, sample in enumerate(self.samples):
            if self.observation_hashes[index] != sample.observation.observation_hash:
                raise ValueError("OBSERVATION_HASH_MISMATCH")
            if self.behavior_policy_identities[index] != sample.behavior_policy_identity:
                raise ValueError("BEHAVIOR_POLICY_IDENTITY_MISMATCH")
            if self.boundary_types[index] != sample.boundary_type:
                raise ValueError("BOUNDARY_TYPE_MISMATCH")
            if self.bootstrap_state_refs[index] != sample.bootstrap_state_ref_or_null:
                raise ValueError("BOOTSTRAP_STATE_REF_MISMATCH")
            if self.source_fact_hashes[index] != tuple(sample.source_fact_hashes):
                raise ValueError("SOURCE_FACT_HASH_MISMATCH")
            expected_direction_index = DIRECTION_INDEX[sample.nominal_direction]
            if int(self.direction_indices[index].item()) != expected_direction_index:
                raise ValueError("DIRECTION_INDEX_MAPPING_MISMATCH")
            if not math.isclose(float(self.target_risks[index].item()), float(sample.nominal_target_risk), rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("TARGET_RISK_MISMATCH")
            expected_point_mass = sample.risk_measure_kind == "point_mass"
            if bool(self.risk_point_mass_mask[index].item()) is not expected_point_mass:
                raise ValueError("RISK_MEASURE_MASK_MISMATCH")
            if not math.isclose(float(self.behavior_log_mu[index].item()), float(sample.behavior_log_mu), rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("LOG_MU_MISMATCH")
            if not math.isclose(float(self.rewards[index].item()), float(sample.reward), rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("REWARD_MISMATCH")
            if not math.isclose(float(self.discounts[index].item()), float(sample.discount), rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("DISCOUNT_MISMATCH")
            if not math.isclose(float(self.replay_weights[index].item()), float(sample.sampling_probability_or_weight), rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("REPLAY_WEIGHT_MISMATCH")
            if not sample.source_fact_hashes:
                raise ValueError("MISSING_DURABLE_SOURCE_REFERENCE")
            market, account, execution = observation_to_tensors_v1(sample.observation, dtype=self.market_observations.dtype)
            if not torch.equal(market, self.market_observations[index]):
                raise ValueError("MARKET_OBSERVATION_RECONSTRUCTION_MISMATCH")
            if not torch.equal(account, self.account_observations[index]):
                raise ValueError("ACCOUNT_OBSERVATION_RECONSTRUCTION_MISMATCH")
            if not torch.equal(execution, self.execution_observations[index]):
                raise ValueError("EXECUTION_OBSERVATION_RECONSTRUCTION_MISMATCH")
            expected_critic = torch.cat([market, account, execution])
            if not torch.equal(expected_critic, self.critic_observations[index]):
                raise ValueError("CRITIC_OBSERVATION_RECONSTRUCTION_MISMATCH")

    def _validate_time_order(self) -> None:
        last_by_lineage: dict[str, tuple[int, str]] = {}
        for sample in self.samples:
            previous = last_by_lineage.get(sample.account_lineage_id)
            if previous is not None:
                previous_index, previous_time = previous
                if int(sample.decision_index) <= int(previous_index):
                    raise ValueError("TIME_ORDER_VIOLATION_WITHIN_ACCOUNT_LINEAGE")
                if str(sample.environment_time) <= previous_time:
                    raise ValueError("ENVIRONMENT_TIME_ORDER_VIOLATION_WITHIN_ACCOUNT_LINEAGE")
            last_by_lineage[sample.account_lineage_id] = (int(sample.decision_index), str(sample.environment_time))

    def _validate_boundaries(self) -> None:
        for sequence_index, (start, end) in enumerate(self.sequence_offsets):
            sequence_id = self.sequence_ids[sequence_index]
            for index in range(start, end):
                boundary = self.boundary_types[index]
                mechanical = bool(self.mechanical_terminals[index])
                is_last = index == end - 1
                if not isinstance(self.mechanical_terminals[index], bool):
                    raise ValueError("MECHANICAL_TERMINAL_MUST_BE_BOOL")
                if mechanical != bool(self.samples[index].mechanical_terminal):
                    raise ValueError("MECHANICAL_TERMINAL_TENSOR_MISMATCH")
                boundary_class = classify_boundary_v1(boundary, mechanical_terminal=mechanical)
                if not is_last and boundary_class != BOUNDARY_CLASS_CONTINUE:
                    raise ValueError("NON_FINAL_BOUNDARY_MUST_BE_CONTINUE")
                if is_last and boundary_class == BOUNDARY_CLASS_CONTINUE:
                    raise ValueError("SEQUENCE_FINAL_BOUNDARY_CANNOT_BE_CONTINUE")
            final_index = end - 1
            final_sample = self.samples[final_index]
            if classify_boundary_v1(
                final_sample.boundary_type,
                mechanical_terminal=bool(self.mechanical_terminals[final_index]),
            ) == BOUNDARY_CLASS_TERMINAL:
                if final_sample.bootstrap_state_ref_or_null is not None:
                    raise ValueError("TERMINAL_BOUNDARY_MUST_NOT_HAVE_BOOTSTRAP_STATE")
            else:
                if not final_sample.bootstrap_state_ref_or_null:
                    raise ValueError("TRUNCATION_BOUNDARY_REQUIRES_BOOTSTRAP_STATE")

    def _validate_bootstrap_observations(self) -> None:
        bootstrap_indices = tuple(int(item) for item in self.bootstrap_sequence_indices)
        if len(set(bootstrap_indices)) != len(bootstrap_indices):
            raise ValueError("bootstrap sequence index duplicate")
        required_indices: list[int] = []
        for sequence_index, (start, end) in enumerate(self.sequence_offsets):
            del start
            if classify_boundary_v1(
                self.samples[end - 1].boundary_type,
                mechanical_terminal=bool(self.mechanical_terminals[end - 1]),
            ) == BOUNDARY_CLASS_TRUNCATION:
                required_indices.append(sequence_index)
        if set(bootstrap_indices) != set(required_indices):
            raise ValueError("bootstrap sequence index set mismatch")
        if not required_indices:
            if self.bootstrap_observations is not None:
                raise ValueError("UNEXPECTED_BOOTSTRAP_OBSERVATIONS")
            return
        if self.bootstrap_observations is None:
            raise ValueError("BOOTSTRAP_OBSERVATIONS_REQUIRED")
        if self.bootstrap_observations.ndim != 2 or int(self.bootstrap_observations.shape[0]) != len(required_indices):
            raise ValueError("BOOTSTRAP_OBSERVATIONS_SHAPE")
        if int(self.bootstrap_observations.shape[1]) != CANONICAL_MARKET_DIM_V1 + CANONICAL_ACCOUNT_DIM_V1 + CANONICAL_EXECUTION_DIM_V1:
            raise ValueError("BOOTSTRAP_OBSERVATIONS_SHAPE")
        if not torch.isfinite(self.bootstrap_observations).all():
            raise ValueError("BOOTSTRAP_OBSERVATIONS_NONFINITE")

    def to(self, device: torch.device | str) -> "JointActionBatchV1":
        return JointActionBatchV1(
            samples=self.samples,
            provenance=self.provenance,
            sequence_ids=self.sequence_ids,
            sequence_offsets=self.sequence_offsets,
            market_observations=self.market_observations.to(device),
            account_observations=self.account_observations.to(device),
            execution_observations=self.execution_observations.to(device),
            critic_observations=self.critic_observations.to(device),
            direction_indices=self.direction_indices.to(device),
            target_risks=self.target_risks.to(device),
            risk_point_mass_mask=self.risk_point_mass_mask.to(device),
            behavior_log_mu=self.behavior_log_mu.to(device),
            rewards=self.rewards.to(device),
            discounts=self.discounts.to(device),
            replay_weights=self.replay_weights.to(device),
            boundary_types=self.boundary_types,
            mechanical_terminals=self.mechanical_terminals,
            bootstrap_state_refs=self.bootstrap_state_refs,
            observation_hashes=self.observation_hashes,
            behavior_policy_identities=self.behavior_policy_identities,
            source_fact_hashes=self.source_fact_hashes,
            bootstrap_observations=None if self.bootstrap_observations is None else self.bootstrap_observations.to(device),
            bootstrap_sequence_indices=self.bootstrap_sequence_indices,
            target_policy_identity=self.target_policy_identity,
        )

    @property
    def batch_content_sha256(self) -> str:
        from .cc_experience_wire_r0 import content_sha256

        self.validate()
        return content_sha256(
            {
                "materialization_manifest_sha256": self.provenance.materialization_manifest_sha256,
                "target_policy_identity": self.target_policy_identity,
                "sample_hashes": tuple(sample.sample_content_sha256 for sample in self.samples),
                "sequence_offsets": self.sequence_offsets,
                "bootstrap_sequence_indices": self.bootstrap_sequence_indices,
            }
        )


def _manifest_sequence_order(samples: Sequence[DurableJointReplaySampleV1]) -> tuple[str, ...]:
    order: list[str] = []
    for sample in samples:
        if not order or order[-1] != sample.sequence_id:
            if sample.sequence_id in order:
                raise ValueError("SEQUENCE_BLOCK_NOT_CONTIGUOUS")
            order.append(sample.sequence_id)
    return tuple(order)


def _validate_bootstrap_observation_v1(fact: PostCCObservationFactV1) -> None:
    fact.validate()
    canonical_observation_vectors_v1(fact)


def build_joint_batch_v1(
    *,
    samples: Sequence[DurableJointReplaySampleV1],
    provenance: DurableBatchProvenanceV1,
    target_policy_identity: str,
    bootstrap_observations_by_sequence: Mapping[str, PostCCObservationFactV1] | None = None,
) -> JointActionBatchV1:
    """Build the canonical batch from durable materialization output only."""
    ordered = tuple(samples)
    if not ordered:
        raise ValueError("EMPTY_BATCH")
    for sample in ordered:
        sample.validate()
    sequence_ids = _manifest_sequence_order(ordered)
    offsets: list[tuple[int, int]] = []
    starts: dict[str, int] = {}
    for index, sample in enumerate(ordered):
        if sample.sequence_id not in starts:
            starts[sample.sequence_id] = index
    for sequence_id in sequence_ids:
        starts_index = starts[sequence_id]
        end = starts_index
        while end < len(ordered) and ordered[end].sequence_id == sequence_id:
            end += 1
        offsets.append((starts_index, end))

    market_rows = []
    account_rows = []
    execution_rows = []
    direction_indices = []
    target_risks = []
    point_mass_mask = []
    mechanical_terminals = []
    log_mu_values = []
    rewards = []
    discounts = []
    weights = []
    bootstrap_sequence_indices: list[int] = []
    bootstrap_rows: list[torch.Tensor] = []

    mapping = dict(bootstrap_observations_by_sequence or {})
    for sequence_index, sequence_id in enumerate(sequence_ids):
        start, end = offsets[sequence_index]
        for sample in ordered[start:end]:
            market, account, execution = observation_to_tensors_v1(sample.observation)
            market_rows.append(market)
            account_rows.append(account)
            execution_rows.append(execution)
            direction_indices.append(DIRECTION_INDEX[sample.nominal_direction])
            target_risks.append(float(sample.nominal_target_risk))
            point_mass_mask.append(sample.risk_measure_kind == "point_mass")
            mechanical_terminals.append(bool(sample.mechanical_terminal))
            log_mu_values.append(float(sample.behavior_log_mu))
            rewards.append(float(sample.reward))
            discounts.append(float(sample.discount))
            weights.append(float(sample.sampling_probability_or_weight))
        if boundary_requires_bootstrap_v1(
            ordered[end - 1].boundary_type,
            mechanical_terminal=bool(ordered[end - 1].mechanical_terminal),
        ):
            if sequence_id not in mapping:
                raise ValueError(f"BOOTSTRAP_OBSERVATION_REQUIRED:{sequence_id}")
            bootstrap_fact = mapping[sequence_id]
            _validate_bootstrap_observation_v1(bootstrap_fact)
            market_b, account_b, execution_b = observation_to_tensors_v1(bootstrap_fact)
            bootstrap_rows.append(torch.cat([market_b, account_b, execution_b]))
            bootstrap_sequence_indices.append(sequence_index)
        elif sequence_id in mapping:
            raise ValueError(f"UNEXPECTED_BOOTSTRAP_OBSERVATION:{sequence_id}")

    critic_rows = [
        torch.cat([market, account, execution])
        for market, account, execution in zip(market_rows, account_rows, execution_rows)
    ]
    bootstrap_observations = None if not bootstrap_rows else torch.stack(bootstrap_rows, dim=0)
    batch = JointActionBatchV1(
        samples=ordered,
        provenance=provenance,
        sequence_ids=sequence_ids,
        sequence_offsets=tuple(offsets),
        market_observations=torch.stack(market_rows, dim=0),
        account_observations=torch.stack(account_rows, dim=0),
        execution_observations=torch.stack(execution_rows, dim=0),
        critic_observations=torch.stack(critic_rows, dim=0),
        direction_indices=torch.tensor(direction_indices, dtype=torch.long),
        target_risks=torch.tensor(target_risks, dtype=torch.float32),
        risk_point_mass_mask=torch.tensor(point_mass_mask, dtype=torch.bool),
        behavior_log_mu=torch.tensor(log_mu_values, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        discounts=torch.tensor(discounts, dtype=torch.float32),
        replay_weights=torch.tensor(weights, dtype=torch.float32),
        boundary_types=tuple(sample.boundary_type for sample in ordered),
        mechanical_terminals=tuple(mechanical_terminals),
        bootstrap_state_refs=tuple(sample.bootstrap_state_ref_or_null for sample in ordered),
        observation_hashes=tuple(sample.observation.observation_hash for sample in ordered),
        behavior_policy_identities=tuple(sample.behavior_policy_identity for sample in ordered),
        source_fact_hashes=tuple(tuple(sample.source_fact_hashes) for sample in ordered),
        bootstrap_observations=bootstrap_observations,
        bootstrap_sequence_indices=tuple(bootstrap_sequence_indices),
        target_policy_identity=target_policy_identity,
    )
    return batch.validate()
