"""S0-v2 reusable durable-replay learner runtime.

The learner accepts only a validated ``JointActionBatchV1`` whose provenance
says it was materialized from durable stores.  It never consumes collector
rollout records directly, updates only the authorized Brain/Critic paths and
creates a separate child checkpoint through the exactly-once journal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import torch
from torch import nn

from .cc_critic_value_r0 import SeparateCritic, assert_disjoint_parameters
from .cc_experience_wire_r0 import canonical_json_bytes
from .cc_policy_brain_r0 import CCCentralBrain
from .post_cc_critic_vtrace_v1 import audit_gradient_ownership_v1, joint_actor_critic_losses_v1
from .post_cc_joint_batch_v1 import JointActionBatchV1
from .post_cc_joint_policy_loss_v1 import assert_persisted_log_mu_used_v1
from .post_cc_update_transaction_v1 import (
    STATUS_COMMITTED,
    STATUS_PREPARED,
    STATUS_STAGED,
    DurableLearningUpdateV1,
    DurableUpdateStoreV1,
    checkpoint_bundle_bytes_v1,
    checkpoint_sha256_v1,
    decode_checkpoint_bundle_v1,
    encode_tensor_state_v1,
)


class PostCCLearnerError(RuntimeError):
    pass


class InjectedUpdateFaultV1(PostCCLearnerError):
    def __init__(self, fault_at: str):
        super().__init__(f"INJECTED_UPDATE_FAULT:{fault_at}")
        self.fault_at = fault_at


def _parent_checkpoint_bytes_v1(
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer_step: int,
    actor_lr: float,
    critic_lr: float,
) -> bytes:
    payload = {
        "schema": "CB16_R11_S0V2_PARENT_CHECKPOINT_V1",
        "optimizer_step": int(optimizer_step),
        "actor_lr": float(actor_lr),
        "critic_lr": float(critic_lr),
        "actor_state": encode_tensor_state_v1(actor.state_dict()),
        "critic_state": encode_tensor_state_v1(critic.state_dict()),
    }
    return canonical_json_bytes(payload)


def parent_checkpoint_sha256_v1(
    *,
    actor: nn.Module,
    critic: nn.Module,
    optimizer_step: int,
    actor_lr: float,
    critic_lr: float,
) -> str:
    digest = hashlib.sha256(
        _parent_checkpoint_bytes_v1(
            actor=actor,
            critic=critic,
            optimizer_step=optimizer_step,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )
    ).hexdigest()
    return digest


@dataclass(frozen=True)
class DurableUpdateResultV1:
    update_id: str
    applied: bool
    recovered_from_staged: bool
    already_committed: bool
    optimizer_step_before: int
    optimizer_step_after: int
    actor_loss: float
    critic_loss: float
    child_checkpoint_sha256: str
    record: DurableLearningUpdateV1
    gradient_ownership_summary: Mapping[str, Any]
    vtrace_diagnostics: Mapping[str, Any]


class PostCCDurableReplayLearnerV1:
    """One bounded target-policy update with durable exactly-once provenance."""

    def __init__(
        self,
        *,
        target_actor: CCCentralBrain,
        target_critic: SeparateCritic,
        update_store: DurableUpdateStoreV1,
        parent_policy_identity: str,
        target_policy_identity: str,
        science_semantic_version: str,
        actor_lr: float = 0.01,
        critic_lr: float = 0.01,
    ):
        if not isinstance(target_actor, CCCentralBrain):
            raise TypeError("target_actor must be CCCentralBrain")
        if not isinstance(target_critic, SeparateCritic):
            raise TypeError("target_critic must be SeparateCritic")
        assert_disjoint_parameters(target_actor, target_critic)
        target_actor.assert_gradient_ownership()
        for name, value in (
            ("parent_policy_identity", parent_policy_identity),
            ("target_policy_identity", target_policy_identity),
            ("science_semantic_version", science_semantic_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.update_store = update_store
        self.parent_policy_identity = parent_policy_identity
        self.target_policy_identity = target_policy_identity
        self.science_semantic_version = science_semantic_version
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.actor_opt = torch.optim.SGD([p for p in target_actor.parameters() if p.requires_grad], lr=self.actor_lr)
        self.critic_opt = torch.optim.SGD(target_critic.parameters(), lr=self.critic_lr)
        self.optimizer_step = 0
        self.gradient_applications = 0
        self.restart_required = False
        self.restart_required_reason: str | None = None

    def parent_checkpoint_sha256(self) -> str:
        return parent_checkpoint_sha256_v1(
            actor=self.target_actor,
            critic=self.target_critic,
            optimizer_step=self.optimizer_step,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
        )

    def export_parent_checkpoint_bytes(self) -> bytes:
        return _parent_checkpoint_bytes_v1(
            actor=self.target_actor,
            critic=self.target_critic,
            optimizer_step=self.optimizer_step,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
        )

    @classmethod
    def from_parent_checkpoint_bytes_v1(
        cls,
        *,
        payload: bytes,
        target_actor: CCCentralBrain,
        target_critic: SeparateCritic,
        update_store: DurableUpdateStoreV1,
        parent_policy_identity: str,
        target_policy_identity: str,
        science_semantic_version: str,
        actor_lr: float = 0.01,
        critic_lr: float = 0.01,
    ) -> "PostCCDurableReplayLearnerV1":
        import json

        raw = json.loads(bytes(payload).decode("utf-8"))
        if raw.get("schema") != "CB16_R11_S0V2_PARENT_CHECKPOINT_V1":
            raise PostCCLearnerError("PARENT_CHECKPOINT_SCHEMA_MISMATCH")
        from .post_cc_update_transaction_v1 import decode_tensor_state_v1

        target_actor.load_state_dict(decode_tensor_state_v1(raw["actor_state"]))
        target_critic.load_state_dict(decode_tensor_state_v1(raw["critic_state"]))
        learner = cls(
            target_actor=target_actor,
            target_critic=target_critic,
            update_store=update_store,
            parent_policy_identity=parent_policy_identity,
            target_policy_identity=target_policy_identity,
            science_semantic_version=science_semantic_version,
            actor_lr=float(raw.get("actor_lr", actor_lr)),
            critic_lr=float(raw.get("critic_lr", critic_lr)),
        )
        learner.optimizer_step = int(raw["optimizer_step"])
        return learner

    def _deterministic_update_id(self, *, parent_checkpoint_sha256: str, batch: JointActionBatchV1) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema": "CB16_R11_S0V2_UPDATE_ID_V1",
                    "science_semantic_version": self.science_semantic_version,
                    "parent_checkpoint_sha256": parent_checkpoint_sha256,
                    "materialization_manifest_sha256": batch.provenance.materialization_manifest_sha256,
                    "batch_content_sha256": batch.batch_content_sha256,
                    "target_policy_identity": self.target_policy_identity,
                }
            )
        ).hexdigest()

    def _adopt_child_checkpoint(self, record: DurableLearningUpdateV1) -> Mapping[str, Any]:
        assert record.child_checkpoint_sha256 is not None
        bundle = self.update_store.load_child_checkpoint(record.update_id)
        self.target_actor.load_state_dict(bundle["actor_state_tensors"])
        self.target_critic.load_state_dict(bundle["critic_state_tensors"])
        if checkpoint_sha256_v1(self.update_store._checkpoint_path(record.child_checkpoint_sha256).read_bytes()) != record.child_checkpoint_sha256:
            raise PostCCLearnerError("CHILD_CHECKPOINT_HASH_MISMATCH")
        self.optimizer_step = int(record.optimizer_step_after)
        return bundle

    def _frozen_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.target_actor.market_organ.named_parameters()}

    def _assert_authorized_mutation(self, before_frozen: Mapping[str, torch.Tensor]) -> None:
        for name, parameter in self.target_actor.market_organ.named_parameters():
            if name not in before_frozen:
                raise PostCCLearnerError("FROZEN_PARAMETER_SET_CHANGED")
            if not torch.equal(before_frozen[name], parameter.detach()):
                raise PostCCLearnerError(f"FROZEN_MARKET_ORGAN_MUTATED:{name}")

    def apply_durable_update_v1(
        self,
        *,
        batch: JointActionBatchV1,
        fault_at: str | None = None,
    ) -> DurableUpdateResultV1:
        """Apply one update; retries must pass the same durable batch."""
        if self.restart_required:
            raise PostCCLearnerError(
                f"RESTART_REQUIRED_FROM_DURABLE_PARENT:{self.restart_required_reason or 'POST_MUTATION_FAILURE'}"
            )
        batch.validate()
        if not isinstance(batch.provenance.restart_verified, bool) or batch.provenance.restart_verified is not True:
            raise PostCCLearnerError("RESTART_VERIFIED_DURABLE_BATCH_REQUIRED")
        if batch.provenance.durable_replay_only is not True or batch.provenance.collector_private_records_used is not False:
            raise PostCCLearnerError("IN_MEMORY_COLLECTOR_RECORD_SHORTCUT_FORBIDDEN")
        assert_persisted_log_mu_used_v1(batch)
        parent_sha = self.parent_checkpoint_sha256()
        update_id = self._deterministic_update_id(parent_checkpoint_sha256=parent_sha, batch=batch)
        sequence_weights: list[float] = []
        for start, end in batch.sequence_offsets:
            weights = batch.replay_weights[start:end]
            first = float(weights[0].item())
            if not bool((weights == weights[0]).all()):
                raise PostCCLearnerError("SEQUENCE_REPLAY_WEIGHT_MUST_BE_UNIFORM")
            sequence_weights.append(first)
        record = self.update_store.begin(
            update_id=update_id,
            science_semantic_version=self.science_semantic_version,
            parent_checkpoint_sha256=parent_sha,
            parent_policy_identity=self.parent_policy_identity,
            target_policy_identity=self.target_policy_identity,
            materialization_manifest_sha256=batch.provenance.materialization_manifest_sha256,
            batch_content_sha256=batch.batch_content_sha256,
            sampled_sequence_ids=batch.sequence_ids,
            materialized_sample_hashes=tuple(sample.sample_content_sha256 for sample in batch.samples),
            sampling_probabilities_or_weights=tuple(sequence_weights),
            behavior_policy_identities=tuple(dict.fromkeys(sample.behavior_policy_identity for sample in batch.samples)),
            optimizer_step_before=int(self.optimizer_step),
        )

        if record.commit_status == STATUS_COMMITTED:
            self._adopt_child_checkpoint(record)
            assert record.actor_loss is not None and record.critic_loss is not None
            return DurableUpdateResultV1(
                update_id=update_id,
                applied=False,
                recovered_from_staged=False,
                already_committed=True,
                optimizer_step_before=int(record.optimizer_step_before),
                optimizer_step_after=int(record.optimizer_step_after),
                actor_loss=float(record.actor_loss),
                critic_loss=float(record.critic_loss),
                child_checkpoint_sha256=str(record.child_checkpoint_sha256),
                record=record,
                gradient_ownership_summary=dict(record.gradient_ownership_summary),
                vtrace_diagnostics=dict(record.vtrace_diagnostics),
            )

        if record.commit_status == STATUS_STAGED:
            self._adopt_child_checkpoint(record)
            committed = self.update_store.commit(update_id, str(record.child_checkpoint_sha256))
            assert record.actor_loss is not None and record.critic_loss is not None
            return DurableUpdateResultV1(
                update_id=update_id,
                applied=True,
                recovered_from_staged=True,
                already_committed=False,
                optimizer_step_before=int(committed.optimizer_step_before),
                optimizer_step_after=int(committed.optimizer_step_after),
                actor_loss=float(committed.actor_loss),
                critic_loss=float(committed.critic_loss),
                child_checkpoint_sha256=str(committed.child_checkpoint_sha256),
                record=committed,
                gradient_ownership_summary=dict(committed.gradient_ownership_summary),
                vtrace_diagnostics=dict(committed.vtrace_diagnostics),
            )

        if record.commit_status != STATUS_PREPARED:
            raise PostCCLearnerError("UNKNOWN_TRANSACTION_STATUS")

        # From here on the live optimizer may mutate.  Any failure in this
        # region poisons the instance: a retry must reconstruct from the
        # durable parent checkpoint, never apply another gradient in-place.
        self.restart_required = True
        self.restart_required_reason = "POST_MUTATION_FAILURE"
        before_frozen = self._frozen_snapshot()
        losses = joint_actor_critic_losses_v1(self.target_actor, self.target_critic, batch)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        losses.actor_loss.backward()
        losses.critic_loss.backward()
        gradient_summary = dict(audit_gradient_ownership_v1(self.target_actor, self.target_critic))
        self.actor_opt.step()
        self.critic_opt.step()
        self.optimizer_step += 1
        self.gradient_applications += 1
        self._assert_authorized_mutation(before_frozen)

        child_bytes = checkpoint_bundle_bytes_v1(
            actor=self.target_actor,
            critic=self.target_critic,
            optimizer_step=self.optimizer_step,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            update_id=update_id,
            parent_checkpoint_sha256=parent_sha,
        )
        child_sha = checkpoint_sha256_v1(child_bytes)
        if fault_at == "after_gradient_before_stage":
            raise InjectedUpdateFaultV1(fault_at)
        staged = self.update_store.stage(
            update_id,
            child_checkpoint_bytes=child_bytes,
            actor_loss=float(losses.actor_loss.detach().item()),
            critic_loss=float(losses.critic_loss.detach().item()),
            vtrace_diagnostics=dict(losses.diagnostics),
            gradient_ownership_summary=gradient_summary,
            optimizer_step_after=int(self.optimizer_step),
        )
        if fault_at == "after_stage_before_commit":
            raise InjectedUpdateFaultV1(fault_at)
        committed = self.update_store.commit(update_id, child_sha)
        if fault_at == "after_commit_before_ack":
            raise InjectedUpdateFaultV1(fault_at)
        self.restart_required = False
        self.restart_required_reason = None
        return DurableUpdateResultV1(
            update_id=update_id,
            applied=True,
            recovered_from_staged=False,
            already_committed=False,
            optimizer_step_before=int(record.optimizer_step_before),
            optimizer_step_after=int(committed.optimizer_step_after),
            actor_loss=float(committed.actor_loss),
            critic_loss=float(committed.critic_loss),
            child_checkpoint_sha256=str(committed.child_checkpoint_sha256),
            record=committed,
            gradient_ownership_summary=dict(committed.gradient_ownership_summary),
            vtrace_diagnostics=dict(committed.vtrace_diagnostics),
        )
