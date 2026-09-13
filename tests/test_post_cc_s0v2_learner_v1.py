from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import pytest
import torch

from cb16_local_opt.cc_critic_value_r0 import SeparateCritic
from cb16_local_opt.post_cc_generation_continuity_v1 import parameter_state_sha256_v1
from cb16_local_opt.post_cc_learner_v1 import (
    InjectedUpdateFaultV1,
    PostCCDurableReplayLearnerV1,
    PostCCLearnerError,
)
from cb16_local_opt.post_cc_update_transaction_v1 import (
    STATUS_COMMITTED,
    STATUS_PREPARED,
    STATUS_STAGED,
    DurableUpdateCorruptionError,
    DurableUpdateSemanticConflict,
    DurableUpdateStoreV1,
)
from tests.cc_s0v2_support import (
    SCIENCE_SEMANTIC_VERSION,
    collect_durable_sequence_v1,
    make_batch_v1,
    make_brain,
    make_joint_sample_v1,
    materialize_v1,
)


def _setup_batch(root: Path):
    collected = collect_durable_sequence_v1(root)
    materialized = materialize_v1(root, collected.sequence_id, restart_verified=True)
    return materialized.to_batch()


def _make_learner(root: Path, *, target=None, critic=None, parent="cc-s0v2-policy-g0", target_id="cc-s0v2-target-policy-g1"):
    store = DurableUpdateStoreV1(root / "updates")
    learner = PostCCDurableReplayLearnerV1(
        target_actor=target if target is not None else make_brain(),
        target_critic=critic if critic is not None else SeparateCritic(7, 8),
        update_store=store,
        parent_policy_identity=parent,
        target_policy_identity=target_id,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
    )
    return learner, store


def _restart(learner: PostCCDurableReplayLearnerV1, root: Path, payload: bytes):
    return PostCCDurableReplayLearnerV1.from_parent_checkpoint_bytes_v1(
        payload=payload,
        target_actor=make_brain(),
        target_critic=SeparateCritic(7, 8),
        update_store=DurableUpdateStoreV1(root / "updates"),
        parent_policy_identity=learner.parent_policy_identity,
        target_policy_identity=learner.target_policy_identity,
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
    )


def test_happy_path_creates_distinct_child_and_keeps_behavior_checkpoint_immutable(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    behavior = make_brain()
    behavior_sha = parameter_state_sha256_v1(behavior)
    learner, store = _make_learner(tmp_path, target=make_brain())
    learner.target_actor.load_state_dict(behavior.state_dict())
    target_before = parameter_state_sha256_v1(learner.target_actor)
    result = learner.apply_durable_update_v1(batch=batch)
    assert result.applied is True
    assert result.optimizer_step_after == 1
    assert result.child_checkpoint_sha256 != result.record.parent_checkpoint_sha256
    assert store.get_record(result.update_id).commit_status == STATUS_COMMITTED
    assert parameter_state_sha256_v1(behavior) == behavior_sha
    assert parameter_state_sha256_v1(learner.target_actor) != target_before
    assert learner.gradient_applications == 1
    assert result.gradient_ownership_summary["frozen_market_organ_grad_zero"] is True
    assert result.gradient_ownership_summary["critic_nonzero_gradient_parameter_count"] > 0


def test_in_memory_collector_shortcut_batch_is_rejected(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, _store = _make_learner(tmp_path)
    shortcut = replace(batch, provenance=replace(batch.provenance, durable_replay_only=False))
    with pytest.raises(ValueError, match="DURABLE_REPLAY_ONLY_REQUIRED"):
        learner.apply_durable_update_v1(batch=shortcut)
    non_restart = replace(batch, provenance=replace(batch.provenance, restart_verified=False))
    with pytest.raises(ValueError, match="RESTART_VERIFIED_REQUIRED"):
        learner.apply_durable_update_v1(batch=non_restart)


def test_fault_after_gradient_before_stage_retries_exactly_once(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    parent_payload = learner.export_parent_checkpoint_bytes()
    with pytest.raises(InjectedUpdateFaultV1):
        learner.apply_durable_update_v1(batch=batch, fault_at="after_gradient_before_stage")
    update_id = next(store.updates_dir.glob("*.json")).stem
    assert store.get_record(update_id).commit_status == STATUS_PREPARED
    restarted = _restart(learner, tmp_path, parent_payload)
    assert restarted.gradient_applications == 0
    result = restarted.apply_durable_update_v1(batch=batch)
    assert result.applied is True
    assert result.recovered_from_staged is False
    assert result.optimizer_step_after == 1
    assert restarted.gradient_applications == 1
    assert store.get_record(update_id).commit_status == STATUS_COMMITTED


def test_fault_after_stage_before_commit_recovers_without_second_gradient(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    parent_payload = learner.export_parent_checkpoint_bytes()
    with pytest.raises(InjectedUpdateFaultV1):
        learner.apply_durable_update_v1(batch=batch, fault_at="after_stage_before_commit")
    update_id = next(store.updates_dir.glob("*.json")).stem
    assert store.get_record(update_id).commit_status == STATUS_STAGED
    staged_child_sha = store.get_record(update_id).child_checkpoint_sha256
    restarted = _restart(learner, tmp_path, parent_payload)
    result = restarted.apply_durable_update_v1(batch=batch)
    assert result.applied is True
    assert result.recovered_from_staged is True
    assert result.optimizer_step_after == 1
    assert restarted.gradient_applications == 0
    assert result.child_checkpoint_sha256 == staged_child_sha
    assert parameter_state_sha256_v1(restarted.target_actor) == parameter_state_sha256_v1(learner.target_actor)


def test_fault_after_commit_before_ack_retry_does_not_reapply(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    parent_payload = learner.export_parent_checkpoint_bytes()
    with pytest.raises(InjectedUpdateFaultV1):
        learner.apply_durable_update_v1(batch=batch, fault_at="after_commit_before_ack")
    update_id = next(store.updates_dir.glob("*.json")).stem
    assert store.get_record(update_id).commit_status == STATUS_COMMITTED
    restarted = _restart(learner, tmp_path, parent_payload)
    result = restarted.apply_durable_update_v1(batch=batch)
    assert result.applied is False
    assert result.already_committed is True
    assert result.optimizer_step_after == 1
    assert restarted.gradient_applications == 0
    assert parameter_state_sha256_v1(restarted.target_actor) == parameter_state_sha256_v1(learner.target_actor)


def test_child_checkpoint_tamper_fails_closed(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    result = learner.apply_durable_update_v1(batch=batch)
    checkpoint_path = store._checkpoint_path(result.child_checkpoint_sha256)
    checkpoint_path.write_bytes(b'{"corrupted": true}')
    with pytest.raises(DurableUpdateCorruptionError, match="CHILD_CHECKPOINT_HASH_MISMATCH"):
        store.load_child_checkpoint(result.update_id)


def test_update_id_rebinding_rejected(tmp_path: Path):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    result = learner.apply_durable_update_v1(batch=batch)
    original = store.get_record(result.update_id)
    with pytest.raises(DurableUpdateSemanticConflict, match="UPDATE_ID_REBOUND"):
        store.begin(
            update_id=result.update_id,
            science_semantic_version=original.science_semantic_version,
            parent_checkpoint_sha256=original.parent_checkpoint_sha256,
            parent_policy_identity=original.parent_policy_identity,
            target_policy_identity=original.target_policy_identity,
            materialization_manifest_sha256=original.materialization_manifest_sha256,
            batch_content_sha256="f" * 64,
            sampled_sequence_ids=original.sampled_sequence_ids,
            materialized_sample_hashes=original.materialized_sample_hashes,
            sampling_probabilities_or_weights=original.sampling_probabilities_or_weights,
            behavior_policy_identities=original.behavior_policy_identities,
            optimizer_step_before=original.optimizer_step_before,
        )


@pytest.mark.parametrize(
    "fault_at",
    ("after_gradient_before_stage", "after_stage_before_commit", "after_commit_before_ack"),
)
def test_same_instance_retry_after_mutation_fault_is_rejected(tmp_path: Path, fault_at: str):
    batch = _setup_batch(tmp_path)
    learner, store = _make_learner(tmp_path)
    parent_payload = learner.export_parent_checkpoint_bytes()
    with pytest.raises(InjectedUpdateFaultV1):
        learner.apply_durable_update_v1(batch=batch, fault_at=fault_at)
    update_id = next(store.updates_dir.glob("*.json")).stem
    step_after_fault = learner.optimizer_step
    gradients_after_fault = learner.gradient_applications
    records_after_fault = sorted(store.updates_dir.glob("*.json"))
    assert learner.restart_required is True
    with pytest.raises(PostCCLearnerError, match="RESTART_REQUIRED_FROM_DURABLE_PARENT"):
        learner.apply_durable_update_v1(batch=batch)
    assert learner.optimizer_step == step_after_fault
    assert learner.gradient_applications == gradients_after_fault
    assert sorted(store.updates_dir.glob("*.json")) == records_after_fault

    restarted = _restart(learner, tmp_path, parent_payload)
    assert restarted.restart_required is False
    result = restarted.apply_durable_update_v1(batch=batch)
    assert result.optimizer_step_after == 1
    assert restarted.gradient_applications == (1 if fault_at == "after_gradient_before_stage" else 0)
    assert store.get_record(update_id).commit_status == STATUS_COMMITTED
