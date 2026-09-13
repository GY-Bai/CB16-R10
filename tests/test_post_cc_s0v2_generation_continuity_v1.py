from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from cb16_local_opt.cc_critic_value_r0 import SeparateCritic
from cb16_local_opt.cc_runtime_boundary_r0 import COMPUTE_CHUNK, ECONOMIC_TERMINAL
from cb16_local_opt.post_cc_generation_continuity_v1 import (
    GenerationContinuityError,
    assert_behavior_checkpoint_immutable_v1,
    assert_same_logical_account_continuation_v1,
    commit_child_generation_v1,
    parameter_state_sha256_v1,
)
from cb16_local_opt.post_cc_learner_v1 import PostCCDurableReplayLearnerV1
from cb16_local_opt.post_cc_durable_collection_v1 import (
    DurableObservationCollectorV1,
    canonical_brain_vectors_from_account_v1,
)
from cb16_local_opt.post_cc_update_transaction_v1 import DurableUpdateStoreV1
from tests.cc_s0v2_support import (
    DEMO_MARKET_SOURCE_IDENTITY,
    DEMO_MARKET_SOURCE_VERSION,
    SCIENCE_SEMANTIC_VERSION,
    collect_durable_sequence_v1,
    make_account,
    make_brain,
    make_interval,
    make_policy_rng,
    materialize_v1,
)


def _updated_learner(tmp_path: Path, collected):
    batch = materialize_v1(tmp_path, collected.sequence_id, restart_verified=True).to_batch()
    target = make_brain()
    target.load_state_dict(collected.behavior_brain.state_dict())
    target.assert_gradient_ownership()
    learner = PostCCDurableReplayLearnerV1(
        target_actor=target,
        target_critic=SeparateCritic(7, 8),
        update_store=DurableUpdateStoreV1(tmp_path / "updates"),
        parent_policy_identity="cc-s0v2-policy-g0",
        target_policy_identity="cc-s0v2-target-policy-g1",
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
    )
    return learner, learner.apply_durable_update_v1(batch=batch)


def test_authorized_generation_switch_preserves_account_and_child_acts(tmp_path: Path):
    collected = collect_durable_sequence_v1(tmp_path)
    behavior_sha = parameter_state_sha256_v1(collected.behavior_brain)
    learner, result = _updated_learner(tmp_path, collected)
    runtime = collected.runtime
    account_snapshot = copy.deepcopy(runtime.account)
    lineage_before = runtime.account_lineage_id
    receipt = commit_child_generation_v1(
        runtime,
        child_checkpoint_sha256=result.child_checkpoint_sha256,
        new_policy_generation="1",
        new_policy_id="cc-s0v2-policy-g1",
        new_policy_sha256=result.child_checkpoint_sha256,
        expected_account_snapshot=account_snapshot,
        boundary_type=COMPUTE_CHUNK,
        committed_update_record=result.record,
    )
    assert receipt.account_fully_preserved is True
    assert receipt.account_truth_hash_before == receipt.account_truth_hash_after
    assert runtime.account_lineage_id == lineage_before
    assert runtime.policy_id == "cc-s0v2-policy-g1"
    assert parameter_state_sha256_v1(collected.behavior_brain) == behavior_sha
    assert assert_behavior_checkpoint_immutable_v1(collected.behavior_brain, expected_parameter_state_sha256=behavior_sha) is None

    child_rng = make_policy_rng(policy_id="cc-s0v2-policy-g1", account_lineage_id=lineage_before)
    collector = DurableObservationCollectorV1(
        str(tmp_path),
        science_semantic_version=SCIENCE_SEMANTIC_VERSION,
        market_source_identity=DEMO_MARKET_SOURCE_IDENTITY,
        market_source_version=DEMO_MARKET_SOURCE_VERSION,
    )
    box: dict = {}

    def child_callback(account_state, clocks):
        market, account_values, execution_values = canonical_brain_vectors_from_account_v1(account_state)
        logits, risk_loc, risk_log_scale = learner.target_actor(
            torch.tensor(market), torch.tensor(account_values), torch.tensor(execution_values)
        )
        from cb16_local_opt.cc_policy_distribution_r0 import sample_nominal

        nominal = sample_nominal(logits, risk_loc, risk_log_scale, child_rng)
        decision, _fact = collector.capture_policy_decision(
            account_state,
            account_lineage_id=runtime.account_lineage_id,
            decision_index=clocks.policy_decision_index,
            environment_time=clocks.environment_time,
            policy_generation="1",
            policy_id="cc-s0v2-policy-g1",
            policy_sha256=result.child_checkpoint_sha256,
            nominal=nominal,
        )
        box["decision"] = decision
        return decision

    runtime.step(
        make_interval(104.0, boundary_type=COMPUTE_CHUNK),
        child_callback,
        expected_predecessor_token=runtime.predecessor_token,
    )
    assert box["decision"].policy_id == "cc-s0v2-policy-g1"
    assert box["decision"].policy_sha256 == result.child_checkpoint_sha256
    assert runtime.account_lineage_id == lineage_before


def test_switch_at_forbidden_boundary_is_rejected(tmp_path: Path):
    collected = collect_durable_sequence_v1(tmp_path)
    _, result = _updated_learner(tmp_path, collected)
    runtime = collected.runtime
    with pytest.raises(GenerationContinuityError, match="GENERATION_SWITCH_BOUNDARY_FORBIDDEN"):
        commit_child_generation_v1(
            runtime,
            child_checkpoint_sha256=result.child_checkpoint_sha256,
            new_policy_generation="1",
            new_policy_id="p1",
            new_policy_sha256=result.child_checkpoint_sha256,
            expected_account_snapshot=copy.deepcopy(runtime.account),
            boundary_type=ECONOMIC_TERMINAL,
        )


def test_switch_outside_published_boundary_phase_is_rejected(tmp_path: Path):
    collected = collect_durable_sequence_v1(tmp_path)
    runtime = collected.runtime
    runtime.begin_interval(make_interval(101.0), expected_predecessor_token=runtime.predecessor_token)
    with pytest.raises(GenerationContinuityError, match="GENERATION_SWITCH_REQUIRES_PUBLISHED_BOUNDARY"):
        commit_child_generation_v1(
            runtime,
            child_checkpoint_sha256="a" * 64,
            new_policy_generation="1",
            new_policy_id="p1",
            new_policy_sha256="a" * 64,
            expected_account_snapshot=copy.deepcopy(runtime.account),
            boundary_type=COMPUTE_CHUNK,
        )


def test_new_flat_account_substitution_is_rejected(tmp_path: Path):
    collected = collect_durable_sequence_v1(tmp_path)
    account_before = collected.account_after_collection
    fresh_flat = make_account(account_id="new-flat-account")
    with pytest.raises(GenerationContinuityError, match="GENERATION_ACCOUNT_LEDGER_MUTATED_OR_REPLACED"):
        assert_same_logical_account_continuation_v1(
            account_lineage_id_before=collected.account_lineage_id,
            account_lineage_id_after=collected.account_lineage_id,
            account_before=account_before,
            account_after=fresh_flat,
        )
    with pytest.raises(GenerationContinuityError, match="GENERATION_EXPECTED_ACCOUNT_MISMATCH"):
        commit_child_generation_v1(
            collected.runtime,
            child_checkpoint_sha256="a" * 64,
            new_policy_generation="1",
            new_policy_id="p1",
            new_policy_sha256="a" * 64,
            expected_account_snapshot=fresh_flat,
            boundary_type=COMPUTE_CHUNK,
        )
