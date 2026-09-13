"""Shuffled-credit control must provably break action/consequence pairing."""

from __future__ import annotations

from dataclasses import replace
import random
import tempfile

from cb16_local_opt import post_cc_s1_tasks_v1 as T
from cb16_local_opt.post_cc_s1_credit_adapter_v1 import apply_credit_views_v1
from cb16_local_opt.post_cc_s1_training_loop_v1 import _apply_off_policy_reward_shuffle_v1, _materialize_batch_v1
from cb16_local_opt.cc_policy_rng_r0 import PolicyRNG
from cb16_local_opt.post_cc_generation_continuity_v1 import parameter_state_sha256_v1
from cb16_local_opt.post_cc_replay_materializer_v1 import ReplayMaterializerV1


def _collect_episodes(root: str, spec, context, count: int, *, lineage: str):
    account, _provenance = T.establish_account_context_v1(
        spec=spec, context=context, lineage=lineage, setup_root=f"{root}/setup"
    )
    actor = T.make_s1_brain_v1()
    policy_sha = parameter_state_sha256_v1(actor)
    sequence_ids = []
    for index in range(count):
        sequence_id = f"{lineage}-seq-{index}"
        sequence_ids.append(sequence_id)
        T.collect_episode_v1(
            spec=spec,
            context=context,
            account=account,
            root=root,
            lineage=f"{lineage}-{index}",
            policy_generation="0",
            policy_id="cc-s1-pairing-policy",
            policy_sha256=policy_sha,
            actor=actor,
            action_rng=PolicyRNG("cc-s1-pairing-policy", f"{lineage}-{index}", 1701 + index),
            env_rng=random.Random(1701 + index),
            sequence_id=sequence_id,
        )
    return sequence_ids


def test_delayed_shuffle_replaces_consequence_with_other_context_branch():
    spec = T.build_task_specs_v1()[T.TASK_DELAYED_CONSEQUENCE_CREDIT]
    own = spec.contexts[0]
    other = spec.contexts[1]
    with tempfile.TemporaryDirectory() as root:
        account, _provenance = T.establish_account_context_v1(
            spec=spec, context=own, lineage="cc-s1-shuffle-lineage", setup_root=f"{root}/setup"
        )
        actor = T.make_s1_brain_v1()
        overridden = T.collect_episode_v1(
            spec=spec,
            context=own,
            account=account,
            root=root,
            lineage="cc-s1-shuffle-lineage-a",
            policy_generation="0",
            policy_id="cc-s1-pairing-policy",
            policy_sha256=parameter_state_sha256_v1(actor),
            actor=actor,
            action_rng=PolicyRNG("cc-s1-pairing-policy", "cc-s1-shuffle-lineage-a", 1701),
            env_rng=random.Random(42),
            sequence_id="cc-s1-shuffle-sequence",
            branch_override=replace(other.dynamics.stage2_branches[0]),
        )
    own_marks = {branch.terminal_mark for branch in own.dynamics.stage2_branches}
    other_marks = {branch.terminal_mark for branch in other.dynamics.stage2_branches}
    assert own_marks.isdisjoint(other_marks)
    assert overridden.terminal_branch_mark in other_marks
    assert overridden.terminal_branch_mark not in own_marks
    assert overridden.reward != 0.0


def test_off_policy_reward_shuffle_preserves_multiset_and_breaks_pairing():
    spec = T.build_task_specs_v1()[T.TASK_OFF_POLICY_VTRACE_CORRECTION]
    context = spec.contexts[0]
    with tempfile.TemporaryDirectory() as root:
        sequence_ids = _collect_episodes(root, spec, context, 4, lineage="cc-s1-op-shuffle")
        batch, _materialized = _materialize_batch_v1(
            root=root, sequence_ids=tuple(sequence_ids), target_policy_identity="cc-s1-target"
        )
        original_rewards = sorted(float(sample.reward) for sample in batch.samples)
        shuffled_batch, permutation = _apply_off_policy_reward_shuffle_v1(
            batch, control_rng=random.Random(7)
        )
        shuffled_rewards = sorted(float(sample.reward) for sample in shuffled_batch.samples)
        assert shuffled_rewards == original_rewards
        assert sorted(permutation) == list(range(len(batch.samples)))
        assert permutation != list(range(len(batch.samples)))
        assert shuffled_batch.batch_content_sha256 != batch.batch_content_sha256
        for index, sample in enumerate(shuffled_batch.samples):
            assert sample.consequence_context["shuffled_credit_control_index"] == permutation[index]
