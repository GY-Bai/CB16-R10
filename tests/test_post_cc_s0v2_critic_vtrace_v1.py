from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cb16_local_opt.cc_critic_value_r0 import SeparateCritic
from cb16_local_opt.cc_policy_distribution_r0 import joint_log_prob
from cb16_local_opt.post_cc_critic_vtrace_v1 import (
    bootstrap_decision_v1,
    bootstrap_value_v1,
    classify_boundary_v1,
    joint_actor_critic_losses_v1,
)
from cb16_local_opt.post_cc_joint_batch_v1 import observation_to_tensors_v1
from tests.cc_s0v2_support import make_batch_v1, make_brain, make_canonical_observation_v1, make_joint_sample_v1


def _sample_with_policy_log_mu(actor, sample):
    # Force an exact float32 same-policy identity: both the persisted log_mu and
    # the target likelihood then come from the same evaluated vectorized path.
    from cb16_local_opt.post_cc_joint_policy_loss_v1 import target_joint_log_probs_v1

    placeholder = replace(
        sample,
        behavior_log_mu=0.0,
        boundary_type="OBJECTIVE_HORIZON_REACHED",
        bootstrap_state_ref_or_null=None,
    )
    with torch.no_grad():
        value = target_joint_log_probs_v1(actor, make_batch_v1((placeholder,)))[0].item()
    return replace(sample, behavior_log_mu=float(value))


def _force_same_policy_for_samples(actor, samples, bootstrap_map=None):
    from cb16_local_opt.post_cc_joint_policy_loss_v1 import target_joint_log_probs_v1

    placeholders = tuple(replace(sample, behavior_log_mu=0.0) for sample in samples)
    placeholder_batch = make_batch_v1(
        placeholders, bootstrap_observations_by_sequence=bootstrap_map
    )
    with torch.no_grad():
        values = target_joint_log_probs_v1(actor, placeholder_batch)
    return tuple(
        replace(sample, behavior_log_mu=float(values[index].item()))
        for index, sample in enumerate(samples)
    )


def test_boundary_bootstrap_map_is_explicit_and_fail_closed():
    assert bootstrap_value_v1("ECONOMIC_TERMINAL", 9.0) == 0.0
    assert bootstrap_value_v1("OBJECTIVE_HORIZON_REACHED", 9.0) == 0.0
    assert bootstrap_value_v1("COMPUTE_CHUNK", 9.0) == 9.0
    assert bootstrap_value_v1("PAUSE", -2.0) == -2.0
    with pytest.raises(ValueError, match="UNKNOWN_BOUNDARY_TYPE"):
        bootstrap_value_v1("NOT_A_BOUNDARY", 1.0)
    with pytest.raises(ValueError, match="CONTINUE_BOUNDARY_HAS_NO_BOOTSTRAP"):
        bootstrap_decision_v1("CONTINUE", mechanical_terminal=False, next_value=1.0)
    with pytest.raises(ValueError, match="TERMINAL_BOUNDARY_MUST_NOT_SUPPLY_BOOTSTRAP"):
        bootstrap_decision_v1("ECONOMIC_TERMINAL", mechanical_terminal=True, next_value=1.0)
    assert classify_boundary_v1("COMPUTE_CHUNK") == "TRUNCATION"
    assert classify_boundary_v1("OBJECTIVE_HORIZON_REACHED") == "TERMINAL"
    assert classify_boundary_v1("CONTINUE") == "CONTINUE"


def test_same_policy_vtrace_has_unit_ratio_and_finite_losses(tmp_path: Path):
    actor = make_brain()
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG", nominal_target_risk=0.4)
    sample = _sample_with_policy_log_mu(actor, sample)
    batch = make_batch_v1((sample,))
    critic = SeparateCritic(7, 8)
    losses = joint_actor_critic_losses_v1(actor, critic, batch)
    assert losses.diagnostics["rho_mean"] == pytest.approx(1.0)
    assert losses.diagnostics["rho_clip_fraction"] == pytest.approx(0.0)
    assert torch.isfinite(losses.actor_loss).all()
    assert torch.isfinite(losses.critic_loss).all()
    assert losses.vtrace.vs.requires_grad is False


def test_vtrace_recurrence_matches_manual_terminal_calculation(tmp_path: Path):
    actor = make_brain()
    first = make_joint_sample_v1(
        root=tmp_path,
        decision_index=0,
        environment_time=0,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        reward=0.01,
        boundary_type="CONTINUE",
    )
    second = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-vt-1",
        decision_index=1,
        environment_time=1,
        nominal_direction="SHORT",
        nominal_target_risk=0.6,
        reward=0.02,
        boundary_type="ECONOMIC_TERMINAL",
    )
    first, second = _force_same_policy_for_samples(actor, (first, second))
    batch = make_batch_v1((first, second))
    critic = SeparateCritic(7, 8)
    losses = joint_actor_critic_losses_v1(actor, critic, batch)
    with torch.no_grad():
        values = critic(batch.critic_observations)
    rewards = batch.rewards
    discounts = batch.discounts
    manual_last_vs = rewards[1]
    manual_first_vs = rewards[0] + discounts[0] * manual_last_vs
    expected_vs = torch.stack([manual_first_vs, manual_last_vs])
    expected_advantages = torch.stack(
        [
            rewards[0] + discounts[0] * manual_last_vs - values[0],
            rewards[1] - values[1],
        ]
    )
    assert torch.allclose(losses.vtrace.vs, expected_vs, atol=1e-6, rtol=1e-6)
    assert torch.allclose(losses.vtrace.pg_advantages, expected_advantages, atol=1e-6, rtol=1e-6)


def test_truncation_bootstraps_from_durable_next_observation_and_terminal_does_not(tmp_path: Path):
    actor = make_brain()
    first = make_joint_sample_v1(
        root=tmp_path,
        decision_index=0,
        environment_time=0,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        reward=0.0,
        boundary_type="CONTINUE",
    )
    truncation = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-trunc",
        decision_index=1,
        environment_time=1,
        nominal_direction="LONG",
        nominal_target_risk=0.4,
        reward=0.0,
        boundary_type="COMPUTE_CHUNK",
        bootstrap_state_ref_or_null="b" * 64,
    )
    bootstrap_fact = make_canonical_observation_v1(decision_index=2, environment_time=2)
    bootstrap_map = {truncation.sequence_id: bootstrap_fact}
    first, truncation = _force_same_policy_for_samples(actor, (first, truncation), bootstrap_map)
    batch_trunc = make_batch_v1(
        (first, truncation),
        bootstrap_observations_by_sequence=bootstrap_map,
    )
    critic = SeparateCritic(7, 8)
    with torch.no_grad():
        critic.net[-1].bias.fill_(0.5)
    losses_trunc = joint_actor_critic_losses_v1(actor, critic, batch_trunc)
    assert losses_trunc.diagnostics["truncation_sequence_count"] == 1
    assert losses_trunc.diagnostics["terminal_sequence_count"] == 0
    market, account, execution = observation_to_tensors_v1(bootstrap_fact)
    with torch.no_grad():
        next_value = float(critic(torch.cat([market, account, execution]).reshape(1, -1)).item())
    assert losses_trunc.vtrace.vs[-1].item() == pytest.approx(
        float(batch_trunc.discounts[-1].item()) * next_value, abs=1e-6
    )

    terminal = replace(
        truncation,
        transition_id="cc-s0v2-unit-transition-term",
        boundary_type="OBJECTIVE_HORIZON_REACHED",
        bootstrap_state_ref_or_null=None,
    )
    first_term, terminal = _force_same_policy_for_samples(actor, (first, terminal))
    batch_term = make_batch_v1((first_term, terminal))
    losses_term = joint_actor_critic_losses_v1(actor, SeparateCritic(7, 8), batch_term)
    assert losses_term.vtrace.vs[-1].item() == pytest.approx(0.0, abs=1e-6)
