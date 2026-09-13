from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cb16_local_opt.cc_policy_distribution_r0 import joint_log_prob
from cb16_local_opt.post_cc_joint_batch_v1 import JointActionBatchV1, observation_to_tensors_v1
from cb16_local_opt.post_cc_joint_policy_loss_v1 import (
    assert_persisted_log_mu_used_v1,
    scalar_reference_log_prob_v1,
    target_joint_log_probs_v1,
    target_likelihoods_v1,
)
from tests.cc_s0v2_support import make_batch_v1, make_brain, make_joint_sample_v1


def _actor_outputs(actor, fact):
    market, account, execution = observation_to_tensors_v1(fact)
    return actor(market.reshape(1, -1), account.reshape(1, -1), execution.reshape(1, -1))


def _sample_with_reference_log_mu(actor, sample):
    logits, risk_loc, risk_log_scale = _actor_outputs(actor, sample.observation)
    reference = scalar_reference_log_prob_v1(
        direction_logits=logits.reshape(-1),
        risk_loc=risk_loc.reshape(-1),
        risk_log_scale=risk_log_scale.reshape(-1),
        direction=sample.nominal_direction,
        risk=float(sample.nominal_target_risk),
    )
    return replace(sample, behavior_log_mu=float(reference.detach().item()))


def test_flat_point_mass_matches_scalar_reference(tmp_path: Path):
    actor = make_brain()
    sample = make_joint_sample_v1(
        root=tmp_path, nominal_direction="FLAT", nominal_target_risk=0.0, behavior_log_mu=0.0
    )
    batch = make_batch_v1((sample,))
    logits, risk_loc, risk_log_scale = _actor_outputs(actor, sample.observation)
    expected = joint_log_prob(logits.reshape(-1), risk_loc.reshape(-1), risk_log_scale.reshape(-1), "FLAT", 0.0)
    actual = target_joint_log_probs_v1(actor, batch)
    assert torch.allclose(actual, expected.reshape(1), atol=1e-6, rtol=1e-6)


def test_long_and_short_density_match_scalar_reference(tmp_path: Path):
    actor = make_brain()
    long_sample = make_joint_sample_v1(
        root=tmp_path,
        decision_index=0,
        environment_time=0,
        nominal_direction="LONG",
        nominal_target_risk=0.35,
        boundary_type="CONTINUE",
    )
    short_sample = make_joint_sample_v1(
        root=tmp_path,
        transition_id="cc-s0v2-unit-transition-short",
        decision_index=1,
        environment_time=1,
        nominal_direction="SHORT",
        nominal_target_risk=0.65,
        boundary_type="ECONOMIC_TERMINAL",
    )
    batch = make_batch_v1((long_sample, short_sample))
    actual = target_joint_log_probs_v1(actor, batch)
    for index, sample in enumerate((long_sample, short_sample)):
        logits, risk_loc, risk_log_scale = _actor_outputs(actor, sample.observation)
        expected = joint_log_prob(
            logits.reshape(-1),
            risk_loc.reshape(-1),
            risk_log_scale.reshape(-1),
            sample.nominal_direction,
            float(sample.nominal_target_risk),
        )
        assert torch.allclose(actual[index], expected.reshape(()), atol=1e-6, rtol=1e-6)


def test_same_policy_log_pi_equals_persisted_log_mu(tmp_path: Path):
    actor = make_brain()
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG", nominal_target_risk=0.4)
    persisted = _sample_with_reference_log_mu(actor, sample)
    batch = make_batch_v1((persisted,))
    assert_persisted_log_mu_used_v1(batch)
    log_pi = target_joint_log_probs_v1(actor, batch)
    assert torch.allclose(log_pi, batch.behavior_log_mu, atol=1e-6, rtol=1e-6)


def test_off_policy_ratio_is_exp_log_pi_minus_log_mu_in_correct_order(tmp_path: Path):
    behavior_actor = make_brain(seed=1701, direction_bias=(-0.25, -1.0, 0.35))
    target_actor = make_brain(seed=1710, direction_bias=(0.5, -0.8, 0.1))
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG", nominal_target_risk=0.4)
    sample = _sample_with_reference_log_mu(behavior_actor, sample)
    batch = make_batch_v1((sample,))
    likelihoods = target_likelihoods_v1(target_actor, batch)
    assert torch.allclose(
        likelihoods.log_ratio_log_pi_minus_log_mu,
        likelihoods.log_pi - likelihoods.log_mu,
        atol=1e-7,
        rtol=1e-7,
    )
    expected_ratio = torch.exp(likelihoods.log_pi.detach() - likelihoods.log_mu)
    assert torch.allclose(likelihoods.ratio, expected_ratio, atol=1e-6, rtol=1e-6)
    reversed_ratio = torch.exp(likelihoods.log_mu - likelihoods.log_pi.detach())
    assert not torch.allclose(likelihoods.ratio, reversed_ratio, atol=1e-4, rtol=1e-4)


def test_execution_context_clamping_or_rejection_does_not_change_nominal_likelihood(tmp_path: Path):
    actor = make_brain()
    base = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG", nominal_target_risk=0.4)
    executed_flat = replace(base, consequence_context={"executed_direction": "FLAT", "executed_quantity": 0.0})
    executed_clamped = replace(
        base,
        consequence_context={"executed_direction": "LONG", "executed_quantity": 3.0, "execution_clamped": True},
    )
    with torch.no_grad():
        base_log_prob = target_joint_log_probs_v1(actor, make_batch_v1((base,)))
        flat_log_prob = target_joint_log_probs_v1(actor, make_batch_v1((executed_flat,)))
        clamped_log_prob = target_joint_log_probs_v1(actor, make_batch_v1((executed_clamped,)))
    assert torch.allclose(base_log_prob, flat_log_prob, atol=1e-7, rtol=1e-7)
    assert torch.allclose(base_log_prob, clamped_log_prob, atol=1e-7, rtol=1e-7)


def test_reconstructed_or_fabricated_log_mu_is_rejected(tmp_path: Path):
    actor = make_brain()
    sample = make_joint_sample_v1(root=tmp_path, nominal_direction="LONG")
    with pytest.raises(ValueError, match="LOG_MU_NOT_DECISION_TIME_PERSISTED"):
        replace(sample, log_mu_source="RECONSTRUCTED_FROM_REWARD").validate()
    good = _sample_with_reference_log_mu(actor, sample)
    batch = make_batch_v1((good,))
    fabricated = replace(batch, behavior_log_mu=torch.zeros_like(batch.behavior_log_mu))
    with pytest.raises(ValueError, match="LOG_MU_MISMATCH"):
        fabricated.validate()
