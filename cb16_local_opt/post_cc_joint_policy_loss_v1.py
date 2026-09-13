"""S0-v2 canonical target joint likelihood and policy-gradient loss.

Target ``log_pi`` is always evaluated on the nominal behaviour direction and
nominal target risk persisted in durable replay.  Execution outcomes may only
appear as consequence context; they never replace the nominal action.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from .cc_policy_brain_r0 import CCCentralBrain
from .cc_policy_distribution_r0 import INDEX as DIRECTION_INDEX
from .cc_policy_distribution_r0 import joint_log_prob, nonflat_risk_log_density
from .cc_policy_loss_r0 import actor_policy_gradient_loss
from .post_cc_joint_batch_v1 import JointActionBatchV1

FLAT_INDEX_V1 = DIRECTION_INDEX["FLAT"]
_LOG_TWO_PI_OVER_TWO = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class TargetJointLikelihoodsV1:
    log_pi: Tensor
    log_mu: Tensor
    log_ratio_log_pi_minus_log_mu: Tensor
    ratio: Tensor

    def validate(self) -> "TargetJointLikelihoodsV1":
        for name in ("log_pi", "log_mu", "log_ratio_log_pi_minus_log_mu", "ratio"):
            tensor = getattr(self, name)
            if not isinstance(tensor, Tensor) or tensor.ndim != 1:
                raise ValueError(f"{name} must be a 1D tensor")
            if not torch.isfinite(tensor.detach()).all():
                raise ValueError(f"{name} must be finite")
        if not torch.equal(
            self.log_ratio_log_pi_minus_log_mu.detach(), (self.log_pi - self.log_mu).detach()
        ):
            raise ValueError("LOG_RATIO_MUST_BE_LOG_PI_MINUS_LOG_MU")
        return self


@dataclass(frozen=True)
class JointPolicyLossResultV1:
    actor_loss: Tensor
    critic_loss: Tensor
    likelihoods: TargetJointLikelihoodsV1
    values: Tensor
    vtrace_vs: Tensor
    advantages: Tensor
    rhos: Tensor
    clipped_rhos: Tensor
    cs: Tensor
    mask: Tensor
    diagnostics: Mapping[str, Any]


def _validate_actor_outputs(logits: Tensor, loc: Tensor, log_scale: Tensor) -> None:
    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("DIRECTION_LOGITS_SHAPE")
    if loc.shape != logits.shape or log_scale.shape != logits.shape:
        raise ValueError("RISK_HEAD_SHAPE")
    for name, tensor in (("logits", logits), ("risk_loc", loc), ("risk_log_scale", log_scale)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be finite")


def scalar_reference_log_prob_v1(
    *,
    direction_logits: Tensor,
    risk_loc: Tensor,
    risk_log_scale: Tensor,
    direction: str,
    risk: float,
) -> Tensor:
    """Frozen scalar reference used by component known-answer tests."""
    return joint_log_prob(direction_logits, risk_loc, risk_log_scale, direction, risk)


def target_joint_log_probs_v1(actor: CCCentralBrain, batch: JointActionBatchV1) -> Tensor:
    """Vectorized canonical target joint ``log_pi`` on nominal actions.

    Formula identity with the frozen scalar ``joint_log_prob``:
      FLAT       -> log softmax(direction_logits)[FLAT]
      non-FLAT   -> log softmax(direction_logits)[d]
                    + Normal(logit(r); loc_d, scale_d)
                    - log(r) - log1p(-r)
    """
    batch.validate()
    logits, risk_loc, risk_log_scale = actor(
        batch.market_observations, batch.account_observations, batch.execution_observations
    )
    _validate_actor_outputs(logits, risk_loc, risk_log_scale)
    log_direction = torch.log_softmax(logits, dim=-1)
    direction_log_prob = log_direction.gather(1, batch.direction_indices[:, None]).squeeze(1)

    point_mass = batch.risk_point_mass_mask
    flat_index_ok = point_mass & (batch.direction_indices == FLAT_INDEX_V1)
    if not torch.equal(flat_index_ok, point_mass):
        raise ValueError("POINT_MASS_MASK_MUST_BE_FLAT_ONLY")
    if bool(point_mass.any()):
        flat_risks = batch.target_risks[point_mass]
        if not torch.equal(flat_risks, torch.zeros_like(flat_risks)):
            raise ValueError("FLAT_TARGET_RISK_MUST_BE_ZERO")

    non_flat = ~point_mass
    if bool(non_flat.any()):
        risks = batch.target_risks[non_flat]
        if not bool(((risks > 0.0) & (risks < 1.0)).all()):
            raise ValueError("NONFLAT_TARGET_RISK_OUTSIDE_FROZEN_SUPPORT")
        selected_loc = risk_loc[non_flat].gather(
            1, batch.direction_indices[non_flat, None]
        ).squeeze(1)
        selected_log_scale = risk_log_scale[non_flat].gather(
            1, batch.direction_indices[non_flat, None]
        ).squeeze(1)
        scale = torch.exp(selected_log_scale)
        if not torch.isfinite(scale).all() or not bool((scale > 0.0).all()):
            raise ValueError("BAD_RISK_SCALE")
        z = torch.log(risks) - torch.log1p(-risks)
        normal = (
            -0.5 * ((z - selected_loc) / scale) ** 2
            - selected_log_scale
            - _LOG_TWO_PI_OVER_TWO
        )
        jacobian = -torch.log(risks) - torch.log1p(-risks)
        density = normal + jacobian
        density_full = torch.zeros_like(direction_log_prob)
        density_full[non_flat] = density
        return direction_log_prob + density_full
    return direction_log_prob


def target_likelihoods_v1(actor: CCCentralBrain, batch: JointActionBatchV1) -> TargetJointLikelihoodsV1:
    batch.validate()
    log_pi = target_joint_log_probs_v1(actor, batch)
    log_mu = batch.behavior_log_mu
    log_ratio = log_pi - log_mu
    ratio = torch.exp(torch.clamp(log_ratio.detach(), -80.0, 80.0))
    return TargetJointLikelihoodsV1(
        log_pi=log_pi,
        log_mu=log_mu,
        log_ratio_log_pi_minus_log_mu=log_ratio,
        ratio=ratio,
    ).validate()


def assert_persisted_log_mu_used_v1(batch: JointActionBatchV1) -> None:
    """Fail closed if any sample's likelihood input could be reconstructed."""
    batch.validate()
    for index, sample in enumerate(batch.samples):
        if sample.log_mu_source != "DECISION_TIME_PERSISTED":
            raise ValueError("LOG_MU_NOT_DECISION_TIME_PERSISTED")
        if float(batch.behavior_log_mu[index].item()) != float(sample.behavior_log_mu):
            raise ValueError("LOG_MU_TENSOR_FABRICATED")
        if not sample.behavior_policy_id or not sample.behavior_policy_sha256:
            raise ValueError("BEHAVIOR_POLICY_IDENTITY_REQUIRED")
        if sample.consequence_context is not None:
            for forbidden in ("log_mu", "nominal_direction", "nominal_target_risk"):
                if forbidden in sample.consequence_context:
                    raise ValueError("EXECUTION_CONTEXT_MUST_NOT_REPLACE_NOMINAL_ACTION")


def weighted_mean_v1(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("WEIGHTED_MEAN_SHAPE")
    if not torch.isfinite(values.detach()).all() or not torch.isfinite(weights.detach()).all():
        raise ValueError("WEIGHTED_MEAN_NONFINITE")
    if not bool((weights.detach() > 0.0).all()):
        raise ValueError("WEIGHTS_MUST_BE_POSITIVE")
    return torch.sum(values * weights) / torch.sum(weights)


def actor_loss_from_advantages_v1(log_pi: Tensor, advantages: Tensor, weights: Tensor) -> Tensor:
    if log_pi.ndim != 1 or advantages.ndim != 1 or weights.ndim != 1:
        raise ValueError("ACTOR_LOSS_SHAPE")
    if log_pi.shape != advantages.shape or log_pi.shape != weights.shape:
        raise ValueError("ACTOR_LOSS_SHAPE")
    return actor_policy_gradient_loss(log_pi, advantages, mask=weights)


def critic_loss_v1(values: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or targets.ndim != 1 or weights.ndim != 1:
        raise ValueError("CRITIC_LOSS_SHAPE")
    if values.shape != targets.shape or values.shape != weights.shape:
        raise ValueError("CRITIC_LOSS_SHAPE")
    return weighted_mean_v1((values - targets.detach()) ** 2, weights)
