"""S0-v2 Critic, boundary-aware bootstrap and V-trace integration.

Boundary semantics are explicit and fail-closed: economic/task terminals have
zero future value, while compute/replay truncations bootstrap from the durable
next observation.  A single generic ``done`` flag is not accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .cc_critic_value_r0 import SeparateCritic, assert_disjoint_parameters, mean_value_loss
from .cc_policy_brain_r0 import CCCentralBrain
from .cc_vtrace_r0 import VTraceReturns, vtrace
from .post_cc_joint_batch_v1 import (
    BOUNDARY_CLASS_CONTINUE,
    BOUNDARY_CLASS_TERMINAL,
    BOUNDARY_CLASS_TRUNCATION,
    JointActionBatchV1,
    boundary_is_terminal_v1,
    boundary_requires_bootstrap_v1,
    boundary_semantics_v1,
    classify_boundary_v1,
)
from .post_cc_joint_policy_loss_v1 import (
    TargetJointLikelihoodsV1,
    actor_loss_from_advantages_v1,
    critic_loss_v1,
    target_likelihoods_v1,
)


@dataclass(frozen=True)
class BoundaryBootstrapDecisionV1:
    boundary_type: str
    boundary_class: str
    boundary_semantics: str
    bootstrap_value: float
    mechanical_terminal: bool
    uses_durable_next_observation: bool


@dataclass(frozen=True)
class JointActorCriticLossV1:
    actor_loss: Tensor
    critic_loss: Tensor
    likelihoods: TargetJointLikelihoodsV1
    values: Tensor
    vtrace: VTraceReturns
    mask: Tensor
    bootstrap_decisions: tuple[BoundaryBootstrapDecisionV1, ...]
    diagnostics: Mapping[str, Any]


def bootstrap_value_v1(
    boundary_type: str, next_value: float, *, mechanical_terminal: bool = False
) -> float:
    """Explicit per-class bootstrap map; unknown boundaries fail closed."""
    boundary_class = classify_boundary_v1(boundary_type, mechanical_terminal=mechanical_terminal)
    if boundary_class == BOUNDARY_CLASS_TERMINAL:
        return 0.0
    if boundary_class == BOUNDARY_CLASS_TRUNCATION:
        value = float(next_value)
        if not math.isfinite(value):
            raise ValueError("BOOTSTRAP_VALUE_NONFINITE")
        return value
    raise ValueError("CONTINUE_BOUNDARY_HAS_NO_BOOTSTRAP")


def bootstrap_decision_v1(
    boundary_type: str, *, mechanical_terminal: bool, next_value: float | None
) -> BoundaryBootstrapDecisionV1:
    semantics = boundary_semantics_v1(boundary_type, mechanical_terminal=mechanical_terminal)
    boundary_class = classify_boundary_v1(boundary_type, mechanical_terminal=mechanical_terminal)
    if boundary_class == BOUNDARY_CLASS_CONTINUE:
        raise ValueError("CONTINUE_BOUNDARY_HAS_NO_BOOTSTRAP")
    if boundary_class == BOUNDARY_CLASS_TERMINAL:
        if next_value is not None:
            raise ValueError("TERMINAL_BOUNDARY_MUST_NOT_SUPPLY_BOOTSTRAP")
        return BoundaryBootstrapDecisionV1(
            boundary_type=boundary_type,
            boundary_class=boundary_class,
            boundary_semantics=semantics,
            bootstrap_value=0.0,
            mechanical_terminal=bool(mechanical_terminal),
            uses_durable_next_observation=False,
        )
    if next_value is None:
        raise ValueError("TRUNCATION_BOUNDARY_REQUIRES_BOOTSTRAP")
    value = float(next_value)
    if not math.isfinite(value):
        raise ValueError("BOOTSTRAP_VALUE_NONFINITE")
    return BoundaryBootstrapDecisionV1(
        boundary_type=boundary_type,
        boundary_class=boundary_class,
        boundary_semantics=semantics,
        bootstrap_value=value,
        mechanical_terminal=bool(mechanical_terminal),
        uses_durable_next_observation=True,
    )


def _sequence_bootstrap_value(
    critic: SeparateCritic,
    batch: JointActionBatchV1,
    sequence_index: int,
) -> float:
    if sequence_index in batch.bootstrap_sequence_indices:
        row_index = batch.bootstrap_sequence_indices.index(sequence_index)
        if batch.bootstrap_observations is None:
            raise ValueError("BOOTSTRAP_OBSERVATIONS_REQUIRED")
        with torch.no_grad():
            return float(critic(batch.bootstrap_observations[row_index : row_index + 1]).reshape(()).item())
    return 0.0


def joint_actor_critic_losses_v1(
    actor: CCCentralBrain,
    critic: SeparateCritic,
    batch: JointActionBatchV1,
    *,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    pg_rho_bar: float = 1.0,
) -> JointActorCriticLossV1:
    """Compute canonical joint Actor/Critic losses with separate V-trace semantics."""
    assert_disjoint_parameters(actor, critic)
    batch.validate()
    likelihoods = target_likelihoods_v1(actor, batch)
    values = critic(batch.critic_observations)
    if values.ndim != 1 or values.shape[0] != len(batch.samples):
        raise ValueError("CRITIC_VALUE_SHAPE")

    vtrace_chunks: list[VTraceReturns] = []
    bootstrap_decisions: list[BoundaryBootstrapDecisionV1] = []
    for sequence_index, (start, end) in enumerate(batch.sequence_offsets):
        final_sample = batch.samples[end - 1]
        sequence_boundary = final_sample.boundary_type
        sequence_mechanical = bool(batch.mechanical_terminals[end - 1])
        if sequence_mechanical != bool(final_sample.mechanical_terminal):
            raise ValueError("MECHANICAL_TERMINAL_TENSOR_MISMATCH")
        next_value = _sequence_bootstrap_value(critic, batch, sequence_index)
        bootstrap_decision = bootstrap_decision_v1(
            sequence_boundary,
            mechanical_terminal=sequence_mechanical,
            next_value=next_value if boundary_requires_bootstrap_v1(sequence_boundary) else None,
        )
        bootstrap_decisions.append(bootstrap_decision)
        bootstrap_tensor = torch.tensor(
            bootstrap_decision.bootstrap_value,
            dtype=values.dtype,
            device=values.device,
        )
        vtrace_chunks.append(
            vtrace(
                batch.rewards[start:end],
                values[start:end].detach(),
                bootstrap_tensor,
                likelihoods.log_pi[start:end].detach(),
                batch.behavior_log_mu[start:end],
                batch.discounts[start:end],
                rho_bar=float(rho_bar),
                c_bar=float(c_bar),
                pg_rho_bar=float(pg_rho_bar),
            )
        )
    vs = torch.cat([chunk.vs for chunk in vtrace_chunks], dim=0)
    advantages = torch.cat([chunk.pg_advantages for chunk in vtrace_chunks], dim=0)
    rhos = torch.cat([chunk.rhos for chunk in vtrace_chunks], dim=0)
    clipped_rhos = torch.cat([chunk.clipped_rhos for chunk in vtrace_chunks], dim=0)
    cs = torch.cat([chunk.cs for chunk in vtrace_chunks], dim=0)
    vtrace_all = VTraceReturns(vs=vs, pg_advantages=advantages, rhos=rhos, clipped_rhos=clipped_rhos, cs=cs)

    actor_loss = actor_loss_from_advantages_v1(likelihoods.log_pi, advantages, batch.replay_weights)
    critic_loss = critic_loss_v1(values, vs, batch.replay_weights)
    clip_fraction = float((rhos.detach() > 1.0).float().mean().item())
    diagnostics: dict[str, Any] = {
        "rho_mean": float(rhos.detach().mean().item()),
        "rho_clip_fraction": clip_fraction,
        "c_clip_fraction": float((cs.detach() > 1.0).float().mean().item()),
        "sample_count": len(batch.samples),
        "sequence_count": len(batch.sequence_offsets),
        "terminal_sequence_count": int(
            sum(1 for decision in bootstrap_decisions if decision.boundary_class == BOUNDARY_CLASS_TERMINAL)
        ),
        "mechanical_terminal_sequence_count": int(
            sum(1 for decision in bootstrap_decisions if decision.boundary_semantics == "MECHANICAL_TERMINAL")
        ),
        "boundary_semantics_by_sequence": tuple(decision.boundary_semantics for decision in bootstrap_decisions),
        "truncation_sequence_count": int(
            sum(1 for decision in bootstrap_decisions if decision.boundary_class == BOUNDARY_CLASS_TRUNCATION)
        ),
        "same_policy_log_ratio_abs_max": float(
            (likelihoods.log_pi.detach() - batch.behavior_log_mu).abs().max().item()
        ),
    }
    return JointActorCriticLossV1(
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        likelihoods=likelihoods,
        values=values,
        vtrace=vtrace_all,
        mask=batch.replay_weights,
        bootstrap_decisions=tuple(bootstrap_decisions),
        diagnostics=diagnostics,
    )


def audit_gradient_ownership_v1(actor: CCCentralBrain, critic: SeparateCritic) -> Mapping[str, Any]:
    """Fail closed unless the frozen market organ stays frozen and trainable paths get gradients."""
    actor.assert_gradient_ownership()
    assert_disjoint_parameters(actor, critic)
    frozen = actor.market_organ
    for name, parameter in frozen.named_parameters():
        if parameter.requires_grad:
            raise RuntimeError(f"FROZEN_MARKET_ORGAN_REQUIRES_GRAD:{name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"FROZEN_MARKET_ORGAN_GRAD_NONFINITE:{name}")
            if torch.count_nonzero(parameter.grad).item() != 0:
                raise RuntimeError(f"FROZEN_MARKET_ORGAN_GRAD_NONZERO:{name}")

    actor_groups = {
        "account_stem": actor.account_stem,
        "fusion": actor.fusion,
        "direction_head": actor.direction_head,
        "risk_loc_head": actor.risk_loc_head,
        "risk_log_scale": None,
    }
    summary: dict[str, Any] = {"frozen_market_organ_requires_grad": False, "frozen_market_organ_grad_zero": True}
    for name, module in actor_groups.items():
        parameters = (
            [actor.risk_log_scale]
            if module is None
            else [p for p in module.parameters()]
        )
        if any(not parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"TRAINABLE_ACTOR_PATH_FROZEN:{name}")
        nonzero = 0
        for parameter in parameters:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"ACTOR_GRAD_NONFINITE:{name}")
                nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
        summary[f"{name}_nonzero_gradient_parameter_count"] = nonzero
    critic_params = list(critic.parameters())
    if any(not parameter.requires_grad for parameter in critic_params):
        raise RuntimeError("CRITIC_PATH_FROZEN")
    critic_nonzero = 0
    for parameter in critic_params:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError("CRITIC_GRAD_NONFINITE")
            critic_nonzero += int(torch.count_nonzero(parameter.grad).item() > 0)
    summary["critic_nonzero_gradient_parameter_count"] = critic_nonzero
    summary["actor_critic_parameter_alias"] = False
    summary["all_gradients_finite"] = True
    if summary["risk_log_scale_nonzero_gradient_parameter_count"] <= 0:
        raise RuntimeError("RISK_LOG_SCALE_NO_GRADIENT")
    return summary
