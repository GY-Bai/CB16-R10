from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from cb16_local_opt.gradient_ownership_r0 import (
    FROZEN_MARKET_ORGAN,
    GRADIENT_OWNERSHIP_SCHEMA_R0,
    TRAINABLE_ACCOUNT_STEM,
    TRAINABLE_ACTOR,
    TRAINABLE_CRITIC,
    TRAINABLE_FUSION,
    GradientOwnershipRuleR0,
    GradientOwnershipSpecR0,
    apply_gradient_ownership_r0,
    audit_gradients_r0,
    audit_parameter_mutation_r0,
    partition_named_parameters_r0,
    snapshot_parameters_r0,
)


class TinyOwnedActorCritic(nn.Module):
    """BC-032 known-answer ownership fixture, not a production architecture.

    ``market_organs`` stands for the upstream frozen market representation
    producer.  The account stem, fusion, Actor and Critic are deliberately
    separate trainable groups so the gate can positively test every path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.market_organs = nn.Linear(2, 2, bias=False)
        self.account_stem = nn.Linear(2, 2, bias=False)
        self.fusion = nn.Linear(4, 3, bias=False)
        self.actor = nn.Linear(3, 1, bias=False)
        self.critic = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.market_organs.weight.copy_(torch.tensor([[0.5, 0.25], [0.75, 0.5]]))
            self.account_stem.weight.copy_(torch.tensor([[0.4, 0.2], [0.3, 0.6]]))
            self.fusion.weight.copy_(
                torch.tensor(
                    [
                        [0.2, 0.3, 0.4, 0.5],
                        [0.6, 0.5, 0.4, 0.3],
                        [0.25, 0.35, 0.45, 0.55],
                    ]
                )
            )
            self.actor.weight.copy_(torch.tensor([[0.7, 0.6, 0.5]]))
            self.critic.weight.copy_(torch.tensor([[0.4, 0.8, 0.9]]))

    def forward(self, market: torch.Tensor, account: torch.Tensor):
        market_features = self.market_organs(market)
        account_features = self.account_stem(account)
        fused = torch.tanh(self.fusion(torch.cat((market_features, account_features), dim=-1)))
        return self.actor(fused), self.critic(fused)


def _spec() -> GradientOwnershipSpecR0:
    spec = GradientOwnershipSpecR0(
        schema_version=GRADIENT_OWNERSHIP_SCHEMA_R0,
        rules=(
            GradientOwnershipRuleR0("market-organs", "market_organs", FROZEN_MARKET_ORGAN),
            GradientOwnershipRuleR0("account-stem", "account_stem", TRAINABLE_ACCOUNT_STEM),
            GradientOwnershipRuleR0("fusion", "fusion", TRAINABLE_FUSION),
            GradientOwnershipRuleR0("actor", "actor", TRAINABLE_ACTOR),
            GradientOwnershipRuleR0("critic", "critic", TRAINABLE_CRITIC),
        ),
    )
    spec.validate()
    return spec


def _backward_fixture(model: TinyOwnedActorCritic):
    market = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    account = torch.tensor([[0.5, 1.5]], dtype=torch.float32)
    actor_out, critic_out = model(market, account)
    loss = actor_out.square().mean() + critic_out.square().mean()
    loss.backward()
    return loss


def test_frozen_market_organs_have_zero_gradient_and_zero_optimizer_mutation() -> None:
    torch.manual_seed(7)
    model = TinyOwnedActorCritic()
    spec = _spec()
    apply_gradient_ownership_r0(model, spec)
    before = snapshot_parameters_r0(model, spec)

    _backward_fixture(model)
    gradient_report = audit_gradients_r0(model, spec)
    assert gradient_report.passed is True
    frozen = next(group for group in gradient_report.groups if group.role == FROZEN_MARKET_ORGAN)
    assert frozen.parameter_count == 1
    assert frozen.parameters_with_nonzero_gradient == 0
    assert model.market_organs.weight.requires_grad is False
    assert model.market_organs.weight.grad is None

    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
    optimizer.step()
    mutation_report = audit_parameter_mutation_r0(model, spec, before)
    assert mutation_report.passed is True
    frozen_mutation = next(group for group in mutation_report.groups if group.role == FROZEN_MARKET_ORGAN)
    assert frozen_mutation.changed_parameter_count == 0
    assert torch.equal(before["market_organs.weight"], model.market_organs.weight.detach())


def test_account_fusion_actor_and_critic_all_receive_nonzero_gradient_and_update() -> None:
    model = TinyOwnedActorCritic()
    spec = _spec()
    apply_gradient_ownership_r0(model, spec)
    before = snapshot_parameters_r0(model, spec)
    _backward_fixture(model)

    gradient_report = audit_gradients_r0(model, spec)
    by_role = {group.role: group for group in gradient_report.groups}
    for role in (TRAINABLE_ACCOUNT_STEM, TRAINABLE_FUSION, TRAINABLE_ACTOR, TRAINABLE_CRITIC):
        assert by_role[role].parameters_with_nonzero_gradient >= 1

    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.05)
    optimizer.step()
    mutation_report = audit_parameter_mutation_r0(model, spec, before)
    mutation_by_role = {group.role: group for group in mutation_report.groups}
    for role in (TRAINABLE_ACCOUNT_STEM, TRAINABLE_FUSION, TRAINABLE_ACTOR, TRAINABLE_CRITIC):
        assert mutation_by_role[role].changed_parameter_count >= 1


def test_apply_ownership_sets_exact_requires_grad_boundary() -> None:
    model = TinyOwnedActorCritic()
    owned = apply_gradient_ownership_r0(model, _spec())
    by_name = {item.name: item for item in owned}
    assert by_name["market_organs.weight"].parameter.requires_grad is False
    assert by_name["account_stem.weight"].parameter.requires_grad is True
    assert by_name["fusion.weight"].parameter.requires_grad is True
    assert by_name["actor.weight"].parameter.requires_grad is True
    assert by_name["critic.weight"].parameter.requires_grad is True


def test_spec_requires_frozen_market_and_every_trainable_role() -> None:
    base = _spec()
    for missing_role in (
        FROZEN_MARKET_ORGAN,
        TRAINABLE_ACCOUNT_STEM,
        TRAINABLE_FUSION,
        TRAINABLE_ACTOR,
        TRAINABLE_CRITIC,
    ):
        candidate = replace(base, rules=tuple(rule for rule in base.rules if rule.role != missing_role))
        with pytest.raises(RuntimeError):
            candidate.validate()


def test_prefix_overlap_and_duplicate_or_unknown_ownership_fail_closed() -> None:
    base = _spec()
    overlapping = replace(
        base,
        rules=base.rules
        + (GradientOwnershipRuleR0("nested", "fusion.weight", TRAINABLE_FUSION),),
    )
    with pytest.raises(RuntimeError, match="PREFIX_OVERLAP"):
        overlapping.validate()

    duplicate_group = replace(
        base,
        rules=base.rules
        + (GradientOwnershipRuleR0("actor", "extra", TRAINABLE_ACTOR),),
    )
    with pytest.raises(RuntimeError, match="GROUP_NAME_DUPLICATE"):
        duplicate_group.validate()

    with pytest.raises(RuntimeError, match="ROLE_INVALID"):
        GradientOwnershipRuleR0("bad", "bad", "TRAINABLE_EVERYTHING").validate()


def test_unowned_parameter_and_empty_rule_match_fail_closed() -> None:
    model = TinyOwnedActorCritic()
    model.extra = nn.Linear(1, 1, bias=False)
    with pytest.raises(RuntimeError, match="PARAMETER_UNOWNED:extra.weight"):
        partition_named_parameters_r0(model, _spec())

    spec = replace(
        _spec(),
        rules=_spec().rules
        + (GradientOwnershipRuleR0("ghost-market", "ghost_market", FROZEN_MARKET_ORGAN),),
    )
    with pytest.raises(RuntimeError, match="RULE_MATCHED_NO_PARAMETERS"):
        partition_named_parameters_r0(TinyOwnedActorCritic(), spec)


def test_gradient_audit_rejects_nonfinite_trainable_gradient() -> None:
    model = TinyOwnedActorCritic()
    spec = _spec()
    apply_gradient_ownership_r0(model, spec)
    _backward_fixture(model)
    model.actor.weight.grad = torch.full_like(model.actor.weight, float("nan"))
    with pytest.raises(RuntimeError, match="TRAINABLE_GRAD_NONFINITE"):
        audit_gradients_r0(model, spec)


def test_gradient_audit_rejects_missing_eligible_path() -> None:
    model = TinyOwnedActorCritic()
    spec = _spec()
    apply_gradient_ownership_r0(model, spec)
    _backward_fixture(model)
    model.critic.weight.grad = None
    with pytest.raises(RuntimeError, match="REQUIRED_NONZERO_GRADIENT_MISSING:TRAINABLE_CRITIC"):
        audit_gradients_r0(model, spec)


def test_mutation_audit_rejects_frozen_market_change() -> None:
    model = TinyOwnedActorCritic()
    spec = _spec()
    apply_gradient_ownership_r0(model, spec)
    before = snapshot_parameters_r0(model, spec)
    with torch.no_grad():
        model.market_organs.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="FROZEN_PARAMETER_MUTATED"):
        audit_parameter_mutation_r0(model, spec, before)


def test_mutation_audit_rejects_snapshot_with_wrong_parameter_set() -> None:
    model = TinyOwnedActorCritic()
    spec = _spec()
    before = snapshot_parameters_r0(model, spec)
    before.pop("actor.weight")
    with pytest.raises(RuntimeError, match="SNAPSHOT_PARAMETER_SET_MISMATCH"):
        audit_parameter_mutation_r0(model, spec, before)


def test_spec_hash_is_deterministic_and_changes_with_ownership_semantics() -> None:
    spec = _spec()
    assert spec.semantic_sha256 == _spec().semantic_sha256
    changed_rules = tuple(
        replace(rule, group_name="actor-head") if rule.role == TRAINABLE_ACTOR else rule
        for rule in spec.rules
    )
    changed = replace(spec, rules=changed_rules)
    changed.validate()
    assert changed.semantic_sha256 != spec.semantic_sha256
