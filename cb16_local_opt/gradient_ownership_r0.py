from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn


GRADIENT_OWNERSHIP_SCHEMA_R0 = "CB16_R11_BC_GRADIENT_OWNERSHIP_V1_R0"
FROZEN_MARKET_ORGAN = "FROZEN_MARKET_ORGAN"
TRAINABLE_ACCOUNT_STEM = "TRAINABLE_ACCOUNT_STEM"
TRAINABLE_FUSION = "TRAINABLE_FUSION"
TRAINABLE_ACTOR = "TRAINABLE_ACTOR"
TRAINABLE_CRITIC = "TRAINABLE_CRITIC"
GRADIENT_OWNERSHIP_ROLES_R0 = (
    FROZEN_MARKET_ORGAN,
    TRAINABLE_ACCOUNT_STEM,
    TRAINABLE_FUSION,
    TRAINABLE_ACTOR,
    TRAINABLE_CRITIC,
)
TRAINABLE_ROLES_R0 = (
    TRAINABLE_ACCOUNT_STEM,
    TRAINABLE_FUSION,
    TRAINABLE_ACTOR,
    TRAINABLE_CRITIC,
)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _prefix_matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


@dataclass(frozen=True)
class GradientOwnershipRuleR0:
    group_name: str
    parameter_prefix: str
    role: str

    def validate(self) -> None:
        if not isinstance(self.group_name, str) or not self.group_name.strip():
            raise RuntimeError("ACGRAD_R0_GROUP_NAME_INVALID")
        if not isinstance(self.parameter_prefix, str) or not self.parameter_prefix.strip():
            raise RuntimeError("ACGRAD_R0_PARAMETER_PREFIX_INVALID")
        if self.parameter_prefix.startswith(".") or self.parameter_prefix.endswith("."):
            raise RuntimeError("ACGRAD_R0_PARAMETER_PREFIX_INVALID")
        if self.role not in GRADIENT_OWNERSHIP_ROLES_R0:
            raise RuntimeError("ACGRAD_R0_ROLE_INVALID")


@dataclass(frozen=True)
class GradientOwnershipSpecR0:
    schema_version: str
    rules: tuple[GradientOwnershipRuleR0, ...]

    def validate(self) -> None:
        if self.schema_version != GRADIENT_OWNERSHIP_SCHEMA_R0:
            raise RuntimeError("ACGRAD_R0_SCHEMA_MISMATCH")
        if not self.rules:
            raise RuntimeError("ACGRAD_R0_RULES_EMPTY")
        for rule in self.rules:
            rule.validate()
        names = tuple(rule.group_name for rule in self.rules)
        prefixes = tuple(rule.parameter_prefix for rule in self.rules)
        if len(set(names)) != len(names):
            raise RuntimeError("ACGRAD_R0_GROUP_NAME_DUPLICATE")
        if len(set(prefixes)) != len(prefixes):
            raise RuntimeError("ACGRAD_R0_PARAMETER_PREFIX_DUPLICATE")
        for left_index, left in enumerate(prefixes):
            for right in prefixes[left_index + 1 :]:
                if _prefix_matches(left, right) or _prefix_matches(right, left):
                    raise RuntimeError("ACGRAD_R0_PARAMETER_PREFIX_OVERLAP")
        roles = {rule.role for rule in self.rules}
        if FROZEN_MARKET_ORGAN not in roles:
            raise RuntimeError("ACGRAD_R0_FROZEN_MARKET_ORGAN_REQUIRED")
        for role in TRAINABLE_ROLES_R0:
            if role not in roles:
                raise RuntimeError(f"ACGRAD_R0_REQUIRED_ROLE_MISSING:{role}")

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "rules": [
                    {
                        "group_name": rule.group_name,
                        "parameter_prefix": rule.parameter_prefix,
                        "role": rule.role,
                    }
                    for rule in self.rules
                ],
            }
        )


@dataclass(frozen=True)
class OwnedParameterR0:
    name: str
    group_name: str
    role: str
    parameter: nn.Parameter


@dataclass(frozen=True)
class GradientGroupAuditR0:
    group_name: str
    role: str
    parameter_count: int
    parameters_with_gradient: int
    parameters_with_nonzero_gradient: int


@dataclass(frozen=True)
class GradientAuditReportR0:
    spec_sha256: str
    groups: tuple[GradientGroupAuditR0, ...]

    @property
    def passed(self) -> bool:
        return True


@dataclass(frozen=True)
class MutationGroupAuditR0:
    group_name: str
    role: str
    parameter_count: int
    changed_parameter_count: int


@dataclass(frozen=True)
class MutationAuditReportR0:
    spec_sha256: str
    groups: tuple[MutationGroupAuditR0, ...]

    @property
    def passed(self) -> bool:
        return True


def partition_named_parameters_r0(
    module: nn.Module,
    spec: GradientOwnershipSpecR0,
) -> tuple[OwnedParameterR0, ...]:
    spec.validate()
    named_parameters = tuple(module.named_parameters())
    if not named_parameters:
        raise RuntimeError("ACGRAD_R0_MODULE_HAS_NO_PARAMETERS")
    owned: list[OwnedParameterR0] = []
    matched_groups: dict[str, int] = {rule.group_name: 0 for rule in spec.rules}
    for name, parameter in named_parameters:
        matches = [rule for rule in spec.rules if _prefix_matches(name, rule.parameter_prefix)]
        if not matches:
            raise RuntimeError(f"ACGRAD_R0_PARAMETER_UNOWNED:{name}")
        if len(matches) != 1:
            raise RuntimeError(f"ACGRAD_R0_PARAMETER_MULTI_OWNED:{name}")
        rule = matches[0]
        matched_groups[rule.group_name] += 1
        owned.append(
            OwnedParameterR0(
                name=name,
                group_name=rule.group_name,
                role=rule.role,
                parameter=parameter,
            )
        )
    empty_groups = [name for name, count in matched_groups.items() if count == 0]
    if empty_groups:
        raise RuntimeError("ACGRAD_R0_RULE_MATCHED_NO_PARAMETERS:" + ",".join(sorted(empty_groups)))
    return tuple(owned)


def apply_gradient_ownership_r0(
    module: nn.Module,
    spec: GradientOwnershipSpecR0,
) -> tuple[OwnedParameterR0, ...]:
    owned = partition_named_parameters_r0(module, spec)
    for item in owned:
        item.parameter.requires_grad_(item.role != FROZEN_MARKET_ORGAN)
        item.parameter.grad = None
    return owned


def snapshot_parameters_r0(
    module: nn.Module,
    spec: GradientOwnershipSpecR0,
) -> dict[str, torch.Tensor]:
    owned = partition_named_parameters_r0(module, spec)
    return {item.name: item.parameter.detach().clone() for item in owned}


def _gradient_is_finite(gradient: torch.Tensor) -> bool:
    return bool(torch.isfinite(gradient).all().item())


def _gradient_is_nonzero(gradient: torch.Tensor) -> bool:
    return bool(torch.count_nonzero(gradient).item() > 0)


def audit_gradients_r0(
    module: nn.Module,
    spec: GradientOwnershipSpecR0,
) -> GradientAuditReportR0:
    owned = partition_named_parameters_r0(module, spec)
    by_group: dict[str, list[OwnedParameterR0]] = {}
    for item in owned:
        by_group.setdefault(item.group_name, []).append(item)

    audits: list[GradientGroupAuditR0] = []
    nonzero_roles: set[str] = set()
    for rule in spec.rules:
        items = by_group[rule.group_name]
        with_gradient = 0
        with_nonzero = 0
        for item in items:
            gradient = item.parameter.grad
            if rule.role == FROZEN_MARKET_ORGAN:
                if item.parameter.requires_grad:
                    raise RuntimeError(f"ACGRAD_R0_FROZEN_REQUIRES_GRAD:{item.name}")
                if gradient is not None:
                    if not _gradient_is_finite(gradient):
                        raise RuntimeError(f"ACGRAD_R0_FROZEN_GRAD_NONFINITE:{item.name}")
                    if _gradient_is_nonzero(gradient):
                        raise RuntimeError(f"ACGRAD_R0_FROZEN_GRAD_NONZERO:{item.name}")
                    with_gradient += 1
                continue

            if not item.parameter.requires_grad:
                raise RuntimeError(f"ACGRAD_R0_TRAINABLE_REQUIRES_GRAD_FALSE:{item.name}")
            if gradient is None:
                continue
            with_gradient += 1
            if not _gradient_is_finite(gradient):
                raise RuntimeError(f"ACGRAD_R0_TRAINABLE_GRAD_NONFINITE:{item.name}")
            if _gradient_is_nonzero(gradient):
                with_nonzero += 1
                nonzero_roles.add(rule.role)
        audits.append(
            GradientGroupAuditR0(
                group_name=rule.group_name,
                role=rule.role,
                parameter_count=len(items),
                parameters_with_gradient=with_gradient,
                parameters_with_nonzero_gradient=with_nonzero,
            )
        )

    for role in TRAINABLE_ROLES_R0:
        if role not in nonzero_roles:
            raise RuntimeError(f"ACGRAD_R0_REQUIRED_NONZERO_GRADIENT_MISSING:{role}")
    return GradientAuditReportR0(spec_sha256=spec.semantic_sha256, groups=tuple(audits))


def audit_parameter_mutation_r0(
    module: nn.Module,
    spec: GradientOwnershipSpecR0,
    before: Mapping[str, torch.Tensor],
) -> MutationAuditReportR0:
    owned = partition_named_parameters_r0(module, spec)
    names = {item.name for item in owned}
    if set(before) != names:
        raise RuntimeError("ACGRAD_R0_SNAPSHOT_PARAMETER_SET_MISMATCH")

    by_group: dict[str, list[OwnedParameterR0]] = {}
    for item in owned:
        by_group.setdefault(item.group_name, []).append(item)

    audits: list[MutationGroupAuditR0] = []
    changed_roles: set[str] = set()
    for rule in spec.rules:
        items = by_group[rule.group_name]
        changed = 0
        for item in items:
            old = before[item.name]
            new = item.parameter.detach()
            if old.shape != new.shape or old.dtype != new.dtype:
                raise RuntimeError(f"ACGRAD_R0_PARAMETER_STRUCTURE_CHANGED:{item.name}")
            if not bool(torch.equal(old, new)):
                changed += 1
                if rule.role == FROZEN_MARKET_ORGAN:
                    raise RuntimeError(f"ACGRAD_R0_FROZEN_PARAMETER_MUTATED:{item.name}")
                changed_roles.add(rule.role)
        audits.append(
            MutationGroupAuditR0(
                group_name=rule.group_name,
                role=rule.role,
                parameter_count=len(items),
                changed_parameter_count=changed,
            )
        )

    for role in TRAINABLE_ROLES_R0:
        if role not in changed_roles:
            raise RuntimeError(f"ACGRAD_R0_REQUIRED_TRAINABLE_MUTATION_MISSING:{role}")
    return MutationAuditReportR0(spec_sha256=spec.semantic_sha256, groups=tuple(audits))
