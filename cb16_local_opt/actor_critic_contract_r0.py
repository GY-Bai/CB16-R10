from __future__ import annotations

"""Fail-closed science-version registry for the R11 Actor-Critic lane.

This module freezes scientific semantic identities only.  A code or Git revision
is deliberately represented separately: changing a commit does not by itself
change the scientific contract, and changing a scientific contract must not be
hidden behind an unchanged code-revision label.

Canonical serialization, semantic hashing, and lineage binding belong to
AC-002 and are intentionally not implemented here.
"""

from dataclasses import dataclass
from typing import Mapping


SCIENCE_SEMANTIC_VERSION_R0 = "CB16_R11_ACTOR_CRITIC_SCIENCE_R0"
OBSERVATION_VERSION_R0 = "CB16_R11_AC_OBSERVATION_V1_R0"
ACTION_VERSION_R0 = "CB16_R11_TARGET_POSITION_ACTION_V1_R0"
PERMISSION_EXECUTION_VERSION_R0 = "CB16_R11_AC_PERMISSION_EXECUTION_V1_R0"
REWARD_VERSION_R0 = "CB16_R11_ARITHMETIC_EQUITY_REWARD_V1_R0"
TRAJECTORY_VERSION_R0 = "CB16_R11_AC_TRAJECTORY_V1_R0"
ACTOR_DISTRIBUTION_VERSION_R0 = "CB16_R11_AC_ACTOR_DISTRIBUTION_V1_R0"
CRITIC_VALUE_VERSION_R0 = "CB16_R11_AC_CRITIC_VALUE_V1_R0"
REPLAY_COMPATIBILITY_VERSION_R0 = "CB16_R11_AC_REPLAY_COMPATIBILITY_V1_R0"
CHECKPOINT_BUNDLE_VERSION_R0 = "CB16_R11_AC_CHECKPOINT_BUNDLE_V1_R0"

SCIENCE_VERSION_FIELDS_R0 = (
    "science_semantic_version",
    "observation_version",
    "action_version",
    "permission_execution_version",
    "reward_version",
    "trajectory_version",
    "actor_distribution_version",
    "critic_value_version",
    "replay_compatibility_version",
    "checkpoint_bundle_version",
)


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


@dataclass(frozen=True)
class ActorCriticScienceContractR0:
    """Exact scientific semantic identity of the Actor-Critic R0 lane."""

    science_semantic_version: str
    observation_version: str
    action_version: str
    permission_execution_version: str
    reward_version: str
    trajectory_version: str
    actor_distribution_version: str
    critic_value_version: str
    replay_compatibility_version: str
    checkpoint_bundle_version: str

    def validate(self) -> None:
        expected = ACTOR_CRITIC_SCIENCE_CONTRACT_R0
        for field_name in SCIENCE_VERSION_FIELDS_R0:
            value = _require_nonempty_string(
                getattr(self, field_name),
                code=f"ACSCI_VERSION_EMPTY:{field_name}",
            )
            if value != getattr(expected, field_name):
                raise RuntimeError(f"ACSCI_VERSION_MISMATCH:{field_name}")


ACTOR_CRITIC_SCIENCE_CONTRACT_R0 = ActorCriticScienceContractR0(
    science_semantic_version=SCIENCE_SEMANTIC_VERSION_R0,
    observation_version=OBSERVATION_VERSION_R0,
    action_version=ACTION_VERSION_R0,
    permission_execution_version=PERMISSION_EXECUTION_VERSION_R0,
    reward_version=REWARD_VERSION_R0,
    trajectory_version=TRAJECTORY_VERSION_R0,
    actor_distribution_version=ACTOR_DISTRIBUTION_VERSION_R0,
    critic_value_version=CRITIC_VALUE_VERSION_R0,
    replay_compatibility_version=REPLAY_COMPATIBILITY_VERSION_R0,
    checkpoint_bundle_version=CHECKPOINT_BUNDLE_VERSION_R0,
)


@dataclass(frozen=True)
class CodeRevisionR0:
    """Operational code identity, explicitly outside scientific semantics."""

    code_revision: str

    def validate(self) -> None:
        _require_nonempty_string(
            self.code_revision,
            code="ACSCI_CODE_REVISION_EMPTY",
        )


def validate_actor_critic_science_contract_r0(
    contract: ActorCriticScienceContractR0 | Mapping[str, object],
) -> ActorCriticScienceContractR0:
    """Validate an exact R0 contract and reject missing/unknown/mixed versions."""

    if isinstance(contract, ActorCriticScienceContractR0):
        candidate = contract
    elif isinstance(contract, Mapping):
        required = set(SCIENCE_VERSION_FIELDS_R0)
        actual = set(contract.keys())
        missing = sorted(required - actual)
        unknown = sorted(actual - required)
        if missing:
            raise RuntimeError(f"ACSCI_VERSION_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACSCI_VERSION_FIELDS_UNKNOWN:{','.join(unknown)}")

        values: dict[str, str] = {}
        for field_name in SCIENCE_VERSION_FIELDS_R0:
            values[field_name] = _require_nonempty_string(
                contract[field_name],
                code=f"ACSCI_VERSION_INVALID:{field_name}",
            )
        candidate = ActorCriticScienceContractR0(**values)
    else:
        raise RuntimeError("ACSCI_CONTRACT_TYPE_INVALID")

    candidate.validate()
    return candidate


def validate_code_revision_r0(code_revision: object) -> CodeRevisionR0:
    """Validate operational revision metadata without changing science identity."""

    revision = CodeRevisionR0(
        code_revision=_require_nonempty_string(
            code_revision,
            code="ACSCI_CODE_REVISION_INVALID",
        )
    )
    revision.validate()
    return revision
