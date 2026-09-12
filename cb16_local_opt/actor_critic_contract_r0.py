from __future__ import annotations

"""Fail-closed science identity for the R11 Actor-Critic lane.

AC-001 freezes the scientific version registry.  AC-002 adds canonical science
serialization, semantic hashing, and required lineage binding.  Code/Git
revision remains deliberately separate from scientific identity: a new commit
does not by itself define new science, and a changed scientific identity cannot
be hidden behind incidental process or revision state.
"""

from dataclasses import dataclass
import hashlib
import json
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
SCIENCE_IDENTITY_SCHEMA_R0 = "CB16_R11_ACTOR_CRITIC_SCIENCE_IDENTITY_V1_R0"

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

SCIENCE_LINEAGE_FIELDS_R0 = (
    "account_lineage_id",
    "policy_lineage_id",
    "execution_lineage_id",
    "data_source_lineage_id",
)


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _mapping_key_diff(
    mapping: Mapping[object, object],
    required_fields: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    required = set(required_fields)
    actual = set(mapping.keys())
    missing = sorted(required - actual)
    unknown = sorted(str(key) for key in actual if key not in required)
    return missing, unknown


@dataclass(frozen=True)
class ActorCriticScienceContractR0:
    """Exact scientific semantic version identity of the Actor-Critic R0 lane."""

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


@dataclass(frozen=True)
class ActorCriticScienceLineageR0:
    """Required scientific lineage carried by every hashed science identity."""

    account_lineage_id: str
    policy_lineage_id: str
    execution_lineage_id: str
    data_source_lineage_id: str

    def validate(self) -> None:
        for field_name in SCIENCE_LINEAGE_FIELDS_R0:
            _require_nonempty_string(
                getattr(self, field_name),
                code=f"ACSCI_LINEAGE_EMPTY:{field_name}",
            )


@dataclass(frozen=True)
class ActorCriticScienceIdentityR0:
    """Version contract plus required lineage, excluding incidental code state."""

    contract: ActorCriticScienceContractR0
    lineage: ActorCriticScienceLineageR0

    def validate(self) -> None:
        self.contract.validate()
        self.lineage.validate()

    @property
    def semantic_sha256(self) -> str:
        return science_identity_sha256_r0(self.contract, self.lineage)


def validate_actor_critic_science_contract_r0(
    contract: ActorCriticScienceContractR0 | Mapping[str, object],
) -> ActorCriticScienceContractR0:
    """Validate an exact R0 contract and reject missing/unknown/mixed versions."""

    if isinstance(contract, ActorCriticScienceContractR0):
        candidate = contract
    elif isinstance(contract, Mapping):
        missing, unknown = _mapping_key_diff(contract, SCIENCE_VERSION_FIELDS_R0)
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


def validate_actor_critic_science_lineage_r0(
    lineage: ActorCriticScienceLineageR0 | Mapping[str, object],
) -> ActorCriticScienceLineageR0:
    """Require exact account/policy/execution/data-source lineage."""

    if isinstance(lineage, ActorCriticScienceLineageR0):
        candidate = lineage
    elif isinstance(lineage, Mapping):
        missing, unknown = _mapping_key_diff(lineage, SCIENCE_LINEAGE_FIELDS_R0)
        if missing:
            raise RuntimeError(f"ACSCI_LINEAGE_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACSCI_LINEAGE_FIELDS_UNKNOWN:{','.join(unknown)}")

        values: dict[str, str] = {}
        for field_name in SCIENCE_LINEAGE_FIELDS_R0:
            values[field_name] = _require_nonempty_string(
                lineage[field_name],
                code=f"ACSCI_LINEAGE_INVALID:{field_name}",
            )
        candidate = ActorCriticScienceLineageR0(**values)
    else:
        raise RuntimeError("ACSCI_LINEAGE_TYPE_INVALID")

    candidate.validate()
    return candidate


def science_identity_payload_r0(
    contract: ActorCriticScienceContractR0 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR0 | Mapping[str, object],
) -> dict[str, object]:
    """Return the exact semantic payload committed by the science hash."""

    valid_contract = validate_actor_critic_science_contract_r0(contract)
    valid_lineage = validate_actor_critic_science_lineage_r0(lineage)
    return {
        "schema": SCIENCE_IDENTITY_SCHEMA_R0,
        "contract": {
            field_name: getattr(valid_contract, field_name)
            for field_name in SCIENCE_VERSION_FIELDS_R0
        },
        "lineage": {
            field_name: getattr(valid_lineage, field_name)
            for field_name in SCIENCE_LINEAGE_FIELDS_R0
        },
    }


def canonical_science_identity_json_r0(
    contract: ActorCriticScienceContractR0 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR0 | Mapping[str, object],
) -> str:
    """Canonical UTF-8 JSON text for an Actor-Critic science identity."""

    return json.dumps(
        science_identity_payload_r0(contract, lineage),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def science_identity_sha256_r0(
    contract: ActorCriticScienceContractR0 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR0 | Mapping[str, object],
) -> str:
    """Stable semantic hash, independent of mapping order and code revision."""

    canonical = canonical_science_identity_json_r0(contract, lineage).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
