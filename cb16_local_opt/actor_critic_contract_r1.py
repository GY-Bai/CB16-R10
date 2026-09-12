from __future__ import annotations

"""Round-2 Actor-Critic scientific semantic identity.

BC-004 creates a new semantic lane rather than mutating Actor-Critic R0.
Corrected observation, action, permission/execution, account economics,
environment profile, reward, trajectory, Actor, Critic, replay and checkpoint
semantics are first-class version axes. Operational code revision remains
outside scientific identity.

There is intentionally no implicit R0->R1 compatibility adapter here. A mixed
R0/R1 semantic payload fails closed.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

from .actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_VERSION_FIELDS_R0,
)


SCIENCE_SEMANTIC_VERSION_R1 = "CB16_R11_BC_ROUND2_ACTOR_CRITIC_SCIENCE_R1"
OBSERVATION_VERSION_R1 = "CB16_R11_BC_ROUND2_OBSERVATION_V1_R1"
ACTION_VERSION_R1 = "CB16_R11_BC_ROUND2_TARGET_POSITION_ACTION_V1_R1"
PERMISSION_EXECUTION_VERSION_R1 = "CB16_R11_BC_ROUND2_PERMISSION_EXECUTION_V1_R1"
ACCOUNT_ECONOMICS_VERSION_R1 = "CB16_R11_BC_ROUND2_ACCOUNT_ECONOMICS_V1_R1"
ENVIRONMENT_PROFILE_VERSION_R1 = "CB16_R11_BC_ROUND2_ENVIRONMENT_PROFILE_V1_R1"
REWARD_VERSION_R1 = "CB16_R11_BC_ROUND2_ARITHMETIC_EQUITY_REWARD_V1_R1"
TRAJECTORY_VERSION_R1 = "CB16_R11_BC_ROUND2_TRAJECTORY_V1_R1"
ACTOR_DISTRIBUTION_VERSION_R1 = "CB16_R11_BC_ROUND2_ACTOR_DISTRIBUTION_V1_R1"
CRITIC_VALUE_VERSION_R1 = "CB16_R11_BC_ROUND2_CRITIC_VALUE_V1_R1"
REPLAY_COMPATIBILITY_VERSION_R1 = "CB16_R11_BC_ROUND2_REPLAY_COMPATIBILITY_V1_R1"
CHECKPOINT_BUNDLE_VERSION_R1 = "CB16_R11_BC_ROUND2_CHECKPOINT_BUNDLE_V1_R1"
SCIENCE_IDENTITY_SCHEMA_R1 = "CB16_R11_BC_ROUND2_SCIENCE_IDENTITY_V1_R1"
SCIENCE_CONTRACT_SCHEMA_R1 = "CB16_R11_BC_ROUND2_SCIENCE_CONTRACT_V1_R1"

SCIENCE_VERSION_FIELDS_R1 = (
    "science_semantic_version",
    "observation_version",
    "action_version",
    "permission_execution_version",
    "account_economics_version",
    "environment_profile_version",
    "reward_version",
    "trajectory_version",
    "actor_distribution_version",
    "critic_value_version",
    "replay_compatibility_version",
    "checkpoint_bundle_version",
)

SCIENCE_LINEAGE_FIELDS_R1 = (
    "account_lineage_id",
    "policy_lineage_id",
    "execution_lineage_id",
    "data_source_lineage_id",
)

# Compatibility must be an explicit future artifact, not an inference made by
# this contract validator.
EXPLICIT_COMPATIBILITY_ADAPTER_IDS_R1: tuple[str, ...] = ()


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


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r0_semantic_values() -> frozenset[str]:
    return frozenset(
        str(getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R0, field_name))
        for field_name in SCIENCE_VERSION_FIELDS_R0
    )


@dataclass(frozen=True)
class ActorCriticScienceContractR1:
    """Exact Round-2 semantic version registry."""

    science_semantic_version: str
    observation_version: str
    action_version: str
    permission_execution_version: str
    account_economics_version: str
    environment_profile_version: str
    reward_version: str
    trajectory_version: str
    actor_distribution_version: str
    critic_value_version: str
    replay_compatibility_version: str
    checkpoint_bundle_version: str

    def validate(self) -> None:
        expected = ACTOR_CRITIC_SCIENCE_CONTRACT_R1
        r0_values = _r0_semantic_values()
        for field_name in SCIENCE_VERSION_FIELDS_R1:
            value = _require_nonempty_string(
                getattr(self, field_name),
                code=f"ACSCI_R1_VERSION_EMPTY:{field_name}",
            )
            expected_value = getattr(expected, field_name)
            if value == expected_value:
                continue
            if value in r0_values:
                raise RuntimeError(f"ACSCI_R1_R0_COMPONENT_MIX_FORBIDDEN:{field_name}")
            raise RuntimeError(f"ACSCI_R1_VERSION_MISMATCH:{field_name}")


ACTOR_CRITIC_SCIENCE_CONTRACT_R1 = ActorCriticScienceContractR1(
    science_semantic_version=SCIENCE_SEMANTIC_VERSION_R1,
    observation_version=OBSERVATION_VERSION_R1,
    action_version=ACTION_VERSION_R1,
    permission_execution_version=PERMISSION_EXECUTION_VERSION_R1,
    account_economics_version=ACCOUNT_ECONOMICS_VERSION_R1,
    environment_profile_version=ENVIRONMENT_PROFILE_VERSION_R1,
    reward_version=REWARD_VERSION_R1,
    trajectory_version=TRAJECTORY_VERSION_R1,
    actor_distribution_version=ACTOR_DISTRIBUTION_VERSION_R1,
    critic_value_version=CRITIC_VALUE_VERSION_R1,
    replay_compatibility_version=REPLAY_COMPATIBILITY_VERSION_R1,
    checkpoint_bundle_version=CHECKPOINT_BUNDLE_VERSION_R1,
)


@dataclass(frozen=True)
class CodeRevisionR1:
    """Operational revision metadata; deliberately outside science identity."""

    code_revision: str

    def validate(self) -> None:
        _require_nonempty_string(self.code_revision, code="ACSCI_R1_CODE_REVISION_EMPTY")


@dataclass(frozen=True)
class ActorCriticScienceLineageR1:
    account_lineage_id: str
    policy_lineage_id: str
    execution_lineage_id: str
    data_source_lineage_id: str

    def validate(self) -> None:
        for field_name in SCIENCE_LINEAGE_FIELDS_R1:
            _require_nonempty_string(
                getattr(self, field_name),
                code=f"ACSCI_R1_LINEAGE_EMPTY:{field_name}",
            )


@dataclass(frozen=True)
class ActorCriticScienceIdentityR1:
    contract: ActorCriticScienceContractR1
    lineage: ActorCriticScienceLineageR1

    def validate(self) -> None:
        self.contract.validate()
        self.lineage.validate()

    @property
    def semantic_sha256(self) -> str:
        return science_identity_sha256_r1(self.contract, self.lineage)


def validate_actor_critic_science_contract_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object],
) -> ActorCriticScienceContractR1:
    if isinstance(contract, ActorCriticScienceContractR1):
        candidate = contract
    elif isinstance(contract, Mapping):
        missing, unknown = _mapping_key_diff(contract, SCIENCE_VERSION_FIELDS_R1)
        if missing:
            raise RuntimeError(f"ACSCI_R1_VERSION_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACSCI_R1_VERSION_FIELDS_UNKNOWN:{','.join(unknown)}")
        values: dict[str, str] = {}
        for field_name in SCIENCE_VERSION_FIELDS_R1:
            values[field_name] = _require_nonempty_string(
                contract[field_name],
                code=f"ACSCI_R1_VERSION_INVALID:{field_name}",
            )
        candidate = ActorCriticScienceContractR1(**values)
    else:
        raise RuntimeError("ACSCI_R1_CONTRACT_TYPE_INVALID")

    candidate.validate()
    return candidate


def validate_code_revision_r1(code_revision: object) -> CodeRevisionR1:
    revision = CodeRevisionR1(
        code_revision=_require_nonempty_string(
            code_revision,
            code="ACSCI_R1_CODE_REVISION_INVALID",
        )
    )
    revision.validate()
    return revision


def validate_actor_critic_science_lineage_r1(
    lineage: ActorCriticScienceLineageR1 | Mapping[str, object],
) -> ActorCriticScienceLineageR1:
    if isinstance(lineage, ActorCriticScienceLineageR1):
        candidate = lineage
    elif isinstance(lineage, Mapping):
        missing, unknown = _mapping_key_diff(lineage, SCIENCE_LINEAGE_FIELDS_R1)
        if missing:
            raise RuntimeError(f"ACSCI_R1_LINEAGE_FIELDS_MISSING:{','.join(missing)}")
        if unknown:
            raise RuntimeError(f"ACSCI_R1_LINEAGE_FIELDS_UNKNOWN:{','.join(unknown)}")
        values: dict[str, str] = {}
        for field_name in SCIENCE_LINEAGE_FIELDS_R1:
            values[field_name] = _require_nonempty_string(
                lineage[field_name],
                code=f"ACSCI_R1_LINEAGE_INVALID:{field_name}",
            )
        candidate = ActorCriticScienceLineageR1(**values)
    else:
        raise RuntimeError("ACSCI_R1_LINEAGE_TYPE_INVALID")

    candidate.validate()
    return candidate


def science_contract_payload_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object] = ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
) -> dict[str, object]:
    valid = validate_actor_critic_science_contract_r1(contract)
    return {
        "schema": SCIENCE_CONTRACT_SCHEMA_R1,
        "contract": {
            field_name: getattr(valid, field_name)
            for field_name in SCIENCE_VERSION_FIELDS_R1
        },
    }


def canonical_science_contract_json_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object] = ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
) -> str:
    return _canonical_json(science_contract_payload_r1(contract))


def science_contract_sha256_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object] = ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
) -> str:
    return _sha256_text(canonical_science_contract_json_r1(contract))


def science_identity_payload_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR1 | Mapping[str, object],
) -> dict[str, object]:
    valid_contract = validate_actor_critic_science_contract_r1(contract)
    valid_lineage = validate_actor_critic_science_lineage_r1(lineage)
    return {
        "schema": SCIENCE_IDENTITY_SCHEMA_R1,
        "contract": {
            field_name: getattr(valid_contract, field_name)
            for field_name in SCIENCE_VERSION_FIELDS_R1
        },
        "lineage": {
            field_name: getattr(valid_lineage, field_name)
            for field_name in SCIENCE_LINEAGE_FIELDS_R1
        },
    }


def canonical_science_identity_json_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR1 | Mapping[str, object],
) -> str:
    return _canonical_json(science_identity_payload_r1(contract, lineage))


def science_identity_sha256_r1(
    contract: ActorCriticScienceContractR1 | Mapping[str, object],
    lineage: ActorCriticScienceLineageR1 | Mapping[str, object],
) -> str:
    return _sha256_text(canonical_science_identity_json_r1(contract, lineage))
