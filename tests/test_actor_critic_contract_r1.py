from __future__ import annotations

from dataclasses import asdict

import pytest

from cb16_local_opt.actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_VERSION_FIELDS_R0,
)
from cb16_local_opt.actor_critic_contract_r1 import (
    ACCOUNT_ECONOMICS_VERSION_R1,
    ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
    ENVIRONMENT_PROFILE_VERSION_R1,
    EXPLICIT_COMPATIBILITY_ADAPTER_IDS_R1,
    SCIENCE_LINEAGE_FIELDS_R1,
    SCIENCE_VERSION_FIELDS_R1,
    ActorCriticScienceContractR1,
    ActorCriticScienceIdentityR1,
    ActorCriticScienceLineageR1,
    canonical_science_contract_json_r1,
    canonical_science_identity_json_r1,
    science_contract_sha256_r1,
    science_identity_sha256_r1,
    validate_actor_critic_science_contract_r1,
    validate_actor_critic_science_lineage_r1,
    validate_code_revision_r1,
)


def declared_payload() -> dict[str, str]:
    return asdict(ACTOR_CRITIC_SCIENCE_CONTRACT_R1)


def declared_lineage() -> dict[str, str]:
    return {
        "account_lineage_id": "round2-account-lineage-r1",
        "policy_lineage_id": "round2-policy-lineage-r1",
        "execution_lineage_id": "round2-execution-lineage-r1",
        "data_source_lineage_id": "round2-historical-market-lineage-r1",
    }


def test_declared_contract_covers_all_round2_semantic_axes() -> None:
    contract = validate_actor_critic_science_contract_r1(ACTOR_CRITIC_SCIENCE_CONTRACT_R1)

    assert tuple(asdict(contract).keys()) == SCIENCE_VERSION_FIELDS_R1
    assert len(SCIENCE_VERSION_FIELDS_R1) == 12
    assert contract.account_economics_version == ACCOUNT_ECONOMICS_VERSION_R1
    assert contract.environment_profile_version == ENVIRONMENT_PROFILE_VERSION_R1
    assert "code_revision" not in SCIENCE_VERSION_FIELDS_R1


def test_same_declared_contract_is_order_independent() -> None:
    payload = declared_payload()
    reversed_payload = dict(reversed(list(payload.items())))

    assert validate_actor_critic_science_contract_r1(payload) == ACTOR_CRITIC_SCIENCE_CONTRACT_R1
    assert validate_actor_critic_science_contract_r1(reversed_payload) == ACTOR_CRITIC_SCIENCE_CONTRACT_R1
    assert canonical_science_contract_json_r1(payload) == canonical_science_contract_json_r1(reversed_payload)
    assert science_contract_sha256_r1(payload) == science_contract_sha256_r1(reversed_payload)
    assert len(science_contract_sha256_r1(payload)) == 64


def test_missing_unknown_or_invalid_version_fails_closed() -> None:
    missing = declared_payload()
    missing.pop("account_economics_version")
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_FIELDS_MISSING:account_economics_version"):
        validate_actor_critic_science_contract_r1(missing)

    unknown = declared_payload()
    unknown["compatibility_adapter_id"] = "implicit-adapter"
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_FIELDS_UNKNOWN:compatibility_adapter_id"):
        validate_actor_critic_science_contract_r1(unknown)

    invalid = declared_payload()
    invalid["environment_profile_version"] = ""
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_INVALID:environment_profile_version"):
        validate_actor_critic_science_contract_r1(invalid)


def test_r0_component_versions_cannot_be_mixed_into_r1_payload() -> None:
    r0_payload = asdict(ACTOR_CRITIC_SCIENCE_CONTRACT_R0)
    shared_fields = set(SCIENCE_VERSION_FIELDS_R0) & set(SCIENCE_VERSION_FIELDS_R1)

    assert shared_fields
    for field_name in sorted(shared_fields):
        mixed = declared_payload()
        mixed[field_name] = r0_payload[field_name]
        with pytest.raises(
            RuntimeError,
            match=f"ACSCI_R1_R0_COMPONENT_MIX_FORBIDDEN:{field_name}",
        ):
            validate_actor_critic_science_contract_r1(mixed)


def test_unknown_r1_semantic_value_fails_closed() -> None:
    payload = declared_payload()
    payload["reward_version"] = "CB16_R11_BC_ROUND2_REWARD_UNKNOWN"
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_MISMATCH:reward_version"):
        validate_actor_critic_science_contract_r1(payload)


def test_no_implicit_compatibility_adapter_exists_at_bc004() -> None:
    assert EXPLICIT_COMPATIBILITY_ADAPTER_IDS_R1 == ()


def test_account_economics_and_environment_profile_change_science_contract_hash() -> None:
    baseline = science_contract_sha256_r1()

    economics = declared_payload()
    economics["account_economics_version"] = "UNKNOWN_ECONOMICS"
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_MISMATCH:account_economics_version"):
        validate_actor_critic_science_contract_r1(economics)

    environment = declared_payload()
    environment["environment_profile_version"] = "UNKNOWN_ENVIRONMENT"
    with pytest.raises(RuntimeError, match="ACSCI_R1_VERSION_MISMATCH:environment_profile_version"):
        validate_actor_critic_science_contract_r1(environment)

    assert science_contract_sha256_r1() == baseline


def test_code_revision_is_separate_from_science_identity() -> None:
    lineage = declared_lineage()
    identity = ActorCriticScienceIdentityR1(
        contract=ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
        lineage=ActorCriticScienceLineageR1(**lineage),
    )
    identity.validate()
    before = identity.semantic_sha256

    revision_a = validate_code_revision_r1("3b561d6384af53a3bdf74f928797caf47bd58c5f")
    revision_b = validate_code_revision_r1("deadbeef")

    assert revision_a != revision_b
    assert identity.semantic_sha256 == before
    assert "code_revision" not in canonical_science_identity_json_r1(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
        lineage,
    )


def test_empty_code_revision_fails_closed_without_mutating_science_contract() -> None:
    with pytest.raises(RuntimeError, match="ACSCI_R1_CODE_REVISION_INVALID"):
        validate_code_revision_r1("")
    assert validate_actor_critic_science_contract_r1(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R1
    ) == ACTOR_CRITIC_SCIENCE_CONTRACT_R1


def test_lineage_requires_exact_account_policy_execution_and_data_source_fields() -> None:
    lineage = validate_actor_critic_science_lineage_r1(declared_lineage())
    assert tuple(asdict(lineage).keys()) == SCIENCE_LINEAGE_FIELDS_R1

    missing = declared_lineage()
    missing.pop("account_lineage_id")
    with pytest.raises(RuntimeError, match="ACSCI_R1_LINEAGE_FIELDS_MISSING:account_lineage_id"):
        validate_actor_critic_science_lineage_r1(missing)

    unknown = declared_lineage()
    unknown["code_revision"] = "deadbeef"
    with pytest.raises(RuntimeError, match="ACSCI_R1_LINEAGE_FIELDS_UNKNOWN:code_revision"):
        validate_actor_critic_science_lineage_r1(unknown)


def test_identity_hash_is_order_independent_and_lineage_sensitive() -> None:
    lineage = declared_lineage()
    reversed_lineage = dict(reversed(list(lineage.items())))
    reversed_contract = dict(reversed(list(declared_payload().items())))

    assert science_identity_sha256_r1(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
        lineage,
    ) == science_identity_sha256_r1(reversed_contract, reversed_lineage)

    changed = dict(lineage)
    changed["execution_lineage_id"] = "round2-execution-lineage-r1-changed"
    assert science_identity_sha256_r1(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
        lineage,
    ) != science_identity_sha256_r1(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
        changed,
    )


def test_manual_dataclass_with_mixed_r0_component_fails_closed() -> None:
    payload = declared_payload()
    payload["action_version"] = ACTOR_CRITIC_SCIENCE_CONTRACT_R0.action_version
    candidate = ActorCriticScienceContractR1(**payload)
    with pytest.raises(RuntimeError, match="ACSCI_R1_R0_COMPONENT_MIX_FORBIDDEN:action_version"):
        validate_actor_critic_science_contract_r1(candidate)
