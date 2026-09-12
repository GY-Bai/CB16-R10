from __future__ import annotations

from dataclasses import asdict

import pytest

from cb16_local_opt.actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_LINEAGE_FIELDS_R0,
    SCIENCE_VERSION_FIELDS_R0,
    ActorCriticScienceContractR0,
    ActorCriticScienceIdentityR0,
    ActorCriticScienceLineageR0,
    canonical_science_identity_json_r0,
    science_identity_sha256_r0,
    validate_actor_critic_science_contract_r0,
    validate_actor_critic_science_lineage_r0,
    validate_code_revision_r0,
)


def declared_payload() -> dict[str, str]:
    return asdict(ACTOR_CRITIC_SCIENCE_CONTRACT_R0)


def declared_lineage() -> dict[str, str]:
    return {
        "account_lineage_id": "account-state-lineage-r0",
        "policy_lineage_id": "policy-lineage-r0",
        "execution_lineage_id": "execution-authority-lineage-r0",
        "data_source_lineage_id": "historical-market-authority-lineage-r0",
    }


def test_declared_contract_covers_all_science_version_axes() -> None:
    contract = validate_actor_critic_science_contract_r0(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0
    )
    assert tuple(asdict(contract).keys()) == SCIENCE_VERSION_FIELDS_R0
    assert len(SCIENCE_VERSION_FIELDS_R0) == 10


def test_same_declared_contract_validates_deterministically() -> None:
    payload = declared_payload()
    first = validate_actor_critic_science_contract_r0(payload)
    second = validate_actor_critic_science_contract_r0(dict(reversed(list(payload.items()))))
    assert first == second == ACTOR_CRITIC_SCIENCE_CONTRACT_R0


def test_missing_version_fails_closed() -> None:
    payload = declared_payload()
    payload.pop("trajectory_version")
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_FIELDS_MISSING:trajectory_version"):
        validate_actor_critic_science_contract_r0(payload)


def test_unknown_science_semantic_version_fails_closed() -> None:
    payload = declared_payload()
    payload["science_semantic_version"] = "CB16_R11_ACTOR_CRITIC_SCIENCE_UNKNOWN"
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_MISMATCH:science_semantic_version"):
        validate_actor_critic_science_contract_r0(payload)


def test_mixed_component_versions_fail_closed() -> None:
    payload = declared_payload()
    payload["action_version"] = "CB16_R11_TARGET_POSITION_ACTION_V2_R0"
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_MISMATCH:action_version"):
        validate_actor_critic_science_contract_r0(payload)


def test_unknown_contract_field_fails_closed() -> None:
    payload = declared_payload()
    payload["git_revision"] = "deadbeef"
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_FIELDS_UNKNOWN:git_revision"):
        validate_actor_critic_science_contract_r0(payload)


def test_empty_and_non_string_versions_fail_closed() -> None:
    empty = declared_payload()
    empty["reward_version"] = ""
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_INVALID:reward_version"):
        validate_actor_critic_science_contract_r0(empty)

    non_string = declared_payload()
    non_string["reward_version"] = 1  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_INVALID:reward_version"):
        validate_actor_critic_science_contract_r0(non_string)


def test_code_revision_is_structurally_separate_from_science_semantics() -> None:
    contract_before = validate_actor_critic_science_contract_r0(declared_payload())
    revision_a = validate_code_revision_r0("25d5df8337905232ca76b7d334c390d5deeb1a07")
    revision_b = validate_code_revision_r0("ce99b9d896c8669bed532ba63abe9a885ce64ff5")
    contract_after = validate_actor_critic_science_contract_r0(declared_payload())

    assert revision_a != revision_b
    assert contract_before == contract_after == ACTOR_CRITIC_SCIENCE_CONTRACT_R0
    assert "code_revision" not in SCIENCE_VERSION_FIELDS_R0


def test_empty_code_revision_fails_closed_without_changing_contract() -> None:
    with pytest.raises(RuntimeError, match="ACSCI_CODE_REVISION_INVALID"):
        validate_code_revision_r0("")
    assert validate_actor_critic_science_contract_r0(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0
    ) == ACTOR_CRITIC_SCIENCE_CONTRACT_R0


def test_contract_type_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="ACSCI_CONTRACT_TYPE_INVALID"):
        validate_actor_critic_science_contract_r0(  # type: ignore[arg-type]
            ["not", "a", "contract"]
        )


def test_manual_dataclass_with_unknown_component_fails_closed() -> None:
    payload = declared_payload()
    payload["checkpoint_bundle_version"] = "UNKNOWN"
    candidate = ActorCriticScienceContractR0(**payload)
    with pytest.raises(RuntimeError, match="ACSCI_VERSION_MISMATCH:checkpoint_bundle_version"):
        validate_actor_critic_science_contract_r0(candidate)


def test_lineage_requires_account_policy_execution_and_data_source() -> None:
    lineage = validate_actor_critic_science_lineage_r0(declared_lineage())
    assert tuple(asdict(lineage).keys()) == SCIENCE_LINEAGE_FIELDS_R0


@pytest.mark.parametrize("field_name", SCIENCE_LINEAGE_FIELDS_R0)
def test_absent_lineage_field_fails_closed(field_name: str) -> None:
    lineage = declared_lineage()
    lineage.pop(field_name)
    with pytest.raises(RuntimeError, match=f"ACSCI_LINEAGE_FIELDS_MISSING:{field_name}"):
        validate_actor_critic_science_lineage_r0(lineage)


def test_unknown_or_empty_lineage_fails_closed() -> None:
    unknown = declared_lineage()
    unknown["code_revision"] = "deadbeef"
    with pytest.raises(RuntimeError, match="ACSCI_LINEAGE_FIELDS_UNKNOWN:code_revision"):
        validate_actor_critic_science_lineage_r0(unknown)

    empty = declared_lineage()
    empty["policy_lineage_id"] = ""
    with pytest.raises(RuntimeError, match="ACSCI_LINEAGE_INVALID:policy_lineage_id"):
        validate_actor_critic_science_lineage_r0(empty)


def test_canonical_science_identity_is_order_independent() -> None:
    contract = declared_payload()
    lineage = declared_lineage()
    reversed_contract = dict(reversed(list(contract.items())))
    reversed_lineage = dict(reversed(list(lineage.items())))

    assert canonical_science_identity_json_r0(contract, lineage) == canonical_science_identity_json_r0(
        reversed_contract,
        reversed_lineage,
    )
    assert science_identity_sha256_r0(contract, lineage) == science_identity_sha256_r0(
        reversed_contract,
        reversed_lineage,
    )


def test_changed_semantic_lineage_changes_hash() -> None:
    baseline = declared_lineage()
    changed = dict(baseline)
    changed["execution_lineage_id"] = "execution-authority-lineage-r1"

    assert science_identity_sha256_r0(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
        baseline,
    ) != science_identity_sha256_r0(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
        changed,
    )


def test_code_revision_does_not_enter_semantic_hash() -> None:
    lineage = declared_lineage()
    identity = ActorCriticScienceIdentityR0(
        contract=ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
        lineage=ActorCriticScienceLineageR0(**lineage),
    )
    identity.validate()
    before = identity.semantic_sha256

    validate_code_revision_r0("1111111")
    validate_code_revision_r0("2222222")

    assert identity.semantic_sha256 == before
    assert identity.semantic_sha256 == science_identity_sha256_r0(
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
        lineage,
    )
    assert len(identity.semantic_sha256) == 64
