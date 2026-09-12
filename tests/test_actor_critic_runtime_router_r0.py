from __future__ import annotations

from dataclasses import replace
import json

import pytest

from cb16_local_opt.actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_VERSION_FIELDS_R0,
)
from cb16_local_opt.actor_critic_runtime_router_r0 import (
    ACTOR_CRITIC_R0,
    LEGACY_PROTOCOL_ID_R0,
    LEGACY_R11,
    RUNTIME_ROUTER_SCHEMA_R0,
    RuntimeLaneReceiptR0,
    canonical_runtime_lane_contract_json_r0,
    runtime_lane_contract_payload_r0,
    runtime_lane_contract_sha256_r0,
    select_runtime_lane_r0,
)


@pytest.mark.parametrize("lane", [LEGACY_R11, ACTOR_CRITIC_R0])
def test_explicit_lane_selection_emits_contract_hash_receipt(lane: str) -> None:
    expected_hash = runtime_lane_contract_sha256_r0(lane)
    receipt = select_runtime_lane_r0(
        requested_lane=lane,
        declared_science_contract_sha256=expected_hash,
    )

    receipt.validate()
    assert receipt.selected_lane == lane
    assert receipt.selected_science_contract_sha256 == expected_hash
    assert receipt.router_schema_version == RUNTIME_ROUTER_SCHEMA_R0
    payload = receipt.to_payload()
    assert payload["selected_lane"] == lane
    assert payload["selected_science_contract_sha256"] == expected_hash
    assert payload["selected_contract_payload"] == runtime_lane_contract_payload_r0(lane)


def test_actor_critic_lane_hash_commits_full_science_version_registry() -> None:
    payload = runtime_lane_contract_payload_r0(ACTOR_CRITIC_R0)
    contract = payload["actor_critic_science_contract"]
    assert isinstance(contract, dict)
    assert contract == {
        field_name: getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R0, field_name)
        for field_name in SCIENCE_VERSION_FIELDS_R0
    }


def test_legacy_lane_is_explicit_protocol_not_actor_fallback() -> None:
    payload = runtime_lane_contract_payload_r0(LEGACY_R11)
    assert payload == {
        "router_schema_version": RUNTIME_ROUTER_SCHEMA_R0,
        "runtime_lane": LEGACY_R11,
        "protocol_id": LEGACY_PROTOCOL_ID_R0,
    }
    assert runtime_lane_contract_sha256_r0(LEGACY_R11) != runtime_lane_contract_sha256_r0(
        ACTOR_CRITIC_R0
    )


@pytest.mark.parametrize("lane", [None, "", "legacy", "ACTOR_CRITIC", "AUTO"])
def test_unknown_or_implicit_lane_fails_closed(lane: object) -> None:
    with pytest.raises(RuntimeError, match="ACROUTER_LANE"):
        runtime_lane_contract_payload_r0(lane)


@pytest.mark.parametrize("declared_hash", [None, "", "abc", "0" * 63, "G" * 64])
def test_missing_or_malformed_contract_hash_fails_closed(declared_hash: object) -> None:
    with pytest.raises(RuntimeError, match="ACROUTER_DECLARED_CONTRACT_HASH_INVALID"):
        select_runtime_lane_r0(
            requested_lane=ACTOR_CRITIC_R0,
            declared_science_contract_sha256=declared_hash,
        )


def test_cross_lane_hash_cannot_trigger_fallback() -> None:
    actor_hash = runtime_lane_contract_sha256_r0(ACTOR_CRITIC_R0)
    legacy_hash = runtime_lane_contract_sha256_r0(LEGACY_R11)

    with pytest.raises(RuntimeError, match="ACROUTER_DECLARED_CONTRACT_HASH_MISMATCH"):
        select_runtime_lane_r0(
            requested_lane=ACTOR_CRITIC_R0,
            declared_science_contract_sha256=legacy_hash,
        )
    with pytest.raises(RuntimeError, match="ACROUTER_DECLARED_CONTRACT_HASH_MISMATCH"):
        select_runtime_lane_r0(
            requested_lane=LEGACY_R11,
            declared_science_contract_sha256=actor_hash,
        )


def test_contract_json_is_canonical_and_stable() -> None:
    encoded_a = canonical_runtime_lane_contract_json_r0(ACTOR_CRITIC_R0)
    encoded_b = canonical_runtime_lane_contract_json_r0(ACTOR_CRITIC_R0)
    assert encoded_a == encoded_b
    assert json.dumps(
        json.loads(encoded_a),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) == encoded_a


def test_receipt_tampering_fails_closed() -> None:
    receipt = select_runtime_lane_r0(
        requested_lane=ACTOR_CRITIC_R0,
        declared_science_contract_sha256=runtime_lane_contract_sha256_r0(ACTOR_CRITIC_R0),
    )

    with pytest.raises(RuntimeError, match="ACROUTER_PROTOCOL_ID_MISMATCH"):
        replace(receipt, selected_protocol_id=LEGACY_PROTOCOL_ID_R0).validate()
    with pytest.raises(RuntimeError, match="ACROUTER_CONTRACT_HASH_MISMATCH"):
        replace(receipt, selected_science_contract_sha256="0" * 64).validate()
    with pytest.raises(RuntimeError, match="ACROUTER_CONTRACT_PAYLOAD_MISMATCH"):
        replace(
            receipt,
            selected_contract_payload_json=canonical_runtime_lane_contract_json_r0(LEGACY_R11),
        ).validate()


def test_receipt_schema_mismatch_fails_closed() -> None:
    receipt = select_runtime_lane_r0(
        requested_lane=LEGACY_R11,
        declared_science_contract_sha256=runtime_lane_contract_sha256_r0(LEGACY_R11),
    )
    with pytest.raises(RuntimeError, match="ACROUTER_RECEIPT_SCHEMA_MISMATCH"):
        replace(receipt, router_schema_version="CB16_R11_RUNTIME_LANE_ROUTER_V2_R0").validate()


def test_directly_constructed_receipt_cannot_hide_noncanonical_payload() -> None:
    payload = runtime_lane_contract_payload_r0(LEGACY_R11)
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    forged = RuntimeLaneReceiptR0(
        router_schema_version=RUNTIME_ROUTER_SCHEMA_R0,
        selected_lane=LEGACY_R11,
        selected_protocol_id=LEGACY_PROTOCOL_ID_R0,
        selected_science_contract_sha256=runtime_lane_contract_sha256_r0(LEGACY_R11),
        selected_contract_payload_json=pretty,
    )
    with pytest.raises(RuntimeError, match="ACROUTER_CONTRACT_PAYLOAD_MISMATCH"):
        forged.validate()
