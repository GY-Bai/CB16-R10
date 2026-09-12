from __future__ import annotations

from dataclasses import replace
import json

import pytest

from cb16_local_opt.actor_critic_contract_r1 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
    SCIENCE_SEMANTIC_VERSION_R1,
    SCIENCE_VERSION_FIELDS_R1,
    science_contract_sha256_r1,
)
from cb16_local_opt.actor_critic_runtime_router_r0 import (
    ACTOR_CRITIC_R0,
    LEGACY_R11,
    runtime_lane_contract_payload_r0,
    runtime_lane_contract_sha256_r0,
)
from cb16_local_opt.actor_critic_runtime_router_r1 import (
    BC_ROUND2_R1,
    RUNTIME_LANES_R1,
    RUNTIME_ROUTER_SCHEMA_R1,
    RuntimeLaneReceiptR1,
    canonical_runtime_lane_contract_json_r1,
    runtime_lane_contract_payload_r1,
    runtime_lane_contract_sha256_r1,
    select_runtime_lane_r1,
)


@pytest.mark.parametrize("lane", RUNTIME_LANES_R1)
def test_each_lane_requires_explicit_exact_contract_hash(lane: str) -> None:
    expected_hash = runtime_lane_contract_sha256_r1(lane)
    receipt = select_runtime_lane_r1(
        requested_lane=lane,
        declared_lane_contract_sha256=expected_hash,
    )

    receipt.validate()
    assert receipt.router_schema_version == RUNTIME_ROUTER_SCHEMA_R1
    assert receipt.selected_lane == lane
    assert receipt.selected_lane_contract_sha256 == expected_hash
    assert receipt.to_payload()["selected_contract_payload"] == runtime_lane_contract_payload_r1(lane)


def test_round2_lane_receipt_binds_exact_r1_science_contract_hash_and_registry() -> None:
    payload = runtime_lane_contract_payload_r1(BC_ROUND2_R1)
    receipt = select_runtime_lane_r1(
        requested_lane=BC_ROUND2_R1,
        declared_lane_contract_sha256=runtime_lane_contract_sha256_r1(BC_ROUND2_R1),
    )

    assert payload["protocol_id"] == SCIENCE_SEMANTIC_VERSION_R1
    assert payload["round2_science_contract_sha256"] == science_contract_sha256_r1()
    assert payload["round2_science_contract"] == {
        field_name: getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R1, field_name)
        for field_name in SCIENCE_VERSION_FIELDS_R1
    }
    assert receipt.selected_round2_science_contract_sha256 == science_contract_sha256_r1()


@pytest.mark.parametrize("lane", [LEGACY_R11, ACTOR_CRITIC_R0])
def test_legacy_and_r0_lanes_bind_existing_r0_protocol_without_round2_hash(lane: str) -> None:
    payload = runtime_lane_contract_payload_r1(lane)
    receipt = select_runtime_lane_r1(
        requested_lane=lane,
        declared_lane_contract_sha256=runtime_lane_contract_sha256_r1(lane),
    )

    assert payload["upstream_r0_lane_contract"] == runtime_lane_contract_payload_r0(lane)
    assert payload["upstream_r0_lane_contract_sha256"] == runtime_lane_contract_sha256_r0(lane)
    assert "round2_science_contract" not in payload
    assert receipt.selected_round2_science_contract_sha256 is None


def test_all_three_lane_contract_hashes_are_distinct() -> None:
    hashes = {lane: runtime_lane_contract_sha256_r1(lane) for lane in RUNTIME_LANES_R1}
    assert len(set(hashes.values())) == 3


@pytest.mark.parametrize("lane", [None, "", "AUTO", "ACTOR_CRITIC", "ROUND2", "BC_ROUND2"])
def test_unknown_or_implicit_lane_fails_closed(lane: object) -> None:
    with pytest.raises(RuntimeError, match="ACROUTER_R1_LANE"):
        runtime_lane_contract_payload_r1(lane)


@pytest.mark.parametrize("declared_hash", [None, "", "abc", "0" * 63, "G" * 64])
def test_missing_or_malformed_lane_contract_hash_fails_closed(declared_hash: object) -> None:
    with pytest.raises(RuntimeError, match="ACROUTER_R1_DECLARED_LANE_CONTRACT_HASH_INVALID"):
        select_runtime_lane_r1(
            requested_lane=BC_ROUND2_R1,
            declared_lane_contract_sha256=declared_hash,
        )


@pytest.mark.parametrize(
    ("requested_lane", "wrong_lane"),
    [
        (BC_ROUND2_R1, ACTOR_CRITIC_R0),
        (BC_ROUND2_R1, LEGACY_R11),
        (ACTOR_CRITIC_R0, BC_ROUND2_R1),
        (LEGACY_R11, BC_ROUND2_R1),
    ],
)
def test_cross_lane_hash_never_triggers_fallback(requested_lane: str, wrong_lane: str) -> None:
    with pytest.raises(RuntimeError, match="ACROUTER_R1_DECLARED_LANE_CONTRACT_HASH_MISMATCH"):
        select_runtime_lane_r1(
            requested_lane=requested_lane,
            declared_lane_contract_sha256=runtime_lane_contract_sha256_r1(wrong_lane),
        )


def test_round2_science_hash_cannot_substitute_for_lane_contract_hash() -> None:
    assert science_contract_sha256_r1() != runtime_lane_contract_sha256_r1(BC_ROUND2_R1)
    with pytest.raises(RuntimeError, match="ACROUTER_R1_DECLARED_LANE_CONTRACT_HASH_MISMATCH"):
        select_runtime_lane_r1(
            requested_lane=BC_ROUND2_R1,
            declared_lane_contract_sha256=science_contract_sha256_r1(),
        )


def test_contract_json_is_canonical_and_stable() -> None:
    encoded = canonical_runtime_lane_contract_json_r1(BC_ROUND2_R1)
    assert encoded == canonical_runtime_lane_contract_json_r1(BC_ROUND2_R1)
    assert json.dumps(
        json.loads(encoded),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) == encoded


def test_receipt_tampering_fails_closed() -> None:
    receipt = select_runtime_lane_r1(
        requested_lane=BC_ROUND2_R1,
        declared_lane_contract_sha256=runtime_lane_contract_sha256_r1(BC_ROUND2_R1),
    )

    with pytest.raises(RuntimeError, match="ACROUTER_R1_PROTOCOL_ID_MISMATCH"):
        replace(receipt, selected_protocol_id="WRONG").validate()
    with pytest.raises(RuntimeError, match="ACROUTER_R1_LANE_CONTRACT_HASH_MISMATCH"):
        replace(receipt, selected_lane_contract_sha256="0" * 64).validate()
    with pytest.raises(RuntimeError, match="ACROUTER_R1_ROUND2_SCIENCE_HASH_MISMATCH"):
        replace(receipt, selected_round2_science_contract_sha256="0" * 64).validate()
    with pytest.raises(RuntimeError, match="ACROUTER_R1_CONTRACT_PAYLOAD_MISMATCH"):
        replace(
            receipt,
            selected_contract_payload_json=canonical_runtime_lane_contract_json_r1(ACTOR_CRITIC_R0),
        ).validate()


def test_round2_hash_is_forbidden_on_nonround2_receipt() -> None:
    receipt = select_runtime_lane_r1(
        requested_lane=ACTOR_CRITIC_R0,
        declared_lane_contract_sha256=runtime_lane_contract_sha256_r1(ACTOR_CRITIC_R0),
    )
    with pytest.raises(RuntimeError, match="ACROUTER_R1_ROUND2_SCIENCE_HASH_ON_NONROUND2_LANE"):
        replace(receipt, selected_round2_science_contract_sha256=science_contract_sha256_r1()).validate()


def test_direct_receipt_cannot_hide_noncanonical_payload() -> None:
    payload = runtime_lane_contract_payload_r1(LEGACY_R11)
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    forged = RuntimeLaneReceiptR1(
        router_schema_version=RUNTIME_ROUTER_SCHEMA_R1,
        selected_lane=LEGACY_R11,
        selected_protocol_id=str(payload["protocol_id"]),
        selected_lane_contract_sha256=runtime_lane_contract_sha256_r1(LEGACY_R11),
        selected_round2_science_contract_sha256=None,
        selected_contract_payload_json=pretty,
    )
    with pytest.raises(RuntimeError, match="ACROUTER_R1_CONTRACT_PAYLOAD_MISMATCH"):
        forged.validate()
