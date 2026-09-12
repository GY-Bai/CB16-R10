from __future__ import annotations

"""Explicit runtime-lane selection for the Round-2 semantic migration.

BC-005 preserves the existing LEGACY_R11 and ACTOR_CRITIC_R0 protocols while
adding BC_ROUND2_R1 as a third, explicit lane.  This module only selects and
receipts a protocol; it does not instantiate a runtime and never falls back to
another lane.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

from .actor_critic_contract_r1 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R1,
    SCIENCE_SEMANTIC_VERSION_R1,
    SCIENCE_VERSION_FIELDS_R1,
    science_contract_sha256_r1,
)
from .actor_critic_runtime_router_r0 import (
    ACTOR_CRITIC_R0,
    LEGACY_PROTOCOL_ID_R0,
    LEGACY_R11,
    runtime_lane_contract_payload_r0,
    runtime_lane_contract_sha256_r0,
)


BC_ROUND2_R1 = "BC_ROUND2_R1"
RUNTIME_LANES_R1 = (LEGACY_R11, ACTOR_CRITIC_R0, BC_ROUND2_R1)
RUNTIME_ROUTER_SCHEMA_R1 = "CB16_R11_BC_ROUND2_RUNTIME_LANE_ROUTER_V1_R1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _require_nonempty_string(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _canonical_json(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ACROUTER_R1_CONTRACT_PAYLOAD_INVALID") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, code: str) -> str:
    text = _require_nonempty_string(value, code=code)
    if _HEX64.fullmatch(text) is None:
        raise RuntimeError(code)
    return text


def _validate_lane(value: object) -> str:
    lane = _require_nonempty_string(value, code="ACROUTER_R1_LANE_INVALID")
    if lane not in RUNTIME_LANES_R1:
        raise RuntimeError("ACROUTER_R1_LANE_UNKNOWN")
    return lane


def _round2_science_contract_payload() -> dict[str, str]:
    ACTOR_CRITIC_SCIENCE_CONTRACT_R1.validate()
    return {
        field_name: getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R1, field_name)
        for field_name in SCIENCE_VERSION_FIELDS_R1
    }


def runtime_lane_contract_payload_r1(lane: object) -> dict[str, object]:
    """Return the exact descriptor for one explicitly selected runtime lane."""

    selected = _validate_lane(lane)
    if selected in (LEGACY_R11, ACTOR_CRITIC_R0):
        upstream = runtime_lane_contract_payload_r0(selected)
        return {
            "router_schema_version": RUNTIME_ROUTER_SCHEMA_R1,
            "runtime_lane": selected,
            "protocol_id": str(upstream["protocol_id"]),
            "upstream_r0_lane_contract_sha256": runtime_lane_contract_sha256_r0(selected),
            "upstream_r0_lane_contract": upstream,
        }

    round2_hash = science_contract_sha256_r1()
    return {
        "router_schema_version": RUNTIME_ROUTER_SCHEMA_R1,
        "runtime_lane": BC_ROUND2_R1,
        "protocol_id": SCIENCE_SEMANTIC_VERSION_R1,
        "round2_science_contract_sha256": round2_hash,
        "round2_science_contract": _round2_science_contract_payload(),
    }


def canonical_runtime_lane_contract_json_r1(lane: object) -> str:
    return _canonical_json(runtime_lane_contract_payload_r1(lane))


def runtime_lane_contract_sha256_r1(lane: object) -> str:
    return _sha256_text(canonical_runtime_lane_contract_json_r1(lane))


@dataclass(frozen=True)
class RuntimeLaneReceiptR1:
    """Immutable receipt for one explicit legacy/R0/Round-2 lane selection."""

    router_schema_version: str
    selected_lane: str
    selected_protocol_id: str
    selected_lane_contract_sha256: str
    selected_round2_science_contract_sha256: str | None
    selected_contract_payload_json: str

    def validate(self) -> None:
        if self.router_schema_version != RUNTIME_ROUTER_SCHEMA_R1:
            raise RuntimeError("ACROUTER_R1_RECEIPT_SCHEMA_MISMATCH")
        lane = _validate_lane(self.selected_lane)
        expected_payload = runtime_lane_contract_payload_r1(lane)
        expected_json = _canonical_json(expected_payload)
        expected_lane_hash = _sha256_text(expected_json)
        expected_protocol_id = str(expected_payload["protocol_id"])

        if self.selected_protocol_id != expected_protocol_id:
            raise RuntimeError("ACROUTER_R1_PROTOCOL_ID_MISMATCH")
        if _require_sha256(
            self.selected_lane_contract_sha256,
            code="ACROUTER_R1_LANE_CONTRACT_HASH_INVALID",
        ) != expected_lane_hash:
            raise RuntimeError("ACROUTER_R1_LANE_CONTRACT_HASH_MISMATCH")
        if self.selected_contract_payload_json != expected_json:
            raise RuntimeError("ACROUTER_R1_CONTRACT_PAYLOAD_MISMATCH")

        if lane == BC_ROUND2_R1:
            expected_science_hash = science_contract_sha256_r1()
            if self.selected_round2_science_contract_sha256 is None:
                raise RuntimeError("ACROUTER_R1_ROUND2_SCIENCE_HASH_MISSING")
            if _require_sha256(
                self.selected_round2_science_contract_sha256,
                code="ACROUTER_R1_ROUND2_SCIENCE_HASH_INVALID",
            ) != expected_science_hash:
                raise RuntimeError("ACROUTER_R1_ROUND2_SCIENCE_HASH_MISMATCH")
            if expected_payload.get("round2_science_contract_sha256") != expected_science_hash:
                raise RuntimeError("ACROUTER_R1_ROUND2_PAYLOAD_SCIENCE_HASH_MISMATCH")
        elif self.selected_round2_science_contract_sha256 is not None:
            raise RuntimeError("ACROUTER_R1_ROUND2_SCIENCE_HASH_ON_NONROUND2_LANE")

        try:
            decoded = json.loads(self.selected_contract_payload_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ACROUTER_R1_CONTRACT_PAYLOAD_INVALID") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("ACROUTER_R1_CONTRACT_PAYLOAD_INVALID")
        if _canonical_json(decoded) != self.selected_contract_payload_json:
            raise RuntimeError("ACROUTER_R1_CONTRACT_PAYLOAD_NONCANONICAL")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "router_schema_version": self.router_schema_version,
            "selected_lane": self.selected_lane,
            "selected_protocol_id": self.selected_protocol_id,
            "selected_lane_contract_sha256": self.selected_lane_contract_sha256,
            "selected_round2_science_contract_sha256": self.selected_round2_science_contract_sha256,
            "selected_contract_payload": json.loads(self.selected_contract_payload_json),
        }


def select_runtime_lane_r1(
    *,
    requested_lane: object,
    declared_lane_contract_sha256: object,
) -> RuntimeLaneReceiptR1:
    """Select exactly one lane after the caller commits to its exact descriptor."""

    lane = _validate_lane(requested_lane)
    declared_hash = _require_sha256(
        declared_lane_contract_sha256,
        code="ACROUTER_R1_DECLARED_LANE_CONTRACT_HASH_INVALID",
    )
    expected_payload = runtime_lane_contract_payload_r1(lane)
    expected_json = _canonical_json(expected_payload)
    expected_lane_hash = _sha256_text(expected_json)
    if declared_hash != expected_lane_hash:
        raise RuntimeError("ACROUTER_R1_DECLARED_LANE_CONTRACT_HASH_MISMATCH")

    receipt = RuntimeLaneReceiptR1(
        router_schema_version=RUNTIME_ROUTER_SCHEMA_R1,
        selected_lane=lane,
        selected_protocol_id=str(expected_payload["protocol_id"]),
        selected_lane_contract_sha256=expected_lane_hash,
        selected_round2_science_contract_sha256=(
            science_contract_sha256_r1() if lane == BC_ROUND2_R1 else None
        ),
        selected_contract_payload_json=expected_json,
    )
    receipt.validate()
    return receipt
