from __future__ import annotations

"""Fail-closed runtime-lane selection for the R11 migration.

This module does not instantiate either runtime.  It freezes the protocol-routing
boundary only: callers must explicitly select one lane and must present the
canonical hash of that lane's declared science/runtime contract.  A mismatch is
an error, never a request to fall back to the other lane.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping

from .actor_critic_contract_r0 import (
    ACTOR_CRITIC_SCIENCE_CONTRACT_R0,
    SCIENCE_SEMANTIC_VERSION_R0,
    SCIENCE_VERSION_FIELDS_R0,
)


LEGACY_R11 = "LEGACY_R11"
ACTOR_CRITIC_R0 = "ACTOR_CRITIC_R0"
RUNTIME_LANES_R0 = (LEGACY_R11, ACTOR_CRITIC_R0)
RUNTIME_ROUTER_SCHEMA_R0 = "CB16_R11_RUNTIME_LANE_ROUTER_V1_R0"
LEGACY_PROTOCOL_ID_R0 = "CB16_R11_LEGACY_FROZEN_PROTOCOL_AC003_R0"
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
        raise RuntimeError("ACROUTER_CONTRACT_PAYLOAD_INVALID") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, code: str) -> str:
    text = _require_nonempty_string(value, code=code)
    if _HEX64.fullmatch(text) is None:
        raise RuntimeError(code)
    return text


def _validate_lane(value: object) -> str:
    lane = _require_nonempty_string(value, code="ACROUTER_LANE_INVALID")
    if lane not in RUNTIME_LANES_R0:
        raise RuntimeError("ACROUTER_LANE_UNKNOWN")
    return lane


def runtime_lane_contract_payload_r0(lane: object) -> dict[str, object]:
    """Return the exact canonical protocol descriptor for one explicit lane."""

    selected = _validate_lane(lane)
    if selected == LEGACY_R11:
        return {
            "router_schema_version": RUNTIME_ROUTER_SCHEMA_R0,
            "runtime_lane": LEGACY_R11,
            "protocol_id": LEGACY_PROTOCOL_ID_R0,
        }

    ACTOR_CRITIC_SCIENCE_CONTRACT_R0.validate()
    return {
        "router_schema_version": RUNTIME_ROUTER_SCHEMA_R0,
        "runtime_lane": ACTOR_CRITIC_R0,
        "protocol_id": SCIENCE_SEMANTIC_VERSION_R0,
        "actor_critic_science_contract": {
            field_name: getattr(ACTOR_CRITIC_SCIENCE_CONTRACT_R0, field_name)
            for field_name in SCIENCE_VERSION_FIELDS_R0
        },
    }


def canonical_runtime_lane_contract_json_r0(lane: object) -> str:
    return _canonical_json(runtime_lane_contract_payload_r0(lane))


def runtime_lane_contract_sha256_r0(lane: object) -> str:
    return _sha256_text(canonical_runtime_lane_contract_json_r0(lane))


@dataclass(frozen=True)
class RuntimeLaneReceiptR0:
    """Immutable evidence of the explicitly selected runtime/science protocol."""

    router_schema_version: str
    selected_lane: str
    selected_protocol_id: str
    selected_science_contract_sha256: str
    selected_contract_payload_json: str

    def validate(self) -> None:
        if self.router_schema_version != RUNTIME_ROUTER_SCHEMA_R0:
            raise RuntimeError("ACROUTER_RECEIPT_SCHEMA_MISMATCH")
        lane = _validate_lane(self.selected_lane)
        expected_payload = runtime_lane_contract_payload_r0(lane)
        expected_json = _canonical_json(expected_payload)
        expected_hash = _sha256_text(expected_json)
        expected_protocol_id = str(expected_payload["protocol_id"])

        if self.selected_protocol_id != expected_protocol_id:
            raise RuntimeError("ACROUTER_PROTOCOL_ID_MISMATCH")
        if _require_sha256(
            self.selected_science_contract_sha256,
            code="ACROUTER_CONTRACT_HASH_INVALID",
        ) != expected_hash:
            raise RuntimeError("ACROUTER_CONTRACT_HASH_MISMATCH")
        if self.selected_contract_payload_json != expected_json:
            raise RuntimeError("ACROUTER_CONTRACT_PAYLOAD_MISMATCH")

        try:
            decoded = json.loads(self.selected_contract_payload_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ACROUTER_CONTRACT_PAYLOAD_INVALID") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("ACROUTER_CONTRACT_PAYLOAD_INVALID")
        if _canonical_json(decoded) != self.selected_contract_payload_json:
            raise RuntimeError("ACROUTER_CONTRACT_PAYLOAD_NONCANONICAL")

    def to_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "router_schema_version": self.router_schema_version,
            "selected_lane": self.selected_lane,
            "selected_protocol_id": self.selected_protocol_id,
            "selected_science_contract_sha256": self.selected_science_contract_sha256,
            "selected_contract_payload": json.loads(self.selected_contract_payload_json),
        }


def select_runtime_lane_r0(
    *,
    requested_lane: object,
    declared_science_contract_sha256: object,
) -> RuntimeLaneReceiptR0:
    """Select exactly one lane, requiring its caller-declared contract hash.

    There is deliberately no default lane and no fallback behavior.  The caller
    must first obtain/record the expected hash for the protocol it intends to run.
    """

    lane = _validate_lane(requested_lane)
    declared_hash = _require_sha256(
        declared_science_contract_sha256,
        code="ACROUTER_DECLARED_CONTRACT_HASH_INVALID",
    )
    expected_payload = runtime_lane_contract_payload_r0(lane)
    expected_json = _canonical_json(expected_payload)
    expected_hash = _sha256_text(expected_json)
    if declared_hash != expected_hash:
        raise RuntimeError("ACROUTER_DECLARED_CONTRACT_HASH_MISMATCH")

    receipt = RuntimeLaneReceiptR0(
        router_schema_version=RUNTIME_ROUTER_SCHEMA_R0,
        selected_lane=lane,
        selected_protocol_id=str(expected_payload["protocol_id"]),
        selected_science_contract_sha256=expected_hash,
        selected_contract_payload_json=expected_json,
    )
    receipt.validate()
    return receipt
