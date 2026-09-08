from __future__ import annotations

"""Stage-4 fail-closed permission boundary for CB16 R11.

This module does not define Permission, Supervisor policy, or Physics semantics.  It
only (1) validates/normalizes the semantic ActionIntent contract and (2) ensures
that execution is reached exclusively by invoking the already-frozen Supervisor
chain in-process.  Caller-supplied Supervisor decisions, ExecutableAction objects,
permission tokens, or account transitions are never accepted as authority.
"""

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping

SHORT, FLAT, LONG = 0, 1, 2
_ACTION_KEYS = frozenset({"schema", "direction", "requested_risk_multiplier", "lineage"})
_LINEAGE_KEYS = frozenset({"generator", "source_snapshot_sha256", "market_packet_id", "account_id", "intent_step"})
_COMPAT_KEYS = frozenset({"direction", "requested_risk"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

_PRIVILEGED_KEYS = frozenset({
    "allowed_action", "permission", "permission_granted", "permission_token", "permitted",
    "approved", "approval", "supervisor_approved", "supervisor_decision", "physics_approved",
    "executable", "executable_action", "execute", "execution_authority", "account_transition",
    "physics_transition", "confidence", "risk_confidence",
})


class Stage4PermissionBoundaryError(RuntimeError):
    """Fail-closed S4E boundary violation."""


@dataclass(frozen=True)
class PermissionChainReceiptR11:
    """Engineering receipt proving call ordering; it is not Permission authority."""

    intent: Mapping[str, Any]
    supervisor_decision: Mapping[str, Any]
    executable_action: Mapping[str, Any]
    physics_result: Mapping[str, Any]
    scientific_semantics_changed: bool = False
    permission_minted_by_stage4: bool = False


def _fail(code: str) -> None:
    raise Stage4PermissionBoundaryError(code)


def _scan_for_privileged_fields(value: Any, path: str = "$", *, depth: int = 0) -> None:
    if depth > 8:
        _fail("S4E_INPUT_NESTING_TOO_DEEP")
    if isinstance(value, Mapping):
        for key, child in value.items():
            k = str(key)
            if k in _PRIVILEGED_KEYS:
                _fail(f"S4E_PRIVILEGED_FIELD_REJECTED:{path}.{k}")
            _scan_for_privileged_fields(child, f"{path}.{k}", depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _scan_for_privileged_fields(child, f"{path}[{i}]", depth=depth + 1)


def _validate_direction(direction: Any) -> int:
    if isinstance(direction, bool) or not isinstance(direction, int) or direction not in (SHORT, FLAT, LONG):
        _fail("S4E_ACTION_INTENT_DIRECTION_INVALID")
    return int(direction)


def _validate_risk(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("S4E_ACTION_INTENT_RISK_NOT_NUMERIC")
    risk = float(value)
    if not math.isfinite(risk):
        _fail("S4E_ACTION_INTENT_RISK_NOT_FINITE")
    if not 0.0 <= risk <= 1.0:
        _fail("S4E_ACTION_INTENT_RISK_OUT_OF_RANGE")
    return risk


def _validate_lineage(lineage: Any) -> dict[str, Any]:
    if not isinstance(lineage, Mapping):
        _fail("S4E_ACTION_INTENT_LINEAGE_REQUIRED")
    if set(lineage) != _LINEAGE_KEYS:
        _fail("S4E_ACTION_INTENT_LINEAGE_SHAPE_INVALID")
    generator = lineage["generator"]
    account_id = lineage["account_id"]
    source_snapshot = lineage["source_snapshot_sha256"]
    market_packet = lineage["market_packet_id"]
    intent_step = lineage["intent_step"]
    if not isinstance(generator, str) or not generator:
        _fail("S4E_ACTION_INTENT_GENERATOR_INVALID")
    if not isinstance(account_id, str) or not account_id:
        _fail("S4E_ACTION_INTENT_ACCOUNT_ID_INVALID")
    if not isinstance(source_snapshot, str) or _HEX64.fullmatch(source_snapshot) is None:
        _fail("S4E_ACTION_INTENT_SOURCE_SNAPSHOT_INVALID")
    if not isinstance(market_packet, str) or _HEX64.fullmatch(market_packet) is None:
        _fail("S4E_ACTION_INTENT_MARKET_PACKET_INVALID")
    if isinstance(intent_step, bool) or not isinstance(intent_step, int) or intent_step < 0:
        _fail("S4E_ACTION_INTENT_STEP_INVALID")
    return {
        "generator": generator,
        "source_snapshot_sha256": source_snapshot,
        "market_packet_id": market_packet,
        "account_id": account_id,
        "intent_step": int(intent_step),
    }


def validate_action_intent_r11(intent: Any) -> dict[str, Any]:
    """Validate the frozen ActionIntentV1 semantic contract exactly."""
    if not isinstance(intent, Mapping):
        _fail("S4E_ACTION_INTENT_MAPPING_REQUIRED")
    _scan_for_privileged_fields(intent)
    if set(intent) != _ACTION_KEYS:
        _fail("S4E_ACTION_INTENT_SHAPE_INVALID")
    if intent.get("schema") != "ActionIntentV1":
        _fail("S4E_ACTION_INTENT_SCHEMA_INVALID")
    direction = _validate_direction(intent.get("direction"))
    risk = _validate_risk(intent.get("requested_risk_multiplier"))
    if direction == FLAT and risk != 0.0:
        _fail("S4E_FLAT_REQUESTED_RISK_MUST_BE_ZERO")
    return {
        "schema": "ActionIntentV1",
        "direction": direction,
        "requested_risk_multiplier": risk,
        "lineage": _validate_lineage(intent.get("lineage")),
    }


def brain_action_to_action_intent_r11(*, direction: Any, requested_risk: Any, lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Representation-only Brain output conversion; requested risk is never confidence/Permission."""
    direction_v = _validate_direction(direction)
    risk_v = _validate_risk(requested_risk)
    if direction_v == FLAT and risk_v != 0.0:
        _fail("S4E_FLAT_REQUESTED_RISK_MUST_BE_ZERO")
    return validate_action_intent_r11({
        "schema": "ActionIntentV1",
        "direction": direction_v,
        "requested_risk_multiplier": risk_v,
        "lineage": _validate_lineage(lineage),
    })


def compatibility_action_to_action_intent_r11(raw: Any, *, lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Translate legacy representation without upgrading it into authority."""
    if not isinstance(raw, Mapping):
        _fail("S4E_COMPATIBILITY_MAPPING_REQUIRED")
    _scan_for_privileged_fields(raw)
    if set(raw) != _COMPAT_KEYS:
        _fail("S4E_COMPATIBILITY_SHAPE_INVALID")
    return brain_action_to_action_intent_r11(
        direction=raw["direction"], requested_risk=raw["requested_risk"], lineage=lineage
    )


def recovery_action_to_action_intent_r11(raw: Any, *, lineage: Mapping[str, Any]) -> dict[str, Any]:
    """Recovery may reconstruct intent representation, never revive old Permission."""
    return compatibility_action_to_action_intent_r11(raw, lineage=lineage)


class FrozenPermissionBoundaryR11:
    """Execution bridge enforcing frozen Supervisor -> executable -> Physics ordering."""

    def __init__(self, frozen_runtime: Any):
        if frozen_runtime is None:
            _fail("S4E_FROZEN_RUNTIME_REQUIRED")
        supervisor = getattr(frozen_runtime, "supervisor", None)
        physics = getattr(frozen_runtime, "physics", None)
        contract = getattr(frozen_runtime, "physics_contract", None)
        if supervisor is None:
            _fail("S4E_FROZEN_SUPERVISOR_REQUIRED")
        if physics is None:
            _fail("S4E_FROZEN_PHYSICS_REQUIRED")
        if not isinstance(contract, Mapping):
            _fail("S4E_FROZEN_PHYSICS_CONTRACT_REQUIRED")
        for name in ("supervise", "executable_action", "execute_physics"):
            if not callable(getattr(supervisor, name, None)):
                _fail(f"S4E_FROZEN_SUPERVISOR_API_MISSING:{name}")
        self._runtime = frozen_runtime
        self._supervisor = supervisor
        self._physics = physics
        self._contract = contract

    def execute_intent(
        self,
        intent: Any,
        *,
        snapshot: Mapping[str, Any],
        risk_authority: Mapping[str, Any],
        market_execution_input: Mapping[str, Any],
    ) -> PermissionChainReceiptR11:
        canonical = validate_action_intent_r11(intent)
        decision = self._supervisor.supervise(canonical, snapshot, risk_authority, self._contract)
        if not isinstance(decision, Mapping) or decision.get("schema") != "RiskSupervisorDecisionV1":
            _fail("S4E_SUPERVISOR_DECISION_INVALID")
        executable = self._supervisor.executable_action(decision, self._contract)
        if not isinstance(executable, Mapping) or executable.get("schema") != "ExecutableActionV1":
            _fail("S4E_SUPERVISOR_EXECUTABLE_INVALID")
        result = self._supervisor.execute_physics(snapshot, executable, market_execution_input, self._contract)
        if not isinstance(result, Mapping):
            _fail("S4E_PHYSICS_RESULT_INVALID")
        return PermissionChainReceiptR11(
            intent=canonical,
            supervisor_decision=dict(decision),
            executable_action=dict(executable),
            physics_result=dict(result),
        )

    def reject_direct_execution_object(self, value: Any) -> None:
        _scan_for_privileged_fields(value)
        _fail("S4E_DIRECT_EXECUTION_OBJECT_FORBIDDEN")
