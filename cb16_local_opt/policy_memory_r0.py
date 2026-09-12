from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Mapping


POLICY_MEMORY_SCHEMA_R0 = "CB16_R11_BC_POLICY_MEMORY_V1_R0"
POLICY_MEMORY_ACTIVATION_EVIDENCE_SCHEMA_R0 = "CB16_R11_BC_POLICY_MEMORY_ACTIVATION_EVIDENCE_V1_R0"
POLICY_MEMORY_SEAL_SCHEMA_R0 = "CB16_R11_BC_POLICY_MEMORY_SEAL_V1_R0"
POLICY_MEMORY_DISABLED = "DISABLED"
POLICY_MEMORY_ENABLED = "ENABLED"
POLICY_MEMORY_MODES_R0 = (POLICY_MEMORY_DISABLED, POLICY_MEMORY_ENABLED)
POLICY_MEMORY_DEFAULT_MODE_R0 = POLICY_MEMORY_DISABLED


def _canonical_json(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("ACMEM_R0_NONCANONICAL_PAYLOAD") from exc


def _sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(value: object, *, code: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(code)
    return value


def _require_text(value: object, *, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(code)
    return value


def _finite_features(values: tuple[float, ...]) -> tuple[float, ...]:
    out = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in out):
        raise RuntimeError("ACMEM_R0_FEATURE_NONFINITE")
    return out


@dataclass(frozen=True)
class PolicyMemoryActivationEvidenceR0:
    schema_version: str
    evidence_id: str
    state_sufficiency_report_sha256: str
    representation_gap_count: int
    causal_history_required: bool
    rationale_sha256: str

    def validate(self) -> None:
        if self.schema_version != POLICY_MEMORY_ACTIVATION_EVIDENCE_SCHEMA_R0:
            raise RuntimeError("ACMEM_R0_EVIDENCE_SCHEMA_MISMATCH")
        _require_text(self.evidence_id, code="ACMEM_R0_EVIDENCE_ID_INVALID")
        _require_sha256(
            self.state_sufficiency_report_sha256,
            code="ACMEM_R0_STATE_SUFFICIENCY_HASH_INVALID",
        )
        _require_sha256(self.rationale_sha256, code="ACMEM_R0_RATIONALE_HASH_INVALID")
        if isinstance(self.representation_gap_count, bool) or self.representation_gap_count <= 0:
            raise RuntimeError("ACMEM_R0_REPRESENTATION_GAP_REQUIRED")
        if self.causal_history_required is not True:
            raise RuntimeError("ACMEM_R0_CAUSAL_HISTORY_EVIDENCE_REQUIRED")

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        return _sha256(dict(self.__dict__))


@dataclass(frozen=True)
class PolicyMemoryConfigR0:
    schema_version: str
    mode: str
    memory_version: str
    activation_evidence: PolicyMemoryActivationEvidenceR0 | None

    def validate(self) -> None:
        if self.schema_version != POLICY_MEMORY_SCHEMA_R0:
            raise RuntimeError("ACMEM_R0_CONFIG_SCHEMA_MISMATCH")
        if self.mode not in POLICY_MEMORY_MODES_R0:
            raise RuntimeError("ACMEM_R0_MODE_INVALID")
        _require_text(self.memory_version, code="ACMEM_R0_MEMORY_VERSION_INVALID")
        if self.mode == POLICY_MEMORY_DISABLED:
            if self.activation_evidence is not None:
                raise RuntimeError("ACMEM_R0_DISABLED_HAS_ACTIVATION_EVIDENCE")
        else:
            if self.activation_evidence is None:
                raise RuntimeError("ACMEM_R0_ENABLED_WITHOUT_EVIDENCE")
            self.activation_evidence.validate()

    @property
    def semantic_sha256(self) -> str:
        self.validate()
        payload = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "memory_version": self.memory_version,
            "activation_evidence_sha256": (
                None if self.activation_evidence is None else self.activation_evidence.semantic_sha256
            ),
        }
        return _sha256(payload)


def disabled_policy_memory_config_r0() -> PolicyMemoryConfigR0:
    config = PolicyMemoryConfigR0(
        schema_version=POLICY_MEMORY_SCHEMA_R0,
        mode=POLICY_MEMORY_DEFAULT_MODE_R0,
        memory_version="CB16_R11_BC_POLICY_MEMORY_DISABLED_V1_R0",
        activation_evidence=None,
    )
    config.validate()
    return config


def enabled_policy_memory_config_r0(
    *,
    memory_version: str,
    activation_evidence: PolicyMemoryActivationEvidenceR0,
) -> PolicyMemoryConfigR0:
    config = PolicyMemoryConfigR0(
        schema_version=POLICY_MEMORY_SCHEMA_R0,
        mode=POLICY_MEMORY_ENABLED,
        memory_version=memory_version,
        activation_evidence=activation_evidence,
    )
    config.validate()
    return config


@dataclass(frozen=True)
class CausalMemoryUpdateR0:
    logical_account_id: str
    decision_index: int
    source_policy_observation_sha256: str
    causal_prefix_sha256: str
    memory_features: tuple[float, ...]

    def validate(self) -> None:
        _require_text(self.logical_account_id, code="ACMEM_R0_ACCOUNT_ID_INVALID")
        if isinstance(self.decision_index, bool) or not isinstance(self.decision_index, int) or self.decision_index < 0:
            raise RuntimeError("ACMEM_R0_DECISION_INDEX_INVALID")
        _require_sha256(
            self.source_policy_observation_sha256,
            code="ACMEM_R0_SOURCE_OBSERVATION_HASH_INVALID",
        )
        _require_sha256(self.causal_prefix_sha256, code="ACMEM_R0_CAUSAL_PREFIX_HASH_INVALID")
        _finite_features(self.memory_features)


@dataclass(frozen=True)
class PolicyMemoryStateR0:
    schema_version: str
    config_sha256: str
    logical_account_id: str
    last_decision_index: int
    causal_prefix_sha256: str | None
    memory_features: tuple[float, ...]

    def validate(self) -> None:
        if self.schema_version != POLICY_MEMORY_SCHEMA_R0:
            raise RuntimeError("ACMEM_R0_STATE_SCHEMA_MISMATCH")
        _require_sha256(self.config_sha256, code="ACMEM_R0_CONFIG_HASH_INVALID")
        _require_text(self.logical_account_id, code="ACMEM_R0_ACCOUNT_ID_INVALID")
        if isinstance(self.last_decision_index, bool) or not isinstance(self.last_decision_index, int):
            raise RuntimeError("ACMEM_R0_LAST_DECISION_INDEX_INVALID")
        if self.last_decision_index < -1:
            raise RuntimeError("ACMEM_R0_LAST_DECISION_INDEX_INVALID")
        if self.last_decision_index == -1:
            if self.causal_prefix_sha256 is not None or self.memory_features:
                raise RuntimeError("ACMEM_R0_UNINITIALIZED_STATE_NOT_EMPTY")
        else:
            _require_sha256(self.causal_prefix_sha256, code="ACMEM_R0_CAUSAL_PREFIX_HASH_INVALID")
            _finite_features(self.memory_features)


def initial_policy_memory_state_r0(
    *,
    config: PolicyMemoryConfigR0,
    logical_account_id: str,
) -> PolicyMemoryStateR0:
    config.validate()
    state = PolicyMemoryStateR0(
        schema_version=POLICY_MEMORY_SCHEMA_R0,
        config_sha256=config.semantic_sha256,
        logical_account_id=_require_text(logical_account_id, code="ACMEM_R0_ACCOUNT_ID_INVALID"),
        last_decision_index=-1,
        causal_prefix_sha256=None,
        memory_features=(),
    )
    state.validate()
    return state


def advance_policy_memory_r0(
    *,
    config: PolicyMemoryConfigR0,
    state: PolicyMemoryStateR0,
    update: CausalMemoryUpdateR0,
) -> PolicyMemoryStateR0:
    config.validate()
    state.validate()
    update.validate()
    if state.config_sha256 != config.semantic_sha256:
        raise RuntimeError("ACMEM_R0_CONFIG_IDENTITY_MISMATCH")
    if state.logical_account_id != update.logical_account_id:
        raise RuntimeError("ACMEM_R0_ACCOUNT_ID_MISMATCH")

    if config.mode == POLICY_MEMORY_DISABLED:
        # Disabled mode does not consume, mutate or expose hidden state.
        return state

    expected_index = state.last_decision_index + 1
    if update.decision_index != expected_index:
        raise RuntimeError("ACMEM_R0_CAUSAL_PREFIX_ORDER_VIOLATION")

    next_state = PolicyMemoryStateR0(
        schema_version=POLICY_MEMORY_SCHEMA_R0,
        config_sha256=config.semantic_sha256,
        logical_account_id=state.logical_account_id,
        last_decision_index=update.decision_index,
        causal_prefix_sha256=update.causal_prefix_sha256,
        memory_features=_finite_features(update.memory_features),
    )
    next_state.validate()
    return next_state


def policy_memory_model_features_r0(
    *,
    config: PolicyMemoryConfigR0,
    state: PolicyMemoryStateR0,
) -> tuple[float, ...]:
    config.validate()
    state.validate()
    if state.config_sha256 != config.semantic_sha256:
        raise RuntimeError("ACMEM_R0_CONFIG_IDENTITY_MISMATCH")
    if config.mode == POLICY_MEMORY_DISABLED:
        return ()
    return _finite_features(state.memory_features)


def seal_policy_memory_state_r0(state: PolicyMemoryStateR0) -> dict[str, object]:
    state.validate()
    payload = {
        "schema_version": POLICY_MEMORY_SEAL_SCHEMA_R0,
        "state": {
            "schema_version": state.schema_version,
            "config_sha256": state.config_sha256,
            "logical_account_id": state.logical_account_id,
            "last_decision_index": state.last_decision_index,
            "causal_prefix_sha256": state.causal_prefix_sha256,
            "memory_features": list(state.memory_features),
        },
    }
    payload["seal_sha256"] = _sha256(payload)
    return payload


def restore_policy_memory_state_r0(seal: Mapping[str, object]) -> PolicyMemoryStateR0:
    if seal.get("schema_version") != POLICY_MEMORY_SEAL_SCHEMA_R0:
        raise RuntimeError("ACMEM_R0_SEAL_SCHEMA_MISMATCH")
    seal_sha = _require_sha256(seal.get("seal_sha256"), code="ACMEM_R0_SEAL_HASH_INVALID")
    unsigned = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if _sha256(unsigned) != seal_sha:
        raise RuntimeError("ACMEM_R0_SEAL_TAMPERED")
    raw_state = seal.get("state")
    if not isinstance(raw_state, Mapping):
        raise RuntimeError("ACMEM_R0_SEAL_STATE_INVALID")
    raw_features = raw_state.get("memory_features")
    if not isinstance(raw_features, list):
        raise RuntimeError("ACMEM_R0_SEAL_FEATURES_INVALID")
    state = PolicyMemoryStateR0(
        schema_version=raw_state.get("schema_version"),
        config_sha256=raw_state.get("config_sha256"),
        logical_account_id=raw_state.get("logical_account_id"),
        last_decision_index=raw_state.get("last_decision_index"),
        causal_prefix_sha256=raw_state.get("causal_prefix_sha256"),
        memory_features=tuple(raw_features),
    )
    state.validate()
    return state
