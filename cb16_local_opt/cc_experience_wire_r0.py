from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

W02_VERSION = "CCEnvironmentTransitionV1"
W03_VERSION = "CCExperienceSequenceV1"
W05_VERSION = "CCEconomicResultV1"


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric value")
        if value == 0.0:
            return 0.0
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def content_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True)
class CCEnvironmentTransitionV1:
    account_lineage_id: str
    decision_index: int
    environment_time_before: str
    environment_time_after: str
    pre_account_truth_hash: str
    policy_decision_ref: str
    permission_status: str
    permission_reason: str
    permitted_target_direction: str
    permitted_target_risk: float
    target_quantity: float
    execution_legs: tuple[Mapping[str, Any], ...]
    fees: float
    funding: float
    realized_pnl: float
    unrealized_pnl_delta: float
    liability_delta: float
    post_account_truth_hash: str
    post_equity: float
    boundary_type: str
    mechanical_terminal: bool
    external_capital_flow_ref_or_null: str | None

    def validate(self) -> "CCEnvironmentTransitionV1":
        for n in ("account_lineage_id", "environment_time_before", "environment_time_after", "pre_account_truth_hash",
                  "policy_decision_ref", "permission_status", "permission_reason", "permitted_target_direction",
                  "post_account_truth_hash", "boundary_type"):
            _require_text(n, getattr(self, n))
        if self.decision_index < 0:
            raise ValueError("decision_index must be >= 0")
        if self.environment_time_after < self.environment_time_before:
            raise ValueError("environment time must not reverse")
        if self.permitted_target_direction not in {"SHORT", "FLAT", "LONG"}:
            raise ValueError("invalid permitted_target_direction")
        if not 0.0 <= float(self.permitted_target_risk) <= 1.0:
            raise ValueError("permitted_target_risk outside [0,1]")
        if self.permitted_target_direction == "FLAT" and float(self.permitted_target_risk) != 0.0:
            raise ValueError("FLAT permitted_target_risk must be 0")
        for n in ("target_quantity", "fees", "funding", "realized_pnl", "unrealized_pnl_delta", "liability_delta", "post_equity"):
            if not math.isfinite(float(getattr(self, n))):
                raise ValueError(f"{n} must be finite")
        if not isinstance(self.execution_legs, tuple):
            raise ValueError("execution_legs must be an immutable tuple")
        for expected_index, leg in enumerate(self.execution_legs):
            if not isinstance(leg, Mapping):
                raise ValueError("each execution leg must be a mapping")
            if leg.get("leg_index") != expected_index:
                raise ValueError("execution legs must be explicitly ordered by contiguous leg_index")
            if "executed_quantity" not in leg:
                raise ValueError("execution leg missing executed_quantity")
            qty = float(leg["executed_quantity"])
            if not math.isfinite(qty):
                raise ValueError("executed_quantity must be finite")
            price = leg.get("execution_price")
            if qty != 0.0:
                if price is None or not math.isfinite(float(price)):
                    raise ValueError("nonzero execution requires finite execution_price")
            elif price is not None and not math.isfinite(float(price)):
                raise ValueError("execution_price must be finite when present")
        return self

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)


@dataclass(frozen=True)
class CCExperienceSequenceV1:
    sequence_id: str
    account_lineage_id: str
    science_semantic_version: str
    market_lineage_id: str
    source_classification: str
    transition_refs: tuple[str, ...]
    first_decision_index: int
    last_decision_index: int
    behavior_policy_identities: tuple[str, ...]
    normalizer_identities: tuple[str, ...]
    chunk_boundary_type: str
    bootstrap_state_ref_or_null: str | None
    raw_fact_content_sha256: str

    def validate(self) -> "CCExperienceSequenceV1":
        for n in ("sequence_id", "account_lineage_id", "science_semantic_version", "market_lineage_id",
                  "source_classification", "chunk_boundary_type", "raw_fact_content_sha256"):
            _require_text(n, getattr(self, n))
        if not self.transition_refs:
            raise ValueError("sequence must contain transition refs")
        if len(set(self.transition_refs)) != len(self.transition_refs):
            raise ValueError("duplicate transition refs")
        if self.first_decision_index < 0 or self.last_decision_index < self.first_decision_index:
            raise ValueError("invalid decision interval")
        return self

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)


@dataclass(frozen=True)
class CCEconomicResultV1:
    evaluation_id: str
    policy_object_type: str
    policy_identity: str
    cohort_id: str
    common_horizon_id: str
    capital_denominator_id: str
    account_results: tuple[Mapping[str, Any], ...]
    mean_arithmetic_return: float
    buy_hold_delta: float
    flat_delta: float
    failure_counts: Mapping[str, int]
    tail_diagnostics: Mapping[str, float]
    result_scope: str

    def validate(self) -> "CCEconomicResultV1":
        for n in ("evaluation_id", "policy_object_type", "policy_identity", "cohort_id", "common_horizon_id",
                  "capital_denominator_id", "result_scope"):
            _require_text(n, getattr(self, n))
        if self.policy_object_type not in {"frozen_checkpoint", "generation_chain", "baseline"}:
            raise ValueError("invalid policy_object_type")
        for n in ("mean_arithmetic_return", "buy_hold_delta", "flat_delta"):
            if not math.isfinite(float(getattr(self, n))):
                raise ValueError(f"{n} must be finite")
        if any(int(v) < 0 for v in self.failure_counts.values()):
            raise ValueError("failure counts must be non-negative")
        return self

    @property
    def content_sha256(self) -> str:
        self.validate()
        return content_sha256(self)
