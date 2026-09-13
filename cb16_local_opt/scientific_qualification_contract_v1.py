"""Shared CB16 scientific qualification contract primitives.

This module is intentionally science-agnostic. Stage-specific rewards, seeds,
thresholds, task generators and oracles do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PROFILE_SCHEMA_V1 = "CB16_QUALIFICATION_PROFILE_V1"

ALLOWED_VERDICTS_V1 = (
    "PASS",
    "SCIENTIFIC_FAIL",
    "CONTRACT_MISMATCH",
    "EXECUTION_BLOCKED",
    "HARDWARE_LIMIT",
    "EVIDENCE_INSUFFICIENT",
)

ALLOWED_EDGE_DECISIONS_V1 = (
    "PROCESS",
    "ZERO_CONTRIBUTION",
    "FAIL_CLOSED",
    "SCIENTIFIC_FAIL",
)


class QualificationContractError(ValueError):
    pass


@dataclass(frozen=True)
class ProofObligationV1:
    obligation_id: str
    producer: str
    consumer: str
    gate_id: str
    artifact_proof: str
    counterexample_id: str
    mandatory: bool = True

    def validate(self) -> "ProofObligationV1":
        for field_name in (
            "obligation_id",
            "producer",
            "consumer",
            "gate_id",
            "artifact_proof",
            "counterexample_id",
        ):
            if not str(getattr(self, field_name)).strip():
                raise QualificationContractError(f"EMPTY_{field_name.upper()}")
        if type(self.mandatory) is not bool:
            raise QualificationContractError("MANDATORY_FLAG_NOT_BOOL")
        return self


@dataclass(frozen=True)
class CapabilityClaimV1:
    claim_id: str
    proof_obligation_ids: tuple[str, ...]

    def validate(self) -> "CapabilityClaimV1":
        if not str(self.claim_id).strip():
            raise QualificationContractError("EMPTY_CLAIM_ID")
        if not self.proof_obligation_ids:
            raise QualificationContractError("CLAIM_WITHOUT_PROOF_OBLIGATIONS")
        if len(set(self.proof_obligation_ids)) != len(self.proof_obligation_ids):
            raise QualificationContractError("DUPLICATE_PROOF_OBLIGATION_REFERENCE")
        return self


def validate_profile_structure_v1(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate shared structural proof requirements only.

    The validator is deliberately incapable of changing stage science.
    """

    required = ("schema", "stage", "claims", "proof_obligations", "edge_cases")
    for key in required:
        if key not in profile:
            raise QualificationContractError(f"MISSING_PROFILE_FIELD:{key}")
    if profile["schema"] != PROFILE_SCHEMA_V1:
        raise QualificationContractError("PROFILE_SCHEMA_MISMATCH")
    if not str(profile["stage"]).strip():
        raise QualificationContractError("EMPTY_STAGE")
    if not profile["claims"]:
        raise QualificationContractError("PROFILE_WITHOUT_CLAIMS")
    if not profile["proof_obligations"]:
        raise QualificationContractError("PROFILE_WITHOUT_PROOF_OBLIGATIONS")

    obligations = [ProofObligationV1(**dict(item)).validate() for item in profile["proof_obligations"]]
    obligation_ids = [item.obligation_id for item in obligations]
    if len(set(obligation_ids)) != len(obligation_ids):
        raise QualificationContractError("DUPLICATE_PROOF_OBLIGATION_ID")
    known = set(obligation_ids)

    claims = [CapabilityClaimV1(**dict(item)).validate() for item in profile["claims"]]
    claim_ids = [item.claim_id for item in claims]
    if len(set(claim_ids)) != len(claim_ids):
        raise QualificationContractError("DUPLICATE_CLAIM_ID")
    for claim in claims:
        missing = set(claim.proof_obligation_ids) - known
        if missing:
            raise QualificationContractError(f"UNKNOWN_PROOF_OBLIGATION:{sorted(missing)}")

    for case_name, decision in dict(profile["edge_cases"]).items():
        if decision not in ALLOWED_EDGE_DECISIONS_V1:
            raise QualificationContractError(f"INVALID_EDGE_DECISION:{case_name}:{decision}")

    return profile


def classify_qualification_v1(
    *,
    contract_violations: Sequence[str] = (),
    execution_blocked: bool = False,
    hardware_limit: bool = False,
    evidence_insufficient: bool = False,
    mandatory_scientific_gates: Mapping[str, bool] | None = None,
) -> str:
    """Apply shared verdict precedence without stage-specific rescue logic."""

    if contract_violations:
        return "CONTRACT_MISMATCH"

    operational_flags = {
        "execution_blocked": execution_blocked,
        "hardware_limit": hardware_limit,
        "evidence_insufficient": evidence_insufficient,
    }
    if any(type(value) is not bool for value in operational_flags.values()):
        return "CONTRACT_MISMATCH"
    if sum(1 for value in operational_flags.values() if value) > 1:
        return "CONTRACT_MISMATCH"
    if hardware_limit:
        return "HARDWARE_LIMIT"
    if execution_blocked:
        return "EXECUTION_BLOCKED"
    if evidence_insufficient:
        return "EVIDENCE_INSUFFICIENT"

    gates = dict(mandatory_scientific_gates or {})
    if not gates:
        return "EVIDENCE_INSUFFICIENT"
    if any(type(value) is not bool for value in gates.values()):
        return "CONTRACT_MISMATCH"
    if all(gates.values()):
        return "PASS"
    return "SCIENTIFIC_FAIL"
