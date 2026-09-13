"""Science-agnostic proof-evidence aggregation for CB16 qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .scientific_qualification_contract_v1 import validate_profile_structure_v1


class QualificationEvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class ProofEvidenceV1:
    obligation_id: str
    passed: bool
    evidence_refs: tuple[str, ...]
    detail: str = ""

    def validate(self) -> "ProofEvidenceV1":
        if not self.obligation_id.strip():
            raise QualificationEvidenceError("EMPTY_OBLIGATION_ID")
        if type(self.passed) is not bool:
            raise QualificationEvidenceError("PROOF_STATUS_NOT_BOOL")
        if self.passed and not self.evidence_refs:
            raise QualificationEvidenceError("PASS_WITHOUT_EVIDENCE_REFERENCE")
        if any(not str(item).strip() for item in self.evidence_refs):
            raise QualificationEvidenceError("EMPTY_EVIDENCE_REFERENCE")
        return self


def audit_proof_evidence_v1(
    *,
    profile: Mapping[str, Any],
    evidence: Sequence[ProofEvidenceV1 | Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Check coverage of declared proof obligations without inferring science.

    A failed or missing mandatory proof obligation is a contract violation. This
    function does not decide scientific success criteria; those remain stage gates.
    """

    validate_profile_structure_v1(profile)
    normalized = [
        item.validate() if isinstance(item, ProofEvidenceV1)
        else ProofEvidenceV1(**dict(item)).validate()
        for item in evidence
    ]
    ids = [item.obligation_id for item in normalized]
    if len(set(ids)) != len(ids):
        raise QualificationEvidenceError("DUPLICATE_PROOF_EVIDENCE_ID")

    declared = {str(item["obligation_id"]): dict(item) for item in profile["proof_obligations"]}
    observed = {item.obligation_id: item for item in normalized}
    unknown = sorted(set(observed) - set(declared))
    violations: list[str] = [f"UNDECLARED_PROOF_EVIDENCE:{item}" for item in unknown]
    passed: list[str] = []
    optional_failed: list[str] = []

    for obligation_id, contract in declared.items():
        result = observed.get(obligation_id)
        mandatory = bool(contract.get("mandatory", True))
        if result is None:
            if mandatory:
                violations.append(f"MISSING_MANDATORY_PROOF:{obligation_id}")
            continue
        if result.passed:
            passed.append(obligation_id)
        elif mandatory:
            violations.append(f"MANDATORY_PROOF_FAILED:{obligation_id}")
        else:
            optional_failed.append(obligation_id)

    mandatory_ids = {
        obligation_id for obligation_id, item in declared.items()
        if bool(item.get("mandatory", True))
    }
    all_mandatory_pass = mandatory_ids.issubset(set(passed)) and not violations

    return {
        "schema": "CB16_QUALIFICATION_PROOF_EVIDENCE_AUDIT_V1",
        "all_mandatory_pass": bool(all_mandatory_pass),
        "passed_obligations": sorted(passed),
        "optional_failed_obligations": sorted(optional_failed),
        "contract_violations": sorted(violations),
        "unknown_evidence_obligations": unknown,
    }
