import pytest

from cb16_local_opt.scientific_qualification_contract_v1 import (
    QualificationContractError,
    validate_profile_structure_v1,
)
from cb16_local_opt.scientific_qualification_evidence_v1 import (
    QualificationEvidenceError,
    audit_proof_evidence_v1,
)
from cb16_local_opt.scientific_qualification_identity_v1 import (
    QualificationIdentityError,
    audit_exact_identity_bindings_v1,
)


def _profile():
    return {
        "schema": "CB16_QUALIFICATION_PROFILE_V1",
        "stage": "TEST",
        "claims": [{"claim_id": "C1", "proof_obligation_ids": ["P1"]}],
        "proof_obligations": [{
            "obligation_id": "P1",
            "producer": "producer",
            "consumer": "consumer",
            "gate_id": "G1",
            "artifact_proof": "artifact/P1",
            "counterexample_id": "X1",
            "mandatory": True,
        }],
        "edge_cases": {"ordinary": "PROCESS"},
    }


def test_null_expected_and_observed_cannot_match():
    with pytest.raises(QualificationIdentityError, match="EXPECTED_NOT_STRING"):
        audit_exact_identity_bindings_v1([{
            "identity_id": "implementation_sha",
            "expected": None,
            "observed": None,
            "expected_source_ref": "review",
            "observed_source_ref": "checkout",
            "required": True,
        }])


def test_null_identity_source_reference_is_rejected():
    with pytest.raises(QualificationIdentityError, match="EXPECTED_SOURCE_REF_NOT_STRING"):
        audit_exact_identity_bindings_v1([{
            "identity_id": "implementation_sha",
            "expected": "a" * 40,
            "observed": "a" * 40,
            "expected_source_ref": None,
            "observed_source_ref": "checkout",
            "required": True,
        }])


def test_passed_proof_cannot_use_null_evidence_reference():
    with pytest.raises(QualificationEvidenceError, match="EVIDENCE_REFERENCE_NOT_STRING"):
        audit_proof_evidence_v1(
            profile=_profile(),
            evidence=[{"obligation_id": "P1", "passed": True, "evidence_refs": [None]}],
        )


def test_evidence_reference_container_cannot_be_a_string():
    with pytest.raises(QualificationEvidenceError, match="EVIDENCE_REFS_NOT_SEQUENCE"):
        audit_proof_evidence_v1(
            profile=_profile(),
            evidence=[{"obligation_id": "P1", "passed": True, "evidence_refs": "artifact/P1"}],
        )


def test_profile_stage_null_is_rejected_instead_of_stringified():
    profile = _profile()
    profile["stage"] = None
    with pytest.raises(QualificationContractError, match="STAGE_NOT_STRING"):
        validate_profile_structure_v1(profile)


def test_profile_obligation_null_string_field_is_rejected():
    profile = _profile()
    profile["proof_obligations"][0]["producer"] = None
    with pytest.raises(QualificationContractError, match="PRODUCER_NOT_STRING"):
        validate_profile_structure_v1(profile)


def test_profile_claim_null_obligation_reference_is_rejected():
    profile = _profile()
    profile["claims"][0]["proof_obligation_ids"] = [None]
    with pytest.raises(QualificationContractError, match="PROOF_OBLIGATION_REFERENCE_NOT_STRING"):
        validate_profile_structure_v1(profile)
