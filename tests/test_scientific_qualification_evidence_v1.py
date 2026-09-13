from cb16_local_opt.scientific_qualification_evidence_v1 import audit_proof_evidence_v1


def _profile():
    return {
        "schema": "CB16_QUALIFICATION_PROFILE_V1",
        "stage": "TEST",
        "claims": [{"claim_id": "C1", "proof_obligation_ids": ["P1"]}],
        "proof_obligations": [{
            "obligation_id": "P1", "producer": "p", "consumer": "c",
            "gate_id": "G1", "artifact_proof": "artifact/P1",
            "counterexample_id": "X1", "mandatory": True,
        }],
        "edge_cases": {"ordinary": "PROCESS"},
    }


def test_mandatory_proof_passes_only_with_evidence_reference():
    result = audit_proof_evidence_v1(
        profile=_profile(),
        evidence=[{"obligation_id": "P1", "passed": True, "evidence_refs": ["artifact/P1"]}],
    )
    assert result["all_mandatory_pass"] is True
    assert result["contract_violations"] == []


def test_missing_mandatory_proof_is_contract_violation():
    result = audit_proof_evidence_v1(profile=_profile(), evidence=[])
    assert result["all_mandatory_pass"] is False
    assert result["contract_violations"] == ["MISSING_MANDATORY_PROOF:P1"]


def test_failed_mandatory_proof_is_contract_violation():
    result = audit_proof_evidence_v1(
        profile=_profile(),
        evidence=[{"obligation_id": "P1", "passed": False, "evidence_refs": ["artifact/P1"]}],
    )
    assert result["contract_violations"] == ["MANDATORY_PROOF_FAILED:P1"]
