from cb16_local_opt.scientific_qualification_contract_v1 import classify_qualification_v1
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


def test_missing_mandatory_proof_cannot_be_hidden_by_green_science_gate():
    audit = audit_proof_evidence_v1(profile=_profile(), evidence=[])
    verdict = classify_qualification_v1(
        contract_violations=audit["contract_violations"],
        mandatory_scientific_gates={"SCIENCE": True},
    )
    assert verdict == "CONTRACT_MISMATCH"


def test_valid_contract_allows_real_scientific_fail():
    audit = audit_proof_evidence_v1(
        profile=_profile(),
        evidence=[{"obligation_id": "P1", "passed": True, "evidence_refs": ["artifact/P1"]}],
    )
    verdict = classify_qualification_v1(
        contract_violations=audit["contract_violations"],
        mandatory_scientific_gates={"SCIENCE": False},
    )
    assert verdict == "SCIENTIFIC_FAIL"
