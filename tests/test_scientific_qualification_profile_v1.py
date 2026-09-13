import pytest

from cb16_local_opt.scientific_qualification_contract_v1 import QualificationContractError, validate_profile_structure_v1


def test_profile_requires_known_obligation():
    profile = {
        "schema": "CB16_QUALIFICATION_PROFILE_V1",
        "stage": "TEST",
        "claims": [{"claim_id": "C1", "proof_obligation_ids": ["MISSING"]}],
        "proof_obligations": [{
            "obligation_id": "P1", "producer": "p", "consumer": "c",
            "gate_id": "G1", "artifact_proof": "a", "counterexample_id": "X1",
            "mandatory": True,
        }],
        "edge_cases": {"ordinary": "PROCESS"},
    }
    with pytest.raises(QualificationContractError, match="UNKNOWN_PROOF_OBLIGATION"):
        validate_profile_structure_v1(profile)
