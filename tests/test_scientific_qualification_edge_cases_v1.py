import pytest

from cb16_local_opt.scientific_qualification_contract_v1 import QualificationContractError, validate_profile_structure_v1


def test_profile_rejects_unrecognized_edge_decision():
    profile = {
        "schema": "CB16_QUALIFICATION_PROFILE_V1",
        "stage": "TEST",
        "claims": [{"claim_id": "C1", "proof_obligation_ids": ["P1"]}],
        "proof_obligations": [{
            "obligation_id": "P1", "producer": "p", "consumer": "c",
            "gate_id": "G1", "artifact_proof": "a", "counterexample_id": "X1",
            "mandatory": True,
        }],
        "edge_cases": {"all_flat": "SKIP_UPDATE"},
    }
    with pytest.raises(QualificationContractError, match="INVALID_EDGE_DECISION"):
        validate_profile_structure_v1(profile)
