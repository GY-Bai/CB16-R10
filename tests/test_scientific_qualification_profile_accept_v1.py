from cb16_local_opt.scientific_qualification_contract_v1 import validate_profile_structure_v1


def test_complete_profile_is_accepted():
    profile = {
        "schema": "CB16_QUALIFICATION_PROFILE_V1",
        "stage": "TEST",
        "claims": [{"claim_id": "C1", "proof_obligation_ids": ["P1"]}],
        "proof_obligations": [{
            "obligation_id": "P1", "producer": "p", "consumer": "c",
            "gate_id": "G1", "artifact_proof": "a", "counterexample_id": "X1",
            "mandatory": True,
        }],
        "edge_cases": {"ordinary": "PROCESS"},
    }
    assert validate_profile_structure_v1(profile) is profile
