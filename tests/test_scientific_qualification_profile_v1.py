import pytest

from cb16_local_opt.scientific_qualification_contract_v1 import QualificationContractError, validate_profile_structure_v1


def test_profile_requires_known_obligation():
    profile = {"schema":"V1","stage":"T","claims":[{"claim_id":"C","proof_obligation_ids":["X"]}],"proof_obligations":[],"edge_cases":{}}
    with pytest.raises(QualificationContractError):
        validate_profile_structure_v1(profile)
