import pytest

from cb16_local_opt.scientific_qualification_contract_v1 import QualificationContractError, validate_profile_structure_v1


def test_profile_rejects_unrecognized_edge_decision():
    profile = {"schema":"V1","stage":"T","claims":[],"proof_obligations":[],"edge_cases":{"all_flat":"SKIP_UPDATE"}}
    with pytest.raises(QualificationContractError):
        validate_profile_structure_v1(profile)
