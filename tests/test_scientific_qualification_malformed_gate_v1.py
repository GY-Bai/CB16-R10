from cb16_local_opt.scientific_qualification_contract_v1 import classify_qualification_v1


def test_non_boolean_gate_is_contract_mismatch():
    assert classify_qualification_v1(mandatory_scientific_gates={"G": 1}) == "CONTRACT_MISMATCH"
