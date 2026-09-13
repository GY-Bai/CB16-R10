from cb16_local_opt.scientific_qualification_contract_v1 import classify_qualification_v1


def test_pass():
    assert classify_qualification_v1(mandatory_scientific_gates={"G": True}) == "PASS"
