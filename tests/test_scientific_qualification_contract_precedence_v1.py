from cb16_local_opt.scientific_qualification_contract_v1 import classify_qualification_v1


def test_contract_mismatch_precedes_science():
    verdict = classify_qualification_v1(
        contract_violations=["gap"],
        mandatory_scientific_gates={"G": True},
    )
    assert verdict == "CONTRACT_MISMATCH"


def test_mutually_exclusive_operational_states_fail_closed():
    verdict = classify_qualification_v1(
        execution_blocked=True,
        hardware_limit=True,
        mandatory_scientific_gates={"G": True},
    )
    assert verdict == "CONTRACT_MISMATCH"
