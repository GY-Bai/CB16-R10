from cb16_local_opt.scientific_qualification_identity_v1 import audit_exact_identity_bindings_v1


def test_required_identity_mismatch_is_contract_violation():
    result = audit_exact_identity_bindings_v1([{
        "identity_id": "implementation_sha",
        "expected": "a" * 40,
        "observed": "b" * 40,
        "expected_source_ref": "review",
        "observed_source_ref": "checkout",
        "required": True,
    }])
    assert result["all_required_exact_bindings_match"] is False
    assert result["contract_violations"] == ["EXACT_IDENTITY_MISMATCH:implementation_sha"]


def test_required_identity_match_passes_exactly():
    value = "a" * 64
    result = audit_exact_identity_bindings_v1([{
        "identity_id": "manifest_sha256",
        "expected": value,
        "observed": value,
        "expected_source_ref": "profile",
        "observed_source_ref": "artifact",
        "required": True,
    }])
    assert result["all_required_exact_bindings_match"] is True
