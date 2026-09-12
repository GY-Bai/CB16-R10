from __future__ import annotations

import pytest

from cb16_local_opt.bc_round2_receipt_r0 import (
    ResultClassification,
    compile_bc_round2_receipt_r0,
    make_dependency_receipt_r0,
)


CODE_SHA = "1" * 40
SCIENCE_HASH = "2" * 64
DEP_SHA = "3" * 64


def _dep(task_id: str = "BC-004", result: str = "PASS"):
    return make_dependency_receipt_r0(
        task_id=task_id,
        result_classification=result,
        receipt_sha256=DEP_SHA,
    )


def _compile(**overrides):
    kwargs = dict(
        task_id="BC-008",
        code_sha=CODE_SHA,
        science_hash=SCIENCE_HASH,
        required_dependency_ids=("BC-004", "BC-007"),
        dependency_receipts={"BC-004": _dep("BC-004"), "BC-007": _dep("BC-007")},
        required_checks=("component-tests", "repo-guard"),
        completed_checks=("component-tests", "repo-guard"),
        maximum_justified_evidence_level="COMPONENT",
        evidence_claim_level="COMPONENT",
        source_artifacts=("artifact://bc-008/component-result.json",),
        final_holdout_untouched=True,
        fresh_data_download=False,
        result_classification="PASS",
    )
    kwargs.update(overrides)
    return compile_bc_round2_receipt_r0(**kwargs)


def test_pass_receipt_binds_all_required_round2_fields() -> None:
    receipt = _compile()
    payload = receipt.payload()
    assert payload["task_id"] == "BC-008"
    assert payload["code_sha"] == CODE_SHA
    assert payload["science_hash"] == SCIENCE_HASH
    assert [dep["task_id"] for dep in payload["dependencies"]] == ["BC-004", "BC-007"]
    assert payload["required_checks"] == ["component-tests", "repo-guard"]
    assert payload["completed_checks"] == ["component-tests", "repo-guard"]
    assert payload["evidence"]["claimed_level"] == "COMPONENT"
    assert payload["final_holdout_untouched"] is True
    assert payload["fresh_data_download"] is False
    assert payload["result_classification"] == "PASS"
    assert len(receipt.receipt_sha256()) == 64


def test_missing_dependency_prevents_receipt_compilation() -> None:
    with pytest.raises(RuntimeError, match="MISSING_DEPENDENCY"):
        _compile(dependency_receipts={"BC-004": _dep("BC-004")})


def test_unknown_dependency_prevents_receipt_compilation() -> None:
    deps = {
        "BC-004": _dep("BC-004"),
        "BC-007": _dep("BC-007"),
        "BC-099": _dep("BC-099"),
    }
    with pytest.raises(RuntimeError, match="UNKNOWN_DEPENDENCY"):
        _compile(dependency_receipts=deps)


def test_nonpass_dependency_prevents_pass_receipt() -> None:
    deps = {
        "BC-004": _dep("BC-004"),
        "BC-007": _dep("BC-007", "EXECUTION_BLOCKED"),
    }
    with pytest.raises(RuntimeError, match="PASS_WITH_NONPASS_DEPENDENCY"):
        _compile(dependency_receipts=deps)


def test_skipped_required_check_prevents_pass_receipt() -> None:
    with pytest.raises(RuntimeError, match="PASS_WITH_SKIPPED_REQUIRED_CHECK"):
        _compile(completed_checks=("component-tests",))


def test_unknown_completed_check_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_COMPLETED_CHECK"):
        _compile(completed_checks=("component-tests", "repo-guard", "not-declared"))


def test_final_holdout_or_fresh_download_prevents_pass() -> None:
    with pytest.raises(RuntimeError, match="FINAL_HOLDOUT_TOUCHED"):
        _compile(final_holdout_untouched=False)
    with pytest.raises(RuntimeError, match="FRESH_DATA_DOWNLOAD"):
        _compile(fresh_data_download=True)


def test_component_receipt_cannot_claim_economic_evidence() -> None:
    with pytest.raises(RuntimeError, match="CLAIM_EXCEEDS_JUSTIFIED_LEVEL"):
        _compile(evidence_claim_level="ECONOMIC")


def test_nonpass_outcome_can_preserve_partial_checks_without_becoming_pass() -> None:
    receipt = _compile(
        completed_checks=("component-tests",),
        result_classification="EXECUTION_BLOCKED",
    )
    assert receipt.result_classification is ResultClassification.EXECUTION_BLOCKED
    assert receipt.payload()["completed_checks"] == ["component-tests"]


@pytest.mark.parametrize(
    "classification",
    [
        "PASS",
        "SCIENTIFIC_FAIL",
        "EXECUTION_BLOCKED",
        "HARDWARE_LIMIT",
        "UNRESOLVED_OWNER_DECISION",
    ],
)
def test_all_round2_result_classifications_are_explicit(classification: str) -> None:
    receipt = _compile(result_classification=classification)
    assert receipt.payload()["result_classification"] == classification


def test_invalid_hashes_and_result_classification_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="CODE_SHA_INVALID"):
        _compile(code_sha="deadbeef")
    with pytest.raises(RuntimeError, match="SCIENCE_HASH_INVALID"):
        _compile(science_hash="bad")
    with pytest.raises(RuntimeError, match="RESULT_CLASSIFICATION_INVALID"):
        _compile(result_classification="SUCCESS")


def test_receipt_hash_changes_when_science_identity_changes() -> None:
    first = _compile(science_hash="2" * 64)
    second = _compile(science_hash="4" * 64)
    assert first.receipt_sha256() != second.receipt_sha256()
