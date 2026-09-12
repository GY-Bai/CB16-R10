from __future__ import annotations

from dataclasses import replace

import pytest

from cb16_local_opt.evidence_level_r0 import (
    EVIDENCE_SCHEMA_R0,
    EvidenceLevel,
    make_evidence_receipt_r0,
    parse_evidence_level,
    validate_serialized_evidence_claim_r0,
)


def test_evidence_levels_have_frozen_strict_order() -> None:
    assert [level.name for level in EvidenceLevel] == [
        "CONTRACT",
        "COMPONENT",
        "CLOSED_LOOP",
        "KNOWN_ANSWER",
        "ECONOMIC",
        "TRANSFER",
    ]
    assert EvidenceLevel.CONTRACT < EvidenceLevel.COMPONENT < EvidenceLevel.CLOSED_LOOP
    assert EvidenceLevel.CLOSED_LOOP < EvidenceLevel.KNOWN_ANSWER < EvidenceLevel.ECONOMIC
    assert EvidenceLevel.ECONOMIC < EvidenceLevel.TRANSFER


def test_component_receipt_can_only_claim_component_or_weaker() -> None:
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="COMPONENT",
        source_artifacts=("tests/test_component.py", "artifact://component/result.json"),
    )
    assert receipt.schema_version == EVIDENCE_SCHEMA_R0
    assert receipt.allows("CONTRACT")
    assert receipt.allows("COMPONENT")
    assert not receipt.allows("CLOSED_LOOP")
    assert not receipt.allows("ECONOMIC")
    assert not receipt.allows("TRANSFER")
    payload = receipt.serialize_claim("COMPONENT")
    assert payload["claimed_level"] == "COMPONENT"
    assert payload["maximum_justified_level"] == "COMPONENT"
    assert validate_serialized_evidence_claim_r0(payload) == receipt


@pytest.mark.parametrize("illegal_claim", ["CLOSED_LOOP", "KNOWN_ANSWER", "ECONOMIC", "TRANSFER"])
def test_component_only_fixture_cannot_serialize_stronger_claim(illegal_claim: str) -> None:
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="COMPONENT",
        source_artifacts=("artifact://component-only/result.json",),
    )
    with pytest.raises(RuntimeError, match="CLAIM_EXCEEDS_JUSTIFIED_LEVEL"):
        receipt.serialize_claim(illegal_claim)


def test_tampered_component_payload_cannot_be_upgraded_to_economic_or_transfer() -> None:
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="COMPONENT",
        source_artifacts=("artifact://component-only/result.json",),
    )
    payload = receipt.serialize_claim("COMPONENT")
    for stronger in ("ECONOMIC", "TRANSFER"):
        tampered = dict(payload)
        tampered["claimed_level"] = stronger
        with pytest.raises(RuntimeError, match="CLAIM_EXCEEDS_JUSTIFIED_LEVEL"):
            validate_serialized_evidence_claim_r0(tampered)


def test_source_artifacts_are_required_unique_and_canonical() -> None:
    with pytest.raises(RuntimeError, match="SOURCE_ARTIFACTS_MISSING"):
        make_evidence_receipt_r0(maximum_justified_level="CONTRACT", source_artifacts=())
    with pytest.raises(RuntimeError, match="SOURCE_ARTIFACT_DUPLICATE"):
        make_evidence_receipt_r0(
            maximum_justified_level="COMPONENT",
            source_artifacts=("a", "a"),
        )
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="COMPONENT",
        source_artifacts=("z", "a"),
    )
    assert receipt.source_artifacts == ("a", "z")
    with pytest.raises(RuntimeError, match="SOURCE_ARTIFACTS_NONCANONICAL"):
        replace(receipt, source_artifacts=("z", "a")).validate()


def test_unknown_level_and_payload_shape_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="LEVEL_INVALID"):
        parse_evidence_level("WORKFLOW_PASS")
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="KNOWN_ANSWER",
        source_artifacts=("artifact://known-answer/result.json",),
    )
    payload = receipt.serialize_claim("KNOWN_ANSWER")
    payload["workflow_success"] = True
    with pytest.raises(RuntimeError, match="PAYLOAD_FIELDS_INVALID"):
        validate_serialized_evidence_claim_r0(payload)


def test_claim_hash_is_deterministic_and_bound_to_claim_level() -> None:
    receipt = make_evidence_receipt_r0(
        maximum_justified_level="TRANSFER",
        source_artifacts=("artifact://transfer/result.json", "artifact://cohort/manifest.json"),
    )
    assert receipt.claim_sha256("ECONOMIC") == receipt.claim_sha256("ECONOMIC")
    assert receipt.claim_sha256("ECONOMIC") != receipt.claim_sha256("TRANSFER")
