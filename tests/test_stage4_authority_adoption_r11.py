from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading

import pytest

from cb16_local_opt.stage4_authority_adoption_r11 import (
    ADOPTION_RECEIPT_SCHEMA,
    FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC,
    FAIL_AFTER_TEMP_FSYNC_BEFORE_PUBLISH,
    AdoptionReceiptCorrupt,
    AuthorityAdoptionConflict,
    AuthorityAdoptionContract,
    SourceAuthorityIdentity,
    SourceAuthorityMismatch,
    adopt_authority,
    build_adoption_receipt,
    canonical_json_bytes,
    load_adoption_receipt,
    verify_adoption_receipt,
)

FREEZE_BLOB = "3c401a0a350984381912f7860181e3e96eb8d7cf"


def _h(ch: str) -> str:
    return ch * 64


def source() -> SourceAuthorityIdentity:
    return SourceAuthorityIdentity(
        source_repo="GY-Bai/CB16-R10",
        source_sha="0" * 40,
        semantic_freeze_identity=FREEZE_BLOB,
        source_generation=17,
        champion_identity="champion:G00000017",
        champion_hash=_h("a"),
        checkpoint_identity="checkpoint:G00000017",
        checkpoint_hash=_h("b"),
        evidence_root_identity="evidence-root:accepted-17",
        journal_head_identity="journal-head:accepted-17",
        checkpoint_root_identity="checkpoint-root:accepted-17",
    )


def contract() -> AuthorityAdoptionContract:
    return AuthorityAdoptionContract(source(), "cb16-r11-canonical-authority:v1")


def test_first_valid_adoption_succeeds_without_generation_advance_or_evidence(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    result = adopt_authority(
        source(), contract=contract(), receipt_path=path,
        adoption_timestamp_utc="2026-09-08T04:15:00Z",
    )
    assert result.status == "ADOPTED"
    assert result.adoption_generation == source().source_generation
    receipt = load_adoption_receipt(path, contract=contract())
    assert receipt["schema"] == ADOPTION_RECEIPT_SCHEMA
    assert receipt["target"]["adoption_generation"] == receipt["source"]["source_generation"]
    assert receipt["semantic_guards"] == {
        "scientific_history_rewritten": False,
        "new_evidence_created": False,
        "new_scientific_verdict": False,
        "generation_advanced_by_adoption": False,
    }


def test_identical_second_adoption_is_idempotent_and_does_not_rewrite_bytes(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    first = adopt_authority(
        source(), contract=contract(), receipt_path=path,
        adoption_timestamp_utc="2026-09-08T04:15:00Z",
    )
    before = path.read_bytes()
    second = adopt_authority(
        source(), contract=contract(), receipt_path=path,
        adoption_timestamp_utc="2030-01-01T00:00:00Z",
    )
    assert second.status == "ALREADY_ADOPTED"
    assert second.canonical_content_hash == first.canonical_content_hash
    assert path.read_bytes() == before


def test_timestamp_is_metadata_not_identity_authority() -> None:
    a = build_adoption_receipt(
        source(), target_r11_authority_identity=contract().target_r11_authority_identity,
        adoption_timestamp_utc="2026-09-08T04:15:00Z",
    )
    b = build_adoption_receipt(
        source(), target_r11_authority_identity=contract().target_r11_authority_identity,
        adoption_timestamp_utc="2035-02-03T04:05:06+00:00",
    )
    assert a["metadata"] != b["metadata"]
    assert a["canonical_content_hash"] == b["canonical_content_hash"]
    assert a["identity_rules"]["adoption_timestamp_is_identity_authority"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("champion_identity", "champion:wrong"),
        ("champion_hash", _h("c")),
        ("checkpoint_identity", "checkpoint:wrong"),
        ("checkpoint_hash", _h("d")),
        ("journal_head_identity", "journal-head:wrong"),
        ("semantic_freeze_identity", "1" * 40),
        ("evidence_root_identity", "evidence-root:wrong"),
        ("checkpoint_root_identity", "checkpoint-root:wrong"),
    ],
)
def test_wrong_source_authority_identity_fails_closed_before_write(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "stage4-adoption.json"
    observed = replace(source(), **{field: value})
    with pytest.raises(SourceAuthorityMismatch, match="OBSERVED_SOURCE_MISMATCH"):
        adopt_authority(observed, contract=contract(), receipt_path=path)
    assert not path.exists()


def test_conflicting_second_adoption_fails_closed_and_preserves_first(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    first_contract = contract()
    adopt_authority(source(), contract=first_contract, receipt_path=path)
    before = path.read_bytes()
    conflicting_source = replace(source(), source_generation=18)
    conflicting_contract = AuthorityAdoptionContract(
        conflicting_source, first_contract.target_r11_authority_identity
    )
    with pytest.raises((SourceAuthorityMismatch, AuthorityAdoptionConflict)):
        adopt_authority(conflicting_source, contract=conflicting_contract, receipt_path=path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema":',
        b'{}\n',
        b'{"schema":"CB16_R11_STAGE4_AUTHORITY_ADOPTION_RECEIPT_V1","schema":"duplicate"}\n',
    ],
)
def test_partial_or_corrupt_existing_receipt_fails_closed_without_overwrite(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "stage4-adoption.json"
    path.write_bytes(payload)
    with pytest.raises(AdoptionReceiptCorrupt):
        adopt_authority(source(), contract=contract(), receipt_path=path)
    assert path.read_bytes() == payload


def test_noncanonical_serialization_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    receipt = build_adoption_receipt(
        source(), target_r11_authority_identity=contract().target_r11_authority_identity
    )
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    with pytest.raises(AdoptionReceiptCorrupt, match="NOT_CANONICALLY_SERIALIZED"):
        load_adoption_receipt(path)


def test_tampering_generation_or_semantic_guards_invalidates_receipt() -> None:
    receipt = build_adoption_receipt(
        source(), target_r11_authority_identity=contract().target_r11_authority_identity
    )
    changed_generation = json.loads(json.dumps(receipt))
    changed_generation["target"]["adoption_generation"] += 1
    with pytest.raises(AdoptionReceiptCorrupt, match="GENERATION_CHANGED"):
        verify_adoption_receipt(changed_generation)
    changed_guard = json.loads(json.dumps(receipt))
    changed_guard["semantic_guards"]["new_evidence_created"] = True
    with pytest.raises(AdoptionReceiptCorrupt, match="SEMANTIC_GUARD_VIOLATION"):
        verify_adoption_receipt(changed_guard)


def test_source_root_identities_are_opaque_and_unchanged(tmp_path: Path) -> None:
    before = source()
    path = tmp_path / "stage4-adoption.json"
    adopt_authority(before, contract=contract(), receipt_path=path)
    after = source()
    assert before == after
    receipt = load_adoption_receipt(path)
    assert receipt["source"]["evidence_root_identity"] == before.evidence_root_identity
    assert receipt["source"]["journal_head_identity"] == before.journal_head_identity
    assert receipt["source"]["checkpoint_root_identity"] == before.checkpoint_root_identity


def test_crash_before_publish_leaves_no_authoritative_receipt_and_retry_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"

    def fail(stage: str) -> None:
        if stage == FAIL_AFTER_TEMP_FSYNC_BEFORE_PUBLISH:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        adopt_authority(source(), contract=contract(), receipt_path=path, _fault_hook=fail)
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert adopt_authority(source(), contract=contract(), receipt_path=path).status == "ADOPTED"


def test_crash_after_publish_leaves_one_valid_history_and_retry_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"

    def fail(stage: str) -> None:
        if stage == FAIL_AFTER_PUBLISH_BEFORE_DIRECTORY_FSYNC:
            raise RuntimeError("synthetic crash")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        adopt_authority(source(), contract=contract(), receipt_path=path, _fault_hook=fail)
    persisted = load_adoption_receipt(path, contract=contract())
    expected_hash = persisted["canonical_content_hash"]
    retry = adopt_authority(source(), contract=contract(), receipt_path=path)
    assert retry.status == "ALREADY_ADOPTED"
    assert retry.canonical_content_hash == expected_hash


def test_concurrent_conflicting_publish_can_create_only_one_legitimate_history(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    base = source()
    other = replace(
        base,
        source_sha="1" * 40,
        champion_identity="champion:other",
        champion_hash=_h("c"),
        checkpoint_identity="checkpoint:other",
        checkpoint_hash=_h("d"),
        evidence_root_identity="evidence-root:other",
        journal_head_identity="journal-head:other",
        checkpoint_root_identity="checkpoint-root:other",
    )
    contracts = [
        AuthorityAdoptionContract(base, "cb16-r11-canonical-authority:v1"),
        AuthorityAdoptionContract(other, "cb16-r11-canonical-authority:v1"),
    ]
    sources = [base, other]
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            result = adopt_authority(sources[index], contract=contracts[index], receipt_path=path)
            value = result.status
        except (AuthorityAdoptionConflict, SourceAuthorityMismatch):
            value = "CONFLICT"
        with lock:
            outcomes.append(value)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outcomes) == 2
    assert outcomes.count("ADOPTED") == 1
    assert outcomes.count("CONFLICT") == 1
    persisted = load_adoption_receipt(path)
    assert persisted["canonical_content_hash"] in {
        build_adoption_receipt(sources[0], target_r11_authority_identity=contracts[0].target_r11_authority_identity)["canonical_content_hash"],
        build_adoption_receipt(sources[1], target_r11_authority_identity=contracts[1].target_r11_authority_identity)["canonical_content_hash"],
    }


def test_persisted_receipt_is_byte_canonical(tmp_path: Path) -> None:
    path = tmp_path / "stage4-adoption.json"
    adopt_authority(source(), contract=contract(), receipt_path=path)
    obj = load_adoption_receipt(path)
    assert path.read_bytes() == canonical_json_bytes(obj) + b"\n"
