from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cb16_local_opt.stage4_state_roots_r11 import (
    ALL_OBJECT_CLASSES,
    CONTROL_OBJECT_CLASSES,
    DATA_OBJECT_CLASSES,
    CONTROL_ROOT_MARKER,
    DATA_ROOT_MARKER,
    SEMANTIC_FREEZE_BLOB_SHA,
    SCIENTIFIC_STATUS,
    Stage4ContentIdentityConflict,
    Stage4FrozenRootWritableError,
    Stage4IncompleteObjectError,
    Stage4PathEscapeError,
    Stage4RootOverlapError,
    Stage4RootSetR11,
    Stage4RootTypeError,
    Stage4StateRootsR11,
    Stage4SymlinkError,
    Stage4UnclaimedRootError,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "authority"
    / "rearchitecture_r11"
    / "CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1.json"
)


def _make_frozen(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen-market-raw"
    frozen.mkdir()
    sentinel = frozen / "frozen.identity"
    sentinel.write_text("synthetic-frozen-authority\n", encoding="utf-8")
    sentinel.chmod(0o444)
    frozen.chmod(0o555)
    return frozen


def _new_store(tmp_path: Path) -> tuple[Stage4StateRootsR11, Path, Path, Path]:
    frozen = _make_frozen(tmp_path)
    control = tmp_path / "ssd-control"
    data = tmp_path / "hdd-data"
    roots = Stage4RootSetR11.from_paths(
        control_root=control,
        data_root=data,
        frozen_raw_root=frozen,
        frozen_raw_identity="fixture-frozen-root-v1",
    )
    return Stage4StateRootsR11(roots), control, data, frozen


def _reopen(control: Path, data: Path, frozen: Path) -> Stage4StateRootsR11:
    return Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control,
            data_root=data,
            frozen_raw_root=frozen,
            frozen_raw_identity="fixture-frozen-root-v1",
        )
    )


def test_machine_readable_contract_covers_every_required_object_class() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["schema"] == "CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1"
    assert contract["semantic_freeze_blob_sha"] == SEMANTIC_FREEZE_BLOB_SHA
    assert contract["scientific_status"] == SCIENTIFIC_STATUS
    assert contract["s4f_grants_runtime_authority"] is False
    assert contract["s4f_grants_scientific_evidence_admission"] is False
    assert set(contract["object_classes"]) == set(ALL_OBJECT_CLASSES)
    for name in ALL_OBJECT_CLASSES:
        entry = contract["object_classes"][name]
        for required in (
            "owner",
            "read_write_mode",
            "mutability",
            "creation_authority",
            "sealing_authority",
            "recovery",
            "orphan",
            "torn_write",
            "content_address_identity",
            "startup_verification",
        ):
            assert required in entry, (name, required)


def test_clean_initialization_restart_verification_and_reconstruction(tmp_path: Path) -> None:
    store, control, data, frozen = _new_store(tmp_path)
    first = store.initialize()
    assert first.sealed_object_count == 0
    assert (control / CONTROL_ROOT_MARKER).is_file()
    assert (data / DATA_ROOT_MARKER).is_file()
    for name in CONTROL_OBJECT_CLASSES:
        assert (control / name).is_dir()
    for name in DATA_OBJECT_CLASSES:
        assert (data / name / "sha256").is_dir()

    refs = [
        store.materialize_immutable_payload("evidence_payloads", b"storage bytes only; not evidence admission"),
        store.materialize_immutable_payload("trace_replay_payloads", b"engineering replay; never new evidence"),
        store.materialize_immutable_payload("cold_content_artifacts", b"cold immutable artifact"),
    ]
    duplicate = store.materialize_immutable_payload(
        "evidence_payloads", b"storage bytes only; not evidence admission"
    )
    assert duplicate == refs[0]

    restarted = _reopen(control, data, frozen)
    receipt = restarted.verify_startup()
    assert receipt.sealed_object_count == 3
    assert {(x.object_class, x.content_sha256) for x in receipt.sealed_objects} == {
        (x.object_class, x.content_sha256) for x in refs
    }
    assert all(Path(x.payload_path).stat().st_mode & 0o222 == 0 for x in refs)


def test_wrong_root_type_is_rejected(tmp_path: Path) -> None:
    store, control, data, frozen = _new_store(tmp_path)
    store.initialize()
    wrong = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=data,
            data_root=control,
            frozen_raw_root=frozen,
            frozen_raw_identity="fixture-frozen-root-v1",
        )
    )
    with pytest.raises(Stage4RootTypeError):
        wrong.verify_startup()


def test_nonempty_legacy_writable_root_is_never_auto_adopted(tmp_path: Path) -> None:
    frozen = _make_frozen(tmp_path)
    control = tmp_path / "legacy-control"
    control.mkdir()
    (control / "legacy.sqlite").write_bytes(b"legacy")
    data = tmp_path / "new-data"
    store = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=control,
            data_root=data,
            frozen_raw_root=frozen,
            frozen_raw_identity="fixture-frozen-root-v1",
        )
    )
    with pytest.raises(Stage4UnclaimedRootError):
        store.initialize()
    assert not (control / CONTROL_ROOT_MARKER).exists()


def test_required_frozen_source_becoming_writable_fails_closed(tmp_path: Path) -> None:
    store, control, data, frozen = _new_store(tmp_path)
    store.initialize()
    frozen.chmod(0o755)
    try:
        with pytest.raises(Stage4FrozenRootWritableError):
            _reopen(control, data, frozen).verify_startup()
    finally:
        frozen.chmod(0o555)


def test_orphan_payload_is_never_silently_sealed_or_authoritative(tmp_path: Path) -> None:
    store, _control, data, _frozen = _new_store(tmp_path)
    store.initialize()
    raw = b"crash-after-payload-before-seal"
    digest = hashlib.sha256(raw).hexdigest()
    orphan = data / "evidence_payloads" / "sha256" / digest[:2] / f"{digest}.blob"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(raw)
    with pytest.raises(Stage4IncompleteObjectError, match="ORPHAN_PAYLOAD"):
        store.verify_startup()
    with pytest.raises(Stage4IncompleteObjectError, match="ORPHAN_PAYLOAD"):
        store.materialize_immutable_payload("evidence_payloads", raw)


def test_incomplete_seal_is_never_authoritative(tmp_path: Path) -> None:
    store, control, _data, _frozen = _new_store(tmp_path)
    store.initialize()
    digest = hashlib.sha256(b"missing-payload").hexdigest()
    seal = control / "object_seals" / "cold_content_artifacts" / digest[:2] / f"{digest}.json"
    seal.parent.mkdir(parents=True, exist_ok=True)
    seal.write_text(
        json.dumps(
            {
                "schema": "CB16_R11_STAGE4_STORAGE_IDENTITY_SEAL_V1",
                "object_class": "cold_content_artifacts",
                "content_sha256": digest,
                "byte_count": len(b"missing-payload"),
                "data_root_id": json.loads((store.data_root / DATA_ROOT_MARKER).read_text())["root_id"],
                "identity_basis": "SHA256_EXACT_PAYLOAD_BYTES",
                "storage_identity_only": True,
                "scientific_evidence_admission": False,
                "generation_advancement": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Stage4IncompleteObjectError, match="INCOMPLETE_SEAL"):
        store.verify_startup()


def test_conflicting_content_identity_fails_closed(tmp_path: Path) -> None:
    store, _control, _data, _frozen = _new_store(tmp_path)
    store.initialize()
    ref = store.materialize_immutable_payload("cold_content_artifacts", b"original")
    payload = Path(ref.payload_path)
    payload.chmod(0o644)
    payload.write_bytes(b"tampered")
    payload.chmod(0o444)
    with pytest.raises(Stage4ContentIdentityConflict, match="CONTENT_IDENTITY_CONFLICT"):
        store.verify_startup()


def test_path_escape_and_symlink_are_rejected(tmp_path: Path) -> None:
    store, control, data, _frozen = _new_store(tmp_path)
    store.initialize()
    with pytest.raises(Stage4PathEscapeError):
        store.control_path("../outside")
    with pytest.raises(Stage4PathEscapeError):
        store.data_path(Path("/") / "outside")

    outside = tmp_path / "outside"
    outside.mkdir()
    link = data / "cold_content_artifacts" / "sha256" / "escape-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unsupported on this test platform")
    with pytest.raises(Stage4SymlinkError):
        store.verify_startup()
    assert control.exists()


def test_control_state_inside_frozen_raw_root_is_rejected_before_mutation(tmp_path: Path) -> None:
    frozen = _make_frozen(tmp_path)
    forbidden_control = frozen / "r11-control"
    data = tmp_path / "data"
    store = Stage4StateRootsR11(
        Stage4RootSetR11.from_paths(
            control_root=forbidden_control,
            data_root=data,
            frozen_raw_root=frozen,
            frozen_raw_identity="fixture-frozen-root-v1",
        )
    )
    with pytest.raises(Stage4RootOverlapError):
        store.initialize()
    assert not forbidden_control.exists()
    assert not data.exists()


def test_torn_partial_file_fails_restart_and_is_not_repaired(tmp_path: Path) -> None:
    store, _control, data, _frozen = _new_store(tmp_path)
    store.initialize()
    partial = data / "trace_replay_payloads" / "sha256" / "aa" / "fault.stage4-partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"torn")
    with pytest.raises(Stage4IncompleteObjectError, match="TORN_OR_PARTIAL_OBJECT"):
        store.verify_startup()
    assert partial.read_bytes() == b"torn"


def test_root_marker_corruption_fails_closed(tmp_path: Path) -> None:
    store, control, _data, _frozen = _new_store(tmp_path)
    store.initialize()
    marker = control / CONTROL_ROOT_MARKER
    marker.chmod(0o644)
    marker.write_text("{}\n", encoding="utf-8")
    marker.chmod(0o444)
    with pytest.raises(Stage4RootTypeError, match="ROOT_MARKER_MISMATCH"):
        store.verify_startup()
