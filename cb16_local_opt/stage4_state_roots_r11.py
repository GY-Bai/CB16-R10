from __future__ import annotations

"""Stage-4 canonical persistent state-root ownership contract for CB16 R11.

This module is deliberately storage-semantic only.  It does not mint/admit Evidence,
advance a generation, grant Permission, acquire runtime authority, or reinterpret replay.
It defines and verifies where later integrated R11 authority holders may persist control
metadata and immutable payload bytes.

Legacy R11 stores remain engineering/reference implementations until final Stage-4
integration explicitly wires them through this contract.  Import/call reachability is not
authority.
"""

import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

STAGE4_GATEWORK_BASE_SHA = "0f18e08ec7250b9b4e45c62803c25be966834390"
SEMANTIC_FREEZE_BLOB_SHA = "3c401a0a350984381912f7860181e3e96eb8d7cf"
SCIENTIFIC_STATUS = "DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE"
ROOT_MARKER_SCHEMA = "CB16_R11_STAGE4_STATE_ROOT_MARKER_V1"
STORAGE_SEAL_SCHEMA = "CB16_R11_STAGE4_STORAGE_IDENTITY_SEAL_V1"
CONTRACT_SCHEMA = "CB16_R11_STAGE4_STATE_ROOT_CONTRACT_V1"
AUTHORITY_NAMESPACE = "CB16_R11_CANONICAL_STAGE4"

CONTROL_ROOT_MARKER = ".cb16_r11_stage4_control_root_v1.json"
DATA_ROOT_MARKER = ".cb16_r11_stage4_data_root_v1.json"
PARTIAL_SUFFIX = ".stage4-partial"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

CONTROL_OBJECT_CLASSES = (
    "authoritative_journal_metadata",
    "checkpoint_metadata",
    "runtime_lease_fencing_state",
    "hot_indexes",
    "recovery_metadata",
    "adoption_control_metadata",
)
DATA_OBJECT_CLASSES = (
    "evidence_payloads",
    "trace_replay_payloads",
    "cold_content_artifacts",
)
ALL_OBJECT_CLASSES = CONTROL_OBJECT_CLASSES + DATA_OBJECT_CLASSES + ("frozen_market_raw_authority",)


class Stage4StateRootError(RuntimeError):
    """Fail-closed base exception for canonical root verification."""


class Stage4RootOverlapError(Stage4StateRootError):
    pass


class Stage4RootTypeError(Stage4StateRootError):
    pass


class Stage4FrozenRootWritableError(Stage4StateRootError):
    pass


class Stage4PathEscapeError(Stage4StateRootError):
    pass


class Stage4SymlinkError(Stage4StateRootError):
    pass


class Stage4IncompleteObjectError(Stage4StateRootError):
    pass


class Stage4ContentIdentityConflict(Stage4StateRootError):
    pass


class Stage4UnclaimedRootError(Stage4StateRootError):
    pass


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _resolved(path: str | Path) -> Path:
    return _lexical_absolute(path).resolve(strict=False)


def _overlap(a: Path, b: Path) -> bool:
    a = a.resolve(strict=False)
    b = b.resolve(strict=False)
    return a == b or a in b.parents or b in a.parents


def _write_bits(mode: int) -> bool:
    return bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_new(path: Path, payload: bytes, *, readonly_after: bool = False) -> None:
    """Create ``path`` atomically; never replace an existing authority object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=PARTIAL_SUFFIX, dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # link() is an atomic no-replace publication on one filesystem.  It leaves the
        # temp inode in place until unlink below; a crash leaves a detectable partial.
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise
        _fsync_dir(path.parent)
        if readonly_after:
            os.chmod(path, 0o444)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=PARTIAL_SUFFIX, dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_component_name(value: str, *, field: str) -> str:
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise Stage4PathEscapeError(f"STAGE4_INVALID_{field.upper()}:{value!r}")
    return value


@dataclass(frozen=True)
class Stage4RootSetR11:
    control_root: Path
    data_root: Path
    frozen_raw_root: Path
    frozen_raw_identity: str

    @classmethod
    def from_paths(
        cls,
        *,
        control_root: str | Path,
        data_root: str | Path,
        frozen_raw_root: str | Path,
        frozen_raw_identity: str,
    ) -> "Stage4RootSetR11":
        identity = str(frozen_raw_identity).strip()
        if not identity:
            raise Stage4StateRootError("STAGE4_FROZEN_RAW_IDENTITY_REQUIRED")
        return cls(
            control_root=_lexical_absolute(control_root),
            data_root=_lexical_absolute(data_root),
            frozen_raw_root=_lexical_absolute(frozen_raw_root),
            frozen_raw_identity=identity,
        )


@dataclass(frozen=True)
class Stage4StorageObjectRefR11:
    object_class: str
    content_sha256: str
    byte_count: int
    payload_path: str
    seal_path: str


@dataclass(frozen=True)
class Stage4StartupReceiptR11:
    control_root_id: str
    data_root_id: str
    frozen_raw_identity: str
    sealed_object_count: int
    sealed_objects: tuple[Stage4StorageObjectRefR11, ...]


class Stage4StateRootsR11:
    """Initialize and verify a canonical Stage-4 root set.

    The class grants no authority.  Its write helper materializes only raw immutable
    content plus a *storage-integrity* seal.  Scientific Evidence admission and all
    lifecycle/lease/permission semantics remain outside S4F and must be wired by final
    integration.
    """

    _CONTROL_DIRS = CONTROL_OBJECT_CLASSES + ("object_seals",)
    _DATA_DIRS = DATA_OBJECT_CLASSES

    def __init__(self, roots: Stage4RootSetR11):
        self.roots = roots

    @property
    def control_root(self) -> Path:
        return self.roots.control_root

    @property
    def data_root(self) -> Path:
        return self.roots.data_root

    @property
    def frozen_raw_root(self) -> Path:
        return self.roots.frozen_raw_root

    def _root_marker_path(self, role: str) -> Path:
        if role == "CONTROL_PLANE":
            return self.control_root / CONTROL_ROOT_MARKER
        if role == "DATA_PLANE":
            return self.data_root / DATA_ROOT_MARKER
        raise Stage4RootTypeError(f"STAGE4_UNKNOWN_ROOT_ROLE:{role}")

    def _root_identity_basis(self, *, role: str, path: Path) -> dict[str, Any]:
        return {
            "schema": ROOT_MARKER_SCHEMA,
            "authority_namespace": AUTHORITY_NAMESPACE,
            "role": role,
            "canonical_path": str(path.resolve(strict=False)),
            "gatework_base_sha": STAGE4_GATEWORK_BASE_SHA,
            "semantic_freeze_blob_sha": SEMANTIC_FREEZE_BLOB_SHA,
            "frozen_raw_path": str(self.frozen_raw_root.resolve(strict=False)),
            "frozen_raw_identity": self.roots.frozen_raw_identity,
            "scientific_status": SCIENTIFIC_STATUS,
        }

    def _expected_root_marker(self, *, role: str, path: Path) -> dict[str, Any]:
        basis = self._root_identity_basis(role=role, path=path)
        return {**basis, "root_id": sha256_bytes(canonical_json_bytes(basis))}

    def _validate_root_boundaries_before_mutation(self) -> None:
        frozen = self.frozen_raw_root
        if not frozen.exists() or not frozen.is_dir():
            raise Stage4RootTypeError(f"STAGE4_FROZEN_RAW_ROOT_NOT_DIRECTORY:{frozen}")
        if frozen.is_symlink():
            raise Stage4SymlinkError(f"STAGE4_FROZEN_RAW_ROOT_SYMLINK:{frozen}")
        self._assert_frozen_read_only()

        for candidate, label in (
            (self.control_root, "CONTROL"),
            (self.data_root, "DATA"),
        ):
            if candidate.exists() and candidate.is_symlink():
                raise Stage4SymlinkError(f"STAGE4_{label}_ROOT_SYMLINK:{candidate}")
            if candidate.exists() and not candidate.is_dir():
                raise Stage4RootTypeError(f"STAGE4_{label}_ROOT_NOT_DIRECTORY:{candidate}")

        pairs = (
            (self.control_root, self.data_root, "CONTROL_DATA"),
            (self.control_root, frozen, "CONTROL_FROZEN"),
            (self.data_root, frozen, "DATA_FROZEN"),
        )
        for left, right, label in pairs:
            if _overlap(left, right):
                raise Stage4RootOverlapError(
                    f"STAGE4_ROOT_OVERLAP_{label}:{left.resolve(strict=False)}:{right.resolve(strict=False)}"
                )

    def _assert_frozen_read_only(self) -> None:
        root = self.frozen_raw_root
        mode = root.lstat().st_mode
        if _write_bits(mode):
            raise Stage4FrozenRootWritableError(f"STAGE4_FROZEN_RAW_ROOT_WRITABLE:{root}")

    def _claim_or_verify_root(self, *, role: str, root: Path, marker_name: str) -> str:
        expected = self._expected_root_marker(role=role, path=root)
        marker = root / marker_name
        entries = list(root.iterdir()) if root.exists() else []
        if marker.exists():
            if marker.is_symlink() or not marker.is_file():
                raise Stage4RootTypeError(f"STAGE4_ROOT_MARKER_INVALID:{marker}")
            try:
                actual = json.loads(marker.read_text(encoding="utf-8"))
            except Exception as exc:
                raise Stage4RootTypeError(f"STAGE4_ROOT_MARKER_CORRUPT:{marker}") from exc
            if actual != expected:
                raise Stage4RootTypeError(
                    f"STAGE4_ROOT_MARKER_MISMATCH:{role}:{actual.get('role')}:{actual.get('root_id')}"
                )
            return str(expected["root_id"])

        if entries:
            raise Stage4UnclaimedRootError(f"STAGE4_NONEMPTY_UNCLAIMED_ROOT:{role}:{root}")
        _atomic_write_new(marker, canonical_json_bytes(expected) + b"\n", readonly_after=True)
        return str(expected["root_id"])

    def initialize(self) -> Stage4StartupReceiptR11:
        """Claim empty control/data roots and create only Stage-4 namespaced layout."""
        self._validate_root_boundaries_before_mutation()
        self.control_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

        control_id = self._claim_or_verify_root(
            role="CONTROL_PLANE", root=self.control_root, marker_name=CONTROL_ROOT_MARKER
        )
        data_id = self._claim_or_verify_root(
            role="DATA_PLANE", root=self.data_root, marker_name=DATA_ROOT_MARKER
        )

        for name in self._CONTROL_DIRS:
            path = self.control_root / name
            if path.exists() and (path.is_symlink() or not path.is_dir()):
                raise Stage4RootTypeError(f"STAGE4_CONTROL_CLASS_PATH_INVALID:{name}")
            path.mkdir(exist_ok=True)
        for name in self._DATA_DIRS:
            class_root = self.data_root / name
            if class_root.exists() and (class_root.is_symlink() or not class_root.is_dir()):
                raise Stage4RootTypeError(f"STAGE4_DATA_CLASS_PATH_INVALID:{name}")
            (class_root / "sha256").mkdir(parents=True, exist_ok=True)

        return self.verify_startup(expected_control_id=control_id, expected_data_id=data_id)

    def _read_marker(self, *, role: str, root: Path, marker_name: str) -> dict[str, Any]:
        marker = root / marker_name
        if not marker.is_file() or marker.is_symlink():
            raise Stage4RootTypeError(f"STAGE4_ROOT_MARKER_MISSING_OR_INVALID:{marker}")
        try:
            actual = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise Stage4RootTypeError(f"STAGE4_ROOT_MARKER_CORRUPT:{marker}") from exc
        expected = self._expected_root_marker(role=role, path=root)
        if actual != expected:
            raise Stage4RootTypeError(f"STAGE4_ROOT_MARKER_MISMATCH:{role}")
        return actual

    @staticmethod
    def _assert_no_symlink_tree(root: Path) -> None:
        if root.is_symlink():
            raise Stage4SymlinkError(f"STAGE4_ROOT_SYMLINK:{root}")
        for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
            current_path = Path(current)
            for name in list(dirs) + list(files):
                path = current_path / name
                if path.is_symlink():
                    raise Stage4SymlinkError(f"STAGE4_SYMLINK_FORBIDDEN:{path}")

    @staticmethod
    def _assert_no_partial_files(root: Path) -> None:
        for current, _dirs, files in os.walk(root, topdown=True, followlinks=False):
            for name in files:
                if name.endswith(PARTIAL_SUFFIX):
                    raise Stage4IncompleteObjectError(
                        f"STAGE4_TORN_OR_PARTIAL_OBJECT:{Path(current) / name}"
                    )

    def _safe_under(self, root: Path, relative: str | Path) -> Path:
        rel = Path(relative)
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise Stage4PathEscapeError(f"STAGE4_PATH_ESCAPE:{relative}")
        candidate = root.joinpath(*rel.parts)
        resolved_root = root.resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
            raise Stage4PathEscapeError(f"STAGE4_PATH_ESCAPE:{relative}")
        # Existing symlink components inside the canonical root are never traversed.
        cursor = root
        for part in rel.parts:
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                raise Stage4SymlinkError(f"STAGE4_SYMLINK_FORBIDDEN:{cursor}")
        return candidate

    def control_path(self, relative: str | Path) -> Path:
        return self._safe_under(self.control_root, relative)

    def data_path(self, relative: str | Path) -> Path:
        return self._safe_under(self.data_root, relative)

    def _payload_path(self, object_class: str, digest: str) -> Path:
        object_class = _validate_component_name(object_class, field="object_class")
        if object_class not in DATA_OBJECT_CLASSES:
            raise Stage4StateRootError(f"STAGE4_NOT_DATA_OBJECT_CLASS:{object_class}")
        if not _HEX64.fullmatch(digest):
            raise Stage4ContentIdentityConflict(f"STAGE4_INVALID_SHA256:{digest}")
        return self.data_path(Path(object_class) / "sha256" / digest[:2] / f"{digest}.blob")

    def _seal_path(self, object_class: str, digest: str) -> Path:
        return self.control_path(Path("object_seals") / object_class / digest[:2] / f"{digest}.json")

    def _storage_seal(self, *, object_class: str, digest: str, byte_count: int) -> dict[str, Any]:
        data_marker = self._expected_root_marker(role="DATA_PLANE", path=self.data_root)
        return {
            "schema": STORAGE_SEAL_SCHEMA,
            "object_class": object_class,
            "content_sha256": digest,
            "byte_count": int(byte_count),
            "data_root_id": data_marker["root_id"],
            "identity_basis": "SHA256_EXACT_PAYLOAD_BYTES",
            "storage_identity_only": True,
            "scientific_evidence_admission": False,
            "generation_advancement": False,
        }

    def materialize_immutable_payload(
        self, object_class: str, payload: bytes | bytearray | memoryview
    ) -> Stage4StorageObjectRefR11:
        """Materialize a content-addressed payload and storage-integrity seal.

        This operation does not mint or admit scientific Evidence.  If an unsealed payload
        already exists, the method fails rather than upgrading that orphan to authority.
        """
        self.verify_startup()
        raw = bytes(payload)
        digest = sha256_bytes(raw)
        payload_path = self._payload_path(object_class, digest)
        seal_path = self._seal_path(object_class, digest)
        expected_seal = self._storage_seal(
            object_class=object_class, digest=digest, byte_count=len(raw)
        )

        payload_exists = payload_path.exists()
        seal_exists = seal_path.exists()
        if payload_exists and not seal_exists:
            raise Stage4IncompleteObjectError(f"STAGE4_ORPHAN_PAYLOAD_PRESENT:{payload_path}")
        if seal_exists and not payload_exists:
            raise Stage4IncompleteObjectError(f"STAGE4_INCOMPLETE_SEAL_PRESENT:{seal_path}")
        if payload_exists and seal_exists:
            self._verify_object_pair(payload_path=payload_path, seal_path=seal_path, expected=expected_seal)
            return Stage4StorageObjectRefR11(
                object_class, digest, len(raw), str(payload_path), str(seal_path)
            )

        payload_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_write_new(payload_path, raw, readonly_after=True)
        except FileExistsError:
            raise Stage4ContentIdentityConflict(f"STAGE4_PAYLOAD_PUBLICATION_RACE:{digest}")

        # Deliberate two-phase durability boundary: a crash here leaves an orphan payload.
        # Restart fails closed and never synthesizes the missing seal.
        seal_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_write_new(
                seal_path, canonical_json_bytes(expected_seal) + b"\n", readonly_after=True
            )
        except FileExistsError:
            raise Stage4ContentIdentityConflict(f"STAGE4_SEAL_PUBLICATION_RACE:{digest}")

        self._verify_object_pair(payload_path=payload_path, seal_path=seal_path, expected=expected_seal)
        return Stage4StorageObjectRefR11(
            object_class, digest, len(raw), str(payload_path), str(seal_path)
        )

    def _verify_object_pair(
        self, *, payload_path: Path, seal_path: Path, expected: Mapping[str, Any] | None = None
    ) -> Stage4StorageObjectRefR11:
        if payload_path.is_symlink() or seal_path.is_symlink():
            raise Stage4SymlinkError(f"STAGE4_OBJECT_SYMLINK_FORBIDDEN:{payload_path}:{seal_path}")
        if not payload_path.is_file():
            raise Stage4IncompleteObjectError(f"STAGE4_PAYLOAD_MISSING:{payload_path}")
        if not seal_path.is_file():
            raise Stage4IncompleteObjectError(f"STAGE4_SEAL_MISSING:{seal_path}")
        try:
            seal = json.loads(seal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise Stage4IncompleteObjectError(f"STAGE4_SEAL_CORRUPT:{seal_path}") from exc

        object_class = str(seal.get("object_class", ""))
        digest = str(seal.get("content_sha256", ""))
        byte_count = int(seal.get("byte_count", -1))
        if object_class not in DATA_OBJECT_CLASSES or not _HEX64.fullmatch(digest):
            raise Stage4ContentIdentityConflict(f"STAGE4_SEAL_IDENTITY_INVALID:{seal_path}")
        canonical_payload = self._payload_path(object_class, digest)
        canonical_seal = self._seal_path(object_class, digest)
        if canonical_payload.resolve(strict=False) != payload_path.resolve(strict=False):
            raise Stage4ContentIdentityConflict(f"STAGE4_PAYLOAD_LOCATION_CONFLICT:{payload_path}")
        if canonical_seal.resolve(strict=False) != seal_path.resolve(strict=False):
            raise Stage4ContentIdentityConflict(f"STAGE4_SEAL_LOCATION_CONFLICT:{seal_path}")
        actual_digest = sha256_file(payload_path)
        actual_size = payload_path.stat().st_size
        if actual_digest != digest or actual_size != byte_count:
            raise Stage4ContentIdentityConflict(f"STAGE4_CONTENT_IDENTITY_CONFLICT:{digest}")
        required = self._storage_seal(
            object_class=object_class, digest=digest, byte_count=byte_count
        )
        if seal != required or (expected is not None and dict(expected) != required):
            raise Stage4ContentIdentityConflict(f"STAGE4_STORAGE_SEAL_CONFLICT:{digest}")
        return Stage4StorageObjectRefR11(
            object_class, digest, byte_count, str(payload_path), str(seal_path)
        )

    def _iter_payload_files(self) -> Iterable[tuple[str, Path]]:
        for object_class in DATA_OBJECT_CLASSES:
            root = self.data_root / object_class / "sha256"
            if not root.is_dir() or root.is_symlink():
                raise Stage4RootTypeError(f"STAGE4_DATA_CLASS_LAYOUT_INVALID:{object_class}")
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    continue
                if path.is_symlink():
                    raise Stage4SymlinkError(f"STAGE4_SYMLINK_FORBIDDEN:{path}")
                if path.name.endswith(PARTIAL_SUFFIX):
                    raise Stage4IncompleteObjectError(f"STAGE4_TORN_OR_PARTIAL_OBJECT:{path}")
                if path.suffix != ".blob" or not _HEX64.fullmatch(path.stem):
                    raise Stage4ContentIdentityConflict(f"STAGE4_UNEXPECTED_DATA_OBJECT:{path}")
                digest = path.stem
                expected = self._payload_path(object_class, digest)
                if expected.resolve(strict=False) != path.resolve(strict=False):
                    raise Stage4ContentIdentityConflict(f"STAGE4_NONCANONICAL_DATA_LOCATION:{path}")
                yield object_class, path

    def _iter_seal_files(self) -> Iterable[Path]:
        root = self.control_root / "object_seals"
        if not root.is_dir() or root.is_symlink():
            raise Stage4RootTypeError("STAGE4_OBJECT_SEAL_ROOT_INVALID")
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            if path.is_symlink():
                raise Stage4SymlinkError(f"STAGE4_SYMLINK_FORBIDDEN:{path}")
            if path.name.endswith(PARTIAL_SUFFIX):
                raise Stage4IncompleteObjectError(f"STAGE4_TORN_OR_PARTIAL_OBJECT:{path}")
            if path.suffix != ".json" or not _HEX64.fullmatch(path.stem):
                raise Stage4ContentIdentityConflict(f"STAGE4_UNEXPECTED_STORAGE_SEAL:{path}")
            yield path

    def reconstruct_sealed_payloads(self) -> tuple[Stage4StorageObjectRefR11, ...]:
        payload_pairs: dict[tuple[str, str], Path] = {}
        for object_class, payload_path in self._iter_payload_files():
            digest = payload_path.stem
            key = (object_class, digest)
            payload_pairs[key] = payload_path
            seal_path = self._seal_path(object_class, digest)
            if not seal_path.is_file():
                raise Stage4IncompleteObjectError(f"STAGE4_ORPHAN_PAYLOAD:{payload_path}")

        refs: dict[tuple[str, str], Stage4StorageObjectRefR11] = {}
        for seal_path in self._iter_seal_files():
            try:
                seal = json.loads(seal_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise Stage4IncompleteObjectError(f"STAGE4_SEAL_CORRUPT:{seal_path}") from exc
            object_class = str(seal.get("object_class", ""))
            digest = str(seal.get("content_sha256", ""))
            if object_class not in DATA_OBJECT_CLASSES or not _HEX64.fullmatch(digest):
                raise Stage4ContentIdentityConflict(f"STAGE4_SEAL_IDENTITY_INVALID:{seal_path}")
            expected_seal_path = self._seal_path(object_class, digest)
            if expected_seal_path.resolve(strict=False) != seal_path.resolve(strict=False):
                raise Stage4ContentIdentityConflict(f"STAGE4_NONCANONICAL_SEAL_LOCATION:{seal_path}")
            payload_path = payload_pairs.get((object_class, digest))
            if payload_path is None:
                raise Stage4IncompleteObjectError(f"STAGE4_INCOMPLETE_SEAL:{seal_path}")
            refs[(object_class, digest)] = self._verify_object_pair(
                payload_path=payload_path, seal_path=seal_path
            )

        if set(payload_pairs) != set(refs):
            raise Stage4IncompleteObjectError("STAGE4_PAYLOAD_SEAL_SET_MISMATCH")
        return tuple(refs[key] for key in sorted(refs))

    def verify_startup(
        self,
        *,
        expected_control_id: str | None = None,
        expected_data_id: str | None = None,
    ) -> Stage4StartupReceiptR11:
        self._validate_root_boundaries_before_mutation()
        if not self.control_root.is_dir() or not self.data_root.is_dir():
            raise Stage4RootTypeError("STAGE4_CANONICAL_ROOT_MISSING")
        control_marker = self._read_marker(
            role="CONTROL_PLANE", root=self.control_root, marker_name=CONTROL_ROOT_MARKER
        )
        data_marker = self._read_marker(
            role="DATA_PLANE", root=self.data_root, marker_name=DATA_ROOT_MARKER
        )
        if expected_control_id is not None and control_marker["root_id"] != expected_control_id:
            raise Stage4RootTypeError("STAGE4_CONTROL_ROOT_ID_DRIFT")
        if expected_data_id is not None and data_marker["root_id"] != expected_data_id:
            raise Stage4RootTypeError("STAGE4_DATA_ROOT_ID_DRIFT")

        for name in self._CONTROL_DIRS:
            path = self.control_root / name
            if not path.is_dir() or path.is_symlink():
                raise Stage4RootTypeError(f"STAGE4_CONTROL_CLASS_PATH_INVALID:{name}")
        for name in self._DATA_DIRS:
            path = self.data_root / name / "sha256"
            if not path.is_dir() or path.is_symlink():
                raise Stage4RootTypeError(f"STAGE4_DATA_CLASS_PATH_INVALID:{name}")

        self._assert_no_symlink_tree(self.control_root)
        self._assert_no_symlink_tree(self.data_root)
        self._assert_no_partial_files(self.control_root)
        self._assert_no_partial_files(self.data_root)
        sealed = self.reconstruct_sealed_payloads()
        return Stage4StartupReceiptR11(
            control_root_id=str(control_marker["root_id"]),
            data_root_id=str(data_marker["root_id"]),
            frozen_raw_identity=self.roots.frozen_raw_identity,
            sealed_object_count=len(sealed),
            sealed_objects=sealed,
        )


def expected_object_classes() -> tuple[str, ...]:
    return ALL_OBJECT_CLASSES
