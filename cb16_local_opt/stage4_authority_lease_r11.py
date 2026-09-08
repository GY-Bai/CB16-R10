from __future__ import annotations

"""Single-machine canonical authority lease and fencing primitive for CB16 R11 Stage-4.

This module owns no scientific semantics and grants no domain-specific mutation by itself.
It provides a process-lifetime local kernel lock plus a persistent monotonically increasing
fencing epoch. Production writers are expected to be wired to ``assert_fencing_token``
in the Stage-4 integration wave.
"""

import fcntl
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "CB16_R11_STAGE4_AUTHORITY_LEASE_STATE_V1"
ROOT_SCHEMA = "CB16_R11_STAGE4_AUTHORITY_LEASE_ROOT_V1"
LOCK_SENTINEL = b"CB16_R11_STAGE4_AUTHORITY_LEASE_LOCK_V1\n"
ROOT_FILE = "stage4_authority_lease_root.json"
STATE_FILE = "stage4_authority_lease_state.json"
LOCK_FILE = "stage4_authority_lease.lock"


class AuthorityLeaseError(RuntimeError):
    pass


class AuthorityBusyError(AuthorityLeaseError):
    pass


class AuthorityLeaseNotInitializedError(AuthorityLeaseError):
    pass


class CorruptAuthorityLeaseError(AuthorityLeaseError):
    pass


class AuthorityRecoveryRequiredError(AuthorityLeaseError):
    pass


class AlreadyAcquiredError(AuthorityLeaseError):
    pass


class StaleFencingTokenError(AuthorityLeaseError):
    pass


class AuthorityOwnershipError(AuthorityLeaseError):
    pass


@dataclass(frozen=True, order=True)
class FencingTokenR11:
    epoch: int
    owner_nonce: str

    def __post_init__(self) -> None:
        if int(self.epoch) < 1:
            raise ValueError("STAGE4_FENCE_EPOCH_MUST_BE_POSITIVE")
        if not isinstance(self.owner_nonce, str) or len(self.owner_nonce) != 32:
            raise ValueError("STAGE4_FENCE_NONCE_INVALID")


@dataclass(frozen=True)
class AuthorityLeaseSnapshotR11:
    state: str
    epoch: int
    owner_id: str | None
    owner_pid: int | None
    owner_nonce: str | None

    @property
    def fencing_token(self) -> FencingTokenR11 | None:
        if self.state != "ACTIVE" or self.owner_nonce is None:
            return None
        return FencingTokenR11(self.epoch, self.owner_nonce)


# A forked child must not keep the parent's flock alive. This registry is process-local
# safety state only; persistent authority is the state file plus kernel lock.
_live_fds: set[int] = set()


def _after_fork_child() -> None:
    for fd in tuple(_live_fds):
        try:
            os.close(fd)
        except OSError:
            pass
    _live_fds.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_child)


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_atomic_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.parent / f".stage4_authority_lease_state.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(tmp, flags, 0o600)
    try:
        payload = _canonical_bytes(obj)
        view = memoryview(payload)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _validate_root_path(root: str | Path, *, create: bool) -> Path:
    path = Path(os.path.abspath(os.fspath(root)))
    if path.exists() or path.is_symlink():
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            raise CorruptAuthorityLeaseError(f"STAGE4_LEASE_ROOT_NOT_REAL_DIRECTORY:{path}")
    elif create:
        path.mkdir(parents=True, mode=0o700)
    else:
        raise AuthorityLeaseNotInitializedError(f"STAGE4_LEASE_ROOT_MISSING:{path}")
    return path


def _require_regular_file(path: Path) -> None:
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise AuthorityLeaseNotInitializedError(f"STAGE4_LEASE_METADATA_MISSING:{path.name}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise CorruptAuthorityLeaseError(f"STAGE4_LEASE_METADATA_NOT_REGULAR:{path.name}")


def _read_json_file(path: Path) -> dict[str, Any]:
    _require_regular_file(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CorruptAuthorityLeaseError(f"STAGE4_LEASE_JSON_CORRUPT:{path.name}") from exc
    if not isinstance(raw, dict):
        raise CorruptAuthorityLeaseError(f"STAGE4_LEASE_JSON_NOT_OBJECT:{path.name}")
    return raw


def _validate_state(raw: dict[str, Any]) -> AuthorityLeaseSnapshotR11:
    expected = {"schema", "state", "epoch", "owner_id", "owner_pid", "owner_nonce"}
    if set(raw) != expected or raw.get("schema") != SCHEMA:
        raise CorruptAuthorityLeaseError("STAGE4_LEASE_STATE_SCHEMA_INVALID")
    state = raw.get("state")
    if state not in {"RELEASED", "ACTIVE"}:
        raise CorruptAuthorityLeaseError("STAGE4_LEASE_STATE_VALUE_INVALID")
    epoch = raw.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise CorruptAuthorityLeaseError("STAGE4_LEASE_EPOCH_INVALID")
    owner_id, owner_pid, nonce = raw.get("owner_id"), raw.get("owner_pid"), raw.get("owner_nonce")
    if state == "RELEASED":
        if any(x is not None for x in (owner_id, owner_pid, nonce)):
            raise CorruptAuthorityLeaseError("STAGE4_RELEASED_STATE_HAS_OWNER")
    else:
        if epoch < 1 or not isinstance(owner_id, str) or not owner_id:
            raise CorruptAuthorityLeaseError("STAGE4_ACTIVE_OWNER_INVALID")
        if isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0:
            raise CorruptAuthorityLeaseError("STAGE4_ACTIVE_PID_INVALID")
        if not isinstance(nonce, str) or len(nonce) != 32:
            raise CorruptAuthorityLeaseError("STAGE4_ACTIVE_NONCE_INVALID")
    return AuthorityLeaseSnapshotR11(state, epoch, owner_id, owner_pid, nonce)


class Stage4AuthorityLeaseR11:
    """Local single-owner lease with persistent fencing.

    The kernel lock proves live exclusivity on one machine. The persisted epoch/nonce
    distinguishes successive owners. PID and wall clock are never authority inputs.
    """

    def __init__(self, root: str | Path, *, owner_id: str):
        if not isinstance(owner_id, str) or not owner_id.strip() or len(owner_id) > 256:
            raise ValueError("STAGE4_OWNER_ID_INVALID")
        self.root = _validate_root_path(root, create=False)
        self.owner_id = owner_id.strip()
        self._fd: int | None = None
        self._token: FencingTokenR11 | None = None
        self._owner_process_pid: int | None = None

    @classmethod
    def initialize(cls, root: str | Path) -> Path:
        root_path = _validate_root_path(root, create=True)
        root_file = root_path / ROOT_FILE
        state_file = root_path / STATE_FILE
        lock_file = root_path / LOCK_FILE

        if root_file.exists() or root_file.is_symlink():
            existing = _read_json_file(root_file)
            if existing != {"schema": ROOT_SCHEMA, "single_machine_only": True}:
                raise CorruptAuthorityLeaseError("STAGE4_LEASE_ROOT_MARKER_INVALID")
        else:
            _write_atomic_json(root_file, {"schema": ROOT_SCHEMA, "single_machine_only": True})

        if lock_file.exists() or lock_file.is_symlink():
            _require_regular_file(lock_file)
            try:
                if lock_file.read_bytes() != LOCK_SENTINEL:
                    raise CorruptAuthorityLeaseError("STAGE4_LEASE_LOCK_SENTINEL_INVALID")
            except OSError as exc:
                raise CorruptAuthorityLeaseError("STAGE4_LEASE_LOCK_UNREADABLE") from exc
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(lock_file, flags, 0o600)
            try:
                os.write(fd, LOCK_SENTINEL)
                os.fsync(fd)
            finally:
                os.close(fd)
            _fsync_dir(root_path)

        if state_file.exists() or state_file.is_symlink():
            _validate_state(_read_json_file(state_file))
        else:
            _write_atomic_json(
                state_file,
                {"schema": SCHEMA, "state": "RELEASED", "epoch": 0,
                 "owner_id": None, "owner_pid": None, "owner_nonce": None},
            )
        return root_path

    @property
    def token(self) -> FencingTokenR11:
        if self._token is None or self._owner_process_pid != os.getpid():
            raise AuthorityOwnershipError("STAGE4_LEASE_NOT_OWNED_BY_THIS_PROCESS")
        return self._token

    def _root_marker_ok(self) -> None:
        raw = _read_json_file(self.root / ROOT_FILE)
        if raw != {"schema": ROOT_SCHEMA, "single_machine_only": True}:
            raise CorruptAuthorityLeaseError("STAGE4_LEASE_ROOT_MARKER_INVALID")

    def _read_state(self) -> AuthorityLeaseSnapshotR11:
        self._root_marker_ok()
        return _validate_state(_read_json_file(self.root / STATE_FILE))

    def inspect_current_authority(self) -> AuthorityLeaseSnapshotR11:
        """Inspect durable metadata only; inspection never grants write authority."""
        return self._read_state()

    def _open_and_lock(self) -> int:
        _require_regular_file(self.root / LOCK_FILE)
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.root / LOCK_FILE, flags)
        except OSError as exc:
            raise CorruptAuthorityLeaseError("STAGE4_LEASE_LOCK_OPEN_FAILED") from exc
        try:
            if os.read(fd, len(LOCK_SENTINEL)) != LOCK_SENTINEL:
                raise CorruptAuthorityLeaseError("STAGE4_LEASE_LOCK_SENTINEL_INVALID")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise AuthorityBusyError("STAGE4_AUTHORITATIVE_RUNTIME_ALREADY_LIVE") from exc
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _begin(self, *, recovery: bool) -> FencingTokenR11:
        if self._fd is not None or self._token is not None:
            raise AlreadyAcquiredError("STAGE4_SAME_HANDLE_DOUBLE_ACQUIRE")
        fd = self._open_and_lock()
        try:
            state = self._read_state()
            if recovery:
                if state.state != "ACTIVE":
                    raise AuthorityRecoveryRequiredError("STAGE4_RECOVERY_REQUIRES_CRASHED_ACTIVE_STATE")
            elif state.state == "ACTIVE":
                # Kernel lock is free but durable state still says ACTIVE: prior owner died.
                # Do not infer death from PID or time and do not silently steal authority.
                raise AuthorityRecoveryRequiredError("STAGE4_EXPLICIT_DEAD_OWNER_RECOVERY_REQUIRED")
            epoch = state.epoch + 1
            nonce = secrets.token_hex(16)
            token = FencingTokenR11(epoch, nonce)
            new_state = {
                "schema": SCHEMA, "state": "ACTIVE", "epoch": epoch,
                "owner_id": self.owner_id, "owner_pid": os.getpid(), "owner_nonce": nonce,
            }
            _write_atomic_json(self.root / STATE_FILE, new_state)
            self._fd = fd
            self._token = token
            self._owner_process_pid = os.getpid()
            _live_fds.add(fd)
            return token
        except BaseException:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise

    def acquire_authority(self) -> FencingTokenR11:
        return self._begin(recovery=False)

    def recover_after_dead_owner(self) -> FencingTokenR11:
        """Explicitly recover only after the prior kernel lock is gone.

        A busy lock fails before metadata is considered. PID identity is deliberately ignored.
        """
        return self._begin(recovery=True)

    def validate_fencing_token(self, token: FencingTokenR11) -> bool:
        state = self._read_state()
        return bool(state.state == "ACTIVE" and state.fencing_token == token)

    def assert_fencing_token(self, token: FencingTokenR11) -> None:
        # A copied token is insufficient: the asserting provider must still own the local lock.
        if self._fd is None or self._owner_process_pid != os.getpid() or self._token != token:
            raise StaleFencingTokenError("STAGE4_FENCING_TOKEN_NOT_LIVE_LOCAL_OWNER")
        if not self.validate_fencing_token(token):
            raise StaleFencingTokenError("STAGE4_FENCING_TOKEN_STALE")

    def release_authority(self, token: FencingTokenR11) -> None:
        self.assert_fencing_token(token)
        state = self._read_state()
        released = {
            "schema": SCHEMA, "state": "RELEASED", "epoch": state.epoch,
            "owner_id": None, "owner_pid": None, "owner_nonce": None,
        }
        # Persist RELEASED before unlocking. If this write fails, retain the lock and fail closed.
        _write_atomic_json(self.root / STATE_FILE, released)
        assert self._fd is not None
        fd = self._fd
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            _live_fds.discard(fd)
            os.close(fd)
            self._fd = None
            self._token = None
            self._owner_process_pid = None

    def __enter__(self) -> "Stage4AuthorityLeaseR11":
        self.acquire_authority()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._token is not None and self._owner_process_pid == os.getpid():
            self.release_authority(self._token)
