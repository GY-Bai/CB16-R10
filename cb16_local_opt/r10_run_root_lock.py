from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunRootExclusiveLock:
    """Process-lifetime exclusive lock for one mutable R10 run root.

    This is runtime safety only. The lock file is not part of scientific/cache
    identity and contains no scientific data.
    """

    run_root: Path
    name: str = ".cb16_r104_run.lock"
    _fd: int | None = None

    def acquire(self) -> "RunRootExclusiveLock":
        root = Path(self.run_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / self.name
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        except PermissionError:
            # A lock file may have been created by another local account. flock
            # itself does not require mutating file contents, so read-only open is
            # sufficient for arbitration when the directory remains accessible.
            fd = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(f"R10_4_RUN_ROOT_ALREADY_ACTIVE:{root}") from exc
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "RunRootExclusiveLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
