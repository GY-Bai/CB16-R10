"""R4 write-path instrumentation (measurement-only).

Installed into a Python process tree through ``sitecustomize``. It never
changes S1 runtime logic; it wraps file/sqlite write entry points and writes an
aggregated JSON snapshot per process into ``CB16_R4_METRICS_DIR`` at exit.
"""

from __future__ import annotations

import atexit
import builtins
import io
import json
import os
import pathlib
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_RAW_OPEN = io.open
_RAW_PATH_OPEN = pathlib.Path.open
_RAW_SQLITE_CONNECT = sqlite3.connect
_RAW_FSYNC = os.fsync
_RAW_REPLACE = os.replace
_RAW_RENAME = os.rename
_RAW_OS_OPEN = os.open
_RAW_OS_WRITE = os.write
_RAW_OS_CLOSE = os.close
_RAW_OS_FDOPEN = os.fdopen

_LOCK = threading.Lock()
_START_NS = time.monotonic_ns()
_SUMMARY: dict[str, dict[str, dict[str, Any]]] = {}
_UNIQUE_PATHS: set[str] = set()
_FD_PATHS: dict[int, str] = {}
_EVENT_COUNT = 0
_INSTALLED = False

UPDATE_JOURNAL_RE = re.compile(r"/updates/([0-9a-f]{64})\.json$")


def _monitored_roots() -> tuple[str, ...]:
    raw = os.environ.get("CB16_R4_MONITORED_ROOTS", "")
    return tuple(item for item in raw.split(os.pathsep) if item)


def _metrics_dir() -> Path:
    return Path(os.environ["CB16_R4_METRICS_DIR"])


def classify_path(path: str) -> str:
    """Map an absolute path to an R4 measured write-path surface.

    Scratch paths describe runtime writable surfaces; output paths describe
    exported artifact staging. Generation-switch receipts are matched before
    the generic update-journal rule.
    """
    text = str(path)
    if "/scratch/" in text:
        if "generation_switch_receipts" in text:
            return "generation_continuity"
        if "checkpoints" in text:
            return "checkpoint_store"
        if "/updates/" in text:
            return "update_journal"
        if "observ" in text:
            return "observation_store"
        if "durable_sequences" in text or "replay" in text or "material" in text:
            return "replay_materialization"
        if "index.sqlite3" in text or text.endswith("-wal") or text.endswith("-shm") or "sqlite" in text:
            return "sqlite_index"
        if "durable_index" in text or "update_resolution" in text:
            return "update_journal"
        return "other"
    if "generation_switch_receipts" in text:
        return "generation_continuity"
    if "provenance_staged" in text or "/output/provenance/" in text:
        if "update_journal" in text or "/update_journals/" in text:
            return "artifact_staging"
        return "provenance"
    if "/output/" in text or "checkpoints_staged" in text:
        return "artifact_staging"
    if "checkpoint" in text:
        return "checkpoint_store"
    if "/updates/" in text:
        return "update_journal"
    if "index.sqlite3" in text or text.endswith("-wal") or text.endswith("-shm") or "sqlite" in text:
        return "sqlite_index"
    if "observ" in text:
        return "observation_store"
    if "material" in text or "replay" in text:
        return "replay_materialization"
    if "provenance" in text:
        return "provenance"
    return "other"


def _is_monitored(path: str) -> bool:
    if path.startswith(str(_metrics_dir())):
        return False
    roots = _monitored_roots()
    return any(path.startswith(root) for root in roots)


def _record(op: str, path: str | None = None, bytes_count: int = 0, duration_ns: int = 0, **extra: Any) -> None:
    global _EVENT_COUNT
    category = classify_path(path) if path else "sqlite_control"
    key = f"{category}|{op}"
    with _LOCK:
        bucket = _SUMMARY.setdefault(category, {}).setdefault(
            op,
            {"count": 0, "bytes": 0, "duration_ns": 0, "extra": {}},
        )
        bucket["count"] += 1
        bucket["bytes"] += int(bytes_count)
        bucket["duration_ns"] += int(duration_ns)
        for name, value in extra.items():
            bucket["extra"][name] = bucket["extra"].get(name, 0) + value
        if path:
            _UNIQUE_PATHS.add(f"{category}|{path}")
        _EVENT_COUNT += 1
        if _EVENT_COUNT % 2000 == 0:
            _flush_locked()


class _CountedFile:
    __slots__ = ("_file", "_path")

    def __init__(self, file_obj: Any, path: str) -> None:
        self._file = file_obj
        self._path = path

    def write(self, data: Any) -> int:
        written = self._file.write(data)
        if isinstance(data, str):
            byte_count = len(data.encode("utf-8", errors="replace"))
        else:
            byte_count = len(data)
        _record("write", self._path, bytes_count=byte_count)
        return written

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        _record("flush", self._path)
        return self._file.flush()

    def close(self) -> None:
        return self._file.close()

    def __enter__(self) -> "_CountedFile":
        self._file.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._file.__exit__(*args)

    def __iter__(self) -> Any:
        return iter(self._file)

    def __next__(self) -> Any:
        return next(self._file)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file, name)


class _CountedConnection:
    __slots__ = ("_conn", "_path")

    def __init__(self, conn: Any, path: str) -> None:
        self._conn = conn
        self._path = path

    def _record_sql(self, sql: str, duration_ns: int) -> None:
        statement = sql.strip().split(None, 1)[0].upper() if sql.strip() else "UNKNOWN"
        extra: dict[str, Any] = {}
        if statement == "COMMIT":
            extra["sqlite_commits"] = 1
        _record("sqlite_execute", self._path, duration_ns=duration_ns, **extra)
        _record("sqlite_statement", self._path, statement=0)
        if "WAL" in sql.upper():
            _record("sqlite_wal_pragma", self._path)

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic_ns()
        try:
            return self._conn.execute(sql, *args, **kwargs)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower():
                _record("sqlite_locked_error", self._path)
            raise
        finally:
            self._record_sql(sql, time.monotonic_ns() - start)

    def executemany(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic_ns()
        try:
            return self._conn.executemany(sql, *args, **kwargs)
        finally:
            self._record_sql(sql, time.monotonic_ns() - start)

    def executescript(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        start = time.monotonic_ns()
        try:
            return self._conn.executescript(sql, *args, **kwargs)
        finally:
            self._record_sql(sql, time.monotonic_ns() - start)

    def commit(self) -> None:
        start = time.monotonic_ns()
        try:
            return self._conn.commit()
        finally:
            _record("sqlite_commit", self._path, duration_ns=time.monotonic_ns() - start)

    def rollback(self) -> None:
        _record("sqlite_rollback", self._path)
        return self._conn.rollback()

    def close(self) -> None:
        return self._conn.close()

    def __enter__(self) -> "_CountedConnection":
        self._conn.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._conn.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _instrumented_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    file_obj = _RAW_OPEN(file, mode, *args, **kwargs)
    if isinstance(file, int):
        return file_obj
    write_like = any(flag in mode for flag in ("w", "a", "x", "+"))
    if not write_like:
        return file_obj
    path = str(os.fspath(file))
    if path.startswith(str(_metrics_dir())):
        return file_obj
    return _CountedFile(file_obj, path)


def _instrumented_path_open(
    self: Path,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
) -> Any:
    file_obj = _RAW_PATH_OPEN(self, mode, buffering, encoding, errors, newline)
    write_like = any(flag in mode for flag in ("w", "a", "x", "+"))
    if not write_like:
        return file_obj
    path = str(self)
    if path.startswith(str(_metrics_dir())):
        return file_obj
    return _CountedFile(file_obj, path)


def _instrumented_sqlite_connect(database: Any, *args: Any, **kwargs: Any) -> Any:
    conn = _RAW_SQLITE_CONNECT(database, *args, **kwargs)
    path = os.fspath(database) if isinstance(database, (str, bytes, os.PathLike)) else ""
    return _CountedConnection(conn, str(path)) if path else conn


def _instrumented_fsync(fd: int) -> None:
    start = time.monotonic_ns()
    try:
        return _RAW_FSYNC(fd)
    finally:
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            path = f"fd:{fd}"
        _record("fsync", path, duration_ns=time.monotonic_ns() - start)


def _instrumented_os_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
    fd = _RAW_OS_OPEN(path, flags, *args, **kwargs)
    write_like = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
    if write_like:
        resolved = str(os.fspath(path))
        _FD_PATHS[fd] = resolved
        _record("os_open_write", resolved)
    return fd


def _instrumented_os_write(fd: int, data: Any) -> int:
    written = _RAW_OS_WRITE(fd, data)
    path = _FD_PATHS.get(fd)
    if path is not None:
        _record("os_write", path, bytes_count=written)
    return written


def _instrumented_os_close(fd: int) -> None:
    _FD_PATHS.pop(fd, None)
    return _RAW_OS_CLOSE(fd)


def _instrumented_fdopen(fd: int, *args: Any, **kwargs: Any) -> Any:
    file_obj = _RAW_OS_FDOPEN(fd, *args, **kwargs)
    path = _FD_PATHS.get(fd)
    if path and not path.startswith(str(_metrics_dir())):
        return _CountedFile(file_obj, path)
    return file_obj


def _instrumented_replace(src: Any, dst: Any) -> None:
    _record("replace", str(os.fspath(dst)))
    return _RAW_REPLACE(src, dst)


def _instrumented_rename(src: Any, dst: Any) -> None:
    _record("rename", str(os.fspath(dst)))
    return _RAW_RENAME(src, dst)


def _snapshot() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "start_ns": _START_NS,
        "end_ns": time.monotonic_ns(),
        "summary": _SUMMARY,
        "unique_paths": sorted(_UNIQUE_PATHS),
    }


def _flush_locked() -> None:
    directory = _metrics_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"pid-{os.getpid()}.json"
    tmp = directory / f".pid-{os.getpid()}.json.tmp"
    tmp.write_text(json.dumps(_snapshot(), sort_keys=True), encoding="utf-8")
    _RAW_REPLACE(tmp, target)


def flush() -> None:
    with _LOCK:
        _flush_locked()


def _reset_after_fork() -> None:
    global _SUMMARY, _UNIQUE_PATHS, _FD_PATHS, _EVENT_COUNT, _START_NS
    _SUMMARY = {}
    _UNIQUE_PATHS = set()
    _FD_PATHS = {}
    _EVENT_COUNT = 0
    _START_NS = time.monotonic_ns()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    builtins.open = _instrumented_open
    io.open = _instrumented_open
    pathlib.Path.open = _instrumented_path_open
    sqlite3.connect = _instrumented_sqlite_connect
    os.fsync = _instrumented_fsync
    os.open = _instrumented_os_open
    os.write = _instrumented_os_write
    os.close = _instrumented_os_close
    os.fdopen = _instrumented_fdopen
    os.replace = _instrumented_replace
    os.rename = _instrumented_rename
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_reset_after_fork)
    atexit.register(flush)
