from __future__ import annotations
import json, os, sqlite3, tempfile
from pathlib import Path
from typing import Any, Mapping
from .cc_experience_wire_r0 import canonical_json_bytes, content_sha256

class SemanticConflict(RuntimeError): pass

class RawFactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS facts (logical_id TEXT PRIMARY KEY, content_sha TEXT NOT NULL, object_path TEXT NOT NULL)")

    def _object_path(self, digest: str) -> Path:
        return self.objects / digest[:2] / f"{digest}.json"

    def put(self, logical_id: str, fact: Mapping[str, Any], *, crash_after_object: bool = False) -> str:
        if not logical_id:
            raise ValueError("logical_id required")
        digest = content_sha256(fact)
        path = self._object_path(digest)
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT content_sha FROM facts WHERE logical_id=?", (logical_id,)).fetchone()
            if row:
                if row[0] != digest:
                    raise SemanticConflict("same logical ID with different content")
                return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            fd, tmp = tempfile.mkstemp(prefix=".cc-tmp-", dir=str(path.parent))
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(canonical_json_bytes(fact)); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, path)
                dirfd = os.open(path.parent, os.O_RDONLY)
                try: os.fsync(dirfd)
                finally: os.close(dirfd)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        if crash_after_object:
            raise RuntimeError("FAULT_AFTER_DURABLE_OBJECT")
        with sqlite3.connect(self.db_path) as db:
            try:
                db.execute("INSERT INTO facts(logical_id,content_sha,object_path) VALUES(?,?,?)", (logical_id,digest,str(path)))
            except sqlite3.IntegrityError:
                row = db.execute("SELECT content_sha FROM facts WHERE logical_id=?", (logical_id,)).fetchone()
                if not row or row[0] != digest:
                    raise SemanticConflict("concurrent semantic conflict")
        return digest

    def get(self, logical_id: str) -> Mapping[str, Any]:
        with sqlite3.connect(self.db_path) as db:
            row = db.execute("SELECT object_path FROM facts WHERE logical_id=?", (logical_id,)).fetchone()
        if not row: raise KeyError(logical_id)
        with open(row[0], "r", encoding="utf-8") as f: return json.load(f)

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as db:
            return int(db.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
