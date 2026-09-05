"""CB16 CI Relay V1.1 - FastAPI application.

Endpoints:
  GET  /healthz
  POST /webhook/github
  GET  /api/v1/worker/jobs/next
  POST /api/v1/worker/jobs/{job_id}/claim
  POST /api/v1/worker/jobs/{job_id}/heartbeat
  POST /api/v1/worker/jobs/{job_id}/result
  POST /api/v1/worker/jobs/{job_id}/logs
  GET  /api/v1/worker/bundles/{sha}

CI profile selection is explicit and bounded. A push defaults to ``smoke``.
To request another profile, include one marker in the head commit message:

    [ci:unit] [ci:r102] [ci:r103] [ci:r104] [ci:v63]

The webhook can never supply an arbitrary shell command.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import subprocess
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import zstandard
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

RELAY_ROOT = Path(os.environ.get("CB16_CI_ROOT", "/home/bgy/cb16-ci"))
REPO_MIRROR = RELAY_ROOT / "repo" / "source.git"
BUNDLES_DIR = RELAY_ROOT / "bundles"
JOBS_DIR = RELAY_ROOT / "jobs"
RESULTS_DIR = RELAY_ROOT / "results"
LOGS_DIR = RELAY_ROOT / "logs"
STATE_DIR = RELAY_ROOT / "state"
DB_PATH = STATE_DIR / "relay.db"

REPO_URL = os.environ.get(
    "CB16_REPO_URL", "https://github.com/GY-Bai/CB16-R10.git"
)
ALLOWED_REPOS = {
    x.strip()
    for x in os.environ.get("CB16_ALLOWED_REPOS", "GY-Bai/CB16-R10").split(",")
    if x.strip()
}
ALLOWED_REFS = {"refs/heads/main", "refs/heads/ai/"}
FORBIDDEN_REFS = {"refs/heads/ci-results"}

ALLOWED_CI_PROFILES = {"smoke", "unit", "r102", "r103", "r104", "v63"}
DEFAULT_CI_PROFILE = os.environ.get("CB16_DEFAULT_CI_PROFILE", "smoke")
if DEFAULT_CI_PROFILE not in ALLOWED_CI_PROFILES:
    raise RuntimeError(f"invalid CB16_DEFAULT_CI_PROFILE={DEFAULT_CI_PROFILE!r}")

CI_PROFILE_RE = re.compile(r"\[ci:(smoke|unit|r102|r103|r104|v63)\]", re.IGNORECASE)
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_STALE_SECONDS = int(os.environ.get("CB16_JOB_STALE_SECONDS", "600"))
MAX_JOB_ATTEMPTS = int(os.environ.get("CB16_MAX_JOB_ATTEMPTS", "2"))

for d in (
    RELAY_ROOT,
    REPO_MIRROR.parent,
    BUNDLES_DIR,
    JOBS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    STATE_DIR,
):
    d.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_column(
    conn: sqlite3.Connection, table: str, name: str, declaration: str
) -> None:
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            repository TEXT NOT NULL,
            branch TEXT NOT NULL,
            commit_sha TEXT NOT NULL,
            github_delivery_id TEXT,
            ci_profile TEXT DEFAULT 'smoke',
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            claimed_at TEXT,
            finished_at TEXT,
            worker_id TEXT,
            verdict TEXT,
            bundle_sha256 TEXT,
            bundle_size INTEGER,
            error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS deliveries (
            delivery_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        )
        """
    )
    ensure_column(conn, "jobs", "heartbeat_at", "TEXT")
    ensure_column(conn, "jobs", "attempt_count", "INTEGER NOT NULL DEFAULT 0")
    conn.commit()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def hmac_equal(a: str, b: str) -> bool:
    try:
        return hmac.compare_digest(a, b)
    except Exception:
        return False


def compute_signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def load_secret(name: str, default: str = "") -> str:
    env_path = Path(os.environ.get("CB16_RELAY_ENV", "/etc/cb16-ci/relay.env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                if k == name:
                    return v
    return os.environ.get(name, default)


GITHUB_WEBHOOK_SECRET = load_secret("GITHUB_WEBHOOK_SECRET")
CB16_WORKER_TOKEN = load_secret("CB16_WORKER_TOKEN")
CB16_INTERNAL_SECRET = load_secret("CB16_INTERNAL_SECRET")
GITHUB_RESULT_TOKEN = load_secret("GITHUB_RESULT_TOKEN")


def fetch_repo_mirror() -> None:
    if not (REPO_MIRROR / "HEAD").exists():
        subprocess.run(
            ["git", "clone", "--bare", REPO_URL, str(REPO_MIRROR)],
            check=True,
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                "git", "-C", str(REPO_MIRROR), "fetch", "--prune", REPO_URL,
                "+refs/heads/*:refs/heads/*",
            ],
            check=True,
            capture_output=True,
        )


def sha_exists(sha: str) -> bool:
    if not COMMIT_SHA_RE.fullmatch(sha):
        return False
    try:
        subprocess.run(
            ["git", "-C", str(REPO_MIRROR), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=True,
            capture_output=True,
        )
        return True
    except Exception:
        return False


def create_bundle(sha: str) -> dict[str, Any]:
    if not COMMIT_SHA_RE.fullmatch(sha):
        raise ValueError("invalid commit sha")
    bundle_path = BUNDLES_DIR / f"{sha}.tar.zst"
    if bundle_path.exists():
        data = bundle_path.read_bytes()
        return {"sha": sha, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

    tar_tmp = BUNDLES_DIR / f".{sha}.{uuid.uuid4().hex}.tar"
    zst_tmp = BUNDLES_DIR / f".{sha}.{uuid.uuid4().hex}.tar.zst"
    try:
        with tar_tmp.open("wb") as f:
            subprocess.run(
                ["git", "-C", str(REPO_MIRROR), "archive", "--format=tar", sha],
                stdout=f,
                check=True,
            )
        cctx = zstandard.ZstdCompressor()
        with tar_tmp.open("rb") as fsrc, zst_tmp.open("wb") as fdst:
            cctx.copy_stream(fsrc, fdst)
        zst_tmp.replace(bundle_path)
        data = bundle_path.read_bytes()
        return {"sha": sha, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    finally:
        tar_tmp.unlink(missing_ok=True)
        zst_tmp.unlink(missing_ok=True)


def requested_ci_profile(payload: dict[str, Any]) -> str:
    message = ((payload.get("head_commit") or {}).get("message") or "").strip()
    matches = CI_PROFILE_RE.findall(message)
    if not matches:
        return DEFAULT_CI_PROFILE
    profiles = {m.lower() for m in matches}
    if len(profiles) != 1:
        raise ValueError(f"multiple CI profiles requested: {sorted(profiles)}")
    profile = next(iter(profiles))
    if profile not in ALLOWED_CI_PROFILES:
        raise ValueError(f"CI profile not allowed: {profile}")
    return profile


def ref_allowed(ref: str) -> bool:
    if ref in FORBIDDEN_REFS:
        return False
    return ref in ALLOWED_REFS or any(
        prefix.endswith("/") and ref.startswith(prefix) for prefix in ALLOWED_REFS
    )


def requeue_stale_jobs(conn: sqlite3.Connection) -> None:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(seconds=JOB_STALE_SECONDS)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    rows = conn.execute(
        """
        SELECT id, status, attempt_count
        FROM jobs
        WHERE status IN ('CLAIMED','RUNNING')
          AND COALESCE(heartbeat_at, claimed_at, created_at) < ?
        """,
        (cutoff,),
    ).fetchall()

    for row in rows:
        if int(row["attempt_count"] or 0) >= MAX_JOB_ATTEMPTS:
            conn.execute(
                """
                UPDATE jobs
                SET status='ERROR', finished_at=?, verdict='ERROR',
                    error='STALE_WORKER_MAX_ATTEMPTS'
                WHERE id=?
                """,
                (now_iso(), row["id"]),
            )
        else:
            conn.execute(
                """
                UPDATE jobs
                SET status='PENDING', claimed_at=NULL, heartbeat_at=NULL,
                    worker_id=NULL, error='STALE_WORKER_REQUEUED'
                WHERE id=?
                """,
                (row["id"],),
            )


def create_job_from_webhook(payload: dict[str, Any], delivery_id: str) -> dict[str, Any]:
    repo_full = (payload.get("repository") or {}).get("full_name", "")
    ref = payload.get("ref", "")
    if repo_full not in ALLOWED_REPOS:
        raise ValueError(f"repository not allowed: {repo_full}")
    if not ref_allowed(ref):
        raise ValueError(f"ref not allowed: {ref}")
    if bool(payload.get("deleted", False)):
        raise ValueError("deleted branch push ignored")

    commit_sha = (payload.get("after") or "").strip().lower()
    if not COMMIT_SHA_RE.fullmatch(commit_sha):
        raise ValueError("invalid or missing commit sha")

    profile = requested_ci_profile(payload)
    branch = ref.removeprefix("refs/heads/")
    job_id = str(uuid.uuid4())

    with db() as conn:
        if conn.execute("SELECT 1 FROM deliveries WHERE delivery_id=?", (delivery_id,)).fetchone():
            raise DuplicateDelivery(delivery_id)
        conn.execute(
            "INSERT INTO deliveries(delivery_id, created_at) VALUES (?, ?)",
            (delivery_id, now_iso()),
        )
        existing = conn.execute(
            "SELECT id FROM jobs WHERE repository=? AND commit_sha=? AND ci_profile=?",
            (repo_full, commit_sha, profile),
        ).fetchone()
        if existing:
            return {
                "job_id": existing["id"], "duplicate": True,
                "commit_sha": commit_sha, "ci_profile": profile,
            }

        fetch_repo_mirror()
        if not sha_exists(commit_sha):
            raise ValueError(f"sha not found: {commit_sha}")
        bundle_meta = create_bundle(commit_sha)
        conn.execute(
            """
            INSERT INTO jobs(
                id, repository, branch, commit_sha, github_delivery_id,
                ci_profile, status, created_at, claimed_at, heartbeat_at,
                finished_at, worker_id, verdict, bundle_sha256, bundle_size,
                error, attempt_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id, repo_full, branch, commit_sha, delivery_id, profile,
                "PENDING", now_iso(), None, None, None, None, None,
                bundle_meta["sha256"], bundle_meta["size_bytes"], None, 0,
            ),
        )
        return {
            "job_id": job_id, "duplicate": False,
            "commit_sha": commit_sha, "ci_profile": profile,
        }


class DuplicateDelivery(Exception):
    pass


class InternalPrefixMiddleware:
    """Allows Cloudflare Worker edge auth to call /cb16-internal/api/* safely."""
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith("/cb16-internal/api/"):
                headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
                if not hmac.compare_digest(headers.get("x-cb16-internal-secret", ""), CB16_INTERNAL_SECRET):
                    response = JSONResponse({"error": "forbidden"}, status_code=403)
                    await response(scope, receive, send)
                    return
                scope["path"] = path[len("/cb16-internal"):]
                if b"raw_path" in scope:
                    scope["raw_path"] = scope["path"].encode()
        await self.app(scope, receive, send)


app = FastAPI(title="CB16 CI Relay", version="1.1")
app.add_middleware(InternalPrefixMiddleware)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/webhook/github")
async def github_webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not GITHUB_WEBHOOK_SECRET:
        return JSONResponse({"error": "server not configured"}, status_code=500)
    if not hmac_equal(compute_signature(GITHUB_WEBHOOK_SECRET, raw), sig):
        return JSONResponse({"error": "invalid signature"}, status_code=403)
    if request.headers.get("X-GitHub-Event", "") != "push":
        return JSONResponse({"error": "event not supported"}, status_code=400)
    delivery = request.headers.get("X-GitHub-Delivery", "")
    if not delivery:
        return JSONResponse({"error": "missing delivery id"}, status_code=400)
    try:
        payload = json.loads(raw)
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)
    try:
        result = create_job_from_webhook(payload, delivery)
    except DuplicateDelivery as e:
        return JSONResponse(
            {"status": "duplicate_delivery", "delivery_id": str(e)}, status_code=200
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=202)
    return JSONResponse(
        {
            "status": "created" if not result["duplicate"] else "duplicate_job",
            "job_id": result["job_id"],
            "commit_sha": result["commit_sha"],
            "ci_profile": result["ci_profile"],
        },
        status_code=201 if not result["duplicate"] else 200,
    )


def require_worker_auth(request: Request) -> bool:
    auth = request.headers.get("Authorization", "")
    expected = "Bearer " + CB16_WORKER_TOKEN
    return bool(CB16_WORKER_TOKEN) and hmac.compare_digest(auth, expected)


def _unauthorized():
    return JSONResponse({"error": "unauthorized"}, status_code=401)


@app.get("/api/v1/worker/jobs/next")
async def next_job(request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    with db() as conn:
        requeue_stale_jobs(conn)
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='PENDING' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if not row:
            return {"job": None}
        return {"job": dict(row)}


@app.post("/api/v1/worker/jobs/{job_id}/claim")
async def claim_job(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    body = await request.json()
    worker_id = body.get("worker_id", "unknown")
    timestamp = now_iso()
    with db() as conn:
        updated = conn.execute(
            """
            UPDATE jobs
            SET status='CLAIMED', claimed_at=?, heartbeat_at=?, worker_id=?,
                attempt_count=attempt_count+1
            WHERE id=? AND status='PENDING'
            """,
            (timestamp, timestamp, worker_id, job_id),
        )
        if updated.rowcount != 1:
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return JSONResponse({"error": "job not found"}, status_code=404)
            return JSONResponse(
                {"error": "job not pending", "status": row["status"]}, status_code=409
            )
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return {"job": dict(row)}


@app.post("/api/v1/worker/jobs/{job_id}/heartbeat")
async def heartbeat(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    with db() as conn:
        updated = conn.execute(
            """
            UPDATE jobs SET status='RUNNING', heartbeat_at=?
            WHERE id=? AND status IN ('CLAIMED','RUNNING')
            """,
            (now_iso(), job_id),
        )
        if updated.rowcount != 1:
            row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return JSONResponse({"error": "job not found"}, status_code=404)
            return JSONResponse(
                {"error": "job not active", "status": row["status"]}, status_code=409
            )
        return {"status": "ok"}


@app.post("/api/v1/worker/jobs/{job_id}/result")
async def submit_result(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    payload = await request.json()
    verdict = str(payload.get("verdict", "ERROR")).upper()
    if verdict not in {"PASS", "FAIL", "ERROR", "TIMEOUT"}:
        return JSONResponse({"error": "invalid verdict"}, status_code=400)
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return JSONResponse({"error": "job not found"}, status_code=404)
        if payload.get("job_id") != job_id:
            return JSONResponse({"error": "result job_id mismatch"}, status_code=409)
        if payload.get("commit_sha") != row["commit_sha"]:
            return JSONResponse({"error": "result commit_sha mismatch"}, status_code=409)
        if payload.get("ci_profile", row["ci_profile"]) != row["ci_profile"]:
            return JSONResponse({"error": "result ci_profile mismatch"}, status_code=409)
        result_path = RESULTS_DIR / job_id
        result_path.mkdir(parents=True, exist_ok=True)
        (result_path / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
        conn.execute(
            "UPDATE jobs SET status=?, finished_at=?, verdict=?, heartbeat_at=? WHERE id=?",
            (verdict, now_iso(), verdict, now_iso(), job_id),
        )
        try:
            publish_result_branch(job_id, row["commit_sha"], verdict)
        except Exception as e:
            conn.execute(
                "UPDATE jobs SET error=? WHERE id=?",
                (f"RESULT_PUBLISH_FAILED:{type(e).__name__}", job_id),
            )
            return JSONResponse(
                {"status": verdict, "warning": "result stored on OCI but GitHub publish failed"},
                status_code=202,
            )
        return {"status": verdict}


@app.get("/api/v1/worker/bundles/{sha}")
async def get_bundle(sha: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    if not COMMIT_SHA_RE.fullmatch(sha):
        return JSONResponse({"error": "invalid sha"}, status_code=400)
    bundle = BUNDLES_DIR / f"{sha}.tar.zst"
    if not bundle.exists():
        return JSONResponse({"error": "bundle not found"}, status_code=404)
    return FileResponse(bundle, media_type="application/zstd", filename=bundle.name)


@app.post("/api/v1/worker/jobs/{job_id}/logs")
async def upload_logs(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    with db() as conn:
        if not conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
            return JSONResponse({"error": "job not found"}, status_code=404)
    result_path = RESULTS_DIR / job_id
    result_path.mkdir(parents=True, exist_ok=True)
    form = await request.form()
    for field_name in ("stdout.log", "stderr.log", "REPORT.md", "SHA256SUMS"):
        f = form.get(field_name)
        if f is not None:
            data = await f.read()
            (result_path / field_name).write_bytes(data)
    return {"status": "ok"}


def publish_result_branch(job_id: str, commit_sha: str, verdict: str) -> None:
    """Push sanitized result to ci-results branch using OCI-only credentials."""
    token = GITHUB_RESULT_TOKEN
    pub_root = RELAY_ROOT / "results-publish"
    if not (pub_root / ".git").exists():
        subprocess.run(["git", "init", str(pub_root)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(pub_root), "checkout", "-b", "ci-results"],
            check=True, capture_output=True,
        )
        remote = repo_url_with_token(token)
        subprocess.run(
            ["git", "-C", str(pub_root), "remote", "add", "origin", remote],
            check=True, capture_output=True,
        )
    else:
        remote = repo_url_with_token(token)
        subprocess.run(
            ["git", "-C", str(pub_root), "remote", "set-url", "origin", remote],
            check=True, capture_output=True,
        )
    try:
        subprocess.run(
            ["git", "-C", str(pub_root), "fetch", "origin", "ci-results"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(pub_root), "checkout", "-B", "ci-results", "origin/ci-results"],
            check=True, capture_output=True,
        )
    except Exception:
        subprocess.run(
            ["git", "-C", str(pub_root), "checkout", "-B", "ci-results"],
            check=True, capture_output=True,
        )
    run_dir = pub_root / "runs" / commit_sha
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / job_id
    shutil.copy(result_path / "result.json", run_dir / "result.json")
    for name in ("REPORT.md", "SHA256SUMS"):
        source = result_path / name
        if source.exists():
            shutil.copy(source, run_dir / name)
    subprocess.run(
        ["git", "-C", str(pub_root), "config", "user.name", "CB16 CI Relay"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(pub_root), "config", "user.email", "ci@localhost"],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "-C", str(pub_root), "add", "."], check=True, capture_output=True)
    commit = subprocess.run(
        ["git", "-C", str(pub_root), "commit", "-m", f"ci-result: {commit_sha} {verdict}"],
        capture_output=True, text=True,
    )
    if commit.returncode not in (0, 1):
        raise RuntimeError(f"git commit failed: {commit.stderr.strip()}")
    subprocess.run(
        ["git", "-C", str(pub_root), "push", "origin", "ci-results"],
        check=True, capture_output=True,
    )


def repo_url_with_token(token: str) -> str:
    if not token:
        return "git@github.com:GY-Bai/CB16-R10.git"
    parsed = urlparse(REPO_URL)
    return f"https://x-access-token:{token}@{parsed.netloc}{parsed.path}"
