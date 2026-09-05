"""CB16 CI Relay V1 - FastAPI application.

Endpoints:
  GET  /healthz
  POST /webhook/github
  GET  /api/v1/worker/jobs/next
  POST /api/v1/worker/jobs/{job_id}/claim
  POST /api/v1/worker/jobs/{job_id}/heartbeat
  POST /api/v1/worker/jobs/{job_id}/result
  GET  /api/v1/worker/bundles/{sha}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import zstandard

from fastapi import FastAPI, Request, Response, UploadFile, File, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

RELAY_ROOT = Path(os.environ.get("CB16_CI_ROOT", "/home/bgy/cb16-ci"))
REPO_MIRROR = RELAY_ROOT / "repo" / "source.git"
BUNDLES_DIR = RELAY_ROOT / "bundles"
JOBS_DIR = RELAY_ROOT / "jobs"
RESULTS_DIR = RELAY_ROOT / "results"
LOGS_DIR = RELAY_ROOT / "logs"
STATE_DIR = RELAY_ROOT / "state"
DB_PATH = STATE_DIR / "relay.db"
APP_DIR = Path(__file__).resolve().parent

# Repository authority settings (injected via env for test; default is this repo)
REPO_URL = os.environ.get("CB16_REPO_URL", "https://github.com/GY-Bai/CB16-R10.git")
ALLOWED_REPOS = {x.strip() for x in os.environ.get("CB16_ALLOWED_REPOS", "GY-Bai/CB16-R10").split(",") if x.strip()}
ALLOWED_REFS = {"refs/heads/main", "refs/heads/ai/"}
FORBIDDEN_REFS = {"refs/heads/ci-results"}

for d in (RELAY_ROOT, REPO_MIRROR.parent, BUNDLES_DIR, JOBS_DIR, RESULTS_DIR, LOGS_DIR, STATE_DIR):
    d.mkdir(parents=True, exist_ok=True)


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
            ci_profile TEXT DEFAULT 'default',
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
    conn.commit()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
GITHUB_RESULT_TOKEN = load_secret("GITHUB_RESULT_TOKEN")


def fetch_repo_mirror() -> None:
    if not (REPO_MIRROR / "HEAD").exists():
        subprocess.run(["git", "clone", "--bare", REPO_URL, str(REPO_MIRROR)], check=True, capture_output=True)
    else:
        subprocess.run(["git", "-C", str(REPO_MIRROR), "fetch", "--prune", REPO_URL, "+refs/heads/*:refs/heads/*"], check=True, capture_output=True)


def sha_exists(sha: str) -> bool:
    try:
        subprocess.run(["git", "-C", str(REPO_MIRROR), "cat-file", "-e", f"{sha}^{{commit}}"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def create_bundle(sha: str) -> dict[str, Any]:
    # Exact SHA is the identity; create immutable archive from git object database.
    bundle_path = BUNDLES_DIR / f"{sha}.tar.zst"
    if bundle_path.exists():
        data = bundle_path.read_bytes()
        return {"sha": sha, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    tmp = bundle_path.with_suffix(".tmp")
    try:
        # Always create real zstd archives using the Python zstandard library.
        with open(tmp, "wb") as f:
            subprocess.run(["git", "-C", str(REPO_MIRROR), "archive", "--format=tar", sha], stdout=f, check=True)
        cctx = zstandard.ZstdCompressor()
        with open(tmp, "rb") as fsrc, open(bundle_path, "wb") as fdst:
            cctx.copy_stream(fsrc, fdst)
        tmp.unlink(missing_ok=True)
        data = bundle_path.read_bytes()
        return {"sha": sha, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    finally:
        tmp.unlink(missing_ok=True)


def create_job_from_webhook(payload: dict[str, Any], delivery_id: str) -> dict[str, Any]:
    repo_full = (payload.get("repository") or {}).get("full_name", "")
    ref = payload.get("ref", "")
    if repo_full not in ALLOWED_REPOS:
        raise ValueError(f"repository not allowed: {repo_full}")
    if ref in FORBIDDEN_REFS or (ref not in ALLOWED_REFS and not any(ref.startswith(r) for r in ALLOWED_REFS)):
        raise ValueError(f"ref not allowed: {ref}")
    deleted = bool((payload.get("deleted") or False) or (payload.get("created") is False and payload.get("forced") is False and payload.get("deleted") is True))
    if deleted:
        raise ValueError("deleted branch push ignored")
    commit_sha = (payload.get("after") or "").strip()
    if not commit_sha or commit_sha == "0000000000000000000000000000000000000000":
        raise ValueError("no commit sha")
    branch = ref.removeprefix("refs/heads/")
    job_id = str(uuid.uuid4())
    with db() as conn:
        # idempotency on delivery id and on repository+sha+profile
        row = conn.execute("SELECT delivery_id FROM deliveries WHERE delivery_id=?", (delivery_id,)).fetchone()
        if row:
            raise DuplicateDelivery(row["delivery_id"])
        conn.execute("INSERT INTO deliveries(delivery_id, created_at) VALUES (?, ?)", (delivery_id, now_iso()))
        existing = conn.execute(
            "SELECT id FROM jobs WHERE repository=? AND commit_sha=? AND ci_profile=?",
            (repo_full, commit_sha, "default"),
        ).fetchone()
        if existing:
            return {"job_id": existing["id"], "duplicate": True, "commit_sha": commit_sha}
        # Build bundle before publishing PENDING job.
        fetch_repo_mirror()
        if not sha_exists(commit_sha):
            raise ValueError(f"sha not found: {commit_sha}")
        bundle_meta = create_bundle(commit_sha)
        job = {
            "id": job_id,
            "repository": repo_full,
            "branch": branch,
            "commit_sha": commit_sha,
            "github_delivery_id": delivery_id,
            "ci_profile": "default",
            "status": "PENDING",
            "created_at": now_iso(),
            "claimed_at": None,
            "finished_at": None,
            "worker_id": None,
            "verdict": None,
            "bundle_sha256": bundle_meta["sha256"],
            "bundle_size": bundle_meta["size_bytes"],
            "error": None,
        }
        conn.execute(
            """INSERT INTO jobs(id, repository, branch, commit_sha, github_delivery_id, ci_profile, status,
               created_at, claimed_at, finished_at, worker_id, verdict, bundle_sha256, bundle_size, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            tuple(job.values()),
        )
        return {"job_id": job_id, "duplicate": False, "commit_sha": commit_sha}


class DuplicateDelivery(Exception):
    pass


app = FastAPI(title="CB16 CI Relay", version="1.0")


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
    event = request.headers.get("X-GitHub-Event", "")
    if event != "push":
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
        return JSONResponse({"status": "duplicate", "job_id": str(e)}, status_code=200)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=202)
    return JSONResponse({"status": "created", "job_id": result["job_id"], "commit_sha": result["commit_sha"]}, status_code=201)


def require_worker_auth(request: Request) -> bool:
    # Cloudflare Access is enforced at edge. Application layer also requires bearer token.
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
        row = conn.execute("SELECT * FROM jobs WHERE status='PENDING' ORDER BY created_at LIMIT 1").fetchone()
        if not row:
            return {"job": None}
        return {"job": dict(row)}


@app.post("/api/v1/worker/jobs/{job_id}/claim")
async def claim_job(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    body = await request.json()
    worker_id = body.get("worker_id", "unknown")
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return JSONResponse({"error": "job not found"}, status_code=404)
        if row["status"] != "PENDING":
            return JSONResponse({"error": "job not pending", "status": row["status"]}, status_code=409)
        conn.execute("UPDATE jobs SET status='CLAIMED', claimed_at=?, worker_id=? WHERE id=?", (now_iso(), worker_id, job_id))
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return {"job": dict(row)}


@app.post("/api/v1/worker/jobs/{job_id}/heartbeat")
async def heartbeat(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    with db() as conn:
        row = conn.execute("SELECT id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return JSONResponse({"error": "job not found"}, status_code=404)
        conn.execute("UPDATE jobs SET claimed_at=? WHERE id=?", (now_iso(), job_id))
        return {"status": "ok"}


@app.post("/api/v1/worker/jobs/{job_id}/result")
async def submit_result(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    payload = await request.json()
    verdict = payload.get("verdict", "ERROR")
    result_path = RESULTS_DIR / job_id
    result_path.mkdir(parents=True, exist_ok=True)
    (result_path / "result.json").write_text(json.dumps(payload, indent=2))
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return JSONResponse({"error": "job not found"}, status_code=404)
        status_value = "PASS" if verdict == "PASS" else ("FAIL" if verdict == "FAIL" else "ERROR")
        conn.execute(
            "UPDATE jobs SET status=?, finished_at=?, verdict=? WHERE id=?",
            (status_value, now_iso(), verdict, job_id),
        )
        # publish result branch best-effort
        try:
            publish_result_branch(job_id, row["commit_sha"], verdict)
        except Exception as e:
            pass
        return {"status": status_value}


@app.get("/api/v1/worker/bundles/{sha}")
async def get_bundle(sha: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
    bundle = BUNDLES_DIR / f"{sha}.tar.zst"
    if not bundle.exists():
        return JSONResponse({"error": "bundle not found"}, status_code=404)
    return FileResponse(bundle, media_type="application/zstd", filename=bundle.name)


@app.post("/api/v1/worker/jobs/{job_id}/logs")
async def upload_logs(job_id: str, request: Request):
    if not require_worker_auth(request):
        return _unauthorized()
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
    """Push sanitized result to ci-results branch using OCI-only token."""
    token = GITHUB_RESULT_TOKEN
    pub_root = RELAY_ROOT / "results-publish"
    if not (pub_root / ".git").exists():
        subprocess.run(["git", "init", str(pub_root)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(pub_root), "checkout", "-b", "ci-results"], check=True, capture_output=True)
        # fetch remote branch if exists
        remote = repo_url_with_token(token)
        subprocess.run(["git", "-C", str(pub_root), "remote", "add", "origin", remote], check=True, capture_output=True)
        try:
            subprocess.run(["git", "-C", str(pub_root), "fetch", "origin", "ci-results"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(pub_root), "checkout", "-B", "ci-results", "origin/ci-results"], check=True, capture_output=True)
        except Exception:
            pass
    else:
        remote = repo_url_with_token(token)
        subprocess.run(["git", "-C", str(pub_root), "remote", "set-url", "origin", remote], check=True, capture_output=True)
        try:
            subprocess.run(["git", "-C", str(pub_root), "fetch", "origin", "ci-results"], check=True, capture_output=True)
            subprocess.run(["git", "-C", str(pub_root), "checkout", "-B", "ci-results", "origin/ci-results"], check=True, capture_output=True)
        except Exception:
            pass
    run_dir = pub_root / "runs" / commit_sha
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / job_id
    shutil.copy(result_path / "result.json", run_dir / "result.json")
    # REPORT.md and SHA256SUMS come from worker upload; copy if present.
    report = result_path / "REPORT.md"
    if report.exists():
        shutil.copy(report, run_dir / "REPORT.md")
    sums = result_path / "SHA256SUMS"
    if sums.exists():
        shutil.copy(sums, run_dir / "SHA256SUMS")
    (run_dir / "result.json").write_text(json.dumps(json.loads((run_dir / "result.json").read_text()), indent=2))
    subprocess.run(["git", "-C", str(pub_root), "add", "."], check=True, capture_output=True)
    try:
        subprocess.run(["git", "-C", str(pub_root), "commit", "-m", f"ci-result: {commit_sha} {verdict}"], check=True, capture_output=True)
    except Exception:
        pass  # nothing to commit
    subprocess.run(["git", "-C", str(pub_root), "push", "origin", "ci-results"], check=True, capture_output=True)


def repo_url_with_token(token: str) -> str:
    # If no PAT is present, use the OCI SSH deploy key (git@github.com). Token is never committed.
    if not token:
        return "git@github.com:GY-Bai/CB16-R10.git"
    parsed = urlparse(REPO_URL)
    return f"https://x-access-token:{token}@{parsed.netloc}{parsed.path}"
