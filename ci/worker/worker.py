#!/usr/bin/env python3
"""CB16 CI Worker for Shanxi GPU node.

Polls Japan OCI relay through Cloudflare Tunnel.
Uses:
  - Cloudflare Service Token (CF-Access-Client-Id / CF-Access-Client-Secret)
  - CB16 application bearer token
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import requests

WORKER_ROOT = Path(os.environ.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci"))
WORKSPACES = WORKER_ROOT / "workspaces"
RESULTS_DIR = WORKER_ROOT / "results"
CACHE_DIR = WORKER_ROOT / "cache"
RELAY_URL = os.environ["CB16_RELAY_URL"].rstrip("/")
WORKER_ID = os.environ.get("CB16_WORKER_ID", f"shanxi-{uuid.uuid4().hex[:8]}")
CB16_TOKEN = os.environ["CB16_WORKER_TOKEN"]
CF_CLIENT_ID = os.environ.get("CF_ACCESS_CLIENT_ID", "")
CF_CLIENT_SECRET = os.environ.get("CF_ACCESS_CLIENT_SECRET", "")

for d in (WORKSPACES, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def headers() -> dict:
    h = {"Authorization": f"Bearer {CB16_TOKEN}"}
    if CF_CLIENT_ID and CF_CLIENT_SECRET:
        h["CF-Access-Client-Id"] = CF_CLIENT_ID
        h["CF-Access-Client-Secret"] = CF_CLIENT_SECRET
    return h


def get_next_job():
    r = requests.get(f"{RELAY_URL}/api/v1/worker/jobs/next", headers=headers(), timeout=30)
    r.raise_for_status()
    return r.json().get("job")


def claim(job_id: str):
    r = requests.post(f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/claim", headers=headers(), json={"worker_id": WORKER_ID}, timeout=30)
    r.raise_for_status()
    return r.json()


def heartbeat(job_id: str):
    try:
        requests.post(f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/heartbeat", headers=headers(), json={}, timeout=20)
    except Exception:
        pass


def download_bundle(job: dict, expected_sha256: str) -> Path:
    sha = job["commit_sha"]
    dest = CACHE_DIR / f"{sha}.tar.zst"
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected_sha256:
        return dest
    with requests.get(f"{RELAY_URL}/api/v1/worker/bundles/{sha}", headers=headers(), stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"SHA256_MISMATCH:{sha}:expected={expected_sha256}:actual={actual}")
        tmp.replace(dest)
    return dest


def extract_bundle(bundle: Path, job_id: str) -> Path:
    ws = WORKSPACES / job_id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    if bundle.suffix == ".zst":
        # zstd tar
        import zstandard  # optional; fallback to tar with --zstd if system zstd available
        try:
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with bundle.open("rb") as fsrc, open(ws / "bundle.tar", "wb") as fdst:
                dctx.copy_stream(fsrc, fdst)
            with tarfile.open(ws / "bundle.tar", "r:") as tf:
                tf.extractall(ws, filter="data")
            (ws / "bundle.tar").unlink()
        except ImportError:
            subprocess.run(["tar", "--zstd", "-xf", str(bundle), "-C", str(ws)], check=True)
    else:
        with tarfile.open(bundle, "r:*") as tf:
            tf.extractall(ws, filter="data")
    # If repo root is a single top dir, chdir into it by convention? CI script assumes repo root is current dir.
    # We archive top-level entries directly, so the root is ws.
    return ws


def run_job(job: dict) -> dict:
    job_id = job["id"]
    bundle_sha = job.get("bundle_sha256")
    if not bundle_sha:
        # Webhook creates bundle asynchronously? For v1 require bundle already built by relay before claiming.
        raise RuntimeError("NO_BUNDLE_SHA")
    bundle = download_bundle(job, bundle_sha)
    ws = extract_bundle(bundle, job_id)
    stdout_path = RESULTS_DIR / f"{job_id}.stdout.log"
    stderr_path = RESULTS_DIR / f"{job_id}.stderr.log"
    started = time.time()
    env = os.environ.copy()
    env["CI_JOB_ID"] = job_id
    env["CI_COMMIT_SHA"] = job["commit_sha"]
    env["CI_OUT"] = str(ws / "ci_output")
    env["CI_PYTHON"] = sys.executable
    proc = subprocess.Popen(
        ["bash", "ci/run_shanxi_ci.sh"],
        cwd=ws,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    # heartbeat thread would be better; do simple polling loop
    import threading

    stop = threading.Event()

    def beat():
        while not stop.is_set():
            heartbeat(job_id)
            time.sleep(20)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        out, err = proc.communicate(timeout=3600 * 6)
    finally:
        stop.set()
    stdout_path.write_text(out)
    stderr_path.write_text(err)
    exit_code = proc.returncode
    result_path = ws / "ci_output" / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {"verdict": "FAIL", "error": "no ci_output/result.json", "exit_code": exit_code}
    result["job_id"] = job_id
    result["commit_sha"] = job["commit_sha"]
    result["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["tests_total"] = result.get("tests_total", 0)
    result["tests_pass"] = result.get("tests_pass", 0)
    result["tests_fail"] = result.get("tests_fail", 0)
    result["verdict"] = "PASS" if exit_code == 0 and result.get("verdict") != "FAIL" else "FAIL"
    # Upload logs/report/result to relay.
    report = ws / "ci_output" / "REPORT.md"
    sums = ws / "ci_output" / "SHA256SUMS"
    upload_result(job_id, result, report, sums, stdout_path, stderr_path)
    # Cleanup workspace after successful upload
    shutil.rmtree(ws, ignore_errors=True)
    return result


def upload_result(job_id: str, result: dict, report: Path, sums: Path, stdout_path: Path, stderr_path: Path):
    # Upload full logs/report/sums to OCI first.
    try:
        files = {}
        if stdout_path.exists():
            files["stdout.log"] = ("stdout.log", stdout_path.open("rb"), "text/plain")
        if stderr_path.exists():
            files["stderr.log"] = ("stderr.log", stderr_path.open("rb"), "text/plain")
        if report.exists():
            files["REPORT.md"] = ("REPORT.md", report.open("rb"), "text/markdown")
        if sums.exists():
            files["SHA256SUMS"] = ("SHA256SUMS", sums.open("rb"), "text/plain")
        if files:
            r = requests.post(f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/logs", headers=headers(), files=files, timeout=120)
            r.raise_for_status()
    finally:
        for f in files.values():
            try: f[1].close()
            except Exception: pass
    r = requests.post(f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/result", headers=headers(), json=result, timeout=60)
    r.raise_for_status()


def main_loop():
    while True:
        try:
            job = get_next_job()
            if not job:
                time.sleep(10)
                continue
            claim(job["id"])
            print(f"RUN {job['id']} {job['commit_sha']}", flush=True)
            result = run_job(job)
            print(f"DONE {job['id']} {result.get('verdict')}", flush=True)
        except Exception as e:
            print(f"ERROR {e}", flush=True)
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
