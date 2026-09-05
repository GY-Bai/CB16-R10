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
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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

PROFILE_TIMEOUTS = {
    "smoke": 30 * 60,
    "unit": 2 * 60 * 60,
    "r102": 24 * 60 * 60,
    "r103": 72 * 60 * 60,
    "r104": 7 * 24 * 60 * 60,
    "v63": 72 * 60 * 60,
}
MAX_TIMEOUT_SECONDS = int(os.environ.get("CB16_MAX_JOB_TIMEOUT_SECONDS", 7 * 24 * 60 * 60))

for d in (WORKSPACES, RESULTS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def headers() -> dict[str, str]:
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
    r = requests.post(
        f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/claim",
        headers=headers(),
        json={"worker_id": WORKER_ID},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def heartbeat(job_id: str):
    try:
        requests.post(
            f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/heartbeat",
            headers=headers(),
            json={},
            timeout=20,
        )
    except Exception:
        pass


def download_bundle(job: dict, expected_sha256: str) -> Path:
    sha = job["commit_sha"]
    dest = CACHE_DIR / f"{sha}.tar.zst"
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected_sha256:
        return dest
    with requests.get(
        f"{RELAY_URL}/api/v1/worker/bundles/{sha}",
        headers=headers(),
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        actual = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256_MISMATCH:{sha}:expected={expected_sha256}:actual={actual}"
            )
        tmp.replace(dest)
    return dest


def extract_bundle(bundle: Path, job_id: str) -> Path:
    ws = WORKSPACES / job_id
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)
    if bundle.suffix == ".zst":
        try:
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            with bundle.open("rb") as fsrc, open(ws / "bundle.tar", "wb") as fdst:
                dctx.copy_stream(fsrc, fdst)
            with tarfile.open(ws / "bundle.tar", "r:") as tf:
                tf.extractall(ws, filter="data")
            (ws / "bundle.tar").unlink()
        except ImportError:
            subprocess.run(
                ["tar", "--zstd", "-xf", str(bundle), "-C", str(ws)],
                check=True,
            )
    else:
        with tarfile.open(bundle, "r:*") as tf:
            tf.extractall(ws, filter="data")
    return ws


def profile_timeout(profile: str) -> int:
    if profile not in PROFILE_TIMEOUTS:
        raise RuntimeError(f"UNKNOWN_CI_PROFILE:{profile}")
    env_name = f"CB16_TIMEOUT_{profile.upper()}_SECONDS"
    requested = int(os.environ.get(env_name, PROFILE_TIMEOUTS[profile]))
    if requested <= 0:
        raise RuntimeError(f"INVALID_TIMEOUT:{profile}:{requested}")
    return min(requested, MAX_TIMEOUT_SECONDS)


def write_evidence_hashes(ci_out: Path) -> None:
    lines = []
    for name in ("result.json", "REPORT.md"):
        path = ci_out / name
        if path.exists():
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    (ci_out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def write_synthetic_evidence(
    ci_out: Path,
    *,
    job: dict,
    verdict: str,
    error: str,
    started_at: str,
) -> dict:
    ci_out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "CB16_CI_RESULT_V1",
        "job_id": job["id"],
        "commit_sha": job["commit_sha"],
        "ci_profile": job.get("ci_profile", "smoke"),
        "verdict": verdict,
        "started_at": started_at,
        "finished_at": utc_now(),
        "tests_total": 1,
        "tests_pass": 0,
        "tests_fail": 1,
        "error": error[:4000],
        "model_artifacts": [],
    }
    report = (
        "# CB16 CI Report\n\n"
        f"- commit: {job['commit_sha']}\n"
        f"- profile: {job.get('ci_profile', 'smoke')}\n"
        f"- started: {started_at}\n\n"
        "FAIL worker_runtime\n\n"
        "## Result\n\n"
        f"- verdict: {verdict}\n"
        f"- error: {error[:2000]}\n"
    )
    (ci_out / "REPORT.md").write_text(report)
    (ci_out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    write_evidence_hashes(ci_out)
    return result


def validate_result(job: dict, result: dict) -> None:
    if result.get("job_id") != job["id"]:
        raise RuntimeError(
            f"RESULT_JOB_ID_MISMATCH:expected={job['id']}:actual={result.get('job_id')}"
        )
    if result.get("commit_sha") != job["commit_sha"]:
        raise RuntimeError(
            "RESULT_COMMIT_SHA_MISMATCH:"
            f"expected={job['commit_sha']}:actual={result.get('commit_sha')}"
        )
    expected_profile = job.get("ci_profile", "smoke")
    if result.get("ci_profile", expected_profile) != expected_profile:
        raise RuntimeError(
            f"RESULT_PROFILE_MISMATCH:expected={expected_profile}:"
            f"actual={result.get('ci_profile')}"
        )
    if result.get("verdict") not in {"PASS", "FAIL", "ERROR", "TIMEOUT"}:
        raise RuntimeError(f"INVALID_RESULT_VERDICT:{result.get('verdict')}")


def terminate_process_group(proc: subprocess.Popen, grace_seconds: int = 30) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=grace_seconds)
        return
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def run_provisioner(ws: Path, job: dict, env: dict, stdout_path: Path, stderr_path: Path) -> dict:
    """Run provision/scripts/provision_all.py if present. Returns resolved env dict."""
    provision_all = ws / "provision" / "scripts" / "provision_all.py"
    if not provision_all.exists():
        return {}
    profile = job.get("ci_profile", "smoke")
    job_id = job["id"]
    cmd = [sys.executable, str(provision_all), "--profile", profile, "--job-id", job_id, "--repo", str(ws)]
    with stdout_path.open("a") as out, stderr_path.open("a") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise RuntimeError(f"PROVISION_FAILED:profile={profile}:exit={proc.returncode}")
    prov_dir = Path(env.get("CB16_CI_WORKER_ROOT", "/data/cb16_ci")) / "jobs" / job_id
    prov_json = prov_dir / "provisioning.json"
    if not prov_json.exists():
        raise RuntimeError(f"PROVISION_FAILED:missing_provisioning_json:{prov_json}")
    import json as _json
    prov = _json.loads(prov_json.read_text())
    if prov.get("status") != "READY":
        raise RuntimeError(f"PROVISION_FAILED:status={prov.get('status')}")
    resolved_env = {}
    env_path = prov_dir / "resolved.env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line and "=" in line:
                k, v = line.split("=", 1)
                resolved_env[k] = v
    return resolved_env


def run_job(job: dict) -> dict:
    job_id = job["id"]
    profile = job.get("ci_profile", "smoke")
    timeout_seconds = profile_timeout(profile)
    bundle_sha = job.get("bundle_sha256")
    if not bundle_sha:
        raise RuntimeError("NO_BUNDLE_SHA")

    bundle = download_bundle(job, bundle_sha)
    ws = extract_bundle(bundle, job_id)
    ci_out = ws / "ci_output"
    stdout_path = RESULTS_DIR / f"{job_id}.stdout.log"
    stderr_path = RESULTS_DIR / f"{job_id}.stderr.log"
    started_at = utc_now()

    env = os.environ.copy()
    env["CI_JOB_ID"] = job_id
    env["CI_COMMIT_SHA"] = job["commit_sha"]
    env["CI_PROFILE"] = profile
    env["CI_OUT"] = str(ci_out)
    env["CI_PYTHON"] = sys.executable

    try:
        resolved_env = run_provisioner(ws, job, env, stdout_path, stderr_path)
        env.update(resolved_env)
        if resolved_env.get("CI_PYTHON"):
            env["CI_PYTHON"] = resolved_env["CI_PYTHON"]
    except Exception as e:
        result = write_synthetic_evidence(
            ci_out,
            job=job,
            verdict="ERROR",
            error=f"PROVISION_FAILED: {e}",
            started_at=started_at,
        )
        report = ci_out / "REPORT.md"
        sums = ci_out / "SHA256SUMS"
        upload_result(job_id, result, report, sums, stdout_path, stderr_path)
        shutil.rmtree(ws, ignore_errors=True)
        return result

    import threading

    stop = threading.Event()

    def beat():
        while not stop.wait(20):
            heartbeat(job_id)

    heartbeat(job_id)
    t = threading.Thread(target=beat, daemon=True)
    t.start()

    timed_out = False
    try:
        with stdout_path.open("a") as stdout_file, stderr_path.open("a") as stderr_file:
            proc = subprocess.Popen(
                ["bash", "ci/run_shanxi_ci.sh"],
                cwd=ws,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
                env=env,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process_group(proc)
    finally:
        stop.set()
        t.join(timeout=5)

    if timed_out:
        result = write_synthetic_evidence(
            ci_out,
            job=job,
            verdict="TIMEOUT",
            error=f"CI_TIMEOUT profile={profile} limit_seconds={timeout_seconds}",
            started_at=started_at,
        )
    else:
        result_path = ci_out / "result.json"
        if not result_path.exists():
            result = write_synthetic_evidence(
                ci_out,
                job=job,
                verdict="ERROR",
                error=f"NO_RESULT_JSON exit_code={proc.returncode}",
                started_at=started_at,
            )
        else:
            result = json.loads(result_path.read_text())
            validate_result(job, result)
            if proc.returncode != 0 and result.get("verdict") == "PASS":
                raise RuntimeError(
                    f"PASS_WITH_NONZERO_EXIT:{proc.returncode}:{job_id}"
                )
            # The script is canonical. Do not mutate result.json after hashing.
            write_evidence_hashes(ci_out)

    report = ci_out / "REPORT.md"
    sums = ci_out / "SHA256SUMS"
    upload_result(job_id, result, report, sums, stdout_path, stderr_path)
    shutil.rmtree(ws, ignore_errors=True)
    return result


def upload_result(
    job_id: str,
    result: dict,
    report: Path,
    sums: Path,
    stdout_path: Path,
    stderr_path: Path,
):
    files = {}
    try:
        if stdout_path.exists():
            files["stdout.log"] = ("stdout.log", stdout_path.open("rb"), "text/plain")
        if stderr_path.exists():
            files["stderr.log"] = ("stderr.log", stderr_path.open("rb"), "text/plain")
        if report.exists():
            files["REPORT.md"] = ("REPORT.md", report.open("rb"), "text/markdown")
        if sums.exists():
            files["SHA256SUMS"] = ("SHA256SUMS", sums.open("rb"), "text/plain")
        if files:
            r = requests.post(
                f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/logs",
                headers=headers(),
                files=files,
                timeout=300,
            )
            r.raise_for_status()
    finally:
        for f in files.values():
            try:
                f[1].close()
            except Exception:
                pass

    r = requests.post(
        f"{RELAY_URL}/api/v1/worker/jobs/{job_id}/result",
        headers=headers(),
        json=result,
        timeout=120,
    )
    r.raise_for_status()


def report_worker_error(job: dict, error: Exception) -> None:
    started_at = utc_now()
    scratch = RESULTS_DIR / f"{job['id']}.error-evidence"
    if scratch.exists():
        shutil.rmtree(scratch)
    result = write_synthetic_evidence(
        scratch,
        job=job,
        verdict="ERROR",
        error=f"{type(error).__name__}: {error}",
        started_at=started_at,
    )
    stdout_path = RESULTS_DIR / f"{job['id']}.stdout.log"
    stderr_path = RESULTS_DIR / f"{job['id']}.stderr.log"
    if not stderr_path.exists():
        stderr_path.write_text(f"{type(error).__name__}: {error}\n")
    upload_result(
        job["id"],
        result,
        scratch / "REPORT.md",
        scratch / "SHA256SUMS",
        stdout_path,
        stderr_path,
    )
    shutil.rmtree(scratch, ignore_errors=True)


def clock_canary() -> None:
    """Compare Shanxi UTC clock against the OCI/Cloudflare HTTP Date header."""

    def http_date(url: str, timeout: float = 15.0):
        try:
            r = requests.get(url, timeout=timeout)
            dt = parsedate_to_datetime(r.headers.get("Date", ""))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None

    local_ts = time.time()
    relay_ts = http_date(f"{RELAY_URL}/healthz")
    print(f"CLOCK_CANARY local_unix={int(local_ts)}", flush=True)
    if relay_ts is not None:
        skew = abs(local_ts - relay_ts)
        print(f"CLOCK_CANARY relay_unix={int(relay_ts)} skew={skew:.3f}s", flush=True)
        if skew > 30:
            print(f"CLOCK_SKEW_TOO_LARGE relay skew={skew:.3f}s > 30s", flush=True)

    if os.environ.get("CB16_CLOCK_CANARY_GITHUB", "0") == "1":
        gh_ts = http_date("https://api.github.com")
        if gh_ts is not None:
            skew = abs(local_ts - gh_ts)
            print(f"CLOCK_CANARY github_unix={int(gh_ts)} skew={skew:.3f}s", flush=True)
            if skew > 30:
                print(f"CLOCK_SKEW_TOO_LARGE github skew={skew:.3f}s > 30s", flush=True)


def main_loop():
    clock_canary()
    while True:
        job = None
        try:
            job = get_next_job()
            if not job:
                time.sleep(10)
                continue
            claim(job["id"])
            print(
                f"RUN {job['id']} {job['commit_sha']} profile={job.get('ci_profile', 'smoke')}",
                flush=True,
            )
            result = run_job(job)
            print(f"DONE {job['id']} {result.get('verdict')}", flush=True)
        except Exception as e:
            print(f"ERROR {type(e).__name__}: {e}", flush=True)
            if job is not None:
                try:
                    report_worker_error(job, e)
                except Exception as report_error:
                    print(
                        f"ERROR_REPORT_FAILED {type(report_error).__name__}: {report_error}",
                        flush=True,
                    )
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
