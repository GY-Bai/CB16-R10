#!/usr/bin/env python3
"""Host-side fail-closed gate for CB16 Shanxi self-hosted GitHub runners.

Install a COPY of this file outside every repository checkout and configure it
with ACTIONS_RUNNER_HOOK_JOB_STARTED. It blocks any job before workflow steps
unless:
  1) the caller job depends on the canonical `shanxi-preflight` reusable job;
  2) that reusable job points at the canonical preflight workflow;
  3) the canonical preflight bytes match the pinned git blob below; and
  4) the consolidated Repo Guard, including action-runtime compatibility,
     already completed successfully in this exact GitHub Actions run.

This hook intentionally has no repository write capability.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request

EXPECTED_REPOSITORY = "GY-Bai/CB16-R10"
PREFLIGHT_JOB_ID = "shanxi-preflight"
PREFLIGHT_PATH = ".github/workflows/_cb16-shanxi-preflight.yml"
PREFLIGHT_USES = "./.github/workflows/_cb16-shanxi-preflight.yml"
PREFLIGHT_GIT_BLOB_SHA = "2c663a822e3480e9230f73e9c81409ce906d6a57"
REQUIRED_GATE_NAME = "Repo Guard"


def fail(code: str, detail: str = "") -> None:
    msg = f"CB16_SHANXI_PRE_JOB_GATE=FAIL {code}"
    if detail:
        msg += f" {detail}"
    print(msg, file=sys.stderr)
    raise SystemExit(97)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _job_blocks(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    jobs_at = next(
        (i for i, line in enumerate(lines) if line.strip() == "jobs:" and _indent(line) == 0),
        None,
    )
    if jobs_at is None:
        return {}

    starts: list[tuple[int, str]] = []
    job_re = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$")
    scope_end = len(lines)
    for i in range(jobs_at + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#") and _indent(line) == 0:
            scope_end = i
            break
        m = job_re.match(line)
        if m:
            starts.append((i, m.group(1)))

    out: dict[str, list[str]] = {}
    for n, (start, job_id) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else scope_end
        out[job_id] = lines[start:end]
    return out


def _field(block: list[str], key: str) -> list[str]:
    prefix = f"    {key}:"
    for i, line in enumerate(block):
        if line.startswith(prefix):
            base = _indent(line)
            end = len(block)
            for j in range(i + 1, len(block)):
                if block[j].strip() and not block[j].lstrip().startswith("#") and _indent(block[j]) <= base:
                    end = j
                    break
            return block[i:end]
    return []


def _http_text(url: str) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "cb16-shanxi-pre-job-gate-v1",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        fail("NETWORK_READ_FAILED", f"{url}:{exc}")


def _raw_at(repo: str, sha: str, path: str) -> str:
    safe_path = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return _http_text(f"https://raw.githubusercontent.com/{repo}/{sha}/{safe_path}")


def _git_blob_sha(text: str) -> str:
    data = text.encode("utf-8")
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _jobs(repo: str, run_id: str) -> list[dict]:
    text = _http_text(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    )
    try:
        doc = json.loads(text)
        jobs = doc["jobs"]
        if not isinstance(jobs, list):
            raise TypeError("jobs not list")
        return jobs
    except Exception as exc:
        fail("JOB_METADATA_INVALID", str(exc))


def _workflow_path(repo: str, workflow_ref: str) -> str:
    prefix = repo + "/"
    if not workflow_ref.startswith(prefix) or "@" not in workflow_ref:
        fail("WORKFLOW_REF_INVALID", workflow_ref)
    path = workflow_ref[len(prefix):].rsplit("@", 1)[0]
    if not path.startswith(".github/workflows/"):
        fail("WORKFLOW_PATH_INVALID", path)
    return path


def _gate_job(jobs: list[dict], gate_name: str) -> dict:
    matches = [
        j for j in jobs
        if j.get("name") == gate_name or str(j.get("name", "")).endswith(" / " + gate_name)
    ]
    if len(matches) != 1:
        fail("GATE_JOB_CARDINALITY", f"{gate_name}:{len(matches)}")
    job = matches[0]
    if job.get("status") != "completed" or job.get("conclusion") != "success":
        fail(
            "GATE_NOT_SUCCESS",
            f"{gate_name}:{job.get('status')}:{job.get('conclusion')}",
        )
    return job


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    job_id = os.environ.get("GITHUB_JOB", "")
    workflow_ref = os.environ.get("GITHUB_WORKFLOW_REF", "")
    workflow_sha = os.environ.get("GITHUB_WORKFLOW_SHA", "")

    if repo != EXPECTED_REPOSITORY:
        fail("REPOSITORY_MISMATCH", repo)
    if not run_id.isdigit():
        fail("RUN_ID_INVALID", run_id)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", workflow_sha):
        fail("WORKFLOW_SHA_INVALID", workflow_sha)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        fail("JOB_ID_INVALID", job_id)

    workflow_path = _workflow_path(repo, workflow_ref)
    caller_text = _raw_at(repo, workflow_sha, workflow_path)
    blocks = _job_blocks(caller_text)

    current = blocks.get(job_id)
    if current is None:
        fail("CURRENT_JOB_NOT_DIRECT_CALLER_JOB", job_id)

    needs = "\n".join(_field(current, "needs"))
    if PREFLIGHT_JOB_ID not in needs:
        fail("CURRENT_JOB_MISSING_PREFLIGHT_NEEDS", job_id)

    preflight_caller = blocks.get(PREFLIGHT_JOB_ID)
    if preflight_caller is None:
        fail("PREFLIGHT_CALLER_JOB_MISSING")
    uses = "\n".join(_field(preflight_caller, "uses"))
    if PREFLIGHT_USES not in uses:
        fail("PREFLIGHT_CALLER_NOT_CANONICAL", uses)

    preflight_text = _raw_at(repo, workflow_sha, PREFLIGHT_PATH)
    actual_blob = _git_blob_sha(preflight_text)
    if actual_blob != PREFLIGHT_GIT_BLOB_SHA:
        fail("PREFLIGHT_BLOB_MISMATCH", actual_blob)

    jobs = _jobs(repo, run_id)
    _gate_job(jobs, REQUIRED_GATE_NAME)

    print(
        "CB16_SHANXI_PRE_JOB_GATE=PASS "
        f"run_id={run_id} job={job_id} workflow={workflow_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
