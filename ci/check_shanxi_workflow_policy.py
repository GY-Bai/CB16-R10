#!/usr/bin/env python3
"""Fail-closed GitHub Actions topology policy for CB16 Shanxi self-hosted runners.

Any job that could target a CB16 self-hosted/Shanxi runner must depend on the
canonical reusable preflight. The preflight itself runs A=Repo Guard followed
by B=Node24 Guard on GitHub-hosted runners.

Permissions follow GitHub Actions inheritance semantics: workflow-level
permissions are the default for jobs, while a job-level permissions block
replaces that default for the job. Therefore a sensitive job is accepted when
its *effective* contents permission is read-only, whether supplied at workflow
scope or explicitly at job scope.

This is repository-side defense in depth. The hard host-side boundary is the
ACTIONS_RUNNER_HOOK_JOB_STARTED hook from ci/shanxi_runner_pre_job_gate.py.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PREFLIGHT_JOB_ID = "shanxi-preflight"
PREFLIGHT_USES = "./.github/workflows/_cb16-shanxi-preflight.yml"
SENSITIVE_RUNNER_MARKERS = (
    "self-hosted",
    "shanxi",
    "cb16-r10-canonical",
    "cb16-wss-qualification",
)
MUTATION_PATTERNS = (
    re.compile(r"\bgit\s+push\b"),
    re.compile(r"\bgh\s+pr\s+(?:create|merge|close|reopen)\b"),
    re.compile(r"\bgh\s+api\b[^\n]*(?:-X|--method)\s+(?:POST|PUT|PATCH|DELETE)\b", re.I),
)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _job_blocks(text: str) -> dict[str, list[str]]:
    lines = text.splitlines()
    jobs_at = None
    for i, line in enumerate(lines):
        if line.strip() == "jobs:" and _indent(line) == 0:
            jobs_at = i
            break
    if jobs_at is None:
        return {}

    starts: list[tuple[int, str]] = []
    job_re = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(?:#.*)?$")
    for i in range(jobs_at + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.lstrip().startswith("#") and _indent(line) == 0:
            break
        m = job_re.match(line)
        if m:
            starts.append((i, m.group(1)))

    out: dict[str, list[str]] = {}
    for n, (start, job_id) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        while end > start and end <= len(lines):
            if n + 1 < len(starts):
                break
            probe = None
            for j in range(start + 1, end):
                if lines[j].strip() and not lines[j].lstrip().startswith("#") and _indent(lines[j]) == 0:
                    probe = j
                    break
            if probe is not None:
                end = probe
            break
        out[job_id] = lines[start:end]
    return out


def _field(block: list[str], key: str) -> list[str]:
    """Return one job-level YAML field and its nested lines."""
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


def _workflow_field(text: str, key: str) -> list[str]:
    """Return one top-level YAML field and its nested lines."""
    lines = text.splitlines()
    prefix = f"{key}:"
    for i, line in enumerate(lines):
        if line.startswith(prefix) and _indent(line) == 0:
            base = 0
            end = len(lines)
            for j in range(i + 1, len(lines)):
                if lines[j].strip() and not lines[j].lstrip().startswith("#") and _indent(lines[j]) <= base:
                    end = j
                    break
            return lines[i:end]
    return []


def _permissions_contents_read_only(field: list[str]) -> bool:
    """True when a permissions field grants contents read and not contents write.

    Supports the canonical mapping form and GitHub's ``permissions: read-all``
    shorthand. Job-level permissions replace workflow defaults, so callers must
    decide which field is effective before invoking this helper.
    """
    if not field:
        return False
    text = "\n".join(field)
    first = field[0].split("#", 1)[0].strip().lower()
    if first == "permissions: read-all":
        return True
    if first == "permissions: write-all":
        return False
    has_read = bool(re.search(r"(?m)^\s+contents:\s*read\s*(?:#.*)?$", text))
    has_write = bool(re.search(r"(?m)^\s+contents:\s*write\s*(?:#.*)?$", text))
    return has_read and not has_write


def _effective_contents_read_only(text: str, block: list[str]) -> bool:
    """Resolve permissions using GitHub workflow/job inheritance semantics."""
    job_permissions = _field(block, "permissions")
    if job_permissions:
        return _permissions_contents_read_only(job_permissions)
    return _permissions_contents_read_only(_workflow_field(text, "permissions"))


def _contains_sensitive_runner(block: list[str]) -> tuple[bool, bool]:
    field = _field(block, "runs-on")
    if not field:
        return False, False
    joined = "\n".join(field)
    dynamic = "${{" in joined
    sensitive = any(marker.lower() in joined.lower() for marker in SENSITIVE_RUNNER_MARKERS)
    return sensitive, dynamic


def _needs_preflight(block: list[str]) -> bool:
    field = _field(block, "needs")
    return PREFLIGHT_JOB_ID in "\n".join(field)


def _checkout_persists_credentials(block: list[str]) -> bool:
    """True if any checkout step lacks explicit persist-credentials:false."""
    lines = block
    for i, line in enumerate(lines):
        if "uses:" not in line or "actions/checkout@" not in line:
            continue
        step_indent = _indent(line)
        end = len(lines)
        for j in range(i + 1, len(lines)):
            stripped = lines[j].lstrip()
            if stripped.startswith("- ") and _indent(lines[j]) <= step_indent:
                end = j
                break
        step = "\n".join(lines[i:end])
        if not re.search(r"(?m)^\s+persist-credentials:\s*false\s*(?:#.*)?$", step):
            return True
    return False


def check_workflows(root: Path) -> list[str]:
    errors: list[str] = []
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return ["missing .github/workflows"]

    for path in sorted((*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml"))):
        text = path.read_text(encoding="utf-8")
        blocks = _job_blocks(text)
        if not blocks:
            continue

        preflight = blocks.get(PREFLIGHT_JOB_ID)
        sensitive_jobs: list[tuple[str, list[str]]] = []

        for job_id, block in blocks.items():
            sensitive, dynamic = _contains_sensitive_runner(block)
            if dynamic:
                errors.append(f"{path}: job {job_id}: dynamic runs-on is forbidden by fail-closed runner policy")
            if sensitive:
                sensitive_jobs.append((job_id, block))

        if not sensitive_jobs:
            continue

        if preflight is None:
            errors.append(f"{path}: missing {PREFLIGHT_JOB_ID} reusable preflight job")
        else:
            uses = "\n".join(_field(preflight, "uses"))
            if PREFLIGHT_USES not in uses:
                errors.append(
                    f"{path}: {PREFLIGHT_JOB_ID} must use exactly {PREFLIGHT_USES}"
                )

        for job_id, block in sensitive_jobs:
            if job_id == PREFLIGHT_JOB_ID:
                errors.append(f"{path}: preflight job itself must never target a self-hosted runner")
                continue
            if not _needs_preflight(block):
                errors.append(f"{path}: job {job_id}: missing needs: {PREFLIGHT_JOB_ID}")
            if not _effective_contents_read_only(text, block):
                errors.append(
                    f"{path}: job {job_id}: effective permissions.contents must be read-only "
                    "(workflow default or job-level override)"
                )
            if _checkout_persists_credentials(block):
                errors.append(
                    f"{path}: job {job_id}: every checkout must set persist-credentials: false"
                )
            job_text = "\n".join(block)
            for pat in MUTATION_PATTERNS:
                if pat.search(job_text):
                    errors.append(
                        f"{path}: job {job_id}: repository mutation command forbidden on self-hosted runner"
                    )
                    break

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    errors = check_workflows(args.root.resolve())
    if errors:
        print("SHANXI_WORKFLOW_GATE_POLICY=FAIL")
        for e in errors:
            print(e)
        return 1
    print("SHANXI_WORKFLOW_GATE_POLICY=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
