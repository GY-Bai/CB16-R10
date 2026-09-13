"""Machine proof for the two scope claims serialized by qualification CI.

This is intentionally PR-scope specific, not a reusable semantic verifier.  The
workflow runs every ``test_scientific_qualification_*.py`` test before emitting
its validation receipt, so a PASS receipt is impossible if this diff firewall
finds frozen authority or S1 runtime changes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _qualification_pr_changed_paths() -> list[str]:
    # Fail closed if the canonical base ref is unavailable.  Exact-SHA Shanxi
    # checkout uses fetch-depth: 0, so origin/main is expected to exist.
    _git("rev-parse", "--verify", "origin/main")
    merge_base = _git("merge-base", "HEAD", "origin/main")
    return [
        line
        for line in _git("diff", "--name-only", merge_base, "HEAD").splitlines()
        if line.strip()
    ]


def test_receipt_scope_claims_are_machine_proved_by_git_diff_firewall():
    changed = _qualification_pr_changed_paths()

    science_authority_changes = [path for path in changed if path.startswith("authority/")]

    s1_runtime_changes = [
        path
        for path in changed
        if path.startswith("cb16_local_opt/post_cc_")
        or path == "scripts/run_r11_post_cc_s1_learnability.py"
        or path == "scripts/verify_r11_post_cc_s1_qualification_authorization.py"
        or path == "scripts/verify_r11_post_cc_s1_record_binding.py"
        or path == ".github/workflows/cb16-r11-post-cc-s1-learnability.yml"
    ]

    assert science_authority_changes == [], science_authority_changes
    assert s1_runtime_changes == [], s1_runtime_changes
