"""Verify that the S1 review-candidate record binds this exact repository head.

CI-D uses this before any deterministic scientific artifact reuse.  If the only
net change since the qualified runtime head is the reviewer-owned candidate
record (and PR metadata), the artifacts may be reused; otherwise the script
fails closed and a full re-qualification is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json"
ALLOWED_METADATA_ONLY_PATHS = {
    CANDIDATE_PATH,
}

EXIT_OK = 0
EXIT_BINDING_FAILED = 1


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def verify_record_binding_v1(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    path = root / CANDIDATE_PATH
    if not path.exists():
        raise SystemExit(json.dumps({"status": "CONTRACT_MISMATCH", "error": "REVIEW_CANDIDATE_MISSING"}))
    candidate = json.loads(path.read_text(encoding="utf-8"))
    head_sha = _git(root, "rev-parse", "HEAD")
    head_tree = _git(root, "rev-parse", "HEAD^{tree}")
    checks: dict[str, bool] = {}
    checks["terminal_state_ready_for_sol_review"] = candidate.get("terminal_state") == "READY_FOR_SOL_REVIEW"
    implementation_sha = candidate.get("implementation_candidate_sha")
    implementation_tree = candidate.get("implementation_candidate_tree_sha")
    implementation_is_ancestor = bool(
        implementation_sha
        and subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(implementation_sha), "HEAD"], cwd=root
        ).returncode
        == 0
    )
    checks["implementation_candidate_is_ancestor_of_head"] = implementation_is_ancestor
    checks["implementation_candidate_tree_sha_present"] = bool(
        isinstance(implementation_tree, str) and len(implementation_tree) == 40
    )
    changed_paths: list[str] = []
    metadata_only_change = False
    if implementation_is_ancestor:
        changed_paths = [
            line for line in _git(root, "diff", "--name-only", str(implementation_sha), "HEAD").splitlines() if line.strip()
        ]
        metadata_only_change = all(item in ALLOWED_METADATA_ONLY_PATHS for item in changed_paths)
    checks["only_review_metadata_changed_since_implementation_candidate"] = bool(metadata_only_change)
    manifest_entry = candidate.get("execution_manifest", {})
    manifest_path = root / str(manifest_entry.get("path", ""))
    checks["execution_manifest_file_present"] = manifest_path.exists()
    if manifest_path.exists():
        import hashlib

        checks["execution_manifest_file_sha256_matches"] = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_entry.get("sha256")
        )
        declared_payload_sha = manifest_entry.get("manifest_payload_sha256")
        checks["execution_manifest_payload_sha256_matches"] = bool(
            declared_payload_sha
            and json.loads(manifest_path.read_text(encoding="utf-8")).get("manifest_sha256") == declared_payload_sha
        )
    else:
        checks["execution_manifest_file_sha256_matches"] = False
        checks["execution_manifest_payload_sha256_matches"] = False
    state = candidate.get("candidate_state", {})
    implementation = state.get("implementation", {})
    checks["task_coverage_complete"] = bool(implementation.get("task_coverage_complete") is True)
    checks["shanxi_implementation_ci_present_or_explicitly_pending"] = bool(
        implementation.get("shanxi_ci_state") in ("PASS", "PENDING_SOL_REVIEW")
    )
    passed = all(checks.values())
    return {
        "schema": "CB16_R11_POST_CC_S1_RECORD_BINDING_VERIFICATION_V1",
        "status": "PASS" if passed else "CONTRACT_MISMATCH",
        "head_sha": head_sha,
        "head_tree_sha": head_tree,
        "candidate_path": CANDIDATE_PATH,
        "implementation_candidate_sha": implementation_sha,
        "changed_paths_since_implementation_candidate": changed_paths,
        "checks": checks,
        "reuse_of_qualified_artifacts_allowed": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = verify_record_binding_v1(args.repo_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    print(payload)
    return EXIT_OK if result["status"] == "PASS" else EXIT_BINDING_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
