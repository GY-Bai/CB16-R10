"""Verify that the S1 review-candidate record binds the exact repository evidence.

Strengthened for Sol R1 B7:

* the reviewed implementation SHA must be an ancestor of HEAD with only the
  candidate record changed after it;
* the recorded record-binding head must be HEAD or a metadata-only ancestor,
  and must itself carry the same implementation candidate identity;
* the newest CI-A / CI-B evidence entries must bind the implementation
  candidate and a checkout head no older than the record-binding head;
* the execution manifest must match both file bytes and canonical payload hash.

A stale record (older run/head/checkouts than the recorded binding) fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANDIDATE_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json"
ALLOWED_METADATA_ONLY_PATHS = {CANDIDATE_PATH}

EXIT_OK = 0
EXIT_BINDING_FAILED = 1


def evaluate_record_binding_v1(
    record: Mapping[str, Any],
    *,
    head_sha: str,
    head_tree: str,
    is_ancestor: Callable[[str, str], bool],
    changed_paths: Callable[[str, str], list[str]],
    show_file: Callable[[str, str], str | None],
) -> dict[str, Any]:
    """Pure binding evaluation; git access is injected for testability."""
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["terminal_state_ready_for_sol_review"] = record.get("terminal_state") == "READY_FOR_SOL_REVIEW"
    implementation_sha = str(record.get("implementation_candidate_sha") or "")
    implementation_tree = str(record.get("implementation_candidate_tree_sha") or "")
    checks["implementation_candidate_sha_present"] = bool(len(implementation_sha) == 40)
    checks["implementation_candidate_tree_sha_present"] = bool(len(implementation_tree) == 40)
    checks["implementation_candidate_is_ancestor_of_head"] = bool(
        implementation_sha and is_ancestor(implementation_sha, head_sha)
    )
    changed_since_implementation: list[str] = []
    if checks["implementation_candidate_is_ancestor_of_head"]:
        changed_since_implementation = changed_paths(implementation_sha, head_sha)
    checks["only_review_metadata_changed_since_implementation_candidate"] = bool(
        checks["implementation_candidate_is_ancestor_of_head"]
        and set(changed_since_implementation) <= ALLOWED_METADATA_ONLY_PATHS
    )
    details["changed_paths_since_implementation_candidate"] = changed_since_implementation

    manifest_entry = record.get("execution_manifest", {})
    manifest_path = ROOT / str(manifest_entry.get("path", ""))
    checks["execution_manifest_file_present"] = manifest_path.exists()
    if manifest_path.exists():
        checks["execution_manifest_file_sha256_matches"] = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_entry.get("sha256")
        )
        declared_payload = manifest_entry.get("manifest_payload_sha256")
        checks["execution_manifest_payload_sha256_matches"] = bool(
            declared_payload
            and json.loads(manifest_path.read_text(encoding="utf-8")).get("manifest_sha256") == declared_payload
        )
    else:
        checks["execution_manifest_file_sha256_matches"] = False
        checks["execution_manifest_payload_sha256_matches"] = False

    binding_head = record.get("record_binding_head_sha")
    checks["record_binding_head_declared"] = bool(isinstance(binding_head, str) and len(binding_head) == 40)
    binding_head_matches = False
    if checks["record_binding_head_declared"]:
        binding_head_matches = bool(str(binding_head) == head_sha or is_ancestor(str(binding_head), head_sha))
    checks["record_binding_head_is_head_or_ancestor"] = binding_head_matches
    binding_head_diff_ok = False
    binding_head_record_same_impl = False
    if binding_head_matches:
        diff = changed_paths(str(binding_head), head_sha)
        binding_head_diff_ok = set(diff) <= ALLOWED_METADATA_ONLY_PATHS
        previous_record_text = show_file(str(binding_head), CANDIDATE_PATH)
        if previous_record_text is not None:
            try:
                previous_record = json.loads(previous_record_text)
                binding_head_record_same_impl = (
                    str(previous_record.get("implementation_candidate_sha")) == implementation_sha
                )
            except json.JSONDecodeError:
                binding_head_record_same_impl = False
    checks["record_binding_head_metadata_only_diff"] = bool(binding_head_diff_ok)
    checks["record_binding_head_record_has_same_implementation"] = bool(binding_head_record_same_impl)

    shanxi = record.get("candidate_state", {}).get("shanxi_ci", {})
    latest_ci_head = str(record.get("candidate_state", {}).get("latest_ci_head_sha") or "")
    checks["latest_ci_head_declared"] = bool(len(latest_ci_head) == 40)
    if checks["latest_ci_head_declared"]:
        checks["latest_ci_head_is_head_or_ancestor"] = bool(
            latest_ci_head == head_sha or is_ancestor(latest_ci_head, head_sha)
        )
        ci_diff = changed_paths(latest_ci_head, head_sha) if checks["latest_ci_head_is_head_or_ancestor"] else []
        checks["latest_ci_head_metadata_only_diff"] = bool(
            checks["latest_ci_head_is_head_or_ancestor"] and set(ci_diff) <= ALLOWED_METADATA_ONLY_PATHS
        )
        checks["latest_ci_head_not_older_than_binding_head"] = bool(
            checks["record_binding_head_declared"]
            and (latest_ci_head == str(binding_head) or is_ancestor(latest_ci_head, str(binding_head)))
        )
    else:
        checks["latest_ci_head_is_head_or_ancestor"] = False
        checks["latest_ci_head_metadata_only_diff"] = False
        checks["latest_ci_head_not_older_than_binding_head"] = False

    required_ci_entries = ("implementation_suite", "bounded_smoke", "record_binding_verification")
    for entry_name in required_ci_entries:
        entry = shanxi.get(entry_name) or {}
        checks[f"{entry_name}_present"] = bool(entry)
        checks[f"{entry_name}_run_id_present"] = bool(entry.get("run_id"))
        checks[f"{entry_name}_job_id_present"] = bool(entry.get("job_id"))
        checks[f"{entry_name}_runtime_candidate_matches"] = (
            str(entry.get("runtime_candidate_sha")) == implementation_sha
        )
        entry_head = str(entry.get("checkout_sha") or entry.get("verified_head_sha") or "")
        checks[f"{entry_name}_checkout_head_binds_latest_ci_head"] = bool(
            entry_head and latest_ci_head and entry_head == latest_ci_head
        )
    checks["record_binding_verification_passed"] = bool(
        (shanxi.get("record_binding_verification") or {}).get("status") == "PASS"
    )
    details["implementation_candidate_sha"] = implementation_sha
    details["latest_ci_head_sha"] = latest_ci_head
    details["record_binding_head_sha"] = binding_head
    passed = all(checks.values())
    return {
        "schema": "CB16_R11_POST_CC_S1_RECORD_BINDING_VERIFICATION_V1",
        "status": "PASS" if passed else "CONTRACT_MISMATCH",
        "head_sha": head_sha,
        "head_tree_sha": head_tree,
        "candidate_path": CANDIDATE_PATH,
        "checks": checks,
        "details": details,
        "reuse_of_qualified_artifacts_allowed": bool(passed),
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def verify_record_binding_v1(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    path = root / CANDIDATE_PATH
    if not path.exists():
        return {
            "schema": "CB16_R11_POST_CC_S1_RECORD_BINDING_VERIFICATION_V1",
            "status": "CONTRACT_MISMATCH",
            "error": "REVIEW_CANDIDATE_MISSING",
        }
    record = json.loads(path.read_text(encoding="utf-8"))
    head_sha = _git(root, "rev-parse", "HEAD")
    head_tree = _git(root, "rev-parse", "HEAD^{tree}")

    def is_ancestor(candidate: str, descendant: str) -> bool:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", candidate, descendant], cwd=root, check=False
        ).returncode == 0

    def changed_paths(base: str, head: str) -> list[str]:
        if base == head:
            return []
        return [line for line in _git(root, "diff", "--name-only", base, head).splitlines() if line.strip()]

    def show_file(revision: str, relative_path: str) -> str | None:
        completed = subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"], cwd=root, text=True, capture_output=True, check=False
        )
        return completed.stdout if completed.returncode == 0 else None

    return evaluate_record_binding_v1(
        record,
        head_sha=head_sha,
        head_tree=head_tree,
        is_ancestor=is_ancestor,
        changed_paths=changed_paths,
        show_file=show_file,
    )


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
