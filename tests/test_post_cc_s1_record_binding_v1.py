"""CI-D record-binding verification, including the B7 stale-record counterexample."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.verify_r11_post_cc_s1_record_binding as binding

IMPL = "1" * 40
CI_HEAD = "2" * 40
BINDING_HEAD = "3" * 40
HEAD = "4" * 40
MANIFEST_PATH = "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1.json"


def _manifest_hashes():
    payload = Path(MANIFEST_PATH).read_bytes()
    return hashlib.sha256(payload).hexdigest(), json.loads(payload.decode("utf-8"))["manifest_sha256"]


def _record(*, binding_head=BINDING_HEAD, latest_ci_head=CI_HEAD, runtime=IMPL, stale_entry_head=None):
    file_sha, payload_sha = _manifest_hashes()
    entry_head = stale_entry_head or latest_ci_head
    return {
        "terminal_state": "READY_FOR_SOL_REVIEW",
        "implementation_candidate_sha": IMPL,
        "implementation_candidate_tree_sha": "a" * 40,
        "record_binding_head_sha": binding_head,
        "execution_manifest": {
            "path": MANIFEST_PATH,
            "sha256": file_sha,
            "manifest_payload_sha256": payload_sha,
        },
        "candidate_state": {
            "latest_ci_head_sha": latest_ci_head,
            "shanxi_ci": {
                "implementation_suite": {"run_id": 1, "job_id": 2, "runtime_candidate_sha": runtime, "checkout_sha": entry_head},
                "bounded_smoke": {"run_id": 3, "job_id": 4, "runtime_candidate_sha": runtime, "checkout_sha": entry_head},
                "record_binding_verification": {
                    "run_id": 5,
                    "job_id": 6,
                    "runtime_candidate_sha": runtime,
                    "verified_head_sha": entry_head,
                    "status": "PASS",
                } if entry_head else {},
            },
        },
    }


def _git_model(*, ancestor_pairs, diffs, show_file_fn=None):
    def is_ancestor(candidate, descendant):
        return candidate == descendant or (candidate, descendant) in ancestor_pairs

    def changed_paths(base, head):
        return diffs.get((base, head), [])

    def show_file(revision, relative_path):
        if show_file_fn is None:
            return json.dumps({"implementation_candidate_sha": IMPL})
        return show_file_fn(revision, relative_path)

    return is_ancestor, changed_paths, show_file


def test_fresh_record_binding_passes():
    record = _record()
    is_ancestor, changed_paths, show_file = _git_model(
        ancestor_pairs={(IMPL, HEAD), (CI_HEAD, HEAD), (BINDING_HEAD, HEAD), (CI_HEAD, BINDING_HEAD)},
        diffs={
            (IMPL, HEAD): [binding.CANDIDATE_PATH],
            (CI_HEAD, HEAD): [binding.CANDIDATE_PATH],
            (BINDING_HEAD, HEAD): [binding.CANDIDATE_PATH],
        },
    )
    result = binding.evaluate_record_binding_v1(
        record, head_sha=HEAD, head_tree="b" * 40, is_ancestor=is_ancestor, changed_paths=changed_paths, show_file=show_file
    )
    assert result["status"] == "PASS", [name for name, ok in result["checks"].items() if not ok]


def test_entry_checkout_head_may_be_metadata_only_ancestor_of_latest_ci_head():
    record = _record(latest_ci_head="5" * 40)
    record["candidate_state"]["shanxi_ci"]["implementation_suite"]["checkout_sha"] = CI_HEAD
    record["candidate_state"]["shanxi_ci"]["bounded_smoke"]["checkout_sha"] = CI_HEAD
    record["candidate_state"]["shanxi_ci"]["record_binding_verification"] = {}
    is_ancestor, changed_paths, show_file = _git_model(
        ancestor_pairs={
            (IMPL, HEAD),
            (CI_HEAD, "5" * 40),
            ("5" * 40, HEAD),
            (BINDING_HEAD, HEAD),
            (CI_HEAD, BINDING_HEAD),
            ("5" * 40, BINDING_HEAD),
        },
        diffs={
            (IMPL, HEAD): [binding.CANDIDATE_PATH],
            (CI_HEAD, "5" * 40): [binding.CANDIDATE_PATH],
            ("5" * 40, HEAD): [binding.CANDIDATE_PATH],
            (BINDING_HEAD, HEAD): [binding.CANDIDATE_PATH],
        },
    )
    result = binding.evaluate_record_binding_v1(
        record, head_sha=HEAD, head_tree="b" * 40, is_ancestor=is_ancestor, changed_paths=changed_paths, show_file=show_file
    )
    assert result["status"] == "PASS", [name for name, ok in result["checks"].items() if not ok]


def test_stale_record_binding_is_rejected():
    # newest CI evidence head is newer than the recorded binding head
    record = _record(binding_head=CI_HEAD, latest_ci_head=HEAD)
    is_ancestor, changed_paths, show_file = _git_model(
        ancestor_pairs={(IMPL, HEAD), (CI_HEAD, HEAD)},
        diffs={(IMPL, HEAD): [binding.CANDIDATE_PATH], (CI_HEAD, HEAD): [binding.CANDIDATE_PATH]},
    )
    result = binding.evaluate_record_binding_v1(
        record, head_sha=HEAD, head_tree="b" * 40, is_ancestor=is_ancestor, changed_paths=changed_paths, show_file=show_file
    )
    assert result["status"] == "CONTRACT_MISMATCH"
    assert result["checks"]["latest_ci_head_not_older_than_binding_head"] is False


def test_evidence_from_a_different_runtime_is_rejected():
    record = _record(runtime="9" * 40)
    is_ancestor, changed_paths, show_file = _git_model(
        ancestor_pairs={(IMPL, HEAD), (CI_HEAD, HEAD), (BINDING_HEAD, HEAD), (CI_HEAD, BINDING_HEAD)},
        diffs={
            (IMPL, HEAD): [binding.CANDIDATE_PATH],
            (CI_HEAD, HEAD): [binding.CANDIDATE_PATH],
            (BINDING_HEAD, HEAD): [binding.CANDIDATE_PATH],
        },
    )
    result = binding.evaluate_record_binding_v1(
        record, head_sha=HEAD, head_tree="b" * 40, is_ancestor=is_ancestor, changed_paths=changed_paths, show_file=show_file
    )
    assert result["status"] == "CONTRACT_MISMATCH"
    assert result["checks"]["implementation_suite_runtime_candidate_matches"] is False


def test_dirty_runtime_after_implementation_candidate_is_rejected():
    record = _record()
    is_ancestor, changed_paths, show_file = _git_model(
        ancestor_pairs={(IMPL, HEAD), (CI_HEAD, HEAD), (BINDING_HEAD, HEAD), (CI_HEAD, BINDING_HEAD)},
        diffs={
            (IMPL, HEAD): [binding.CANDIDATE_PATH, "cb16_local_opt/post_cc_s1_tasks_v1.py"],
            (CI_HEAD, HEAD): [binding.CANDIDATE_PATH],
            (BINDING_HEAD, HEAD): [binding.CANDIDATE_PATH],
        },
    )
    result = binding.evaluate_record_binding_v1(
        record, head_sha=HEAD, head_tree="b" * 40, is_ancestor=is_ancestor, changed_paths=changed_paths, show_file=show_file
    )
    assert result["status"] == "CONTRACT_MISMATCH"
    assert result["checks"]["only_review_metadata_changed_since_implementation_candidate"] is False


def test_record_binding_verification_passes_at_current_head():
    result = binding.verify_record_binding_v1(".")
    assert result["status"] == "PASS", [name for name, ok in result["checks"].items() if not ok]
