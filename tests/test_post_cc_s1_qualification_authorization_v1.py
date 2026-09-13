"""B3: qualification authorization must bind runtime SHA/tree and manifest hash."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from cb16_local_opt.post_cc_s1_qualification_v1 import (
    S1QualificationError,
    require_qualification_authorization_v1,
)

RUNTIME_SHA = "a" * 40
RUNTIME_TREE = "b" * 40
MANIFEST_SHA = "c" * 64


def _write_authorization(root: Path, **overrides) -> None:
    payload = {
        "status": "READY_FOR_S1_QUALIFICATION",
        "reviewer_role": "SOL_INDEPENDENT_QUALIFICATION_REVIEW",
        "reviewed_implementation_sha": RUNTIME_SHA,
        "reviewed_implementation_tree_sha": RUNTIME_TREE,
        "execution_manifest_sha256": MANIFEST_SHA,
    }
    payload.update(overrides)
    path = root / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_V1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _authorize(root: Path):
    return require_qualification_authorization_v1(
        root,
        expected_runtime_sha=RUNTIME_SHA,
        expected_runtime_tree_sha=RUNTIME_TREE,
        expected_manifest_sha256=MANIFEST_SHA,
    )


def test_missing_authorization_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("cb16_local_opt.post_cc_s1_qualification_v1._git_changed_paths_v1", lambda *a, **k: [])
    with pytest.raises(S1QualificationError, match="MISSING"):
        _authorize(tmp_path)


def test_authorization_binding_mismatches_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr("cb16_local_opt.post_cc_s1_qualification_v1._git_changed_paths_v1", lambda *a, **k: [])
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})()
    )
    for override, match in (
        ({"status": "DRAFT"}, "STATUS"),
        ({"reviewer_role": "SOMEONE_ELSE"}, "ROLE"),
        ({"reviewed_implementation_sha": "d" * 40}, "SHA"),
        ({"reviewed_implementation_tree_sha": "e" * 40}, "TREE"),
        ({"execution_manifest_sha256": "f" * 64}, "MANIFEST"),
    ):
        _write_authorization(tmp_path, **override)
        with pytest.raises(S1QualificationError, match=match):
            _authorize(tmp_path)


def test_runtime_changed_after_review_fails_closed(tmp_path, monkeypatch):
    _write_authorization(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})()
    )
    monkeypatch.setattr(
        "cb16_local_opt.post_cc_s1_qualification_v1._git_changed_paths_v1",
        lambda *a, **k: ["cb16_local_opt/post_cc_s1_tasks_v1.py"],
    )
    with pytest.raises(S1QualificationError, match="CHANGED_AFTER_REVIEW"):
        _authorize(tmp_path)


def test_valid_authorization_with_metadata_only_diff_passes(tmp_path, monkeypatch):
    _write_authorization(tmp_path)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})()
    )
    monkeypatch.setattr(
        "cb16_local_opt.post_cc_s1_qualification_v1._git_changed_paths_v1",
        lambda *a, **k: ["authority/rearchitecture_r11/CB16_R11_POST_CC_S1_REVIEW_CANDIDATE_V1.json"],
    )
    payload = _authorize(tmp_path)
    assert payload["reviewed_implementation_sha"] == RUNTIME_SHA
