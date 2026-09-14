"""Frozen S1 execution-manifest contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import pytest

from cb16_local_opt.post_cc_s1_execution_manifest_v1 import (
    EXPECTED_AUTHORITY_BLOB_SHA1_V1,
    EXPECTED_MODEL_SOURCE_BLOB_SHA1_V1,
    S1ManifestError,
    build_s1_execution_manifest_v1,
    execution_manifest_payload_v1,
    git_blob_sha1_v1,
    validate_s1_execution_manifest_v1,
)
from cb16_local_opt.post_cc_s1_tasks_v1 import (
    ACTOR_LEARNING_RATE_V1,
    CRITIC_LEARNING_RATE_V1,
    FROZEN_SEEDS_V1,
    build_task_specs_v1,
)


def test_committed_manifest_matches_code_and_frozen_authority():
    manifest = validate_s1_execution_manifest_v1(".")
    built = build_s1_execution_manifest_v1()
    assert manifest == built
    assert manifest["seeds"] == list(FROZEN_SEEDS_V1)
    assert manifest["minimum_positive_seeds_passing"] == 4
    assert manifest["maximum_false_positive_control_seeds"] == 1
    assert manifest["gap_reduction_threshold"] == pytest.approx(0.5)
    assert manifest["model"]["optimizer"]["actor"]["learning_rate"] == pytest.approx(ACTOR_LEARNING_RATE_V1)
    assert manifest["model"]["optimizer"]["critic"]["learning_rate"] == pytest.approx(CRITIC_LEARNING_RATE_V1)
    assert manifest["firewall"]["FINAL_opened"] is False
    assert manifest["firewall"]["historical_market_corpus_accessed"] is False
    assert manifest["manifest_sha256"] == build_s1_execution_manifest_v1()["manifest_sha256"]


def test_manifest_task_entries_are_hashed_from_task_specs():
    manifest = build_s1_execution_manifest_v1()
    specs = build_task_specs_v1()
    assert [task["task_id"] for task in manifest["tasks"]] == list(specs)
    for task in manifest["tasks"]:
        assert task["spec_hash"] == specs[task["task_id"]].spec_hash


def test_manifest_hash_changes_when_payload_changes():
    payload = execution_manifest_payload_v1()
    import hashlib

    from cb16_local_opt.post_cc_s1_execution_manifest_v1 import canonical_json_bytes

    first = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    payload2 = dict(payload)
    payload2["gap_reduction_threshold"] = 0.6
    second = hashlib.sha256(canonical_json_bytes(payload2)).hexdigest()
    assert first != second


def test_model_and_authority_blob_bindings_match_repository():
    root = Path(".")
    for relative, expected in {**EXPECTED_MODEL_SOURCE_BLOB_SHA1_V1, **{
        "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json": EXPECTED_AUTHORITY_BLOB_SHA1_V1["task_registry"],
        "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json": EXPECTED_AUTHORITY_BLOB_SHA1_V1["run_spec"],
    }}.items():
        assert git_blob_sha1_v1(root / relative) == expected


def test_tampered_manifest_fails_closed():
    with tempfile.TemporaryDirectory(prefix="cb16-s1-manifest-tamper-") as temp_root:
        temp = Path(temp_root)
        (temp / "authority").mkdir()
        (temp / "cb16_local_opt").mkdir()
        shutil.copytree(Path("authority/rearchitecture_r11"), temp / "authority/rearchitecture_r11")
        committed = json.loads(
            Path("authority/rearchitecture_r11/CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1.json").read_text(
                encoding="utf-8"
            )
        )
        committed["gap_reduction_threshold"] = 0.4
        (temp / "authority/rearchitecture_r11/CB16_R11_POST_CC_S1_EXECUTION_MANIFEST_V1.json").write_text(
            json.dumps(committed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(S1ManifestError):
            validate_s1_execution_manifest_v1(temp)
