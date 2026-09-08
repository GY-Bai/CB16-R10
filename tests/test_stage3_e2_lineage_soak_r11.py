from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cb16_local_opt.stage3_e2_lineage_soak_r11 import run_lineage_soak


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stage3_e2_100_generations_real_authority_progression(tmp_path):
    report = run_lineage_soak(
        tmp_path / "e2-100",
        generations=100,
        repo_root=REPO_ROOT,
        synchronous="OFF",
    )

    assert report["verdict"] == "PASS", report["first_failure"]
    assert report["correctness_identity"]["status"] == "CORRECTNESS_PASS"
    assert report["runtime_endurance"]["status"] == "CONTROL_PLANE_RUNTIME_PASS"
    assert report["completed_generations"] == 100
    assert len(report["generation_ledger"]) == 100
    assert report["final_authority_identity"]["generation"] == 100
    assert report["decision_coverage"] == {
        "promote": 50,
        "reject": 50,
        "repeated_promote_exercised": True,
        "repeated_reject_exercised": True,
        "alternating_exercised": True,
    }
    assert report["journal_audit"]["pass"] is True
    assert report["runtime_endurance"]["restart_reconstruction_checks"] == 100
    assert report["runtime_endurance"]["attack_rejections"] == 400
    assert report["runtime_endurance"]["idempotent_duplicate_checks"] == 400


def test_stage3_e2_rejected_challenger_never_becomes_later_parent(tmp_path):
    report = run_lineage_soak(
        tmp_path / "e2-reject-lineage",
        generations=30,
        repo_root=REPO_ROOT,
        synchronous="OFF",
    )
    assert report["verdict"] == "PASS", report["first_failure"]

    rejected = set()
    previous_after = None
    for row in report["generation_ledger"]:
        if previous_after is not None:
            assert row["authoritative_champion_before"] == previous_after
        assert row["authoritative_champion_before"] not in rejected
        assert row["snapshot"]["parent_champion_id"] == row["authoritative_champion_before"]
        assert row["snapshot"]["generation"] == row["generation"]
        if row["decision"] == "REJECT":
            rejected.add(row["challenger_id"])
            assert row["authoritative_champion_after"] == row["authoritative_champion_before"]
        else:
            assert row["authoritative_champion_after"] == row["challenger_id"]
        previous_after = row["authoritative_champion_after"]


def test_stage3_e2_report_is_machine_readable_and_freeze_unchanged(tmp_path):
    report_path = tmp_path / "report.json"
    report = run_lineage_soak(
        tmp_path / "e2-report",
        generations=6,
        repo_root=REPO_ROOT,
        synchronous="OFF",
        report_path=report_path,
    )
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["schema"] == "CB16_R11_STAGE3_E2_LINEAGE_SOAK_REPORT_V1"
    assert loaded["verdict"] == "PASS"
    assert loaded["correctness_identity"]["semantic_freeze_sha256_before"]
    assert (
        loaded["correctness_identity"]["semantic_freeze_sha256_before"]
        == loaded["correctness_identity"]["semantic_freeze_sha256_after"]
    )
    assert loaded["repeated_frozen_replay_counts_as_new_scientific_evidence"] is False
    assert loaded["scientific_verdict_created"] is False
    assert loaded["final_holdout_opened"] is False
    assert loaded == report


@pytest.mark.skipif(
    os.environ.get("CB16_STAGE3_E2_RUN_1000") != "1",
    reason="Explicit opt-in logical/control-plane extension; no GPU training is used.",
)
def test_stage3_e2_1000_generation_extension(tmp_path):
    report = run_lineage_soak(
        tmp_path / "e2-1000",
        generations=1000,
        repo_root=REPO_ROOT,
        synchronous="OFF",
    )
    assert report["verdict"] == "PASS", report["first_failure"]
    assert report["completed_generations"] == 1000
    assert report["final_authority_identity"]["generation"] == 1000
    assert report["runtime_endurance"]["restart_reconstruction_checks"] == 1000
    assert report["journal_audit"]["pass"] is True
