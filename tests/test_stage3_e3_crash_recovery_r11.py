from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cb16_local_opt.runtime_events_r11 import TournamentDecision
from cb16_local_opt.stage3_e3_crash_recovery_r11 import (
    BRANCH,
    CrashBoundary,
    SCHEMA,
    repository_guard,
    run_case,
)


def test_e3_boundary_catalog_is_exact_required_ten() -> None:
    assert [x.value for x in CrashBoundary] == [
        "01_BEFORE_SNAPSHOT_SEAL",
        "02_AFTER_SNAPSHOT_SEAL_BEFORE_TRAINING_START",
        "03_AFTER_TRAINING_COMPLETION_BEFORE_VALIDATION",
        "04_AFTER_VALIDATION_BEFORE_TOURNAMENT_DECISION",
        "05_AFTER_TOURNAMENT_DECISION_BEFORE_COMMIT",
        "06_AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL",
        "07_AFTER_CHECKPOINT_SEAL_BEFORE_NEXT_GENERATION_RELEASE",
        "08_DURING_EVIDENCE_STORAGE_WRITE",
        "09_DURING_JOURNAL_WRITE",
        "10_DURING_FINAL_DRAIN_BARRIER",
    ]
    assert SCHEMA == "CB16_R11_STAGE3_E3_CRASH_RECOVERY_REPORT_V1"
    assert BRANCH == "ai/r11-stage3-e3-crash-recovery-r0"


@pytest.mark.parametrize(
    "boundary,decision",
    [
        (CrashBoundary.BEFORE_SNAPSHOT_SEAL, TournamentDecision.PROMOTE),
        (CrashBoundary.AFTER_COMMIT_BEFORE_CHECKPOINT_SEAL, TournamentDecision.PROMOTE),
        (CrashBoundary.AFTER_CHECKPOINT_BEFORE_RELEASE, TournamentDecision.REJECT),
        (CrashBoundary.DURING_EVIDENCE_STORAGE_WRITE, TournamentDecision.PROMOTE),
        (CrashBoundary.DURING_JOURNAL_WRITE, TournamentDecision.REJECT),
        (CrashBoundary.DURING_FINAL_DRAIN_BARRIER, TournamentDecision.PROMOTE),
    ],
)
def test_representative_real_process_crash_recovery_cases(boundary, decision) -> None:
    with tempfile.TemporaryDirectory(prefix="cb16-e3-test-") as td:
        receipt = run_case(Path(td), boundary, decision)
    assert receipt["child_exit_code"] == 86
    assert receipt["pass"] is True, receipt
    assert receipt["failures"] == []
    assert receipt["final_authority"]["generation"] == 2
    assert receipt["audit_result"]["journal"]["pass"] is True
    assert receipt["audit_result"]["checkpoint"]["pass"] is True
    assert receipt["audit_result"]["evidence"]["pass"] is True
    assert receipt["stale_work_rejected"] is True


def test_storage_crash_replay_is_content_addressed_and_not_duplicated() -> None:
    with tempfile.TemporaryDirectory(prefix="cb16-e3-storage-") as td:
        receipt = run_case(
            Path(td),
            CrashBoundary.DURING_EVIDENCE_STORAGE_WRITE,
            TournamentDecision.REJECT,
        )
    assert receipt["pass"] is True, receipt
    replay = receipt["engineering_storage_replay"]
    assert replay["startup_recovered_payloads"] >= 1
    assert replay["exactly_one_payload"] is True
    assert replay["exactly_one_engineering_evidence_row"] is True
    assert replay["second_replay"]["created_payload_count"] == 0
    assert replay["second_replay"]["created_evidence_count"] == 0


def test_journal_crash_rolls_back_unsealed_transaction_then_replays_once() -> None:
    with tempfile.TemporaryDirectory(prefix="cb16-e3-journal-") as td:
        receipt = run_case(
            Path(td),
            CrashBoundary.DURING_JOURNAL_WRITE,
            TournamentDecision.PROMOTE,
        )
    assert receipt["pass"] is True, receipt
    replay = receipt["engineering_journal_replay"]
    assert replay["rows_after_crash_before_replay"] == 0
    assert replay["rows_after_two_replays"] == 1
    assert replay["second_replay_reused_batch"] is True


def test_repository_guard_is_fail_closed_on_current_e3_branch() -> None:
    guard = repository_guard(Path.cwd())
    assert guard["base_is_ancestor"] is True, guard
    assert guard["semantic_freeze_actual_blob"] == guard["semantic_freeze_expected_blob"], guard
    assert guard["forbidden_changed_paths"] == [], guard
    assert guard["unscoped_changed_paths"] == [], guard
    assert guard["pass"] is True, guard
