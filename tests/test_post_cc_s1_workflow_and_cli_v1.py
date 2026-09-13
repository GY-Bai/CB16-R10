"""Workflow topology and CLI no-override contract tests for S1."""

from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/cb16-r11-post-cc-s1-learnability.yml")
RUNNER_SCRIPT = Path("scripts/run_r11_post_cc_s1_learnability.py")


def test_workflow_uses_shared_preflight_and_shanxi_runner():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/_cb16-shanxi-preflight.yml" in text
    assert "runs-on: [self-hosted, shanxi-docker-r11]" in text
    assert "needs: shanxi-preflight" in text
    assert text.count("persist-credentials: false") >= 4
    assert "timeout-minutes:" in text
    assert "contents: read" in text


def test_qualification_is_gated_by_sol_authorization_and_has_no_scientific_overrides():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "inputs.mode == 'qualification'" in text
    assert "CB16_R11_POST_CC_S1_QUALIFICATION_AUTHORIZATION_V1.json" in text
    assert "READY_FOR_S1_QUALIFICATION" in text
    for forbidden in ("seed:", "threshold:", "learning_rate:", "reward:", "budget:"):
        assert forbidden not in text
    assert "--mode qualification" in text


def test_runner_cli_exposes_only_non_scientific_execution_knobs():
    text = RUNNER_SCRIPT.read_text(encoding="utf-8")
    assert 'choices=("smoke", "qualification")' in text
    for forbidden in ("--seed", "--threshold", "--reward", "--learning-rate", "--model", "--budget"):
        assert forbidden not in text
    assert "--output-root" in text and "--workers" in text


def test_smoke_evidence_is_explicitly_not_scientific_qualification():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "bounded smoke" in text.lower()
    assert "engineering" in text.lower()
    qualification_job = text.split("s1-formal-qualification:")[1]
    assert "qualification" in qualification_job
    smoke_job = text.split("s1-bounded-smoke:")[1].split("s1-formal-qualification:")[0]
    assert "--workers 1" in smoke_job
    assert "S1_SMOKE_IS_ENGINEERING_EVIDENCE_ONLY_NOT_SCIENTIFIC_QUALIFICATION" in smoke_job
