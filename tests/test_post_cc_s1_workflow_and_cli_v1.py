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


def test_runner_fails_closed_without_qualification_authorization(tmp_path):
    from cb16_local_opt.post_cc_s1_qualification_v1 import (
        S1QualificationError,
        run_s1_program_v1,
    )

    try:
        run_s1_program_v1(repo_root=".", mode="qualification", output_root=tmp_path, workers=1)
    except S1QualificationError as exc:
        assert "AUTHORIZATION" in str(exc)
    else:  # pragma: no cover - the authorization file must not exist in Phase A
        raise AssertionError("qualification must be blocked without reviewer authorization")


def test_artifact_writer_emits_required_s1_tree(tmp_path):
    from cb16_local_opt.post_cc_s1_qualification_v1 import _write_artifacts_v1
    from cb16_local_opt.post_cc_s1_execution_manifest_v1 import build_s1_execution_manifest_v1

    manifest = build_s1_execution_manifest_v1()
    result = {
        "evidence_class": "SMOKE_ONLY_ENGINEERING_EVIDENCE",
        "decisions_consumed": 16,
        "checkpoints": {
            "INITIAL": {"mean_complete_sample_arithmetic_return": 0.0, "contexts": {}},
            "FINAL": {"mean_complete_sample_arithmetic_return": 0.1, "contexts": {}},
        },
        "oracle": {"mean_oracle_return": 0.18, "contexts": {}},
        "unit_evidence": [
            {
                "materialization_manifest_sha256s": ["a" * 64],
                "child_checkpoint_sha256": "b" * 64,
            }
        ],
        "unit_evidence_sha256s": ["c" * 64],
        "firewall": {"FINAL_opened": False},
    }
    outcome = {
        "status": "OK",
        "job": {"task_id": "DELAYED_CONSEQUENCE_CREDIT", "seed": 1701, "control_id": None, "run_root": str(tmp_path / "run")},
        "result": result,
    }
    _write_artifacts_v1(
        output_root=tmp_path / "artifacts/post_cc_s1",
        manifest=manifest,
        outcomes=[outcome],
        integrity={"attacks": {}, "all_rejected": True},
        objective_audit={"all_checks_pass": True},
        fabricated_audit={"all_checks_pass": True},
        mode="smoke",
        compiled=None,
    )
    root = tmp_path / "artifacts/post_cc_s1"
    for relative in (
        "SMOKE_RESULT.json",
        "S1_REPORT.md",
        "execution_manifest.json",
        "task_results/DELAYED_CONSEQUENCE_CREDIT/1701.json",
        "provenance/task_specs.json",
        "provenance/run_index.json",
        "checkpoints/index.json",
        "replay_manifests/index.json",
    ):
        assert (root / relative).exists(), relative
