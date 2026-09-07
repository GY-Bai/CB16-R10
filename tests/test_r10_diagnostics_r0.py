from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cb16_diagnostics.model_probe import compare_states, semantic_sha256
from cb16_diagnostics.postrun import build_diagnostics, generation_census
from cb16_diagnostics.runtime_observer import RuntimeObserver


def _checkpoint(path: Path, state: dict[str, torch.Tensor], *, generation: int, role: str, parent: str | None = None) -> None:
    torch.save(
        {
            "schema": "TEST_CHECKPOINT",
            "generation": generation,
            "role": role,
            "parent_policy_semantic_sha256": parent,
            "state_dict": state,
        },
        path,
    )


def test_compare_states_detects_group_updates() -> None:
    before = {
        "operator_encoder.weight": torch.ones(2, 2),
        "direction_out.weight": torch.ones(3, 2),
    }
    after = {k: v.clone() for k, v in before.items()}
    after["operator_encoder.weight"][0, 0] += 0.1
    result = compare_states(before, after)
    assert result["compatible"] is True
    assert result["global"]["relative_delta_l2"] > 0
    assert result["groups"]["Operator Brain Stem"]["changed_count"] == 1
    assert result["groups"]["Direction Head"]["changed_count"] == 0


def test_postrun_diagnostics_are_read_only_and_hash_consistent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "diag"
    gd = run / "generations" / "G00"
    gd.mkdir(parents=True)

    parent = {
        "operator_encoder.weight": torch.ones(2, 2),
        "direction_out.weight": torch.ones(3, 2),
    }
    challenger = {k: v.clone() for k, v in parent.items()}
    challenger["operator_encoder.weight"][0, 0] += 0.1
    p_sha = semantic_sha256(parent)
    c_sha = semantic_sha256(challenger)

    start = tmp_path / "start.pt"
    _checkpoint(start, parent, generation=0, role="CHAMPION")
    _checkpoint(gd / "challenger.pt", challenger, generation=1, role="CHALLENGER", parent=p_sha)
    _checkpoint(gd / "champion_after.pt", challenger, generation=1, role="CHAMPION", parent=p_sha)

    (gd / "ON_POLICY_REAL_TRACE_RECEIPT.json").write_text(
        json.dumps({"trace_count": 2, "matured": 2}), encoding="utf-8"
    )
    (gd / "CHALLENGER_TRAINING_RECEIPT_G0.json").write_text(
        json.dumps({
            "parameter_l2_delta": 0.1,
            "gradient_group_norms_last_step": {"Operator Brain Stem": 1.0},
            "update_group_norms": {"Operator Brain Stem": 0.1},
        }),
        encoding="utf-8",
    )
    (gd / "GENERATION_RESULT.json").write_text(
        json.dumps({
            "parent_champion_semantic_sha256": p_sha,
            "challenger": {"semantic_sha256": c_sha},
            "champion_after": {"semantic_sha256": c_sha},
            "tournament": {
                "decision": "PROMOTE",
                "validation_loss_before": 1.0,
                "validation_loss_after": 0.9,
                "relative_improvement": 0.1,
            },
        }),
        encoding="utf-8",
    )
    (run / "FINAL_RESULT_R102.json").write_text(
        json.dumps({
            "final_status": "TEST_PASS",
            "mechanistic_pipeline_pass": True,
            "scientific_controls_status": "DIAGNOSTIC_ONLY",
            "final_champion_semantic_sha256": c_sha,
            "experience_lake_audit": {"pass": True},
            "integrity": {"test": True},
        }),
        encoding="utf-8",
    )

    before_files = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())
    summary = build_diagnostics(
        run_root=run,
        out_dir=out,
        runtime_diag_dir=None,
        start_checkpoint=start,
        source_authority_sha="f056ae6a0722e3e92d71793024a6e6d3fe9af003",
        requested=1,
    )
    after_files = sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file())

    assert before_files == after_files
    assert summary["status"] == "POSTRUN_DIAGNOSTICS_COMPLETE"
    assert summary["diagnostics_are_status_driving"] is False
    assert summary["final_holdout_2025_09_accessed"] is False
    model = json.loads((out / "MODEL_EVOLUTION.json").read_text(encoding="utf-8"))
    assert model["all_parent_hashes_match"] is True
    assert model["all_challenger_hashes_match"] is True
    assert model["all_champion_hashes_match"] is True


def test_generation_census_incomplete_snapshot(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "generations" / "G00").mkdir(parents=True)
    (run / "generations" / "G01").mkdir(parents=True)
    (run / "generations" / "G00" / "GENERATION_RESULT.json").write_text("{}", encoding="utf-8")
    census = generation_census(run, requested=3)
    assert census["present_generation_count"] == 2
    assert census["complete_generation_count"] == 1
    assert census["missing_generation_directories"] == [2]


def test_runtime_observer_refuses_output_inside_campaign(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="DIAGNOSTIC_OUTPUT_MUST_BE_OUTSIDE"):
        RuntimeObserver(run, run / "diagnostics")
