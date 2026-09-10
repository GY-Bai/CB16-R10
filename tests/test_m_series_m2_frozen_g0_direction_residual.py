from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from cb16_local_opt.canonical_state_alignment_falsification_r41 import (
    R41_SCENARIOS,
    shuffle_reduced_targets_by_future_group_r41,
)
from cb16_local_opt.m_series_m2_frozen_g0_direction_residual import (
    M2_RESIDUAL_PARAMETER_COUNT,
    adjudicate_m2,
    build_direction_residual_m2,
    build_objective_target_m2,
    rotate_rich_direction_means_m2,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11


def _prepared_two_groups() -> tuple[PreparedEvidenceR11, np.ndarray]:
    rows = []
    parent_ids = []
    group_ids = []
    rich = []
    for g in range(2):
        for sidx, scenario in enumerate(R41_SCENARIOS):
            packed = np.zeros((107,), dtype=np.float32)
            packed[0] = float(g + 1)
            if g == 0:
                probs = np.asarray([0.70, 0.20, 0.10], dtype=np.float32)
                means = np.asarray([0.03, 0.01, -0.02], dtype=np.float64)
            else:
                probs = np.asarray([0.10, 0.20, 0.70], dtype=np.float32)
                means = np.asarray([-0.02, 0.01, 0.03], dtype=np.float64)
            packed[102:105] = probs
            packed[105] = 0.5
            packed[106] = 1.0
            rows.append(packed)
            parent_ids.append(f"P{g}:{scenario}")
            group_ids.append(f"G{g}")
            rich.append(means + sidx * 1e-6)
    tensor = torch.from_numpy(np.stack(rows))
    prepared = PreparedEvidenceR11(
        parent_ids=tuple(parent_ids),
        dependence_group_ids=tuple(group_ids),
        packed=tensor,
        evidence_hash="test",
        host_to_device_transfers=0,
    )
    prepared.validate()
    return prepared, np.stack(rich)


def test_residual_parameter_count_and_zero_output():
    residual = build_direction_residual_m2(device="cpu")
    assert sum(p.numel() for p in residual.parameters()) == M2_RESIDUAL_PARAMETER_COUNT
    x = torch.randn(7, 256)
    with torch.no_grad():
        delta = residual(x)
    assert torch.equal(delta, torch.zeros_like(delta))


def test_selective_target_preserves_g0_only_on_direction_agreement():
    teacher = torch.tensor(
        [
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ],
        dtype=torch.float32,
    )
    g0 = torch.tensor(
        [
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    target, receipt = build_objective_target_m2(
        objective="SELECTIVE_CHAMPION_PRESERVE_CE",
        teacher_probs=teacher,
        g0_probs=g0,
    )
    assert torch.equal(target[0], g0[0])
    assert torch.equal(target[1], teacher[1])
    assert torch.equal(target[2], g0[2])
    assert receipt["champion_preserve_rows"] == 2
    assert receipt["teacher_learning_rows"] == 1


def test_rich_surface_rotation_matches_reduced_target_mapping_and_multiset():
    prepared, rich = _prepared_two_groups()
    shuffled, receipt = shuffle_reduced_targets_by_future_group_r41(prepared, shift=1)
    rotated, audit = rotate_rich_direction_means_m2(
        rich_means=rich,
        prepared=prepared,
        shuffled_prepared=shuffled,
        shuffle_receipt=receipt,
    )
    assert audit["teacher_surface_multiset_preserved"] is True
    assert audit["same_donor_mapping_as_reduced_targets"] is True
    assert np.allclose(rotated[:6], rich[6:])
    assert np.allclose(rotated[6:], rich[:6])
    assert np.array_equal(
        np.argmax(rotated, axis=1),
        shuffled.direction_target_probs.numpy().argmax(axis=1),
    )


def _arm(gain: float, move: float, agree_change: float):
    return {
        "evaluation": {
            "discrete_champion_relative_gain": gain,
            "teacher_g0_disagreement_move_to_teacher_rate": move,
            "teacher_g0_agreement_change_rate": agree_change,
        }
    }


def test_adjudication_requires_alignment_specific_gain_and_preservation():
    folds = []
    for fold in range(1, 6):
        arms = {
            "SELECTIVE_ALIGNED": _arm(0.02, 0.70, 0.05),
            "ABS_CE_ALIGNED": _arm(0.015, 0.65, 0.15),
        }
        for shift in (1, 7, 13, 23, 31):
            arms[f"SELECTIVE_SHUFFLE_{shift}"] = _arm(0.001, 0.20, 0.10)
            arms[f"ABS_CE_SHUFFLE_{shift}"] = _arm(0.002, 0.25, 0.20)
        folds.append({"fold": fold, "arms": arms})
    out = adjudicate_m2(folds)
    assert out["selective_alignment_pass"] is True
    assert out["preservation_pass"] is True
    assert out["absolute_residual_alignment_secondary_pass"] is True
