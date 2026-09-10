from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from cb16_local_opt.canonical_state_alignment_falsification_r41 import R41_SCENARIOS
from cb16_local_opt.teacher_failure_localization_h51 import (
    CONTROL_FAMILIES,
    adjudicate_h51,
    shuffled_target_feature_blocks_h51,
)


@dataclass(frozen=True)
class FakeParent:
    dependence_group_id: str
    scenario: str


@dataclass(frozen=True)
class FakeIndex:
    parent_ids: tuple[str, ...]
    parent_row_by_id: dict[str, int]
    features: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


def _fake_index():
    parent_ids = []
    parents = {}
    rows = []
    for g in range(2):
        gid = f"FUT:X:{g}"
        for si, scenario in enumerate(R41_SCENARIOS):
            pid = f"P:{g}:{scenario}"
            parent_ids.append(pid)
            parents[pid] = FakeParent(gid, scenario)
            x = np.zeros(102, dtype=np.float64)
            x[:48] = 10.0 + g
            x[48:96] = 20.0 + g
            x[96:] = 100.0 * g + si
            rows.append(x)
    parent_ids.append("TRAIN")
    rows.append(np.arange(102, dtype=np.float64))
    index = FakeIndex(
        parent_ids=tuple(parent_ids),
        parent_row_by_id={p: i for i, p in enumerate(parent_ids)},
        features=np.ascontiguousarray(np.stack(rows)),
    )
    eval_ids = tuple(parent_ids[:-1])
    return index, parents, eval_ids


def test_account_only_shuffle_preserves_market_and_train():
    index, parents, eval_ids = _fake_index()
    out, receipt = shuffled_target_feature_blocks_h51(
        index=index,
        eval_parents=parents,
        eval_parent_ids=eval_ids,
        shift=1,
        blocks=CONTROL_FAMILIES["ACCOUNT6_ONLY"],
    )
    eval_rows = np.asarray([index.parent_row_by_id[p] for p in eval_ids])
    assert np.array_equal(out.features[eval_rows, :96], index.features[eval_rows, :96])
    assert not np.array_equal(out.features[eval_rows, 96:], index.features[eval_rows, 96:])
    assert np.array_equal(out.features[-1], index.features[-1])
    assert receipt["selected_block_multiset_preserved"] is True
    assert receipt["unshuffled_blocks_byte_identical"] is True
    assert receipt["train_features_byte_identical"] is True
    assert receipt["scenario_identity_preserved"] is True
    assert receipt["fixed_points"] == 0


def _fold(fold: int, *, aligned=1.0, full=0.9, market=0.9, account=1.1, operator=0.9, medium=1.1):
    return {
        "fold": fold,
        "aligned": {"qscore": aligned, "central_qscore": 1.0, "tail_qscore": 2.0},
        "control_medians": {
            "FULL_STATE": {"qscore": full, "central_qscore": 2.0, "tail_qscore": 1.0},
            "MARKET96_ONLY": {"qscore": market, "central_qscore": 1.0, "tail_qscore": 1.0},
            "ACCOUNT6_ONLY": {"qscore": account, "central_qscore": 1.0, "tail_qscore": 1.0},
            "OPERATOR48_ONLY": {"qscore": operator, "central_qscore": 1.0, "tail_qscore": 1.0},
            "MEDIUM48_ONLY": {"qscore": medium, "central_qscore": 1.0, "tail_qscore": 1.0},
        },
        "support_geometry": {"guard": "PASS"},
    }


def test_adjudication_localizes_market_operator_and_tail():
    rows = [_fold(i) for i in range(1, 6)]
    out = adjudicate_h51(rows)
    assert out["primary_failure_axis"] == "MARKET_STATE_TRANSPORT_FAILURE"
    assert out["market_sublocalization"] == "OPERATOR48_PRIMARY"
    assert out["tail_calibration_primary"] is True
    assert out["specific_failure_axis_localized"] is True
    assert out["teacher_change_authorized"] is False
    assert out["student_change_authorized"] is False


def test_adjudication_broad_state_geometry():
    rows = [_fold(i, account=0.8, medium=0.8) for i in range(1, 6)]
    out = adjudicate_h51(rows)
    assert out["primary_failure_axis"] == "BROAD_STATE_GEOMETRY_FAILURE"
    assert out["market_sublocalization"] == "OPERATOR_MEDIUM_JOINT"


def test_adjudication_joint_interaction():
    rows = [_fold(i, market=1.1, account=1.1, full=0.9, operator=1.1, medium=1.1) for i in range(1, 6)]
    out = adjudicate_h51(rows)
    assert out["primary_failure_axis"] == "JOINT_MARKET_ACCOUNT_INTERACTION_FAILURE"
    assert out["market_sublocalization"] == "NOT_APPLICABLE"
