from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import torch

from cb16_local_opt.actor_critic_contract_r0 import ACTOR_CRITIC_SCIENCE_CONTRACT_R0
from cb16_local_opt.r102_common import H72, HOUR_MS
from cb16_local_opt.r102_physics import (
    LONG,
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    simulate_h72_branch,
)
from cb16_local_opt.training_runtime_r11 import (
    TIER1_PARAMETER_COUNT_R11,
    R11TrainingConfig,
    assert_tier1_fp32_runtime_r11,
    group_weights_r11,
)
from cb16_local_opt.typed_central_brain_r10 import (
    build_g0_brain_r10,
    parameter_report_r10,
)


ROOT = Path(__file__).resolve().parents[1]

# These are Git blob identities of the frozen legacy surfaces at the AC-003
# authority boundary.  A semantic edit must therefore be an explicit legacy
# requalification event rather than silently riding inside the Actor-Critic lane.
FROZEN_LEGACY_BLOBS_R0 = {
    "cb16_local_opt/typed_central_brain_r10.py": "899065a9b9c015fd5a0e24fc919a79ff7d3a67ff",
    "cb16_local_opt/r102_physics.py": "a9b3696dfa04bf8e5f72bff96f6480397407232d",
    "cb16_local_opt/training_runtime_r11.py": "56c227800d7a5c82ef88c691bb69d7550aee8b39",
    "tests/test_r11_trace_runtime.py": "d35bc1e2e10ceeb1c936522764bbf7d15739d341",
}

LANE_SEPARATION_R0 = {
    "LEGACY_R11": (
        "cb16_local_opt.typed_central_brain_r10",
        "cb16_local_opt.r102_physics",
        "cb16_local_opt.training_runtime_r11",
    ),
    "ACTOR_CRITIC_R0": ("cb16_local_opt.actor_critic_contract_r0",),
}


def _git_blob(path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"HEAD:{path}"],
        cwd=ROOT,
        text=True,
    ).strip()


def _market_arrays(rows: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = 1609459200000
    ts = np.arange(rows, dtype=np.int64) * HOUR_MS + start
    base = 30000.0 + np.arange(rows, dtype=np.float64) * 2.0
    o = base
    c = base + 1.0
    h = np.maximum(o, c) * 1.001
    l = np.minimum(o, c) * 0.999
    v = np.full(rows, 100.0)
    bars = np.stack([o, h, l, c, v], axis=1).astype(np.float32)
    funding = np.zeros(rows, dtype=np.float64)
    funding[::8] = 0.0001
    return ts, bars, funding


def test_frozen_legacy_source_blobs_are_unchanged() -> None:
    actual = {path: _git_blob(path) for path in FROZEN_LEGACY_BLOBS_R0}
    assert actual == FROZEN_LEGACY_BLOBS_R0


def test_legacy_brain_and_historical_training_surface_remain_deterministic() -> None:
    model_a = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    model_b = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")

    assert_tier1_fp32_runtime_r11(model_a)
    assert_tier1_fp32_runtime_r11(model_b)
    R11TrainingConfig().validate()

    report = parameter_report_r10(model_a)
    assert report["total"] == TIER1_PARAMETER_COUNT_R11 == 189_052
    assert report["trainable"] == TIER1_PARAMETER_COUNT_R11
    np.testing.assert_allclose(
        group_weights_r11(("A", "A", "B")),
        np.asarray([0.75, 0.75, 1.5], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    )

    operator48 = torch.zeros((2, 48), dtype=torch.float32)
    medium48 = torch.zeros((2, 48), dtype=torch.float32)
    account6 = torch.zeros((2, 6), dtype=torch.float32)
    out_a = model_a(operator48, medium48, account6)
    out_b = model_b(operator48, medium48, account6)

    for key in ("shared", "direction_logits", "direction_probs", "requested_risk_raw"):
        assert torch.equal(out_a[key], out_b[key])
    assert torch.equal(
        model_a.compose_action(out_a)["direction"],
        model_b.compose_action(out_b)["direction"],
    )


def test_legacy_h72_path_runs_with_frozen_deterministic_semantics() -> None:
    runtime = FrozenPhysicsRuntimeR102.load(ROOT)
    ts, bars, funding = _market_arrays()
    decision_time_ms = int(ts[96])
    parent = build_parent_scenarios(
        runtime,
        symbol="BTCUSDT",
        decision_time_ms=decision_time_ms,
        hourly_ts=ts,
        hourly_ohlcv=bars,
        funding=funding,
        prehistory_hours=96,
    )[0]

    branch_a = simulate_h72_branch(
        runtime,
        parent=parent,
        symbol="BTCUSDT",
        decision_time_ms=decision_time_ms,
        candidate_direction_v55=LONG,
        candidate_risk=0.25,
        hourly_ts=ts,
        hourly_ohlcv=bars,
        funding=funding,
    )
    branch_b = simulate_h72_branch(
        runtime,
        parent=parent,
        symbol="BTCUSDT",
        decision_time_ms=decision_time_ms,
        candidate_direction_v55=LONG,
        candidate_risk=0.25,
        hourly_ts=ts,
        hourly_ohlcv=bars,
        funding=funding,
    )

    assert branch_a == branch_b
    assert branch_a["status"] == "MATURED"
    assert len(branch_a["supervisor_decisions"]) == H72
    assert branch_a["first_step"]["intent"]["direction"] == LONG
    assert branch_a["first_step"]["intent"]["requested_risk_multiplier"] == 0.25


def test_historical_and_actor_critic_lanes_are_explicitly_separate() -> None:
    assert set(LANE_SEPARATION_R0) == {"LEGACY_R11", "ACTOR_CRITIC_R0"}
    assert set(LANE_SEPARATION_R0["LEGACY_R11"]).isdisjoint(
        LANE_SEPARATION_R0["ACTOR_CRITIC_R0"]
    )
    assert (
        ACTOR_CRITIC_SCIENCE_CONTRACT_R0.science_semantic_version
        == "CB16_R11_ACTOR_CRITIC_SCIENCE_R0"
    )
