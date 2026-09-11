from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cb16_local_opt.binance_archive_input_r10 import KlineRecord, MINUTE_MS
from cb16_local_opt.longtraj_infra_closure_r0 import (
    MINUTE_H72_FORBIDDEN_CODE_R0,
    MinuteFrozenPhysicsAdapterR0,
    capture_rng_state_r0,
    load_exact_checkpoint_r0,
    minute_future_lineage_hash_r0,
    recursive_exact_equal_r0,
    recursive_state_sha256_r0,
    restore_exact_checkpoint_r0,
    restore_rng_state_r0,
    save_exact_checkpoint_r0,
    simulate_h72_minute_branch_r0,
    tensor_mapping_sha256_r0,
)


def rec(t: int, p: float) -> KlineRecord:
    return KlineRecord(
        open_time=t,
        open=p,
        high=p + 1.0,
        low=p - 1.0,
        close=p + 0.25,
        volume=10.0,
        close_time=t + MINUTE_MS - 1,
        quote_asset_volume=1000.0,
        number_of_trades=5,
        taker_buy_base_asset_volume=4.0,
        taker_buy_quote_asset_volume=400.0,
    )


def test_recursive_state_hash_and_exact_equality_are_deterministic():
    a = {
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "y": {"z": np.asarray([3, 4], dtype=np.int64)},
        "q": [1, "a", None],
    }
    b = {
        "q": [1, "a", None],
        "y": {"z": np.asarray([3, 4], dtype=np.int64)},
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    assert recursive_exact_equal_r0(a, b)
    assert recursive_state_sha256_r0(a) == recursive_state_sha256_r0(b)
    b["x"][0] = 9.0
    assert not recursive_exact_equal_r0(a, b)
    assert recursive_state_sha256_r0(a) != recursive_state_sha256_r0(b)


def test_rng_capture_restore_roundtrip_exact():
    import random

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    state = capture_rng_state_r0()

    expected = (
        [random.random() for _ in range(4)],
        np.random.random(4).tolist(),
        torch.rand(4).tolist(),
    )
    restore_rng_state_r0(state)
    got = (
        [random.random() for _ in range(4)],
        np.random.random(4).tolist(),
        torch.rand(4).tolist(),
    )
    assert got == expected


def test_checkpoint_restores_model_optimizer_runtime_and_lineage(tmp_path):
    torch.manual_seed(77)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = model(x).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model_hash = tensor_mapping_sha256_r0(model.state_dict())
    optimizer_hash = recursive_state_sha256_r0(optimizer.state_dict())
    path = tmp_path / "checkpoint.pt"
    save_exact_checkpoint_r0(
        path,
        model=model,
        optimizer=optimizer,
        runtime_step_index=17,
        cursor_ms=1_700_000_000_000,
        account_state={"equity": 1.25, "step": 99},
        evidence_hash="e" * 64,
        run_id="run",
        experiment_id="exp",
        spec_sha256="s" * 64,
        archive_identity={"symbol": "BTCUSDT", "hash": "a" * 64},
    )

    with torch.no_grad():
        for p in model.parameters():
            p.add_(100.0)
    optimizer.state.clear()

    runtime = SimpleNamespace(_step_index=0)
    payload = load_exact_checkpoint_r0(path)
    restore_exact_checkpoint_r0(
        payload,
        model=model,
        optimizer=optimizer,
        runtime=runtime,
        device="cpu",
    )

    assert tensor_mapping_sha256_r0(model.state_dict()) == model_hash
    assert recursive_state_sha256_r0(optimizer.state_dict()) == optimizer_hash
    assert runtime._step_index == 17
    assert payload["cursor_ms"] == 1_700_000_000_000
    assert payload["account_state"] == {"equity": 1.25, "step": 99}
    assert payload["evidence_hash"] == "e" * 64


def test_minute_bar_adapter_preserves_ohlcv_and_sets_one_minute_timeframe():
    r = rec(1_700_000_000_000, 100.0)
    bar = MinuteFrozenPhysicsAdapterR0._bar("BTCUSDT", r)
    assert bar["timeframe"] == "1m"
    assert bar["open"] == r.open
    assert bar["high"] == r.high
    assert bar["low"] == r.low
    assert bar["close"] == r.close
    assert bar["volume"] == r.volume
    assert bar["mark_price"] == r.close
    assert bar["index_price"] == r.close


def test_future_lineage_hash_binds_market_and_funding_suffix():
    t0 = 1_700_000_000_000
    rows = [rec(t0 + (i + 1) * MINUTE_MS, 100.0 + i) for i in range(3)]
    base = minute_future_lineage_hash_r0(
        symbol="BTCUSDT",
        parent_time_ms=t0,
        future_rows=rows,
        funding_by_minute={},
    )
    changed_market_rows = list(rows)
    changed_market_rows[-1] = rec(changed_market_rows[-1].open_time, 999.0)
    changed_market = minute_future_lineage_hash_r0(
        symbol="BTCUSDT",
        parent_time_ms=t0,
        future_rows=changed_market_rows,
        funding_by_minute={},
    )
    changed_funding = minute_future_lineage_hash_r0(
        symbol="BTCUSDT",
        parent_time_ms=t0,
        future_rows=rows,
        funding_by_minute={rows[-1].open_time: 0.0001},
    )
    assert base != changed_market
    assert base != changed_funding


def test_open_position_h72_on_one_minute_physics_fails_closed():
    with pytest.raises(RuntimeError, match=MINUTE_H72_FORBIDDEN_CODE_R0):
        simulate_h72_minute_branch_r0()
