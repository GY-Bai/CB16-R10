from __future__ import annotations

import numpy as np

from cb16_local_opt.binance_archive_input_r10 import KlineRecord, MINUTE_MS
from cb16_local_opt.full_minute_long_trajectory_r1 import (
    batch_axis_parity_canary_r1,
    causal_long_trajectory_scan_r1,
    expand_internal_halts_r1,
    make_checkpoint_r1,
)


def rec(t: int, close: float, *, volume: float = 1.0) -> KlineRecord:
    return KlineRecord(
        open_time=t,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
        close_time=t + MINUTE_MS - 1,
        quote_asset_volume=volume * close,
        number_of_trades=1 if volume else 0,
        taker_buy_base_asset_volume=volume / 2,
        taker_buy_quote_asset_volume=volume * close / 2,
    )


def test_internal_gap_is_prefix_price_halt_and_reopen_untouched():
    t0 = 1_700_000_000_000
    out = list(expand_internal_halts_r1([
        rec(t0, 100.0, volume=7.0),
        rec(t0 + 3 * MINUTE_MS, 115.0, volume=9.0),
    ]))
    assert len(out) == 4
    assert [x.halt_imputed for x in out] == [False, True, True, False]
    for x in out[1:3]:
        r = x.record
        assert (r.open, r.high, r.low, r.close) == (100.0, 100.0, 100.0, 100.0)
        assert r.volume == 0.0
        assert r.quote_asset_volume == 0.0
        assert r.number_of_trades == 0
        assert r.taker_buy_base_asset_volume == 0.0
        assert r.taker_buy_quote_asset_volume == 0.0
    assert out[-1].record.close == 115.0
    assert out[-1].record.volume == 9.0


def test_reopening_future_price_cannot_rewrite_prior_halt_minutes():
    t0 = 1_700_000_000_000
    a = list(expand_internal_halts_r1([rec(t0, 100.0), rec(t0 + 4 * MINUTE_MS, 101.0)]))
    b = list(expand_internal_halts_r1([rec(t0, 100.0), rec(t0 + 4 * MINUTE_MS, 9999.0)]))
    assert [(x.record.open_time, x.record.ohlcv().tolist()) for x in a[:-1]] == [
        (x.record.open_time, x.record.ohlcv().tolist()) for x in b[:-1]
    ]
    assert a[-1].record.close != b[-1].record.close


def test_no_prelisting_or_open_ended_fill():
    t0 = 1_700_000_000_000
    out = list(expand_internal_halts_r1([rec(t0, 10.0), rec(t0 + MINUTE_MS, 11.0)]))
    assert out[0].timestamp == t0
    assert out[-1].timestamp == t0 + MINUTE_MS
    assert len(out) == 2


def _run(prices):
    t0 = 1_700_000_000_000
    minutes = [
        x for x in expand_internal_halts_r1([
            rec(t0 + i * MINUTE_MS, float(p)) for i, p in enumerate(prices)
        ])
    ]
    seen = []

    def policy(obs, account, t):
        # Deliberately depend on everything visible, but only visible prefix is provided.
        s = sum(row[3] for row in obs) + float(account[0])
        action = 1 if int(s * 1000) % 2 else -1
        seen.append((t, tuple(obs), tuple(account), action))
        return action

    def transition(account, action, nxt, decision_time):
        # Next price is target/environment side only; policy has already returned.
        last_equity, last_px = account
        ret = nxt.record.close / last_px - 1.0
        return (last_equity * (1.0 + 0.1 * action * ret), nxt.record.close)

    final, events = causal_long_trajectory_scan_r1(
        minutes,
        lookback=3,
        initial_account=(1.0, prices[0]),
        policy=policy,
        transition=transition,
        stop_before_ms=t0 + len(prices) * MINUTE_MS,
    )
    return final, events, seen


def test_future_suffix_mutation_does_not_change_earlier_decisions_or_accounts():
    _, ev_a, seen_a = _run([100, 101, 99, 102, 103, 104, 105, 106])
    _, ev_b, seen_b = _run([100, 101, 99, 102, 103, 999, 777, 555])
    # Decisions through t=4 use no mutated future payload.
    prefix = 3
    assert seen_a[:prefix] == seen_b[:prefix]
    assert [e.action_hash for e in ev_a[:prefix]] == [e.action_hash for e in ev_b[:prefix]]
    assert [e.account_before_hash for e in ev_a[:prefix]] == [e.account_before_hash for e in ev_b[:prefix]]
    assert all(e.observation_max_time_ms == e.decision_time_ms for e in ev_a)
    assert all(e.transition_time_ms == e.decision_time_ms + MINUTE_MS for e in ev_a)


def test_account_is_not_reset_across_halt_minutes():
    t0 = 1_700_000_000_000
    minutes = list(expand_internal_halts_r1([
        rec(t0, 100.0),
        rec(t0 + MINUTE_MS, 101.0),
        rec(t0 + 4 * MINUTE_MS, 110.0),
        rec(t0 + 5 * MINUTE_MS, 111.0),
    ]))

    def policy(obs, account, t):
        return 0

    def transition(account, action, nxt, decision_time):
        # Count every elapsed minute. Halt does not create a new account.
        return {"counter": account["counter"] + 1, "identity": account["identity"]}

    final, events = causal_long_trajectory_scan_r1(
        minutes,
        lookback=1,
        initial_account={"counter": 0, "identity": "ONE_ACCOUNT"},
        policy=policy,
        transition=transition,
        stop_before_ms=t0 + 10 * MINUTE_MS,
    )
    assert final["identity"] == "ONE_ACCOUNT"
    assert final["counter"] == len(events)
    assert any(e.transition_halt_imputed for e in events)


def test_checkpoint_hash_is_exact_for_identical_prefix():
    _, events, _ = _run([100, 101, 102, 103, 104, 105])
    cp1 = make_checkpoint_r1(next_decision_time_ms=123, account_state={"x": 1}, events=events[:2])
    cp2 = make_checkpoint_r1(next_decision_time_ms=123, account_state={"x": 1}, events=events[:2])
    assert cp1 == cp2
    assert cp1.prefix_event_count == 2


def test_batch_axis_parity_known_answer():
    r = batch_axis_parity_canary_r1(batch=128)
    assert r["status"] == "PASS"
    assert r["exact_array_equal"] is True
    assert r["max_abs_diff"] == 0.0
