from __future__ import annotations

import torch

from cb16_local_opt.full_minute_historical_replay_r0 import (
    BRAIN_PARAMETER_TARGET,
    FINAL_HOLDOUT_START_MS,
    MINUTE_MS,
    ActionIntent,
    LOGICAL_PARAMETER_GROUPS,
    MinuteBar,
    PathDependentMinuteReplay,
    ReplayCentralBrain16M,
    brain_parameter_report,
    gradient_boundary_canary,
    iter_rolling_minute_samples,
)


def bars(stamps):
    out = []
    for i, ts in enumerate(stamps):
        p = 100.0 + i
        out.append(MinuteBar(ts, p, p + 1, p - 1, p + 0.5, 10 + i))
    return out


def test_stride_one_uses_every_eligible_minute():
    xs = bars([i * MINUTE_MS for i in range(10)])
    got = list(
        iter_rolling_minute_samples(
            "BTCUSDT", xs, lookback_minutes=3, final_holdout_start_ms=100 * MINUTE_MS
        )
    )
    # 10 rows, causal lookback=3, one next transition required => 7 decisions.
    assert len(got) == 7
    assert [x.decision_time_ms for x in got] == [i * MINUTE_MS for i in range(2, 9)]
    assert all(
        b.open_time <= s.decision_time_ms
        for s in got
        for b in s.market_window
    )
    assert all(s.next_bar.open_time == s.decision_time_ms + MINUTE_MS for s in got)


def test_real_gap_breaks_rolling_windows():
    xs = bars([0, MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS, 6 * MINUTE_MS, 7 * MINUTE_MS, 8 * MINUTE_MS, 9 * MINUTE_MS])
    got = list(
        iter_rolling_minute_samples(
            "SOLUSDT", xs, lookback_minutes=3, final_holdout_start_ms=100 * MINUTE_MS
        )
    )
    assert len(got) == 2
    assert [x.decision_time_ms for x in got] == [2 * MINUTE_MS, 8 * MINUTE_MS]
    assert [x.segment_id for x in got] == [0, 1]
    for s in got:
        stamps = [b.open_time for b in s.market_window] + [s.next_bar.open_time]
        assert all(b - a == MINUTE_MS for a, b in zip(stamps, stamps[1:]))


def test_final_boundary_is_never_buffered_or_used_as_feedback():
    cutoff = 5 * MINUTE_MS
    xs = bars([i * MINUTE_MS for i in range(8)])
    got = list(
        iter_rolling_minute_samples(
            "BTCUSDT", xs, lookback_minutes=2, final_holdout_start_ms=cutoff
        )
    )
    assert got
    assert max(s.decision_time_ms for s in got) == 3 * MINUTE_MS
    assert max(s.next_bar.open_time for s in got) == 4 * MINUTE_MS
    assert all(s.next_bar.open_time < cutoff for s in got)


def test_policy_observation_has_no_future_bar():
    sample = next(
        iter_rolling_minute_samples(
            "BTCUSDT", bars([0, MINUTE_MS, 2 * MINUTE_MS]), lookback_minutes=2,
            final_holdout_start_ms=100 * MINUTE_MS,
        )
    )
    obs = sample.causal_observation({"equity": 1.0})
    assert "next_bar" not in obs
    assert "future" not in obs
    assert obs["market_window"][-1].open_time == sample.decision_time_ms


def _run_replica(direction: str):
    xs = bars([i * MINUTE_MS for i in range(8)])
    samples = list(
        iter_rolling_minute_samples(
            "BTCUSDT", xs, lookback_minutes=2, final_holdout_start_ms=100 * MINUTE_MS
        )
    )
    init_calls = []

    def initial_account(replica_id, sample):
        init_calls.append((replica_id, sample.segment_id))
        return {"position": 0.0, "equity": 1000.0, "steps": 0}

    def policy(obs):
        assert "next_bar" not in obs
        return ActionIntent(direction, 0.0 if direction == "FLAT" else 0.5)

    def supervisor(intent, account):
        return intent

    def physics(account, action, current, nxt):
        sign = {"SHORT": -1.0, "FLAT": 0.0, "LONG": 1.0}[action.direction]
        new_position = account["position"] + sign * action.requested_risk
        pnl = new_position * (nxt.close - current.close)
        return {
            "position": new_position,
            "equity": account["equity"] + pnl,
            "steps": account["steps"] + 1,
        }

    runner = PathDependentMinuteReplay(
        policy=policy,
        supervisor=supervisor,
        physics_step=physics,
        initial_account_factory=initial_account,
    )
    transitions = list(runner.run(samples, replica_id=direction))
    return transitions, init_calls


def test_account_is_history_integral_not_window_local_reset():
    long_path, init_calls = _run_replica("LONG")
    assert len(init_calls) == 1, "account must initialize once, not once per rolling window"
    assert [x.account_before["steps"] for x in long_path] == list(range(len(long_path)))
    assert [x.account_after["steps"] for x in long_path] == list(range(1, len(long_path) + 1))
    assert long_path[-1].account_after["position"] == 0.5 * len(long_path)


def test_same_market_future_different_action_paths_have_different_accounts_but_same_dependence_key():
    long_path, _ = _run_replica("LONG")
    short_path, _ = _run_replica("SHORT")
    assert len(long_path) == len(short_path)
    assert [x.dependence_key for x in long_path] == [x.dependence_key for x in short_path]
    assert long_path[-1].account_before != short_path[-1].account_before
    assert long_path[-1].account_after != short_path[-1].account_after


def test_gap_is_explicit_episode_reset_not_per_window_reset():
    xs = bars([0, MINUTE_MS, 2 * MINUTE_MS, 3 * MINUTE_MS, 8 * MINUTE_MS, 9 * MINUTE_MS, 10 * MINUTE_MS, 11 * MINUTE_MS])
    samples = list(
        iter_rolling_minute_samples(
            "SOLUSDT", xs, lookback_minutes=2, final_holdout_start_ms=100 * MINUTE_MS
        )
    )
    init_calls = []

    runner = PathDependentMinuteReplay(
        policy=lambda obs: ActionIntent("FLAT", 0.0),
        supervisor=lambda action, account: action,
        physics_step=lambda account, action, current, nxt: {"steps": account["steps"] + 1},
        initial_account_factory=lambda rid, s: init_calls.append(s.segment_id) or {"steps": 0},
    )
    out = list(runner.run(samples, replica_id="r0"))
    assert out
    assert init_calls == [0, 1]


def test_16m_brain_parameter_and_ownership_contract_exact():
    torch.manual_seed(1)
    model = ReplayCentralBrain16M()
    report = brain_parameter_report(model)
    assert report["total"] == BRAIN_PARAMETER_TARGET == 16_116_420
    assert report["trainable"] == BRAIN_PARAMETER_TARGET
    assert report["unknown_parameter_names"] == []
    assert set(LOGICAL_PARAMETER_GROUPS) == {
        "operator_brain_stem",
        "medium_brain_stem",
        "account_brain_stem",
        "shared_decision_core",
        "direction_head",
        "requested_risk_head",
    }
    assert all(report[group] > 0 for group in LOGICAL_PARAMETER_GROUPS)
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_external_typed_inputs_block_gradient_and_all_six_brain_groups_receive_gradient():
    torch.manual_seed(2)
    model = ReplayCentralBrain16M()
    receipt = gradient_boundary_canary(model, batch=4)
    assert receipt["external_input_gradients_blocked"] is True
    assert receipt["all_six_groups_receive_gradient"] is True


def test_deterministic_forward_replay_for_same_checkpoint_and_inputs():
    torch.manual_seed(3)
    model = ReplayCentralBrain16M().eval()
    op = torch.randn(2, 48)
    med = torch.randn(2, 48)
    acc = torch.randn(2, 6)
    with torch.no_grad():
        a = model(op, med, acc)
        b = model(op, med, acc)
    assert torch.equal(a["direction_logits"], b["direction_logits"])
    assert torch.equal(a["requested_risk_raw"], b["requested_risk_raw"])
    assert torch.equal(a["requested_risk"], b["requested_risk"])


def test_frozen_final_constant_is_exact():
    assert FINAL_HOLDOUT_START_MS == 1_756_684_800_000
