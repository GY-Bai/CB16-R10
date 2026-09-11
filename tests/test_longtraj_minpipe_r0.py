from __future__ import annotations

from copy import deepcopy

from cb16_local_opt.binance_archive_input_r10 import KlineRecord, MINUTE_MS
from cb16_local_opt.longtraj_minpipe_r0 import (
    make_frozen_physics_scalar_transition_r0,
    prepare_lane_from_observed_r0,
    time_major_batch_scan_r0,
)


def _bar(t: int, p: float, v: float = 1.0) -> KlineRecord:
    return KlineRecord(
        open_time=t,
        open=p,
        high=p,
        low=p,
        close=p,
        volume=v,
        close_time=t + MINUTE_MS - 1,
        quote_asset_volume=v * p,
        number_of_trades=1 if v else 0,
        taker_buy_base_asset_volume=v / 2.0,
        taker_buy_quote_asset_volume=v * p / 2.0,
    )


def _transition(account, action, current, nxt, lane_id):
    # Environment may see nxt only after action is frozen.  Account evolution is
    # deliberately simple so exact lane parity is transparent in this infra test.
    out = deepcopy(account)
    out["steps"] += 1
    out["score"] += float(action["x"])
    out["last_transition_time"] = int(nxt.record.open_time)
    out["lane"] = lane_id
    return out


def test_time_major_batches_lanes_but_not_time_and_freezes_actions_before_future():
    t0 = 1_700_000_000_000
    records_a = [_bar(t0 + i * MINUTE_MS, 100 + i) for i in range(5)]
    records_b = [_bar(t0 + i * MINUTE_MS, 200 + i) for i in range(5)]
    lanes = [
        prepare_lane_from_observed_r0("A", records_a, {"steps": 0, "score": 0.0}),
        prepare_lane_from_observed_r0("B", records_b, {"steps": 0, "score": 0.0}),
    ]
    audit = []
    policy_calls = []

    def hook(stage, t, lane_ids):
        audit.append((stage, t, tuple(lane_ids)))

    def policy_batch(obs):
        policy_calls.append(tuple((x["lane_id"], x["decision_time_ms"]) for x in obs))
        assert all(max(row[0] for row in x["market_window"]) == x["decision_time_ms"] for x in obs)
        assert all("next_bar" not in x and "account_state_t1" not in x and "teacher" not in x and "target" not in x for x in obs)
        return [{"x": 1.0 if x["lane_id"] == "A" else 2.0} for x in obs]

    events = list(
        time_major_batch_scan_r0(
            lanes,
            lookback_minutes=2,
            policy_batch=policy_batch,
            scalar_transition=_transition,
            audit_hook=hook,
        )
    )

    # Three decision clocks, each one policy call for both lanes.
    assert len(policy_calls) == 3
    assert all(len(x) == 2 for x in policy_calls)
    assert len(events) == 6
    assert all(e.policy_batch_size == 2 for e in events)
    assert all(e.observation_max_time_ms == e.decision_time_ms for e in events)
    assert all(e.transition_time_ms == e.decision_time_ms + MINUTE_MS for e in events)

    # For each clock the whole batch is frozen before the first transition begins.
    for t in sorted({e.decision_time_ms for e in events}):
        stages = [x[0] for x in audit if x[1] == t]
        assert stages[:3] == ["OBSERVATIONS_READY", "POLICY_BATCH_BEGIN", "ACTIONS_FROZEN"]
        assert stages.count("TRANSITION_BEGIN") == 2
        assert stages.index("ACTIONS_FROZEN") < stages.index("TRANSITION_BEGIN")


def test_halt_minutes_freeze_price_zero_activity_and_do_not_reset_account():
    t0 = 1_700_100_000_000
    observed = [_bar(t0, 10.0, 3.0), _bar(t0 + 3 * MINUTE_MS, 13.0, 5.0), _bar(t0 + 4 * MINUTE_MS, 14.0, 6.0)]
    lane = prepare_lane_from_observed_r0("HALT", observed, {"steps": 0, "score": 0.0})
    assert len(lane.minutes) == 5
    assert [x.halt_imputed for x in lane.minutes] == [False, True, True, False, False]
    for x in lane.minutes[1:3]:
        r = x.record
        assert (r.open, r.high, r.low, r.close) == (10.0, 10.0, 10.0, 10.0)
        assert r.volume == 0.0
        assert r.number_of_trades == 0

    events = list(
        time_major_batch_scan_r0(
            [lane],
            lookback_minutes=1,
            policy_batch=lambda obs: [{"x": 1.0} for _ in obs],
            scalar_transition=_transition,
        )
    )
    assert [e.account_state_t["steps"] for e in events] == [0, 1, 2, 3]
    assert [e.account_state_t1["steps"] for e in events] == [1, 2, 3, 4]


def test_reopening_future_cannot_rewrite_imputed_halt_or_prior_action():
    t0 = 1_700_200_000_000

    def run(reopen_price: float):
        lane = prepare_lane_from_observed_r0(
            "X",
            [_bar(t0, 50.0), _bar(t0 + 3 * MINUTE_MS, reopen_price), _bar(t0 + 4 * MINUTE_MS, reopen_price + 1)],
            {"steps": 0, "score": 0.0},
        )
        seen = []

        def policy(obs):
            for x in obs:
                last = x["market_window"][-1]
                seen.append((x["decision_time_ms"], last[4], deepcopy(x["account_state_t"])))
            return [{"x": float(x["market_window"][-1][4])} for x in obs]

        events = list(time_major_batch_scan_r0([lane], lookback_minutes=1, policy_batch=policy, scalar_transition=_transition))
        return lane, seen, events

    lane_a, seen_a, events_a = run(80.0)
    lane_b, seen_b, events_b = run(8000.0)
    # The two prefix-only halt rows are identical despite changing reopening future.
    for i in (1, 2):
        assert lane_a.minutes[i].record == lane_b.minutes[i].record
    cutoff = t0 + 2 * MINUTE_MS
    assert [x for x in seen_a if x[0] <= cutoff] == [x for x in seen_b if x[0] <= cutoff]
    assert [e.frozen_action_t for e in events_a if e.decision_time_ms <= cutoff] == [e.frozen_action_t for e in events_b if e.decision_time_ms <= cutoff]


def test_sensory_is_batched_from_causal_payload_only():
    t0 = 1_700_300_000_000
    lanes = [
        prepare_lane_from_observed_r0(str(i), [_bar(t0 + j * MINUTE_MS, 100 * i + j + 1) for j in range(4)], {"steps": 0, "score": 0.0})
        for i in range(3)
    ]
    calls = []

    def sensory(obs):
        calls.append(tuple(x["lane_id"] for x in obs))
        assert all(max(row[0] for row in x["market_window"]) <= x["decision_time_ms"] for x in obs)
        return [{"causal": x["decision_time_ms"]} for x in obs]

    def policy(obs):
        assert all(x["sensory"]["causal"] == x["decision_time_ms"] for x in obs)
        return [{"x": 0.0} for _ in obs]

    events = list(time_major_batch_scan_r0(lanes, lookback_minutes=1, policy_batch=policy, scalar_transition=_transition, sensory_batch_provider=sensory))
    assert len(calls) == 3
    assert all(len(x) == 3 for x in calls)
    assert len(events) == 9


def test_frozen_physics_bridge_constructs_next_market_only_post_action_boundary():
    t0 = 1_700_400_000_000
    order = []
    lane = prepare_lane_from_observed_r0("P", [_bar(t0 + i * MINUTE_MS, 10 + i) for i in range(3)], {"snap": 0})

    def factory(**kw):
        order.append(("factory", kw["decision_minute"].open_time, kw["transition_minute"].open_time))
        return {"bar": {"time": kw["transition_minute"].open_time}}

    def step(snapshot_t, action, market_execution_input, physics_contract):
        order.append(("step", action["x"], market_execution_input["bar"]["time"]))
        return {"snapshot_t1": {"snap": snapshot_t["snap"] + 1}}

    transition = make_frozen_physics_scalar_transition_r0(
        physics_contract={"contract": "frozen"},
        market_execution_input_factory=factory,
        step_account_fn=step,
    )

    def policy(obs):
        order.append(("policy", obs[0]["decision_time_ms"]))
        return [{"x": 7}]

    events = list(time_major_batch_scan_r0([lane], lookback_minutes=1, policy_batch=policy, scalar_transition=transition))
    assert len(events) == 2
    for i in range(0, len(order), 3):
        assert order[i][0] == "policy"
        assert order[i + 1][0] == "factory"
        assert order[i + 2][0] == "step"


def test_final_holdout_is_neither_observation_nor_transition():
    final = 1_700_500_180_000
    t0 = final - 3 * MINUTE_MS
    lane = prepare_lane_from_observed_r0(
        "F",
        [_bar(t0 + i * MINUTE_MS, 1 + i) for i in range(5)],
        {"steps": 0, "score": 0.0},
        final_holdout_start_ms=final,
    )
    events = list(
        time_major_batch_scan_r0(
            [lane],
            lookback_minutes=1,
            policy_batch=lambda obs: [{"x": 0.0} for _ in obs],
            scalar_transition=_transition,
            final_holdout_start_ms=final,
        )
    )
    assert events
    assert max(e.observation_max_time_ms for e in events) < final
    assert max(e.transition_time_ms for e in events) < final
