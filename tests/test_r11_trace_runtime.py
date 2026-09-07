from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cb16_local_opt.market_runtime_cache_r11 import (  # noqa: E402
    MarketRuntimeCacheError,
    MarketRuntimeCacheR11,
)
from cb16_local_opt.r102_common import (  # noqa: E402
    FORBIDDEN_FINAL_START_MS,
    H72,
    HOUR_MS,
    model_state_semantic_sha256,
)
from cb16_local_opt.r102_evidence_cache import ParentContextR102  # noqa: E402
from cb16_local_opt.r102_physics import (  # noqa: E402
    FLAT,
    LONG,
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    simulate_h72_branch,
)
from cb16_local_opt.r102_policy_trace import run_real_on_policy_trace  # noqa: E402
from cb16_local_opt.sharded_experience_lake import ShardedExperienceLake  # noqa: E402
from cb16_local_opt.trace_runtime_r11 import (  # noqa: E402
    H72TraceWorkItemR11,
    TraceRuntimeR11,
    run_real_on_policy_trace_r11,
)
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10  # noqa: E402


def _market_arrays(rows: int = 220):
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


def _write_market(root: Path, symbol: str, ts, bars, funding) -> Path:
    d = root / "market_cache"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{symbol}.hourly_r102.npz"
    np.savez_compressed(p, open_time_ms=ts, ohlcv=bars, funding_rate=funding)
    return p


def _parent_fixture(rt: FrozenPhysicsRuntimeR102, ts, bars, funding):
    t = int(ts[96])
    s = build_parent_scenarios(
        rt,
        symbol="BTCUSDT",
        decision_time_ms=t,
        hourly_ts=ts,
        hourly_ohlcv=bars,
        funding=funding,
        prehistory_hours=96,
    )[0]
    p = ParentContextR102(
        "P0",
        "G0",
        "BTCUSDT",
        t,
        "VALIDATION",
        "CLEAN_FLAT_FULL",
        tuple([0.1] * 48),
        tuple([0.2] * 48),
        tuple(float(x) for x in s["account6"]),
        tuple([0.0] * 30),
        float(s["current_mark"]),
        s["snapshot_sha256"],
        True,
        "market-lineage",
    )
    state = {
        "parent_id": "P0",
        "account_id": s["account_id"],
        "snapshot": s["snapshot"],
        "risk_authority": s["risk_authority"],
        "current_mark": s["current_mark"],
    }
    return t, p, state


def _legacy_snapshot_sequence(rt, parent, ts, bars, funding, t, direction, risk):
    idx = {int(x): i for i, x in enumerate(ts)}
    snap = copy.deepcopy(parent["snapshot"])
    ra = copy.deepcopy(parent["risk_authority"])
    hashes = []
    for j in range(H72):
        now = int(t + j * HOUR_MS)
        i = idx[now]
        d, r = (direction, risk) if j == 0 else (FLAT, 0.0)
        step = rt.step_intent(
            snap,
            ra,
            direction_v55=d,
            risk=r,
            symbol="BTCUSDT",
            open_time_ms=now,
            ohlcv=bars[i],
            funding_rate=float(funding[i]),
            trace_id=f"CF:{parent['account_id']}:{direction}:{risk:.2f}:{j}",
        )
        snap = step["snapshot_t1"]
        hashes.append(rt.physics.sha256_obj(snap))
        if snap["termination_state"]["terminated"]:
            break
    return tuple(hashes)


class TestMarketRuntimeCacheR11(unittest.TestCase):
    def test_load_and_index_once_and_read_only(self):
        ts, bars, funding = _market_arrays()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            cache = MarketRuntimeCacheR11(root)
            a = cache.get("BTCUSDT")
            b = cache.get("BTCUSDT")
            self.assertIs(a, b)
            self.assertEqual(a.index_of(int(ts[96])), 96)
            self.assertEqual(a.h72_slice(int(ts[96])), slice(96, 168))
            stats = cache.stats()
            self.assertEqual(stats.compressed_loads_total, 1)
            self.assertEqual(stats.index_builds_total, 1)
            self.assertEqual(stats.compressed_loads_by_symbol["BTCUSDT"], 1)
            cache.assert_read_only()
            with self.assertRaises(ValueError):
                a.ohlcv[0, 0] = -1.0
            with self.assertRaises(ValueError):
                a.ohlcv.setflags(write=True)

    def test_bad_holdout_manifest_fails_before_npz_open(self):
        ts, bars, funding = _market_arrays()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = _write_market(root, "BTCUSDT", ts, bars, funding)
            manifest = p.with_name("BTCUSDT.manifest_r102.json")
            manifest.write_text(json.dumps({"forbidden_month_opened": True}), encoding="utf-8")
            cache = MarketRuntimeCacheR11(root)
            with mock.patch("cb16_local_opt.market_runtime_cache_r11.np.load") as loader:
                with self.assertRaises(MarketRuntimeCacheError):
                    cache.get("BTCUSDT")
                loader.assert_not_called()

    def test_final_holdout_scoped_path_fails_before_payload_open(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "2025-09" / "market_cache"
            d.mkdir(parents=True)
            (d / "BTCUSDT.hourly_r102.npz").write_bytes(b"not-opened")
            cache = MarketRuntimeCacheR11(d)
            with mock.patch("cb16_local_opt.market_runtime_cache_r11.np.load") as loader:
                with self.assertRaisesRegex(MarketRuntimeCacheError, "FINAL_HOLDOUT_ACCESS_FAIL_CLOSED"):
                    cache.get("BTCUSDT")
                loader.assert_not_called()

    def test_timestamp_at_holdout_boundary_fails_closed(self):
        ts = FORBIDDEN_FINAL_START_MS + np.arange(4, dtype=np.int64) * HOUR_MS
        bars = np.ones((4, 5), dtype=np.float32)
        funding = np.zeros(4, dtype=np.float64)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            with self.assertRaisesRegex(MarketRuntimeCacheError, "FINAL_HOLDOUT_ACCESS_FAIL_CLOSED"):
                MarketRuntimeCacheR11(root).get("BTCUSDT")


class TestTraceRuntimeR11(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rt = FrozenPhysicsRuntimeR102.load(ROOT)

    def test_legacy_r11_exact_branch_snapshots_funding_finalize_and_termination(self):
        ts, bars, funding = _market_arrays()
        t, _, state = _parent_fixture(self.rt, ts, bars, funding)
        legacy = simulate_h72_branch(
            self.rt,
            parent=state,
            symbol="BTCUSDT",
            decision_time_ms=t,
            candidate_direction_v55=LONG,
            candidate_risk=0.25,
            hourly_ts=ts,
            hourly_ohlcv=bars,
            funding=funding,
        )
        expected_snapshots = _legacy_snapshot_sequence(
            self.rt, state, ts, bars, funding, t, LONG, 0.25
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            cache = MarketRuntimeCacheR11(root)
            item = H72TraceWorkItemR11(
                ordinal=0,
                causal_trace_id="TRACE:0",
                account_id=str(state["account_id"]),
                parent_id="P0",
                symbol="BTCUSDT",
                decision_time_ms=t,
                parent_state=state,
                direction_v55=LONG,
                requested_risk=0.25,
            )
            with TraceRuntimeR11(physics=self.rt, market_cache=cache, max_workers=1) as runtime:
                got = runtime.run([item])[0]
            self.assertEqual(got.branch, legacy)
            self.assertEqual(got.snapshot_sha256_sequence, expected_snapshots)
            self.assertEqual(got.branch["supervisor_decisions"], legacy["supervisor_decisions"])
            self.assertEqual(got.branch["utility"], legacy["utility"])
            self.assertEqual(got.branch["terminal_at_step"], legacy["terminal_at_step"])
            self.assertEqual(got.branch["finalize"], legacy["finalize"])
            self.assertEqual(got.branch["first_step"], legacy["first_step"])
            # Nonzero archived funding was supplied; exact branch equality proves
            # R11 consumed it at the same frozen hourly steps.
            self.assertGreater(int(np.count_nonzero(funding[96:168])), 0)

    def test_serial_parallel_set_equality_and_reordered_completion_canonical_identity(self):
        ts, bars, funding = _market_arrays()
        t, _, base_state = _parent_fixture(self.rt, ts, bars, funding)

        def item(ordinal: int, account: str, risk: float):
            state = dict(base_state)
            state["account_id"] = account
            return H72TraceWorkItemR11(
                ordinal=ordinal,
                causal_trace_id=f"TRACE:{ordinal}",
                account_id=account,
                parent_id=f"P{ordinal}",
                symbol="BTCUSDT",
                decision_time_ms=t,
                parent_state=state,
                direction_v55=LONG,
                requested_risk=risk,
            )

        items = [item(0, "SLOW", 0.25), item(1, "FAST1", 0.50), item(2, "FAST2", 0.75)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            serial_cache = MarketRuntimeCacheR11(root)
            with TraceRuntimeR11(physics=self.rt, market_cache=serial_cache, max_workers=1) as serial_rt:
                serial = serial_rt.run(items)

            completion_order = []
            completion_lock = threading.Lock()
            fast_done = threading.Event()

            class ReorderingRuntime(TraceRuntimeR11):
                def _execute_account_group(self, group):
                    ordinal = min(int(x.ordinal) for x in group)
                    if ordinal == 0 and not fast_done.wait(timeout=10.0):
                        raise RuntimeError("TEST_FAST_GROUP_DID_NOT_COMPLETE")
                    rows = super()._execute_account_group(group)
                    with completion_lock:
                        completion_order.append(ordinal)
                    if ordinal == 1:
                        fast_done.set()
                    return rows

            parallel_cache = MarketRuntimeCacheR11(root)
            with ReorderingRuntime(physics=self.rt, market_cache=parallel_cache, max_workers=3) as parallel_rt:
                parallel = parallel_rt.submit(list(reversed(items))).result()

            self.assertNotEqual(completion_order[0], 0)
            self.assertEqual([x.ordinal for x in parallel], [0, 1, 2])
            self.assertEqual(
                [(x.causal_trace_id, x.parent_id, x.branch) for x in parallel],
                [(x.causal_trace_id, x.parent_id, x.branch) for x in serial],
            )
            self.assertEqual(parallel_cache.stats().compressed_loads_total, 1)

    def test_same_account_requests_execute_in_decision_time_order(self):
        ts, bars, funding = _market_arrays(rows=260)
        t, _, state = _parent_fixture(self.rt, ts, bars, funding)
        later = t + HOUR_MS
        items = [
            H72TraceWorkItemR11(0, "TRACE:LATER", "A", "P0", "BTCUSDT", later, state, LONG, 0.25),
            H72TraceWorkItemR11(1, "TRACE:EARLIER", "A", "P1", "BTCUSDT", t, state, LONG, 0.25),
        ]
        starts = []

        def fake_branch(physics, market, *, parent, decision_time_ms, candidate_direction_v55, candidate_risk):
            starts.append(int(decision_time_ms))
            return ({"status": "MATURED", "utility": 0.0}, ())

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            cache = MarketRuntimeCacheR11(root)
            with mock.patch("cb16_local_opt.trace_runtime_r11._simulate_h72_branch_r11", side_effect=fake_branch):
                with TraceRuntimeR11(physics=self.rt, market_cache=cache, max_workers=2) as runtime:
                    runtime.run(items)
        self.assertEqual(starts, [t, later])

    def test_on_policy_old_new_receipt_exact_and_generation_reuses_market(self):
        ts, bars, funding = _market_arrays()
        _, parent, state = _parent_fixture(self.rt, ts, bars, funding)
        parents = {parent.parent_id: parent}
        states = {parent.parent_id: state}
        model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
        policy_hash = model_state_semantic_sha256(model)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            old_lake = ShardedExperienceLake(root / "old_lake", shards=2)
            new_lake = ShardedExperienceLake(root / "new_lake", shards=2)
            try:
                old = run_real_on_policy_trace(
                    model=model,
                    policy_hash=policy_hash,
                    generation=0,
                    parents=parents,
                    parent_states=states,
                    cache_dir=root,
                    physics=self.rt,
                    lake=old_lake,
                    device="cpu",
                    max_groups=1,
                )
                cache = MarketRuntimeCacheR11(root)
                with TraceRuntimeR11(physics=self.rt, market_cache=cache, max_workers=2) as runtime:
                    new = run_real_on_policy_trace_r11(
                        model=model,
                        policy_hash=policy_hash,
                        generation=0,
                        parents=parents,
                        parent_states=states,
                        trace_runtime=runtime,
                        lake=new_lake,
                        device="cpu",
                        max_groups=1,
                    )
                    self.assertEqual(new, old)
                    self.assertEqual(new["traces"][0]["CausalTraceID"], "R102TRACE:G0:P0")
                    self.assertEqual(cache.stats().compressed_loads_total, 1)
                    # A second generation uses the same market object: zero re-decompression.
                    run_real_on_policy_trace_r11(
                        model=model,
                        policy_hash=policy_hash,
                        generation=1,
                        parents=parents,
                        parent_states=states,
                        trace_runtime=runtime,
                        lake=new_lake,
                        device="cpu",
                        max_groups=1,
                    )
                    self.assertEqual(cache.stats().compressed_loads_total, 1)
                    self.assertEqual(cache.stats().index_builds_total, 1)
            finally:
                old_lake.close()
                new_lake.close()


if __name__ == "__main__":
    unittest.main()
