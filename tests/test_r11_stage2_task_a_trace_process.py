from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheR11
from cb16_local_opt.r102_physics import LONG, FrozenPhysicsRuntimeR102, build_parent_scenarios
from cb16_local_opt.trace_process_runtime_r11 import ForkProcessTraceRuntimeR11
from cb16_local_opt.trace_runtime_r11 import H72TraceWorkItemR11, TraceRuntimeR11
from tests.test_r11_trace_runtime import _market_arrays, _write_market


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and "fork" in mp.get_all_start_methods(), "requires POSIX fork")
class TestForkProcessTraceRuntimeR11(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physics = FrozenPhysicsRuntimeR102.load(ROOT)

    def _items(self, ts, bars, funding):
        decision_time_ms = int(ts[96])
        scenarios = build_parent_scenarios(
            self.physics,
            symbol="BTCUSDT",
            decision_time_ms=decision_time_ms,
            hourly_ts=ts,
            hourly_ohlcv=bars,
            funding=funding,
            prehistory_hours=96,
        )[:4]
        items = []
        for ordinal, scenario in enumerate(scenarios):
            state = {
                "account_id": scenario["account_id"],
                "snapshot": scenario["snapshot"],
                "risk_authority": scenario["risk_authority"],
                "current_mark": scenario["current_mark"],
            }
            items.append(
                H72TraceWorkItemR11(
                    ordinal=ordinal,
                    causal_trace_id=f"PROC:{ordinal}",
                    account_id=scenario["account_id"],
                    parent_id=f"P{ordinal}",
                    symbol="BTCUSDT",
                    decision_time_ms=decision_time_ms,
                    parent_state=state,
                    direction_v55=LONG,
                    requested_risk=0.25,
                )
            )
        return items

    def test_serial_process_exact_and_persistent_pool_reuse(self):
        ts, bars, funding = _market_arrays(rows=240)
        items = self._items(ts, bars, funding)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)

            serial_cache = MarketRuntimeCacheR11(root)
            with TraceRuntimeR11(physics=self.physics, market_cache=serial_cache, max_workers=1) as serial_rt:
                expected = serial_rt.run(items)

            process_cache = MarketRuntimeCacheR11(root)
            with ForkProcessTraceRuntimeR11(
                physics=self.physics,
                market_cache=process_cache,
                max_workers=2,
            ) as process_rt:
                first = process_rt.run(items)
                executor_identity = process_rt.executor_identity
                second = process_rt.run(list(reversed(items)))
                self.assertEqual(process_rt.executor_identity, executor_identity)
                self.assertEqual(process_rt.executions, 2)
                self.assertEqual(process_rt.stats().preloaded_symbols, ("BTCUSDT",))

            self.assertEqual(first, expected)
            self.assertEqual(second, expected)
            stats = process_cache.stats()
            self.assertEqual(stats.compressed_loads_total, 1)
            self.assertEqual(stats.index_builds_total, 1)
            process_cache.assert_read_only()

    def test_worker_exception_fails_closed_and_pool_has_no_stale_state(self):
        ts, bars, funding = _market_arrays(rows=240)
        items = self._items(ts, bars, funding)
        bad = H72TraceWorkItemR11(
            ordinal=99,
            causal_trace_id="PROC:BAD",
            account_id=items[0].account_id,
            parent_id="PBAD",
            symbol="BTCUSDT",
            decision_time_ms=int(ts[-10]),
            parent_state=items[0].parent_state,
            direction_v55=LONG,
            requested_risk=0.25,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_market(root, "BTCUSDT", ts, bars, funding)
            cache = MarketRuntimeCacheR11(root)
            with ForkProcessTraceRuntimeR11(
                physics=self.physics,
                market_cache=cache,
                max_workers=2,
            ) as runtime:
                with self.assertRaisesRegex(RuntimeError, "R11_FORK_TRACE_WORKER_EXCEPTION"):
                    runtime.run([bad])
                recovered = runtime.run(items)

            serial_cache = MarketRuntimeCacheR11(root)
            with TraceRuntimeR11(physics=self.physics, market_cache=serial_cache, max_workers=1) as serial_rt:
                expected = serial_rt.run(items)
            self.assertEqual(recovered, expected)


if __name__ == "__main__":
    unittest.main()
