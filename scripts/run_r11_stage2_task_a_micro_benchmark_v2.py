#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict as _dataclass_asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheStatsR11
from scripts import run_r11_stage2_task_a_micro_benchmark as benchmark


def _task_a_asdict(obj):
    if isinstance(obj, MarketRuntimeCacheStatsR11):
        return {
            "compressed_loads_total": int(obj.compressed_loads_total),
            "index_builds_total": int(obj.index_builds_total),
            "loaded_symbols": list(obj.loaded_symbols),
            "compressed_loads_by_symbol": dict(obj.compressed_loads_by_symbol),
        }
    return _dataclass_asdict(obj)


benchmark.asdict = _task_a_asdict


if __name__ == "__main__":
    raise SystemExit(benchmark.main())
