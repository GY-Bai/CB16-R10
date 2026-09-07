#!/usr/bin/env python3
from __future__ import annotations

"""Task F acceptance entrypoint with concrete hardware placement and safe diagnostics."""

from dataclasses import asdict as _dataclass_asdict

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheStatsR11
from cb16_local_opt.training_integration_r11 import IntegratedTrainingRuntimeR11
from scripts import run_r11_task_f_acceptance as acceptance


# Task F owns WHERE computation happens. Replace only the placement-aware runtime
# constructor; all training math, evidence, optimizer, seeds and gates remain Task B's.
acceptance.TrainingRuntimeR11 = IntegratedTrainingRuntimeR11


def _task_f_asdict(obj):
    """Serialize immutable runtime diagnostics without deepcopying MappingProxyType."""
    if isinstance(obj, MarketRuntimeCacheStatsR11):
        return {
            "compressed_loads_total": int(obj.compressed_loads_total),
            "index_builds_total": int(obj.index_builds_total),
            "loaded_symbols": list(obj.loaded_symbols),
            "compressed_loads_by_symbol": dict(obj.compressed_loads_by_symbol),
        }
    return _dataclass_asdict(obj)


# The base acceptance script uses dataclasses.asdict only for evidence/reporting. The
# market-cache stats intentionally expose MappingProxyType to remain immutable, while
# dataclasses.asdict deep-copies and therefore cannot serialize that proxy. Override the
# reporting helper only; the cache and scientific/runtime objects remain unchanged.
acceptance.asdict = _task_f_asdict


if __name__ == "__main__":
    raise SystemExit(acceptance.main())
