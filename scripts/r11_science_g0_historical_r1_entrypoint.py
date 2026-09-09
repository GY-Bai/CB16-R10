#!/usr/bin/env python3
from __future__ import annotations

"""CLI adapter for the R11 G0 Historical R1 implementation.

This file changes no scientific semantics.  It provides two execution-boundary
repairs that are specific to running a script from ``scripts/`` on the Shanxi
Actions runner:

1. bind the repository root before importing ``cb16_local_opt``;
2. convert the read-only MappingProxyType inside MarketRuntimeCacheStatsR11 to a
   plain dict when the implementation asks dataclasses.asdict() to serialize the
   final diagnostic receipt.

The underlying historical execution, Teacher, loss, gradients, Physics and
feedback diagnostics remain in ``r11_science_g0_historical_r1.py`` and the
qualified R11 modules it calls.
"""

from dataclasses import asdict as _dataclass_asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cb16_local_opt.market_runtime_cache_r11 import MarketRuntimeCacheStatsR11
import scripts.r11_science_g0_historical_r1 as _impl


def _receipt_asdict(value):
    if isinstance(value, MarketRuntimeCacheStatsR11):
        return {
            "compressed_loads_total": int(value.compressed_loads_total),
            "index_builds_total": int(value.index_builds_total),
            "loaded_symbols": list(value.loaded_symbols),
            "compressed_loads_by_symbol": dict(value.compressed_loads_by_symbol),
        }
    return _dataclass_asdict(value)


# Narrowly replace only the implementation module's imported serializer alias.
_impl.asdict = _receipt_asdict


if __name__ == "__main__":
    raise SystemExit(_impl.main())
