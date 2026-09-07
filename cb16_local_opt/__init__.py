"""CB16 Shanxi Frozen-Body + Blank-Central-Brain Runtime R10.

Package-root imports are intentionally lightweight.  The R2 storage / diagnostics
paths must be importable on hosts that do not have the ML runtime (PyTorch/CUDA)
installed.  Heavy runtime symbols are therefore exported lazily through
``__getattr__`` while preserving the historical package-root API.
"""

from __future__ import annotations

from importlib import import_module

RUNTIME_VERSION = "CB16_SHANXI_FROZEN_BODY_G0_BRAIN_R10"
RUNTIME_STATUS = "FROZEN_TYPED_SENSORY_BODY_PLUS_BLANK_CENTRAL_BRAIN"

# Preserve the public package-root names without importing torch-heavy modules at
# package initialization time.  This keeps storage, diagnostics, recovery and
# topology tooling usable in a minimal Python environment.
_LAZY_EXPORTS = {
    "BinanceUSDMArchiveSourceR10": (".binance_archive_input_r10", "BinanceUSDMArchiveSourceR10"),
    "SensoryDecisionFrameR10": (".binance_archive_input_r10", "SensoryDecisionFrameR10"),
    "aggregate_1m": (".binance_archive_input_r10", "aggregate_1m"),
    "iter_sensory_frames": (".binance_archive_input_r10", "iter_sensory_frames"),
    "ordered4h30_from_hourly": (".binance_archive_input_r10", "ordered4h30_from_hourly"),
    "FrozenSensoryStackR10": (".frozen_sensory_stack_r10", "FrozenSensoryStackR10"),
    "SensoryAssetPathsR10": (".frozen_sensory_stack_r10", "SensoryAssetPathsR10"),
    "TypedCentralBrainR10": (".typed_central_brain_r10", "TypedCentralBrainR10"),
    "build_g0_brain_r10": (".typed_central_brain_r10", "build_g0_brain_r10"),
    "CB16DecisionRuntimeR10": (".decision_runtime_r10", "CB16DecisionRuntimeR10"),
}

__all__ = ["RUNTIME_VERSION", "RUNTIME_STATUS", *_LAZY_EXPORTS]


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
