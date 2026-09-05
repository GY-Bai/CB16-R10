from __future__ import annotations

"""Resource-guarded delegate for R9 short historical campaign phases.

The wrapper changes no phase semantics.  It only delays phase admission under soft
backpressure and fails closed under hard operational limits.  It cannot drop or
rewrite scientific objects.
"""

import importlib
import time
from pathlib import Path
from typing import Any

from .factory_resource_governor_r8 import (
    FactoryResourceGovernorR8,
    FactoryResourceLimitsR8,
)

PHASE_RESOURCE = {
    "ROLLOUT": "GPU",
    "TEACHER_CREDIT": "CPU_HEAVY",
    "SEAL_SNAPSHOT": "CPU_IO",
    "TRAIN_CHALLENGER": "GPU",
    "TOURNAMENT": "GPU",
    "ADJUDICATE_COMMIT": "CPU_IO",
    "RETENTION": "MAINTENANCE",
}


def _resolve(spec: str):
    module, name = spec.split(":", 1)
    fn = getattr(importlib.import_module(module), name)
    if not callable(fn):
        raise TypeError("delegate not callable")
    return fn


def guarded_phase_plugin_r9(*, phase, cycle_spec, plugin_config, context):
    if phase not in PHASE_RESOURCE:
        raise RuntimeError("R9_UNEXPECTED_PHASE:" + str(phase))
    if "FINAL" in phase.upper() or "HOLDOUT" in phase.upper():
        raise RuntimeError("R9_AUTOMATIC_FINAL_HOLDOUT_FORBIDDEN")

    delegate = plugin_config["delegate_callable"]
    delegate_config = dict(plugin_config.get("delegate_config") or {})
    ssd_root = plugin_config["ssd_root"]
    hdd_root = plugin_config["hdd_root"]
    limits = FactoryResourceLimitsR8(**dict(plugin_config.get("resource_limits") or {}))
    max_wait = float(plugin_config.get("max_soft_backpressure_wait_seconds", 600.0))
    poll = float(plugin_config.get("backpressure_poll_seconds", 5.0))
    governor = FactoryResourceGovernorR8(
        ssd_root=ssd_root,
        hdd_root=hdd_root,
        limits=limits,
    )
    rc = PHASE_RESOURCE[phase]
    t0 = time.monotonic()
    while True:
        snap = governor.snapshot({})
        decision = governor.decide(rc, snap)
        if decision.allowed:
            break
        if decision.hard_stop:
            raise RuntimeError(
                "R9_RESOURCE_HARD_STOP:"
                + ",".join(decision.reasons)
            )
        if time.monotonic() - t0 >= max_wait:
            raise RuntimeError(
                "R9_RESOURCE_BACKPRESSURE_TIMEOUT:"
                + ",".join(decision.reasons)
            )
        time.sleep(max(poll, decision.suggested_sleep_seconds))

    fn = _resolve(delegate)
    return fn(
        phase=phase,
        cycle_spec=cycle_spec,
        plugin_config=delegate_config,
        context=context,
    )
