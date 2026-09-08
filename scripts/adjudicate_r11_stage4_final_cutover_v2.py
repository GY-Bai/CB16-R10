#!/usr/bin/env python3
from __future__ import annotations

"""Qualification-only overlay for the final Stage-4 writer-registry rebuild.

S4A ran independently and therefore could not classify sibling S4B-S4I files that
were absent from its task branch. This overlay changes no runtime authority and
only supplies the final adjudicator with exact accepted Wave-1 path roles.
"""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL = HERE / "adjudicate_r11_stage4_final_cutover.py"
SPEC = importlib.util.spec_from_file_location("cb16_stage4_final_cutover_impl", IMPL)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("FINAL_ADJUDICATOR_IMPLEMENTATION_LOAD_FAILED")
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

WAVE1_PRODUCTION = {
    "cb16_local_opt/stage4_canonical_runtime_r11.py",
    "cb16_local_opt/stage4_authority_adoption_r11.py",
    "cb16_local_opt/stage4_authority_lease_r11.py",
    "cb16_local_opt/stage4_permission_boundary_r11.py",
    "cb16_local_opt/stage4_state_roots_r11.py",
    "cb16_local_opt/stage4_legacy_retirement_r11.py",
}
WAVE1_QUALIFICATION = {
    "cb16_local_opt/stage4_authority_inventory_r11.py",
    "cb16_local_opt/stage4_hostile_cutover_r11.py",
    "cb16_local_opt/stage4_gate_compiler_r11.py",
}

mod.FINAL_PRODUCTION_PATHS = frozenset(set(mod.FINAL_PRODUCTION_PATHS) | WAVE1_PRODUCTION)
mod.FINAL_QUALIFICATION_PATHS = frozenset(set(mod.FINAL_QUALIFICATION_PATHS) | WAVE1_QUALIFICATION)

if __name__ == "__main__":
    raise SystemExit(mod.main())
