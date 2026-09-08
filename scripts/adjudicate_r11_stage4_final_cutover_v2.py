#!/usr/bin/env python3
from __future__ import annotations

"""Qualification-only overlay for the final Stage-4 writer-registry rebuild.

S4A ran independently and therefore could not classify sibling S4B-S4I files that
were absent from its task branch.  This overlay does not alter runtime authority;
it only supplies the final adjudicator with the exact accepted Wave-1 path roles.
"""

import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ns = runpy.run_path(str(HERE / "adjudicate_r11_stage4_final_cutover.py"), run_name="stage4_final_cutover_impl")

wave1_production = {
    "cb16_local_opt/stage4_canonical_runtime_r11.py",
    "cb16_local_opt/stage4_authority_adoption_r11.py",
    "cb16_local_opt/stage4_authority_lease_r11.py",
    "cb16_local_opt/stage4_permission_boundary_r11.py",
    "cb16_local_opt/stage4_state_roots_r11.py",
    "cb16_local_opt/stage4_legacy_retirement_r11.py",
}
wave1_qualification = {
    "cb16_local_opt/stage4_authority_inventory_r11.py",
    "cb16_local_opt/stage4_hostile_cutover_r11.py",
    "cb16_local_opt/stage4_gate_compiler_r11.py",
}

ns["FINAL_PRODUCTION_PATHS"] = frozenset(set(ns["FINAL_PRODUCTION_PATHS"]) | wave1_production)
ns["FINAL_QUALIFICATION_PATHS"] = frozenset(set(ns["FINAL_QUALIFICATION_PATHS"]) | wave1_qualification)

if __name__ == "__main__":
    raise SystemExit(ns["main"]())
