#!/usr/bin/env python3
from __future__ import annotations

"""Integration-only launcher for the qualified Stage-3 E3 crash matrix.

This file exists because E3 uses multiprocessing with the spawn start method.
The parent must therefore be a real importable Python file rather than <stdin>.
No E3 crash/recovery semantics are defined here.
"""

import json
import os
from pathlib import Path

from cb16_local_opt.runtime_events_r11 import TournamentDecision
from cb16_local_opt.stage3_e3_crash_recovery_r11 import CrashBoundary, run_case

SCHEMA = "CB16_R11_STAGE3_INTEGRATION_E3_MATRIX_V1"


def main() -> int:
    root = Path(os.environ["E3_ROOT"]).resolve()
    out_path = Path(
        os.environ.get(
            "E3_REPORT",
            "ci_evidence/stage3_integration_e3_20.json",
        )
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for boundary in CrashBoundary:
        for decision in (TournamentDecision.PROMOTE, TournamentDecision.REJECT):
            receipt = run_case(
                root / f"{boundary.value}-{decision.value}",
                boundary,
                decision,
            )
            rows.append(
                {
                    "boundary": boundary.value,
                    "decision": decision.value,
                    "pass": bool(receipt["pass"]),
                    "failures": list(receipt["failures"]),
                    "child_exit_code": int(receipt["child_exit_code"]),
                }
            )

    report = {
        "schema": SCHEMA,
        "cases_total": len(rows),
        "cases_passed": sum(int(bool(row["pass"])) for row in rows),
        "cases": rows,
    }
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if report["cases_total"] != 20 or report["cases_passed"] != 20:
        print(json.dumps(report, sort_keys=True))
        return 2

    print("CB16_R11_STAGE3_INTEGRATED_E3_20_CASES=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
