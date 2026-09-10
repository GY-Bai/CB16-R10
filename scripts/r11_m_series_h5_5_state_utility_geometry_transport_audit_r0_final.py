#!/usr/bin/env python3
from __future__ import annotations

"""Final H5.5 executor after the pre-valid-result degenerate-rank clarification."""

from cb16_local_opt.state_utility_geometry_transport_audit_h55_clarified import run_fold_h55_clarified
from scripts import r11_m_series_h5_5_state_utility_geometry_transport_audit_r0 as base

CLARIFIED_PREREG_COMMIT = "c545b55dd2e0592b51b6b0426d057a1e8762c819"
CLARIFIED_GATE_BLOB = "54b9492e74d5017711b671837ef402abd604c7c6"
CLARIFIED_HELPER_BLOB = "faa5233170aa421c3bf44f023b1d8e2f7c187568"


def main() -> int:
    base.PREREG_COMMIT = CLARIFIED_PREREG_COMMIT
    base.GATE_BLOB = CLARIFIED_GATE_BLOB
    base.PINNED["gate"] = (
        "authority/rearchitecture_r11/CB16_R11_M_SERIES_H5_5_STATE_UTILITY_GEOMETRY_TRANSPORT_AUDIT_R0_GATE_V1.json",
        CLARIFIED_GATE_BLOB,
    )
    base.PINNED["h55_clarified_helper"] = (
        "cb16_local_opt/state_utility_geometry_transport_audit_h55_clarified.py",
        CLARIFIED_HELPER_BLOB,
    )
    base.run_fold_h55 = run_fold_h55_clarified
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
