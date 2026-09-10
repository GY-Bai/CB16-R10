#!/usr/bin/env python3
from __future__ import annotations

"""Authority-repair replay for H5.15.

The original preregistration, helper, tests, executor, model, labels, controls,
and metrics remain byte-identical.  This wrapper patches only the executor's
chronology receipt to the inherited H5.10 meaning: chronological timestamps are
nondecreasing, ties are legal, and future_group_id (not timestamp) is the unique
dependence identity.
"""

from typing import Any

import scripts.r11_m_series_h5_15_full_state_linear_offset_transport_r0 as base

REPAIR_AUTHORITY_COMMIT = "a9c4816e090927512ce501773da523eb938ca8ff"
REPAIR_AUTHORITY_BLOB = "39733f3259ae278eaa742c645760d966a90bcd5d"
ORIGINAL_EXECUTION_HEAD = "f860256acf4d432fd8c761b9c914394655325aae"


def _payload_order_receipt_repair(payload: dict[str, Any]) -> dict[str, Any]:
    groups = list(payload["groups"])
    timestamps = [int(g["timestamp_ms"]) for g in groups]
    unique_ids = [str(g["future_group_id"]) for g in groups]
    six = all(len(g["scenarios"]) == 6 for g in groups)
    chronological_nondecreasing = all(
        timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
    )
    unique = len(set(unique_ids)) == len(unique_ids)
    tied_adjacent = sum(
        int(timestamps[i] == timestamps[i + 1]) for i in range(len(timestamps) - 1)
    )
    return {
        "fold": int(payload["fold"]),
        "future_group_count": len(groups),
        # Backward-compatible field consumed by the frozen original executor.
        # Its repaired meaning is the inherited H5.10 nondecreasing chronology.
        "strictly_chronological": chronological_nondecreasing,
        "chronological_nondecreasing": chronological_nondecreasing,
        "unique_future_group_ids": unique,
        "all_groups_have_six_scenarios": six,
        "unique_decision_clock_count": len(set(timestamps)),
        "tied_adjacent_timestamp_count": int(tied_adjacent),
        "first_timestamp_ms": timestamps[0] if timestamps else None,
        "last_timestamp_ms": timestamps[-1] if timestamps else None,
        "repair_authority_commit": REPAIR_AUTHORITY_COMMIT,
        "repair_authority_blob": REPAIR_AUTHORITY_BLOB,
        "original_execution_head": ORIGINAL_EXECUTION_HEAD,
        "chronology_semantics": "NONDECREASING_TIMESTAMP__TIES_ALLOWED__UNIQUE_FUTURE_GROUP_ID_IS_DEPENDENCE_IDENTITY",
    }


base._payload_order_receipt = _payload_order_receipt_repair


if __name__ == "__main__":
    raise SystemExit(base.main())
