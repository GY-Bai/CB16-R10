from __future__ import annotations

"""R11 Science G0 R4.1 — post-R4 state-alignment falsification helpers.

R4.1 never changes the production Student or Teacher. It creates matched
training controls by rotating only reduced Teacher targets across whole
independent future-dependence groups while preserving the six AccountState
scenario identities and the exact target multiset.
"""

import hashlib
import json
import math
import statistics
from typing import Any, Mapping

import numpy as np
import torch

from .training_runtime_r11 import PreparedEvidenceR11

R41_RUNTIME = "CB16_R11_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_V1"
R41_SHIFTS = (1, 7, 13, 23, 31)
R41_SCENARIOS = (
    "CLEAN_FLAT_FULL",
    "CLEAN_FLAT_LOW_ENVELOPE",
    "PRIOR_LONG_R025",
    "PRIOR_LONG_R075",
    "PRIOR_SHORT_R025",
    "PRIOR_SHORT_R075",
)
_INPUT_STOP = 102
_TARGET_START = 102
_TARGET_STOP = 106
_WEIGHT_COL = 106


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _scenario_from_parent_id(parent_id: str) -> str:
    value = str(parent_id).rsplit(":", 1)[-1]
    if value not in R41_SCENARIOS:
        raise RuntimeError(f"R11_R41_UNKNOWN_ACCOUNT_SCENARIO:{value}")
    return value


def _group_scenario_rows(prepared: PreparedEvidenceR11) -> tuple[list[str], dict[str, dict[str, int]]]:
    prepared.validate()
    order: list[str] = []
    rows: dict[str, dict[str, int]] = {}
    for idx, (pid, gid0) in enumerate(zip(prepared.parent_ids, prepared.dependence_group_ids)):
        gid = str(gid0)
        if gid not in rows:
            order.append(gid)
            rows[gid] = {}
        scenario = _scenario_from_parent_id(pid)
        if scenario in rows[gid]:
            raise RuntimeError(f"R11_R41_DUPLICATE_SCENARIO_IN_FUTURE_GROUP:{gid}:{scenario}")
        rows[gid][scenario] = idx
    expected = set(R41_SCENARIOS)
    for gid in order:
        observed = set(rows[gid])
        if observed != expected:
            raise RuntimeError(f"R11_R41_SCENARIO_SET_DRIFT:{gid}:{sorted(observed)}")
    return order, rows


def _target_multiset_sha256(packed: torch.Tensor) -> str:
    target = packed[:, _TARGET_START:_TARGET_STOP].detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    rows = sorted(bytes(np.ascontiguousarray(row).tobytes(order="C")) for row in target)
    h = hashlib.sha256(b"CB16_R11_R41_TARGET_MULTISET_V1\0")
    for row in rows:
        h.update(row)
    return h.hexdigest()


def shuffle_reduced_targets_by_future_group_r41(prepared: PreparedEvidenceR11, *, shift: int) -> tuple[PreparedEvidenceR11, dict[str, Any]]:
    """Rotate target donors by future group, mapping rows by AccountState scenario."""
    prepared.validate()
    k0 = int(shift)
    if k0 not in R41_SHIFTS:
        raise RuntimeError(f"R11_R41_UNREGISTERED_SHIFT:{k0}")
    order, rows = _group_scenario_rows(prepared)
    n = len(order)
    if n < 2:
        raise RuntimeError("R11_R41_NEEDS_MULTIPLE_FUTURE_GROUPS")
    k = k0 % n
    if k == 0:
        raise RuntimeError("R11_R41_IDENTITY_GROUP_SHUFFLE_FORBIDDEN")
    before_multiset = _target_multiset_sha256(prepared.packed)
    out_pack = prepared.packed.clone()
    mapping: list[dict[str, str]] = []
    for dst_pos, dst_gid in enumerate(order):
        src_gid = order[(dst_pos + k) % n]
        if src_gid == dst_gid:
            raise RuntimeError("R11_R41_GROUP_SHUFFLE_FIXED_POINT")
        for scenario in R41_SCENARIOS:
            dst_idx = rows[dst_gid][scenario]
            src_idx = rows[src_gid][scenario]
            out_pack[dst_idx, _TARGET_START:_TARGET_STOP] = prepared.packed[src_idx, _TARGET_START:_TARGET_STOP]
        mapping.append({"destination_group": dst_gid, "source_group": src_gid})
    if not torch.equal(out_pack[:, :_INPUT_STOP], prepared.packed[:, :_INPUT_STOP]):
        raise RuntimeError("R11_R41_INPUT_FEATURES_CHANGED_BY_CONTROL")
    if not torch.equal(out_pack[:, _WEIGHT_COL], prepared.packed[:, _WEIGHT_COL]):
        raise RuntimeError("R11_R41_GROUP_WEIGHTS_CHANGED_BY_CONTROL")
    after_multiset = _target_multiset_sha256(out_pack)
    if after_multiset != before_multiset:
        raise RuntimeError("R11_R41_TARGET_MULTISET_CHANGED_BY_CONTROL")
    changed = torch.any(out_pack[:, _TARGET_START:_TARGET_STOP] != prepared.packed[:, _TARGET_START:_TARGET_STOP], dim=1)
    changed_rows = int(changed.sum().detach().cpu().item())
    if changed_rows <= 0:
        raise RuntimeError("R11_R41_TARGET_ALIGNMENT_NOT_BROKEN")
    h = hashlib.sha256()
    h.update(b"CB16_R11_R41_SHUFFLED_PREPARED_EVIDENCE_V1\0")
    h.update(str(prepared.evidence_hash).encode("ascii") + b"\0")
    h.update(_canonical_json_bytes({"shift": k, "mapping": mapping, "scenarios": list(R41_SCENARIOS)}))
    h.update(out_pack[:, _TARGET_START:_TARGET_STOP].detach().cpu().contiguous().numpy().tobytes(order="C"))
    shuffled = PreparedEvidenceR11(
        parent_ids=prepared.parent_ids,
        dependence_group_ids=prepared.dependence_group_ids,
        packed=out_pack,
        evidence_hash=h.hexdigest(),
        host_to_device_transfers=prepared.host_to_device_transfers,
        h2d_strategy=prepared.h2d_strategy,
        h2d_non_blocking=prepared.h2d_non_blocking,
        h2d_latency_ms=prepared.h2d_latency_ms,
        h2d_benchmark=prepared.h2d_benchmark,
    )
    shuffled.validate()
    return shuffled, {
        "schema": "CB16_R11_CANONICAL_STATE_ALIGNMENT_SHUFFLE_R4_1_V1",
        "shift": k,
        "future_group_count": n,
        "rows_per_future_group": len(R41_SCENARIOS),
        "scenario_order": list(R41_SCENARIOS),
        "whole_independent_future_group_shuffle": True,
        "scenario_identity_preserved": True,
        "input_features_byte_identical": True,
        "group_weights_byte_identical": True,
        "target_multiset_sha256_before": before_multiset,
        "target_multiset_sha256_after": after_multiset,
        "target_multiset_exactly_preserved": True,
        "changed_target_rows": changed_rows,
        "fixed_point_future_groups": 0,
        "mapping": mapping,
        "shuffled_evidence_hash": shuffled.evidence_hash,
    }


def summarize_state_alignment_control_r41(aligned_validation: Mapping[str, Any], shuffled_validations: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    required = ("loss", "direction_loss", "sizing_loss")
    for key in required:
        if key not in aligned_validation or not math.isfinite(float(aligned_validation[key])):
            raise RuntimeError(f"R11_R41_NONFINITE_ALIGNED_VALIDATION:{key}")
    if tuple(sorted(int(k) for k in shuffled_validations)) != tuple(sorted(R41_SHIFTS)):
        raise RuntimeError("R11_R41_SHUFFLE_ARM_SET_DRIFT")
    per_shift: list[dict[str, Any]] = []
    for shift in R41_SHIFTS:
        src = shuffled_validations[int(shift)]
        for key in required:
            if key not in src or not math.isfinite(float(src[key])):
                raise RuntimeError(f"R11_R41_NONFINITE_SHUFFLED_VALIDATION:{shift}:{key}")
        row = {"shift": int(shift)}
        for key in required:
            a = float(aligned_validation[key]); s = float(src[key])
            row[f"aligned_minus_shuffled_{key}"] = a - s
            row[f"aligned_lower_{key}"] = a < s
        per_shift.append(row)
    total_deltas = [float(x["aligned_minus_shuffled_loss"]) for x in per_shift]
    direction_deltas = [float(x["aligned_minus_shuffled_direction_loss"]) for x in per_shift]
    sizing_deltas = [float(x["aligned_minus_shuffled_sizing_loss"]) for x in per_shift]
    total_wins = sum(bool(x["aligned_lower_loss"]) for x in per_shift)
    direction_wins = sum(bool(x["aligned_lower_direction_loss"]) for x in per_shift)
    sizing_wins = sum(bool(x["aligned_lower_sizing_loss"]) for x in per_shift)
    if total_wins == len(R41_SHIFTS):
        pattern = "ALIGNED_TOTAL_LOWER_THAN_ALL_PRE_REGISTERED_SHUFFLES"
    elif total_wins == 0:
        pattern = "ALIGNED_TOTAL_NOT_LOWER_THAN_ANY_PRE_REGISTERED_SHUFFLE"
    else:
        pattern = "MIXED_ALIGNED_VS_SHUFFLED_TOTAL_ORDERING"
    return {
        "schema": "CB16_R11_CANONICAL_STATE_ALIGNMENT_FALSIFICATION_R4_1_SUMMARY_V1",
        "basis": "ORIGINAL_UNSHUFFLED_FROZEN_R4_VALIDATION_TEACHER_OBJECTIVE",
        "post_r4_diagnostic_not_status_driving": True,
        "pre_registered_shifts": list(R41_SHIFTS),
        "aligned_validation": {k: float(aligned_validation[k]) for k in required},
        "per_shift": per_shift,
        "aligned_lower_total_count": total_wins,
        "aligned_lower_direction_count": direction_wins,
        "aligned_lower_sizing_count": sizing_wins,
        "median_aligned_minus_shuffled_total": float(statistics.median(total_deltas)),
        "median_aligned_minus_shuffled_direction": float(statistics.median(direction_deltas)),
        "median_aligned_minus_shuffled_sizing": float(statistics.median(sizing_deltas)),
        "descriptive_pattern": pattern,
        "scientific_market_information_verdict": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
    }
