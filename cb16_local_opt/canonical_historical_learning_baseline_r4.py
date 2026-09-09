from __future__ import annotations

"""R11 Science G0 R4 helper contracts.

R4 trains one disposable canonical Challenger but does not advance the canonical
R11 generation.  These helpers seal the bounded training snapshot and produce a
threshold-free shadow Champion/Challenger ranking on the frozen validation
probabilistic-Teacher target loss.
"""

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


R4_RUNTIME = "CB16_R11_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_V1"
R4_SNAPSHOT_SCHEMA = "CB16_R11_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_SHADOW_SNAPSHOT_V1"
R4_RANKING_SCHEMA = "CB16_R11_G0_CANONICAL_HISTORICAL_LEARNING_BASELINE_R4_SHADOW_RANKING_V1"


def canonical_json_sha256_r4(obj: Any) -> str:
    raw = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_shadow_snapshot_r4(
    *,
    champion_policy_hash: str,
    train_evidence_hash: str,
    validation_evidence_hash: str,
    train_teacher_protocol_hash: str,
    validation_teacher_protocol_hash: str,
    train_parent_ids: Sequence[str],
    train_dependence_group_ids: Sequence[str],
    validation_parent_ids: Sequence[str],
    validation_dependence_group_ids: Sequence[str],
) -> dict[str, Any]:
    if not champion_policy_hash or not train_evidence_hash or not validation_evidence_hash:
        raise RuntimeError("R11_R4_EMPTY_SNAPSHOT_IDENTITY")
    if len(train_parent_ids) != len(train_dependence_group_ids) or not train_parent_ids:
        raise RuntimeError("R11_R4_TRAIN_SNAPSHOT_ROW_DRIFT")
    if len(validation_parent_ids) != len(validation_dependence_group_ids) or not validation_parent_ids:
        raise RuntimeError("R11_R4_VALIDATION_SNAPSHOT_ROW_DRIFT")
    train_groups = len(set(str(x) for x in train_dependence_group_ids))
    validation_groups = len(set(str(x) for x in validation_dependence_group_ids))
    if train_groups < 32:
        raise RuntimeError("R11_R4_TRAIN_INDEPENDENT_SUPPORT_BELOW_MINIMUM")
    if validation_groups < 8:
        raise RuntimeError("R11_R4_VALIDATION_INDEPENDENT_SUPPORT_BELOW_MINIMUM")
    core = {
        "schema": R4_SNAPSHOT_SCHEMA,
        "role": "SHADOW_FROZEN_TRAINING_SNAPSHOT__NO_CANONICAL_GENERATION_ADVANCE",
        "canonical_generation": 0,
        "champion_policy_hash": str(champion_policy_hash),
        "train_evidence_hash": str(train_evidence_hash),
        "validation_evidence_hash": str(validation_evidence_hash),
        "train_teacher_protocol_hash": str(train_teacher_protocol_hash),
        "validation_teacher_protocol_hash": str(validation_teacher_protocol_hash),
        "train_rows": len(train_parent_ids),
        "validation_rows": len(validation_parent_ids),
        "train_dependence_groups": train_groups,
        "validation_dependence_groups": validation_groups,
        "train_parent_ids_sha256": canonical_json_sha256_r4(list(train_parent_ids)),
        "validation_parent_ids_sha256": canonical_json_sha256_r4(list(validation_parent_ids)),
        "distributional_r3_artifacts_in_gradient_graph": False,
        "canonical_generation_advanced": False,
        "final_holdout_payload_opened": False,
        "fresh_market_data_downloaded": False,
    }
    return {**core, "snapshot_hash": canonical_json_sha256_r4(core)}


def shadow_champion_challenger_ranking_r4(
    champion_validation: Mapping[str, Any],
    challenger_validation: Mapping[str, Any],
) -> dict[str, Any]:
    required = ("loss", "direction_loss", "sizing_loss")
    for name, src in (("champion", champion_validation), ("challenger", challenger_validation)):
        for key in required:
            if key not in src or not math.isfinite(float(src[key])):
                raise RuntimeError(f"R11_R4_NONFINITE_{name.upper()}_VALIDATION:{key}")
    champion_total = float(champion_validation["loss"])
    challenger_total = float(challenger_validation["loss"])
    delta = challenger_total - champion_total
    relative_improvement = (champion_total - challenger_total) / max(abs(champion_total), 1e-12)
    if challenger_total < champion_total:
        winner = "CHALLENGER"
        ordering = "CHALLENGER_LOWER_VALIDATION_LOSS"
    else:
        winner = "CHAMPION"
        ordering = "CHAMPION_LOWER_OR_EQUAL_VALIDATION_LOSS"
    return {
        "schema": R4_RANKING_SCHEMA,
        "basis": "FROZEN_VALIDATION_PROBABILISTIC_TEACHER_TARGET_LOSS",
        "shadow_winner": winner,
        "ordering": ordering,
        "champion_validation_total_loss": champion_total,
        "challenger_validation_total_loss": challenger_total,
        "absolute_total_loss_delta_challenger_minus_champion": delta,
        "relative_validation_improvement": relative_improvement,
        "direction_loss_delta_challenger_minus_champion": float(challenger_validation["direction_loss"]) - float(champion_validation["direction_loss"]),
        "sizing_loss_delta_challenger_minus_champion": float(challenger_validation["sizing_loss"]) - float(champion_validation["sizing_loss"]),
        "numeric_promotion_threshold_authority": "NOT_SPECIFIED_BY_CB16_R11_SEMANTIC_FREEZE_V1",
        "legacy_r10_0_1_percent_threshold_used": False,
        "canonical_promotion_authorized": False,
        "canonical_generation_advance_authorized": False,
        "scientific_market_verdict": False,
    }
