from __future__ import annotations

"""Helpers for R11 Science G0 R7 full-TRAIN unevaluated candidate materialization."""

import hashlib
import json
from typing import Any, Mapping

import torch

R7_RUNTIME = "CB16_R11_FULL_TRAIN_MULTI_ASSET_CANONICAL_CANDIDATE_R7_V1"
R7_CANDIDATE_SCHEMA = "CB16_R11_R7_FROZEN_UNEVALUATED_CANDIDATE_V1"
R7_SYMBOLS = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT",
)


def canonical_sha256_r7(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def state_dicts_exactly_equal_r7(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> bool:
    if tuple(sorted(a)) != tuple(sorted(b)):
        return False
    return all(
        a[k].dtype == b[k].dtype
        and tuple(a[k].shape) == tuple(b[k].shape)
        and torch.equal(a[k].detach().cpu(), b[k].detach().cpu())
        for k in a
    )


def candidate_identity_r7(
    *,
    g0_policy_hash: str,
    full_train_evidence_hash: str,
    train_teacher_protocol_hash: str,
    candidate_policy_hash: str,
    source_manifest_sha256: str,
    source_parents_sha256: str,
    source_branches_sha256: str,
    execution_head: str,
) -> dict[str, Any]:
    core = {
        "schema": R7_CANDIDATE_SCHEMA,
        "g0_policy_hash": str(g0_policy_hash),
        "full_train_prepared_evidence_hash": str(full_train_evidence_hash),
        "r2_4_train_teacher_protocol_hash": str(train_teacher_protocol_hash),
        "canonical_training_rule": {
            "optimizer": "AdamW_FP32",
            "epochs": 12,
            "batch_size": 512,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "gradient_clip_max_norm": 10.0,
            "generation_seed": 24680,
        },
        "candidate_policy_hash": str(candidate_policy_hash),
        "source_manifest_sha256": str(source_manifest_sha256),
        "source_parents_sha256": str(source_parents_sha256),
        "source_branches_sha256": str(source_branches_sha256),
        "execution_head": str(execution_head),
        "evaluation_state": "UNEVALUATED_OUTSIDE_TRAIN_FIT",
        "promotion_state": "NOT_AUTHORIZED",
    }
    return {**core, "candidate_identity_sha256": canonical_sha256_r7(core)}


__all__ = [
    "R7_RUNTIME",
    "R7_CANDIDATE_SCHEMA",
    "R7_SYMBOLS",
    "canonical_sha256_r7",
    "candidate_identity_r7",
    "state_dicts_exactly_equal_r7",
]
