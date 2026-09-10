from __future__ import annotations

import torch

from cb16_local_opt.full_train_multi_asset_candidate_r7 import (
    R7_CANDIDATE_SCHEMA,
    candidate_identity_r7,
    state_dicts_exactly_equal_r7,
)


def test_state_dict_exact_equality_is_byte_semantic_tensor_equality():
    a = {"x": torch.tensor([1.0, 2.0], dtype=torch.float32), "y": torch.tensor([[3]], dtype=torch.int64)}
    b = {k: v.clone() for k, v in a.items()}
    assert state_dicts_exactly_equal_r7(a, b)
    b["x"][0] += 1.0
    assert not state_dicts_exactly_equal_r7(a, b)


def test_candidate_identity_is_deterministic_and_binds_all_authority_inputs():
    args = dict(
        g0_policy_hash="g0",
        full_train_evidence_hash="ev",
        train_teacher_protocol_hash="teacher",
        candidate_policy_hash="candidate",
        source_manifest_sha256="manifest",
        source_parents_sha256="parents",
        source_branches_sha256="branches",
        execution_head="head",
    )
    a = candidate_identity_r7(**args)
    b = candidate_identity_r7(**args)
    assert a == b
    assert a["schema"] == R7_CANDIDATE_SCHEMA
    assert a["evaluation_state"] == "UNEVALUATED_OUTSIDE_TRAIN_FIT"
    assert a["promotion_state"] == "NOT_AUTHORIZED"
    changed = candidate_identity_r7(**{**args, "candidate_policy_hash": "other"})
    assert changed["candidate_identity_sha256"] != a["candidate_identity_sha256"]
