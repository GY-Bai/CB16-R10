from __future__ import annotations

from types import SimpleNamespace

from cb16_local_opt.probabilistic_teacher_r6 import (
    DependenceAwarePredictiveLawR6,
    DependenceAwareTeacherEvidenceR6,
    EvidenceAdmissionReceiptR6,
)
from cb16_local_opt.science_distribution_boundary_r11 import (
    audit_teacher_action_laws_r11,
    classify_distributional_boundary_r11,
    compare_student_projection_paths_r11,
    projected_target_perturbation_r11,
    tail_only_law_perturbation_r11,
)
from cb16_local_opt.training_runtime_r11 import PreparedEvidenceR11
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10


LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _fixture(rows: int = 8):
    parents = {}
    evidence = []
    grid = ((-1, 0.25), (-1, 0.50), (-1, 0.75), (-1, 1.0), (0, 0.0),
            (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.0))
    for i in range(rows):
        pid = f"P{i:03d}"
        parents[pid] = SimpleNamespace(
            operator48=tuple(0.01 * (i + j + 1) for j in range(48)),
            medium48=tuple(0.02 * (i + j + 1) for j in range(48)),
            account6=tuple(0.03 * (i + j + 1) for j in range(6)),
        )
        laws = []
        for j, (direction, risk) in enumerate(grid):
            center = (i - j) * 1e-4
            quantiles = tuple(center + (k - 3) * 2e-4 for k in range(7))
            laws.append(
                DependenceAwarePredictiveLawR6(
                    direction=direction,
                    requested_risk=risk,
                    mean_utility=center,
                    std_utility=8e-4,
                    quantile_levels=LEVELS,
                    quantiles=quantiles,
                    effective_dependence_n=24.0,
                    unique_dependence_groups=32,
                    nearest_distance=0.5,
                    max_distance_used=2.0,
                    support_dependence_group_hash="SUPPORT",
                )
            )
        admission = EvidenceAdmissionReceiptR6(
            status="ADMITTED",
            lane="CENTER",
            unique_train_dependence_groups=40,
            minimum_action_effective_dependence_n=24.0,
            maximum_action_nearest_distance=0.5,
            reasons=(),
            protocol_hash="PROTO",
        )
        evidence.append(
            DependenceAwareTeacherEvidenceR6(
                evidence_id=f"E{i:03d}",
                parent_id=pid,
                student_context_object_id=f"CTX{i:03d}",
                target_dependence_group_id=f"G{i:03d}",
                timestamp=i,
                teacher_version="TEST",
                teacher_protocol_hash="PROTO",
                train_dependence_group_hash="TRAIN",
                action_laws=tuple(laws),
                direction_target_probs=(0.30, 0.40, 0.30),
                requested_risk_target=0.20,
                direction_weight=1.0,
                sizing_weight=1.0,
                admission=admission,
            )
        )
    return evidence, parents


def test_teacher_action_law_integrity_and_tail_canary_preserves_student_targets():
    evidence, _ = _fixture()
    audit = audit_teacher_action_laws_r11(evidence)
    assert audit["quantile_levels"] == list(LEVELS)
    assert audit["action_laws_per_evidence"] == [9]
    assert audit["quantiles_monotone"] is True

    variant = tail_only_law_perturbation_r11(evidence)
    assert variant[0].content_hash != evidence[0].content_hash
    assert variant[0].direction_target_probs == evidence[0].direction_target_probs
    assert variant[0].requested_risk_target == evidence[0].requested_risk_target
    assert variant[0].action_laws[0].mean_utility == evidence[0].action_laws[0].mean_utility
    assert variant[0].action_laws[0].quantiles != evidence[0].action_laws[0].quantiles


def test_tail_only_distribution_change_is_outside_current_student_gradient_path():
    evidence, parents = _fixture()
    tail = tail_only_law_perturbation_r11(evidence)
    original = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    variant = PreparedEvidenceR11.from_evidence(tail, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    comparison = compare_student_projection_paths_r11(
        model=model, original=original, variant=variant
    )
    assert comparison["prepared_evidence_identity_equal"] is False
    assert comparison["packed_tensor_equal"] is True
    assert comparison["loss_equal_within_atol"] is True
    assert comparison["gradient_equal_within_atol"] is True


def test_projected_target_positive_control_reaches_student_gradient():
    evidence, parents = _fixture()
    target_variant = projected_target_perturbation_r11(evidence)
    original = PreparedEvidenceR11.from_evidence(evidence, parents, device="cpu")
    variant = PreparedEvidenceR11.from_evidence(target_variant, parents, device="cpu")
    model = build_g0_brain_r10("TIER_1", seed=24680, device="cpu")
    comparison = compare_student_projection_paths_r11(
        model=model, original=original, variant=variant
    )
    assert comparison["packed_tensor_equal"] is False
    assert comparison["gradient_equal_within_atol"] is False

    boundary = classify_distributional_boundary_r11(
        teacher_law_audit=audit_teacher_action_laws_r11(evidence),
        tail_projection=compare_student_projection_paths_r11(
            model=model,
            original=original,
            variant=PreparedEvidenceR11.from_evidence(
                tail_only_law_perturbation_r11(evidence), parents, device="cpu"
            ),
        ),
        target_projection_control=comparison,
    )
    assert boundary["diagnosis"] == (
        "DISTRIBUTIONAL_ACTION_LAW_PRESERVED_AS_EVIDENCE_"
        "BUT_NOT_ON_CURRENT_STUDENT_GRADIENT_PATH"
    )
    assert boundary["qr_dqn_conversion_authorized"] is False
