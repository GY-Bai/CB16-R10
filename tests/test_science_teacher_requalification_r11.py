from __future__ import annotations

from dataclasses import replace

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.probabilistic_teacher_r6 import DependenceAwareTeacherConfigR6
from cb16_local_opt.science_teacher_requalification_r11 import (
    DependenceBalancedHistoricalControlSuiteShadowR11,
    compare_control_receipts_r11,
)
from cb16_local_opt.scientific_controls_r6 import (
    DependenceAwareControlSuiteConfigR6,
    DependenceAwareHistoricalControlSuiteR6,
)


GRID = ((-1, 0.25), (-1, 0.50), (-1, 0.75), (-1, 1.0), (0, 0.0),
        (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.0))


def _teacher_config():
    return DependenceAwareTeacherConfigR6(
        teacher_version="R11_R2_2_TEST",
        mode="PREQUENTIAL",
        n_folds=3,
        embargo_groups=0,
        k_dependence_groups=6,
        min_train_dependence_groups=4,
        min_effective_dependence_n=1.0,
        max_nearest_distance=100.0,
        distance_temperature=2.0,
        direction_softmax_temperature=0.01,
        quantile_levels=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        lane="CENTER",
    )


def _control_config():
    return DependenceAwareControlSuiteConfigR6(
        teacher=_teacher_config(),
        market_dim=96,
        account_dim=6,
        bootstrap_reps=100,
        bootstrap_alpha=0.05,
        minimum_scored_dependence_groups=1,
    )


def _market(g: int):
    return tuple(0.001 * (g + 1) * (i + 1) for i in range(96))


def _account(g: int, a: int):
    return (
        float((-1) ** a),
        0.05 * a,
        0.01 * (g + 1),
        float(a + 1),
        -0.02 * a,
        0.5 + 0.1 * a,
    )


def _fixture():
    rows = []
    target_ids = []
    train_deps = []
    for g in range(10):
        dep = f"G{g:02d}"
        if g < 8:
            train_deps.append(dep)
        for a in range(2):
            pid = f"P{g:02d}A{a}"
            context_id = f"CTX:{g:02d}:{a}"
            x = _market(g) + _account(g, a)
            for j, (direction, risk) in enumerate(GRID):
                # deterministic but state/action-sensitive utility surface
                utility = (
                    0.0007 * g
                    + 0.00025 * direction
                    - 0.00012 * risk
                    + 0.00003 * a * direction
                    + 0.00001 * j
                )
                rows.append(CounterfactualBranchSampleR5(
                    parent_id=pid,
                    student_context_object_id=context_id,
                    timestamp=1000 + g,
                    context_features=x,
                    direction=direction,
                    requested_risk=risk,
                    realized_utility=utility,
                    dependence_group_id=dep,
                    market_lineage_hash=f"M{g:02d}",
                ))
            if g >= 8:
                target_ids.append(pid)
    return rows, target_ids, train_deps


def _with_exact_replicas(rows, *, source_parent="P02A0", replicas=20):
    source = [x for x in rows if x.parent_id == source_parent]
    assert len(source) == 9
    out = list(rows)
    for i in range(replicas):
        clone = f"ZZ_REPLICA_{i:03d}:{source_parent}"
        # Preserve Student context ID, features, dependence group and utility.
        out.extend(replace(x, parent_id=clone) for x in source)
    return out


def _q(receipt):
    return {x.formulation: x.qcrps for x in receipt.formulations}


def test_shadow_control_suite_is_exact_replica_invariant_but_current_is_not():
    rows, targets, train_deps = _fixture()
    replicas = _with_exact_replicas(rows)
    config = _control_config()

    current = DependenceAwareHistoricalControlSuiteR6(config)
    shadow = DependenceBalancedHistoricalControlSuiteShadowR11(config)

    current_base = current.evaluate(rows, target_parent_ids=targets,
                                    eligible_train_dependence_group_ids=train_deps)
    current_replica = current.evaluate(replicas, target_parent_ids=targets,
                                       eligible_train_dependence_group_ids=train_deps)
    shadow_base = shadow.evaluate(rows, target_parent_ids=targets,
                                  eligible_train_dependence_group_ids=train_deps)
    shadow_replica = shadow.evaluate(replicas, target_parent_ids=targets,
                                     eligible_train_dependence_group_ids=train_deps)

    assert current_base.status == current_replica.status == "PASS"
    assert shadow_base.status == shadow_replica.status == "PASS"
    assert any(abs(_q(current_replica)[f] - _q(current_base)[f]) > 0.0 for f in _q(current_base))
    assert _q(shadow_base) == _q(shadow_replica)
    assert shadow_base.f2_minus_f0.mean_delta == shadow_replica.f2_minus_f0.mean_delta
    assert shadow_base.f3_minus_f2.mean_delta == shadow_replica.f3_minus_f2.mean_delta
    assert shadow_base.f1_minus_f0.mean_delta == shadow_replica.f1_minus_f0.mean_delta


def test_shadow_and_current_are_nearly_equivalent_on_balanced_input():
    rows, targets, train_deps = _fixture()
    config = _control_config()
    current = DependenceAwareHistoricalControlSuiteR6(config).evaluate(
        rows, target_parent_ids=targets, eligible_train_dependence_group_ids=train_deps
    )
    shadow = DependenceBalancedHistoricalControlSuiteShadowR11(config).evaluate(
        rows, target_parent_ids=targets, eligible_train_dependence_group_ids=train_deps
    )
    comparison = compare_control_receipts_r11(current, shadow)
    assert comparison["maximum_abs_qcrps_delta"] < 1e-12
    assert abs(comparison["f3_minus_f2_delta_difference"]) < 1e-12
