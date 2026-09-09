from __future__ import annotations

from dataclasses import asdict, replace

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.r102_learning import TRAIN_TEACHER_CONFIG_R102, VAL_TEACHER_CONFIG_R102
from cb16_local_opt.r11_teacher_authority_candidate import (
    R11_TRAIN_TEACHER_CONFIG,
    R11_VALIDATION_TEACHER_CONFIG,
    DependenceBalancedExactIncrementalTeacherR11,
    DependenceBalancedProbabilisticTeacherR11,
)
from cb16_local_opt.science_teacher_requalification_r11 import (
    DependenceBalancedProbabilisticTeacherShadowR11,
)

GRID = ((-1, 0.25), (-1, 0.50), (-1, 0.75), (-1, 1.0), (0, 0.0),
        (1, 0.25), (1, 0.50), (1, 0.75), (1, 1.0))


def _fixture():
    rows = []
    train_deps = []
    for g in range(10):
        dep = f"G{g:02d}"
        if g < 8:
            train_deps.append(dep)
        market = tuple(0.001 * (g + 1) * (i + 1) for i in range(96))
        for a in range(2):
            pid = f"P{g:02d}A{a}"
            account = (float((-1) ** a), 0.05 * a, 0.01 * (g + 1),
                       float(a + 1), -0.02 * a, 0.5 + 0.1 * a)
            context = market + account
            for j, (direction, risk) in enumerate(GRID):
                utility = (0.0007 * g + 0.00025 * direction - 0.00012 * risk
                           + 0.00003 * a * direction + 0.00001 * j)
                rows.append(CounterfactualBranchSampleR5(
                    parent_id=pid,
                    student_context_object_id=f"CTX:{g:02d}:{a}",
                    timestamp=1000 + g,
                    context_features=context,
                    direction=direction,
                    requested_risk=risk,
                    realized_utility=utility,
                    dependence_group_id=dep,
                    market_lineage_hash=f"M{g:02d}",
                ))
    return rows, set(train_deps)


def _replicated(rows, source_parent="P02A0", replicas=20):
    source = [x for x in rows if x.parent_id == source_parent]
    out = list(rows)
    for i in range(replicas):
        pid = f"ZZ_REPLICA_{i:03d}:{source_parent}"
        out.extend(replace(x, parent_id=pid) for x in source)
    return out


def _law_semantics(evidence):
    return [asdict(x) for x in evidence.action_laws]


def _admission_semantics(evidence):
    x = asdict(evidence.admission)
    x.pop("protocol_hash")
    return x


def test_r11_candidate_changes_identity_but_not_hyperparameters():
    old_train = asdict(TRAIN_TEACHER_CONFIG_R102)
    new_train = asdict(R11_TRAIN_TEACHER_CONFIG)
    old_val = asdict(VAL_TEACHER_CONFIG_R102)
    new_val = asdict(R11_VALIDATION_TEACHER_CONFIG)
    assert old_train.pop("teacher_version") != new_train.pop("teacher_version")
    assert old_val.pop("teacher_version") != new_val.pop("teacher_version")
    assert old_train == new_train
    assert old_val == new_val
    assert TRAIN_TEACHER_CONFIG_R102.content_hash != R11_TRAIN_TEACHER_CONFIG.content_hash
    assert VAL_TEACHER_CONFIG_R102.content_hash != R11_VALIDATION_TEACHER_CONFIG.content_hash


def test_candidate_matches_r22_shadow_semantics_with_new_protocol_identity():
    rows, train_deps = _fixture()
    target = "P09A0"
    old = DependenceBalancedProbabilisticTeacherShadowR11(VAL_TEACHER_CONFIG_R102)
    new = DependenceBalancedProbabilisticTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
    idx_old = old.index(rows)
    idx_new = new.index(rows)
    e_old = old.compile_one(target_parent=target, index=idx_old,
                            eligible_train_dependence_groups=train_deps)
    e_new = new.compile_one(target_parent=target, index=idx_new,
                            eligible_train_dependence_groups=train_deps)
    assert _law_semantics(e_old) == _law_semantics(e_new)
    assert e_old.direction_target_probs == e_new.direction_target_probs
    assert e_old.requested_risk_target == e_new.requested_risk_target
    assert _admission_semantics(e_old) == _admission_semantics(e_new)
    assert e_old.teacher_protocol_hash != e_new.teacher_protocol_hash
    assert e_old.evidence_id != e_new.evidence_id
    assert e_new.teacher_protocol_hash == R11_VALIDATION_TEACHER_CONFIG.content_hash


def test_candidate_is_exact_replica_invariant():
    rows, train_deps = _fixture()
    replica_rows = _replicated(rows)
    target = "P09A0"
    teacher = DependenceBalancedProbabilisticTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
    base_idx = teacher.index(rows)
    replica_idx = teacher.index(replica_rows)
    base = teacher.compile_one(target_parent=target, index=base_idx,
                               eligible_train_dependence_groups=train_deps)
    replica = teacher.compile_one(target_parent=target, index=replica_idx,
                                  eligible_train_dependence_groups=train_deps)
    assert _law_semantics(base) == _law_semantics(replica)
    assert base.direction_target_probs == replica.direction_target_probs
    assert base.requested_risk_target == replica.requested_risk_target
    assert asdict(base.admission) == asdict(replica.admission)


def test_incremental_candidate_matches_nonincremental_candidate():
    rows, train_deps = _fixture()
    target = "P09A1"
    direct = DependenceBalancedProbabilisticTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
    incremental = DependenceBalancedExactIncrementalTeacherR11(R11_VALIDATION_TEACHER_CONFIG)
    idx = direct.index(rows)
    a = direct.compile_one(target_parent=target, index=idx,
                           eligible_train_dependence_groups=train_deps)
    b = incremental.compile_one(target_parent=target, index=idx,
                                eligible_train_dependence_groups=train_deps)
    assert _law_semantics(a) == _law_semantics(b)
    assert a.direction_target_probs == b.direction_target_probs
    assert a.requested_risk_target == b.requested_risk_target
    assert asdict(a.admission) == asdict(b.admission)
    assert a.teacher_protocol_hash == b.teacher_protocol_hash
