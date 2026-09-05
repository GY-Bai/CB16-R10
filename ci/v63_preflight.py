#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cb16_local_opt.probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from cb16_local_opt.probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
)
from cb16_local_opt.r102_learning import parameter_group_norms
from cb16_local_opt.typed_central_brain_r10 import build_g0_brain_r10

ROOT = Path(__file__).resolve().parents[1]


def teacher_distribution_canary() -> dict:
    cfg = DependenceAwareTeacherConfigR6(
        teacher_version="CB16_INFRA_V6_3_PREFLIGHT_TEACHER_R0",
        mode="PREQUENTIAL",
        n_folds=5,
        embargo_groups=0,
        k_dependence_groups=64,
        min_train_dependence_groups=1,
        min_effective_dependence_n=1.0,
        max_nearest_distance=1e6,
        distance_temperature=2.0,
        direction_softmax_temperature=0.01,
        lane="CENTER",
    )
    teacher = DependenceAwareProbabilisticTeacherR6(cfg)
    samples: list[CounterfactualBranchSampleR5] = []

    # Forty independent future groups, each with two account replicas. The replicas
    # must never count as independent support by themselves.
    for g in range(40):
        dep = f"DEP_{g:03d}"
        ts = 1_700_000_000 + g
        base = (g - 19.5) * 0.001
        for replica in range(2):
            parent = f"P_{g:03d}_R{replica}"
            feats = (g / 10.0, float(replica))
            for direction, risk, utility in (
                (-1, 0.5, -base + replica * 0.0001),
                (0, 0.0, 0.0002 * math.sin(g)),
                (1, 0.5, base + replica * 0.0001),
            ):
                samples.append(
                    CounterfactualBranchSampleR5(
                        parent_id=parent,
                        student_context_object_id=f"CTX_{parent}",
                        timestamp=ts,
                        context_features=feats,
                        direction=direction,
                        requested_risk=risk,
                        realized_utility=utility,
                        dependence_group_id=dep,
                        market_lineage_hash=f"MKT_{dep}",
                    )
                )

    # Target group is strictly later and therefore must be excluded from its own
    # PREQUENTIAL teacher support.
    target_dep = "DEP_TARGET"
    target_parent = "P_TARGET"
    target_ts = 1_700_000_100
    for direction, risk, utility in ((-1, 0.5, -0.01), (0, 0.0, 0.0), (1, 0.5, 0.01)):
        samples.append(
            CounterfactualBranchSampleR5(
                parent_id=target_parent,
                student_context_object_id="CTX_TARGET",
                timestamp=target_ts,
                context_features=(4.0, 0.0),
                direction=direction,
                requested_risk=risk,
                realized_utility=utility,
                dependence_group_id=target_dep,
                market_lineage_hash="MKT_TARGET",
            )
        )

    index = teacher.index(samples)
    train_deps = teacher.train_dependence_groups_for_target(
        target_parent=target_parent,
        index=index,
    )
    assert target_dep not in train_deps, "TARGET_DEPENDENCE_GROUP_LEAKED_INTO_SUPPORT"
    assert len(train_deps) == 40, f"EXPECTED_40_INDEPENDENT_GROUPS_GOT_{len(train_deps)}"

    law = teacher.predictive_law(
        target_features=(4.0, 0.0),
        train_deps=train_deps,
        index=index,
        direction=1,
        risk=0.5,
        equal_weight_climatology=True,
    )
    assert law is not None, "NO_PREDICTIVE_LAW"
    assert law.unique_dependence_groups == 40, "ACCOUNT_REPLICAS_INFLATED_SUPPORT"
    assert abs(law.effective_dependence_n - 40.0) < 1e-9, "EFFECTIVE_N_NOT_GROUP_BASED"
    assert law.std_utility > 0.0, "PREDICTIVE_LAW_COLLAPSED_TO_POINT_TARGET"
    assert law.quantiles[-1] > law.quantiles[0], "PREDICTIVE_QUANTILES_COLLAPSED"

    return {
        "status": "PASS",
        "independent_groups": len(train_deps),
        "effective_dependence_n": law.effective_dependence_n,
        "std_utility": law.std_utility,
        "q05": law.quantiles[0],
        "q95": law.quantiles[-1],
    }


def brain_authority_canary() -> dict:
    model = build_g0_brain_r10("TIER_1", seed=6301, device="cpu")
    op = torch.randn(8, 48, requires_grad=True)
    med = torch.randn(8, 48, requires_grad=True)
    acc = torch.randn(8, 6, requires_grad=True)

    out = model(op, med, acc)
    loss = -F.log_softmax(out["direction_logits"], dim=-1)[:, 2].mean()
    loss = loss + out["requested_risk_raw"].mean()
    loss.backward()

    norms = parameter_group_norms(model, gradients=True)
    assert all(math.isfinite(v) and v > 0.0 for v in norms.values()), f"BRAIN_GRADIENT_DISCONNECT:{norms}"
    assert op.grad is None and med.grad is None and acc.grad is None, "FROZEN_ORGAN_INPUTS_OWNED_BY_AUTOGRAD"

    manual = {
        "direction_probs": torch.tensor(
            [[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]],
            dtype=torch.float32,
        ),
        "requested_risk_raw": torch.tensor([0.2, 0.9, 0.8], dtype=torch.float32),
    }
    action = model.compose_action(manual)
    assert action["direction"].tolist() == [-1, 0, 1], "DIRECTION_CLASS_MAPPING_BROKEN"
    expected_risk = torch.tensor([0.2, 0.0, 0.8], dtype=torch.float32)
    assert torch.allclose(action["requested_risk"], expected_risk), "FLAT_RISK_NOT_ZEROED"

    return {
        "status": "PASS",
        "gradient_group_norms": norms,
        "frozen_input_gradients": "NONE",
        "direction_mapping": [-1, 0, 1],
        "flat_risk": 0.0,
    }


def holdout_lock_canary() -> dict:
    p = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "final_holdout_LOCKED.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    msg = (p.stdout or "") + (p.stderr or "")
    assert p.returncode != 0, "FINAL_HOLDOUT_LOCK_SCRIPT_UNEXPECTEDLY_SUCCEEDED"
    assert "FINAL_HOLDOUT_LOCKED" in msg, "FINAL_HOLDOUT_LOCK_MARKER_MISSING"
    return {"status": "PASS", "returncode": p.returncode, "marker": "FINAL_HOLDOUT_LOCKED"}


def main() -> int:
    result = {
        "schema": "CB16_INFRA_V6_3_PREFLIGHT_R0",
        "teacher_distribution": teacher_distribution_canary(),
        "brain_authority": brain_authority_canary(),
        "final_holdout_lock": holdout_lock_canary(),
    }
    result["status"] = "PASS"
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
