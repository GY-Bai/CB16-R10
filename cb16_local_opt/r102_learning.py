from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .probabilistic_teacher_r6 import (
    DependenceAwareProbabilisticTeacherR6, DependenceAwareTeacherConfigR6,
    DependenceAwareTeacherEvidenceR6,
)
from .r102_common import (
    atomic_write_json, clone_state_dict, model_parameter_l2_delta,
    model_state_semantic_sha256, sha256_obj,
)
from .r102_evidence_cache import ParentContextR102
from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake
from .typed_central_brain_r10 import TypedCentralBrainR10


TRAIN_TEACHER_CONFIG_R102 = DependenceAwareTeacherConfigR6(
    teacher_version="CB16_R10_2_BLOCKED_CROSSFIT_TEACHER_V1",
    mode="BLOCKED_CROSSFIT", n_folds=5, embargo_groups=1,
    k_dependence_groups=64, min_train_dependence_groups=32,
    min_effective_dependence_n=12.0, max_nearest_distance=8.0,
    distance_temperature=2.0, direction_softmax_temperature=0.002,
    lane="CENTER", direction_weight=1.0, sizing_weight=1.0,
)
VAL_TEACHER_CONFIG_R102 = DependenceAwareTeacherConfigR6(
    teacher_version="CB16_R10_2_PREQUENTIAL_VALIDATION_TEACHER_V1",
    mode="PREQUENTIAL", n_folds=5, embargo_groups=0,
    k_dependence_groups=64, min_train_dependence_groups=32,
    min_effective_dependence_n=12.0, max_nearest_distance=8.0,
    distance_temperature=2.0, direction_softmax_temperature=0.002,
    lane="CENTER", direction_weight=1.0, sizing_weight=1.0,
)


def compile_teacher_evidence(samples, parents: Mapping[str, ParentContextR102]):
    train_parent_ids = sorted(p.parent_id for p in parents.values() if p.split == "TRAIN")
    val_parent_ids = sorted(p.parent_id for p in parents.values() if p.split == "VALIDATION")
    train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    train_teacher = DependenceAwareProbabilisticTeacherR6(TRAIN_TEACHER_CONFIG_R102)
    val_teacher = DependenceAwareProbabilisticTeacherR6(VAL_TEACHER_CONFIG_R102)
    idx_train = train_teacher.index(samples)
    idx_val = val_teacher.index(samples)
    train_e = [train_teacher.compile_one(target_parent=p, index=idx_train, eligible_train_dependence_groups=train_groups) for p in train_parent_ids if p in idx_train.rows_by_parent]
    val_e = [val_teacher.compile_one(target_parent=p, index=idx_val, eligible_train_dependence_groups=train_groups) for p in val_parent_ids if p in idx_val.rows_by_parent]
    return train_e, val_e


def evidence_summary(rows: Sequence[DependenceAwareTeacherEvidenceR6]) -> dict[str, Any]:
    admitted = [e for e in rows if e.admission.admitted]
    reasons = {}
    for e in rows:
        for r in e.admission.reasons:
            reasons[r] = reasons.get(r, 0) + 1
    groups = {e.target_dependence_group_id for e in admitted}
    return {
        "total": len(rows), "admitted": len(admitted), "admitted_dependence_groups": len(groups),
        "rejected": len(rows) - len(admitted), "rejection_reasons": reasons,
        "teacher_protocol_hashes": sorted({e.teacher_protocol_hash for e in rows}),
    }


def _group_weights(evidence: Sequence[DependenceAwareTeacherEvidenceR6]) -> np.ndarray:
    counts = {}
    for e in evidence:
        counts[e.target_dependence_group_id] = counts.get(e.target_dependence_group_id, 0) + 1
    w = np.asarray([1.0 / counts[e.target_dependence_group_id] for e in evidence], dtype=np.float32)
    return w / max(float(w.mean()), 1e-12)


def _evidence_tensors(evidence, parents, device):
    ev = [e for e in evidence if e.admission.admitted]
    if not ev:
        raise RuntimeError("NO_ADMITTED_EVIDENCE")
    op = torch.tensor([parents[e.parent_id].operator48 for e in ev], dtype=torch.float32, device=device)
    med = torch.tensor([parents[e.parent_id].medium48 for e in ev], dtype=torch.float32, device=device)
    acc = torch.tensor([parents[e.parent_id].account6 for e in ev], dtype=torch.float32, device=device)
    dp = torch.tensor([e.direction_target_probs for e in ev], dtype=torch.float32, device=device)
    rt = torch.tensor([e.requested_risk_target for e in ev], dtype=torch.float32, device=device)
    w = torch.tensor(_group_weights(ev), dtype=torch.float32, device=device)
    return ev, op, med, acc, dp, rt, w


def soft_teacher_loss(model, evidence, parents, *, device) -> tuple[torch.Tensor, dict[str, float]]:
    ev, op, med, acc, dp, rt, w = _evidence_tensors(evidence, parents, device)
    out = model(op, med, acc)
    logp = F.log_softmax(out["direction_logits"], dim=-1)
    direction = -(dp * logp).sum(-1)
    sizing = F.smooth_l1_loss(out["requested_risk_raw"], rt, reduction="none", beta=0.05)
    loss = ((direction + sizing) * w).sum() / w.sum()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "direction_loss": float((direction * w).sum().detach().cpu() / w.sum().detach().cpu()),
        "sizing_loss": float((sizing * w).sum().detach().cpu() / w.sum().detach().cpu()),
        "admitted_rows": len(ev), "independent_groups": len({e.target_dependence_group_id for e in ev}),
    }


def parameter_group_norms(model: TypedCentralBrainR10, *, gradients: bool) -> dict[str, float]:
    mapping = {
        "Operator Brain Stem": ("operator_encoder",),
        "Medium Brain Stem": ("medium_encoder",),
        "Account Brain Stem": ("account_encoder",),
        "Shared Decision Core": ("shared_core",),
        "Direction Head": ("direction_body", "direction_out"),
        "Requested-Risk Head": ("direction_embedding", "sizing_body", "sizing_out"),
    }
    out = {k: 0.0 for k in mapping}
    for role, prefixes in mapping.items():
        s = 0.0
        for n, p in model.named_parameters():
            if n.startswith(prefixes):
                x = p.grad if gradients else p.detach()
                if x is not None:
                    s += float(torch.sum(x.detach().double() ** 2).cpu())
        out[role] = math.sqrt(s)
    return out


def state_group_update_norms(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> dict[str, float]:
    mapping = {
        "Operator Brain Stem": ("operator_encoder",),
        "Medium Brain Stem": ("medium_encoder",),
        "Account Brain Stem": ("account_encoder",),
        "Shared Decision Core": ("shared_core",),
        "Direction Head": ("direction_body", "direction_out"),
        "Requested-Risk Head": ("direction_embedding", "sizing_body", "sizing_out"),
    }
    out = {}
    for role, prefixes in mapping.items():
        s = 0.0
        for n, b in before.items():
            if n.startswith(prefixes):
                d = after[n].detach().cpu().double() - b.detach().cpu().double()
                s += float(torch.sum(d * d))
        out[role] = math.sqrt(s)
    return out


def persist_generation_snapshot(
    *, lake: ShardedExperienceLake, generation: int, champion_hash: str,
    train_evidence: Sequence[DependenceAwareTeacherEvidenceR6], parents: Mapping[str, ParentContextR102],
):
    refs = []
    for e in train_evidence:
        if not e.admission.admitted:
            continue
        p = parents[e.parent_id]
        oid = f"R102:G{generation}:{e.evidence_id}"
        payload = {
            "schema": "CB16_R10_2_EVIDENCE_PACKAGE_V1", "generation": generation,
            "parent_id": e.parent_id, "dependence_group_id": e.target_dependence_group_id,
            "student_context_object_id": e.student_context_object_id,
            "operator48": list(p.operator48), "medium48": list(p.medium48), "account6": list(p.account6),
            "direction_target_probs": list(e.direction_target_probs),
            "requested_risk_target": e.requested_risk_target,
            "action_laws": [asdict(x) for x in e.action_laws],
            "admission": asdict(e.admission),
            "teacher_protocol_hash": e.teacher_protocol_hash,
        }
        obj = ExperienceObject(
            object_id=oid, object_type="EVIDENCE_PACKAGE", generation=generation,
            policy_weight_hash=champion_hash, snapshot_hash=p.snapshot_sha256,
            lineage_hash=e.content_hash, payload=payload,
        )
        ref, _ = lake.put(obj); refs.append(ref)
    snap = lake.seal_snapshot(
        snapshot_id=f"R102_G{generation}_TRAINING_SNAPSHOT",
        parent_generation=generation, parent_policy_hash=champion_hash, refs=refs,
    )
    return snap, refs


def train_challenger(
    *, model: TypedCentralBrainR10, train_evidence, val_evidence, parents,
    device: str, generation: int, snapshot_hash: str, receipt_dir: str | Path,
    epochs: int = 12, batch_size: int = 512, lr: float = 3e-4, weight_decay: float = 1e-4,
) -> dict[str, Any]:
    receipt_dir = Path(receipt_dir); receipt_dir.mkdir(parents=True, exist_ok=True)
    consume_path = receipt_dir / f"SNAPSHOT_CONSUMPTION_G{generation}.json"
    training_receipt_path = receipt_dir / f"CHALLENGER_TRAINING_RECEIPT_G{generation}.json"
    recovery_path = receipt_dir / f"CHALLENGER_TRAINED_RECOVERY_G{generation}.pt"
    if consume_path.exists():
        old_consume = json.loads(consume_path.read_text())
        if old_consume.get("snapshot_hash") != snapshot_hash:
            raise RuntimeError(f"SNAPSHOT_CONSUMPTION_RECEIPT_CONFLICT:{consume_path}")
        if not (training_receipt_path.is_file() and recovery_path.is_file()):
            raise RuntimeError(f"SNAPSHOT_CONSUMED_WITHOUT_RECOVERABLE_CHALLENGER:{snapshot_hash}")
        recovered = torch.load(recovery_path, map_location="cpu", weights_only=True)
        state = recovered.get("state_dict") if isinstance(recovered, dict) else recovered
        model.load_state_dict(state, strict=True); model.to(device)
        return json.loads(training_receipt_path.read_text())

    model.to(device); model.train()
    before = clone_state_dict(model)
    val_before_loss, val_before = soft_teacher_loss(model, val_evidence, parents, device=device)
    train_admitted = [e for e in train_evidence if e.admission.admitted]
    if len({e.target_dependence_group_id for e in train_admitted}) < 32:
        raise RuntimeError("INSUFFICIENT_INDEPENDENT_TRAIN_GROUPS_FOR_R102")
    ev, op, med, acc, dp, rt, w = _evidence_tensors(train_evidence, parents, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device="cpu"); gen.manual_seed(24680 + int(generation))
    last_grad = None
    steps = 0
    for epoch in range(int(epochs)):
        perm = torch.randperm(len(ev), generator=gen)
        for start in range(0, len(ev), int(batch_size)):
            ids = perm[start:start + int(batch_size)].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(op[ids], med[ids], acc[ids])
            direction = -(dp[ids] * F.log_softmax(out["direction_logits"], -1)).sum(-1)
            sizing = F.smooth_l1_loss(out["requested_risk_raw"], rt[ids], reduction="none", beta=0.05)
            bw = w[ids]
            loss = ((direction + sizing) * bw).sum() / bw.sum()
            if not torch.isfinite(loss): raise RuntimeError("NONFINITE_TRAIN_LOSS")
            loss.backward()
            last_grad = parameter_group_norms(model, gradients=True)
            # Every legal Brain group must remain connected on the real evidence batch.
            if any(v <= 0.0 or not math.isfinite(v) for v in last_grad.values()):
                raise RuntimeError(f"REAL_EVIDENCE_GRADIENT_DISCONNECT:{last_grad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step(); steps += 1
    after = clone_state_dict(model)
    update_norms = state_group_update_norms(before, after)
    if any(v <= 0.0 for v in update_norms.values()):
        raise RuntimeError(f"BRAIN_GROUP_NOT_UPDATED:{update_norms}")
    val_after_loss, val_after = soft_teacher_loss(model, val_evidence, parents, device=device)
    delta = model_parameter_l2_delta(before, after)
    semantic = model_state_semantic_sha256(model)
    receipt = {
        "schema": "CB16_R10_2_CHALLENGER_TRAINING_RECEIPT_V1", "generation": generation,
        "snapshot_hash": snapshot_hash, "optimizer": "AdamW_FP32", "amp": False,
        "epochs": int(epochs), "batch_size": int(batch_size), "optimizer_steps": int(steps),
        "lr": float(lr), "weight_decay": float(weight_decay),
        "parameter_l2_delta": delta, "gradient_group_norms_last_step": last_grad,
        "update_group_norms": update_norms, "challenger_semantic_sha256": semantic,
        "validation_before": val_before, "validation_after": val_after,
        "external_frozen_organ_gradients": "NOT_IN_AUTOGRAD_GRAPH__INPUTS_DETACHED",
    }
    atomic_write_json(training_receipt_path, receipt)
    tmp = recovery_path.with_suffix(recovery_path.suffix + ".tmp")
    torch.save({"schema":"CB16_R10_2_TRAINED_CHALLENGER_RECOVERY_V1","snapshot_hash":snapshot_hash,"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}, tmp)
    os.replace(tmp, recovery_path)
    atomic_write_json(consume_path, {"schema": "CB16_R10_2_SNAPSHOT_CONSUMPTION_V1", "snapshot_hash": snapshot_hash, "generation": generation, "status": "CONSUMED_EXACTLY_ONCE", "recovery_checkpoint": str(recovery_path), "training_receipt": str(training_receipt_path)})
    return receipt


def policy_behavior_fingerprint(model: TypedCentralBrainR10, evidence, parents, *, device: str) -> dict[str, Any]:
    ev = [e for e in evidence if e.admission.admitted]
    if not ev: return {"rows": 0}
    op = torch.tensor([parents[e.parent_id].operator48 for e in ev], dtype=torch.float32, device=device)
    med = torch.tensor([parents[e.parent_id].medium48 for e in ev], dtype=torch.float32, device=device)
    acc = torch.tensor([parents[e.parent_id].account6 for e in ev], dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        out = model(op, med, acc); act = model.compose_action(out)
    arr = np.concatenate([
        out["direction_probs"].detach().cpu().numpy(),
        out["requested_risk_raw"].detach().cpu().numpy()[:, None],
        act["direction"].detach().cpu().numpy().astype(np.float32)[:, None],
    ], axis=1).astype(np.float32)
    return {
        "rows": len(ev), "sha256": sha256_obj(arr.tolist()),
        "mean_direction_probs": arr[:, :3].mean(0).tolist(),
        "mean_requested_risk": float(arr[:, 3].mean()),
        "long_rate": float(np.mean(arr[:, 4] == 1)), "flat_rate": float(np.mean(arr[:, 4] == 0)), "short_rate": float(np.mean(arr[:, 4] == -1)),
    }
