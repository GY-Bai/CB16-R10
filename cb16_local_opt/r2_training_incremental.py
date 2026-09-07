from __future__ import annotations

"""R2-only prepared evidence and fused policy evaluation.

This module preserves the frozen R10.2 training mathematics while removing runtime
recomputation that is irrelevant to scientific identity:

- immutable Teacher evidence is converted to device tensors once per campaign;
- validation loss and behavior fingerprint share one forward pass;
- train_challenger reuses the prepared train/validation tensors;
- the campaign-provided pre-training validation metrics are reused instead of being
  recomputed inside the trainer.

The legacy r102_learning.py implementation is intentionally left untouched and remains
the equivalence authority for qualification.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .probabilistic_teacher_r6 import DependenceAwareTeacherEvidenceR6
from .r102_common import (
    atomic_write_json,
    clone_state_dict,
    model_parameter_l2_delta,
    model_state_semantic_sha256,
    sha256_obj,
)
from .r102_evidence_cache import ParentContextR102
from .r102_learning import parameter_group_norms, state_group_update_norms
from .typed_central_brain_r10 import TypedCentralBrainR10


@dataclass(frozen=True)
class PreparedEvidenceBatchR2:
    evidence: tuple[DependenceAwareTeacherEvidenceR6, ...]
    operator48: torch.Tensor
    medium48: torch.Tensor
    account6: torch.Tensor
    direction_target_probs: torch.Tensor
    requested_risk_target: torch.Tensor
    group_weights: torch.Tensor
    independent_groups: int
    device: str

    @property
    def rows(self) -> int:
        return len(self.evidence)


def _group_weights(evidence: Sequence[DependenceAwareTeacherEvidenceR6]) -> np.ndarray:
    counts: dict[str, int] = {}
    for e in evidence:
        counts[e.target_dependence_group_id] = counts.get(e.target_dependence_group_id, 0) + 1
    w = np.asarray(
        [1.0 / counts[e.target_dependence_group_id] for e in evidence],
        dtype=np.float32,
    )
    return w / max(float(w.mean()), 1e-12)


def prepare_evidence_batch_r2(
    evidence: Sequence[DependenceAwareTeacherEvidenceR6],
    parents: Mapping[str, ParentContextR102],
    *,
    device: str,
) -> PreparedEvidenceBatchR2:
    ev = tuple(e for e in evidence if e.admission.admitted)
    if not ev:
        raise RuntimeError("NO_ADMITTED_EVIDENCE")
    return PreparedEvidenceBatchR2(
        evidence=ev,
        operator48=torch.tensor(
            [parents[e.parent_id].operator48 for e in ev], dtype=torch.float32, device=device
        ),
        medium48=torch.tensor(
            [parents[e.parent_id].medium48 for e in ev], dtype=torch.float32, device=device
        ),
        account6=torch.tensor(
            [parents[e.parent_id].account6 for e in ev], dtype=torch.float32, device=device
        ),
        direction_target_probs=torch.tensor(
            [e.direction_target_probs for e in ev], dtype=torch.float32, device=device
        ),
        requested_risk_target=torch.tensor(
            [e.requested_risk_target for e in ev], dtype=torch.float32, device=device
        ),
        group_weights=torch.tensor(_group_weights(ev), dtype=torch.float32, device=device),
        independent_groups=len({e.target_dependence_group_id for e in ev}),
        device=str(device),
    )


def _loss_from_output(out: Mapping[str, torch.Tensor], batch: PreparedEvidenceBatchR2):
    logp = F.log_softmax(out["direction_logits"], dim=-1)
    direction = -(batch.direction_target_probs * logp).sum(-1)
    sizing = F.smooth_l1_loss(
        out["requested_risk_raw"],
        batch.requested_risk_target,
        reduction="none",
        beta=0.05,
    )
    w = batch.group_weights
    loss = ((direction + sizing) * w).sum() / w.sum()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "direction_loss": float((direction * w).sum().detach().cpu() / w.sum().detach().cpu()),
        "sizing_loss": float((sizing * w).sum().detach().cpu() / w.sum().detach().cpu()),
        "admitted_rows": batch.rows,
        "independent_groups": batch.independent_groups,
    }
    return loss, metrics


def _behavior_from_output(
    model: TypedCentralBrainR10,
    out: Mapping[str, torch.Tensor],
    batch: PreparedEvidenceBatchR2,
) -> dict[str, Any]:
    act = model.compose_action(out)
    arr = np.concatenate(
        [
            out["direction_probs"].detach().cpu().numpy(),
            out["requested_risk_raw"].detach().cpu().numpy()[:, None],
            act["direction"].detach().cpu().numpy().astype(np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "rows": batch.rows,
        "sha256": sha256_obj(arr.tolist()),
        "mean_direction_probs": arr[:, :3].mean(0).tolist(),
        "mean_requested_risk": float(arr[:, 3].mean()),
        "long_rate": float(np.mean(arr[:, 4] == 1)),
        "flat_rate": float(np.mean(arr[:, 4] == 0)),
        "short_rate": float(np.mean(arr[:, 4] == -1)),
    }


def evaluate_policy_r2(
    model: TypedCentralBrainR10,
    batch: PreparedEvidenceBatchR2,
) -> tuple[dict[str, float], dict[str, Any]]:
    """One forward produces both frozen-validation metrics and behavior fingerprint."""
    model.eval()
    with torch.inference_mode():
        out = model(batch.operator48, batch.medium48, batch.account6)
        _loss, metrics = _loss_from_output(out, batch)
        behavior = _behavior_from_output(model, out, batch)
    return metrics, behavior


def train_challenger_r2(
    *,
    model: TypedCentralBrainR10,
    train_batch: PreparedEvidenceBatchR2,
    val_batch: PreparedEvidenceBatchR2,
    validation_before: Mapping[str, float],
    device: str,
    generation: int,
    snapshot_hash: str,
    receipt_dir: str | Path,
    epochs: int = 12,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Exact R10.2 challenger training with R2 runtime-only reuse.

    Returns (training_receipt, behavior_after).  On recovery of an already trained
    challenger the behavior fingerprint is returned as None; callers may evaluate it
    once from the recovered model if the generation result was not yet committed.
    """
    receipt_dir = Path(receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
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
        model.load_state_dict(state, strict=True)
        model.to(device)
        return json.loads(training_receipt_path.read_text()), None

    model.to(device)
    model.train()
    before = clone_state_dict(model)
    val_before = dict(validation_before)

    if train_batch.independent_groups < 32:
        raise RuntimeError("INSUFFICIENT_INDEPENDENT_TRAIN_GROUPS_FOR_R102")

    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(24680 + int(generation))
    last_grad = None
    steps = 0

    for _epoch in range(int(epochs)):
        perm = torch.randperm(train_batch.rows, generator=gen)
        for start in range(0, train_batch.rows, int(batch_size)):
            ids = perm[start : start + int(batch_size)].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(
                train_batch.operator48[ids],
                train_batch.medium48[ids],
                train_batch.account6[ids],
            )
            direction = -(
                train_batch.direction_target_probs[ids]
                * F.log_softmax(out["direction_logits"], -1)
            ).sum(-1)
            sizing = F.smooth_l1_loss(
                out["requested_risk_raw"],
                train_batch.requested_risk_target[ids],
                reduction="none",
                beta=0.05,
            )
            bw = train_batch.group_weights[ids]
            loss = ((direction + sizing) * bw).sum() / bw.sum()
            if not torch.isfinite(loss):
                raise RuntimeError("NONFINITE_TRAIN_LOSS")
            loss.backward()
            last_grad = parameter_group_norms(model, gradients=True)
            if any(v <= 0.0 or not math.isfinite(v) for v in last_grad.values()):
                raise RuntimeError(f"REAL_EVIDENCE_GRADIENT_DISCONNECT:{last_grad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            steps += 1

    after = clone_state_dict(model)
    update_norms = state_group_update_norms(before, after)
    if any(v <= 0.0 for v in update_norms.values()):
        raise RuntimeError(f"BRAIN_GROUP_NOT_UPDATED:{update_norms}")

    val_after, behavior_after = evaluate_policy_r2(model, val_batch)
    delta = model_parameter_l2_delta(before, after)
    semantic = model_state_semantic_sha256(model)
    receipt = {
        "schema": "CB16_R10_2_CHALLENGER_TRAINING_RECEIPT_V1",
        "generation": generation,
        "snapshot_hash": snapshot_hash,
        "optimizer": "AdamW_FP32",
        "amp": False,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "optimizer_steps": int(steps),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "parameter_l2_delta": delta,
        "gradient_group_norms_last_step": last_grad,
        "update_group_norms": update_norms,
        "challenger_semantic_sha256": semantic,
        "validation_before": val_before,
        "validation_after": val_after,
        "external_frozen_organ_gradients": "NOT_IN_AUTOGRAD_GRAPH__INPUTS_DETACHED",
    }
    atomic_write_json(training_receipt_path, receipt)
    tmp = recovery_path.with_suffix(recovery_path.suffix + ".tmp")
    torch.save(
        {
            "schema": "CB16_R10_2_TRAINED_CHALLENGER_RECOVERY_V1",
            "snapshot_hash": snapshot_hash,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        tmp,
    )
    os.replace(tmp, recovery_path)
    atomic_write_json(
        consume_path,
        {
            "schema": "CB16_R10_2_SNAPSHOT_CONSUMPTION_V1",
            "snapshot_hash": snapshot_hash,
            "generation": generation,
            "status": "CONSUMED_EXACTLY_ONCE",
            "recovery_checkpoint": str(recovery_path),
            "training_receipt": str(training_receipt_path),
        },
    )
    return receipt, behavior_after
