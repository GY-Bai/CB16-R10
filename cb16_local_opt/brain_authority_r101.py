from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import math
import torch


AUTHORITY_ID_R101 = "CB16_R10_1_CENTRAL_BRAIN_STEM_AUTHORITY_V1"

# R10 kept historical attribute names *_encoder. R10.1 resolves the naming collision
# without changing checkpoint keys or tensor values: these modules are Central-Brain-owned
# trainable read-in stems, not frozen sensory organs and not an upstream Account encoder.
LOGICAL_PARAMETER_GROUPS_R101: Dict[str, tuple[str, ...]] = {
    "operator_brain_stem": ("operator_encoder.",),
    "medium_brain_stem": ("medium_encoder.",),
    "account_brain_stem": ("account_encoder.",),
    "shared_decision_core": ("shared_core.",),
    "direction_head": ("direction_body.", "direction_out."),
    "requested_risk_head": ("direction_embedding.", "sizing_body.", "sizing_out."),
}

EXPECTED_PARAMETER_COUNTS_R101 = {
    "operator_brain_stem": 3136,
    "medium_brain_stem": 3136,
    "account_brain_stem": 224,
    "shared_decision_core": 107008,
    "direction_head": 33283,
    "requested_risk_head": 42265,
    "total": 189052,
    "trainable": 189052,
}

FROZEN_EXTERNAL_AUTHORITIES_R101 = (
    "Operator48 organ: Kronos-small prefix + tokenizer encode prefix + frozen Micro24/Macro24 reducers",
    "Medium48 organ: TimesFM Layer3 prefix + frozen Nonlinear48 seed24680",
    "Ordered4H30 risk sidecar: SHADOW/TASK_GATED, not nominal Brain input",
    "Frozen Supervisor",
    "Frozen Physics",
)


@dataclass(frozen=True)
class GradientAuthorityReceiptR101:
    authority_id: str
    parameter_counts: dict
    grad_norms: dict
    update_norms: dict
    external_input_gradients_blocked: bool
    all_brain_groups_received_gradient: bool
    all_brain_groups_updated: bool
    status: str


def _matches(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(p) for p in prefixes)


def logical_parameter_report_r101(model: torch.nn.Module) -> dict:
    out = {k: 0 for k in LOGICAL_PARAMETER_GROUPS_R101}
    out["total"] = 0
    out["trainable"] = 0
    unknown = []
    for name, p in model.named_parameters():
        n = p.numel()
        out["total"] += n
        if p.requires_grad:
            out["trainable"] += n
        matched = False
        for group, prefixes in LOGICAL_PARAMETER_GROUPS_R101.items():
            if _matches(name, prefixes):
                out[group] += n
                matched = True
                break
        if not matched:
            unknown.append(name)
    out["unknown_parameter_names"] = unknown
    return out


def validate_brain_authority_r101(model: torch.nn.Module) -> dict:
    report = logical_parameter_report_r101(model)
    failures = []
    for k, expected in EXPECTED_PARAMETER_COUNTS_R101.items():
        if report.get(k) != expected:
            failures.append(f"PARAM_COUNT:{k}:{report.get(k)}!={expected}")
    if report["unknown_parameter_names"]:
        failures.append("UNKNOWN_PARAMETER_NAMES:" + ",".join(report["unknown_parameter_names"]))
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        failures.append("BRAIN_PARAMETER_FROZEN:" + ",".join(frozen))
    return {
        "authority_id": AUTHORITY_ID_R101,
        "classification": {
            "operator_encoder.*": "CENTRAL_BRAIN_OPERATOR_STEM__TRAINABLE",
            "medium_encoder.*": "CENTRAL_BRAIN_MEDIUM_STEM__TRAINABLE",
            "account_encoder.*": "CENTRAL_BRAIN_ACCOUNT_STEM__TRAINABLE",
            "shared_core.*": "CENTRAL_BRAIN_SHARED_DECISION_CORE__TRAINABLE",
            "direction_body.*|direction_out.*": "CENTRAL_BRAIN_DIRECTION_HEAD__TRAINABLE",
            "direction_embedding.*|sizing_body.*|sizing_out.*": "CENTRAL_BRAIN_REQUESTED_RISK_HEAD__TRAINABLE",
        },
        "frozen_external_authorities": list(FROZEN_EXTERNAL_AUTHORITIES_R101),
        "parameter_report": report,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def _group_tensor_norm(named_tensors, group: str) -> float:
    prefixes = LOGICAL_PARAMETER_GROUPS_R101[group]
    s = 0.0
    for name, tensor in named_tensors:
        if _matches(name, prefixes) and tensor is not None:
            x = tensor.detach().double()
            s += float(torch.sum(x * x).item())
    return math.sqrt(s)


def gradient_authority_canary_r101(model: torch.nn.Module, *, device: str = "cpu") -> dict:
    """Synthetic, no-market optimizer canary. Operates on a model copy supplied by caller.

    PASS means all six logical Central Brain groups receive and apply a legal update while
    gradients are stopped at all three typed external observation boundaries.
    """
    model = model.to(device)
    model.train()
    auth = validate_brain_authority_r101(model)
    if auth["status"] != "PASS":
        raise RuntimeError("BRAIN_AUTHORITY_INVALID:" + ";".join(auth["failures"]))

    torch.manual_seed(10101)
    operator = torch.randn(32, 48, device=device, requires_grad=True)
    medium = torch.randn(32, 48, device=device, requires_grad=True)
    account = torch.randn(32, 6, device=device, requires_grad=True)
    target_probs = torch.softmax(torch.randn(32, 3, device=device), dim=-1)
    target_risk = torch.sigmoid(torch.randn(32, device=device))

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    out = model(operator, medium, account)
    logp = torch.log_softmax(out["direction_logits"], dim=-1)
    direction_loss = -(target_probs * logp).sum(dim=-1).mean()
    sizing_loss = torch.mean((out["requested_risk_raw"] - target_risk) ** 2)
    loss = direction_loss + sizing_loss
    loss.backward()

    grad_norms = {
        g: _group_tensor_norm(((n, p.grad) for n, p in model.named_parameters()), g)
        for g in LOGICAL_PARAMETER_GROUPS_R101
    }
    external_blocked = all(x.grad is None or float(x.grad.detach().abs().max().item()) == 0.0 for x in (operator, medium, account))
    all_grad = all(math.isfinite(v) and v > 0.0 for v in grad_norms.values())
    opt.step()
    update_norms = {}
    for g, prefixes in LOGICAL_PARAMETER_GROUPS_R101.items():
        s = 0.0
        for n, p in model.named_parameters():
            if _matches(n, prefixes):
                d = (p.detach() - before[n]).double()
                s += float(torch.sum(d * d).item())
        update_norms[g] = math.sqrt(s)
    all_update = all(math.isfinite(v) and v > 0.0 for v in update_norms.values())
    status = "PASS" if external_blocked and all_grad and all_update else "FAIL"
    return {
        "schema": "CB16_R10_1_GRADIENT_AUTHORITY_CANARY_V1",
        "authority_id": AUTHORITY_ID_R101,
        "loss": float(loss.detach().cpu()),
        "parameter_counts": auth["parameter_report"],
        "grad_norms": grad_norms,
        "update_norms": update_norms,
        "external_input_gradients_blocked": external_blocked,
        "all_brain_groups_received_gradient": all_grad,
        "all_brain_groups_updated": all_update,
        "status": status,
    }
