from __future__ import annotations

"""R2-only prepared on-policy trace context.

The legacy R10.2 trace function intentionally remains unchanged as qualification
reference.  This module removes generation-invariant runtime work only:

- deterministic selection of at most one CLEAN_FLAT_FULL parent per dependence group;
- market-cache NPZ decompression/loading;
- Student-visible Operator48/Medium48/Account6 tensor construction.

Champion inference and exact H72 Physics are still executed for every generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .r102_common import sha256_obj
from .r102_evidence_cache import ParentContextR102
from .r102_physics import FrozenPhysicsRuntimeR102, simulate_h72_branch
from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake


@dataclass(frozen=True)
class PreparedTraceParentR2:
    parent: ParentContextR102
    parent_state: Mapping[str, Any]
    operator48: torch.Tensor
    medium48: torch.Tensor
    account6: torch.Tensor


@dataclass(frozen=True)
class PreparedOnPolicyTraceContextR2:
    parents: tuple[PreparedTraceParentR2, ...]
    market_by_symbol: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    device: str
    max_groups: int

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.market_by_symbol))


def _select_trace_parents(
    parents: Mapping[str, ParentContextR102], *, max_groups: int
) -> list[ParentContextR102]:
    eligible = [p for p in parents.values() if p.eligible_for_economic_evidence]
    eligible.sort(
        key=lambda p: (
            0 if p.split == "VALIDATION" else 1,
            p.decision_time_ms,
            p.symbol,
            p.scenario,
        )
    )
    chosen = []
    seen = set()
    for p in eligible:
        if p.dependence_group_id in seen:
            continue
        if p.scenario != "CLEAN_FLAT_FULL":
            continue
        chosen.append(p)
        seen.add(p.dependence_group_id)
        if len(chosen) >= int(max_groups):
            break
    if not chosen:
        raise RuntimeError("NO_ON_POLICY_TRACE_CONTEXTS")
    return chosen


def prepare_on_policy_trace_context_r2(
    *,
    parents: Mapping[str, ParentContextR102],
    parent_states: Mapping[str, Mapping[str, Any]],
    cache_dir: str | Path,
    device: str,
    max_groups: int = 24,
) -> PreparedOnPolicyTraceContextR2:
    chosen = _select_trace_parents(parents, max_groups=int(max_groups))
    market_by_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    prepared = []
    for p in chosen:
        if p.symbol not in market_by_symbol:
            hp = Path(cache_dir) / "market_cache" / f"{p.symbol}.hourly_r102.npz"
            with np.load(hp, allow_pickle=False) as z:
                market_by_symbol[p.symbol] = (
                    z["open_time_ms"].copy(),
                    z["ohlcv"].copy(),
                    z["funding_rate"].copy(),
                )
        state = dict(parent_states[p.parent_id])
        state.setdefault("account_id", state["risk_authority"]["account_id"])
        prepared.append(
            PreparedTraceParentR2(
                parent=p,
                parent_state=state,
                operator48=torch.tensor([p.operator48], dtype=torch.float32, device=device),
                medium48=torch.tensor([p.medium48], dtype=torch.float32, device=device),
                account6=torch.tensor([p.account6], dtype=torch.float32, device=device),
            )
        )
    return PreparedOnPolicyTraceContextR2(
        parents=tuple(prepared),
        market_by_symbol=market_by_symbol,
        device=str(device),
        max_groups=int(max_groups),
    )


def run_real_on_policy_trace_r2(
    *,
    model,
    policy_hash: str,
    generation: int,
    prepared: PreparedOnPolicyTraceContextR2,
    physics: FrozenPhysicsRuntimeR102,
    lake: ShardedExperienceLake,
) -> dict[str, Any]:
    traces = []
    model.eval()
    for item in prepared.parents:
        p = item.parent
        # Match the legacy shallow-copy semantics so every generation starts from
        # the same frozen parent state even if downstream Physics mutates top-level fields.
        state = dict(item.parent_state)
        hts, hbar, funding = prepared.market_by_symbol[p.symbol]
        with torch.inference_mode():
            out = model(item.operator48, item.medium48, item.account6)
            act = model.compose_action(out)
        direction = int(act["direction"].item())
        requested_risk = float(act["requested_risk"].item())
        branch = simulate_h72_branch(
            physics,
            parent=state,
            symbol=p.symbol,
            decision_time_ms=p.decision_time_ms,
            candidate_direction_v55=direction + 1,
            candidate_risk=requested_risk,
            hourly_ts=hts,
            hourly_ohlcv=hbar,
            funding=funding,
        )
        trace_id = f"R102TRACE:G{generation}:{p.parent_id}"
        decision_payload = {
            "schema":"CB16_R10_2_DECISION_EVENT_V1", "CausalTraceID":trace_id,
            "generation":generation, "policy_hash":policy_hash, "parent_id":p.parent_id,
            "dependence_group_id":p.dependence_group_id, "symbol":p.symbol,
            "decision_time_ms":p.decision_time_ms, "direction":direction,
            "requested_risk":requested_risk,
            "direction_probs":out["direction_probs"].detach().cpu().numpy()[0].tolist(),
            "ordered4h30_visible_to_nominal_brain":False,
        }
        d_obj = ExperienceObject(
            object_id=f"DECISION:{trace_id}", object_type="DECISION_EVENT", generation=generation,
            policy_weight_hash=policy_hash, snapshot_hash=p.snapshot_sha256,
            lineage_hash=sha256_obj(decision_payload), payload=decision_payload,
        )
        dref, _ = lake.put(d_obj)
        outcome_payload = {
            "schema":"CB16_R10_2_OUTCOME_SAMPLE_V1", "CausalTraceID":trace_id,
            "generation":generation, "parent_id":p.parent_id, "dependence_group_id":p.dependence_group_id,
            "status":branch["status"], "realized_utility":branch.get("utility"),
            "w0":branch.get("w0"), "wt":branch.get("wt"), "terminal_at_step":branch.get("terminal_at_step"),
            "first_step":branch.get("first_step"), "evaluation_finalize":branch.get("finalize"),
            "realized_outcome_is_stochastic_sample_not_correct_action_label":True,
        }
        o_obj = ExperienceObject(
            object_id=f"OUTCOME:{trace_id}", object_type="OUTCOME_SAMPLE", generation=generation,
            policy_weight_hash=policy_hash, snapshot_hash=p.snapshot_sha256,
            lineage_hash=sha256_obj(outcome_payload), payload=outcome_payload,
        )
        oref, _ = lake.put(o_obj)
        traces.append(
            {
                "CausalTraceID":trace_id,
                "decision_ref":dref.identity_hash,
                "outcome_ref":oref.identity_hash,
                "status":branch["status"],
                "utility":branch.get("utility"),
                "direction":direction,
                "requested_risk":requested_risk,
            }
        )
    matured = sum(x["status"] == "MATURED" for x in traces)
    return {
        "schema":"CB16_R10_2_ON_POLICY_REAL_TRACE_RECEIPT_V1", "generation":generation,
        "trace_count":len(traces), "matured":matured,
        "unique_dependence_groups":len({x['CausalTraceID'].split(':',2)[-1] for x in traces}),
        "Brain_to_ActionIntent_to_Supervisor_to_exact_Physics_to_H72_Outcome":"PASS" if matured==len(traces) else "PARTIAL",
        "traces":traces,
    }
