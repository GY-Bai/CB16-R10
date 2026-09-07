from __future__ import annotations

"""R11 H72 / on-policy trace orchestration over frozen R10.2 authority.

No Physics or permission rule is implemented here.  Every action is formed as an
ActionIntent by ``FrozenPhysicsRuntimeR102.step_intent`` which calls the frozen
Supervisor and then the vendored Physics authority.  R11 only removes repeated
market decompression/index construction and parallelizes independent account
trace groups while preserving in-account and per-step chronology.
"""

import copy
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .market_runtime_cache_r11 import MarketRuntimeCacheR11, MarketRuntimeSymbolR11
from .r102_common import H72, sha256_obj
from .r102_evidence_cache import ParentContextR102
from .r102_physics import FLAT, FrozenPhysicsRuntimeR102
from .sharded_experience_lake import ExperienceObject, ShardedExperienceLake


@dataclass(frozen=True)
class H72TraceWorkItemR11:
    ordinal: int
    causal_trace_id: str
    account_id: str
    parent_id: str
    symbol: str
    decision_time_ms: int
    parent_state: Mapping[str, Any]
    direction_v55: int
    requested_risk: float


@dataclass(frozen=True)
class H72TraceExecutionR11:
    ordinal: int
    causal_trace_id: str
    account_id: str
    parent_id: str
    symbol: str
    branch: Mapping[str, Any]
    snapshot_sha256_sequence: tuple[str, ...]


class TraceBatchFutureR11:
    """Future-like batch whose result is canonicalized independently of completion order."""

    def __init__(self, futures: Mapping[Future, tuple[int, str]]):
        self._futures = dict(futures)

    def result(self, timeout: float | None = None) -> list[H72TraceExecutionR11]:
        # ``as_completed`` deliberately accepts arbitrary completion order; the
        # final stable sort is part of the R11 output identity contract.
        rows: list[H72TraceExecutionR11] = []
        for future in as_completed(tuple(self._futures), timeout=timeout):
            rows.extend(future.result())
        rows.sort(key=lambda x: (int(x.ordinal), x.causal_trace_id, x.parent_id))
        ordinals = [int(x.ordinal) for x in rows]
        if len(ordinals) != len(set(ordinals)):
            raise RuntimeError("R11_DUPLICATE_TRACE_ORDINAL")
        trace_ids = [x.causal_trace_id for x in rows]
        if len(trace_ids) != len(set(trace_ids)):
            raise RuntimeError("R11_DUPLICATE_CAUSAL_TRACE_ID")
        return rows


def _simulate_h72_branch_r11(
    physics: FrozenPhysicsRuntimeR102,
    market: MarketRuntimeSymbolR11,
    *,
    parent: Mapping[str, Any],
    decision_time_ms: int,
    candidate_direction_v55: int,
    candidate_risk: float,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Strict orchestration adapter for legacy ``simulate_h72_branch``.

    The only hot-path difference is that the 72 indices were built once by the
    campaign market cache.  Supervisor and Physics calls, action order, funding,
    termination, utility and evaluation-finalize behavior intentionally mirror
    the legacy oracle exactly.
    """
    window = market.h72_slice(int(decision_time_ms))
    indices = range(int(window.start), int(window.stop))
    snap = copy.deepcopy(parent["snapshot"])
    ra = copy.deepcopy(parent["risk_authority"])
    w0 = physics.equity_at_mark(snap, float(parent["current_mark"]))
    if not math.isfinite(w0) or w0 <= 0:
        return (
            {"status": "CENSORED_NONPOSITIVE_INITIAL_EQUITY", "utility": None, "w0": w0},
            (),
        )

    decisions: list[Any] = []
    snapshot_hashes: list[str] = []
    first_step_receipt = None
    terminal_at = None
    last_index = None
    for j, i in enumerate(indices):
        d, r = (candidate_direction_v55, candidate_risk) if j == 0 else (FLAT, 0.0)
        t = int(market.open_time_ms[i])
        step = physics.step_intent(
            snap,
            ra,
            direction_v55=int(d),
            risk=float(r),
            symbol=market.symbol,
            open_time_ms=t,
            ohlcv=market.ohlcv[i],
            funding_rate=float(market.funding_rate[i]),
            # Preserve the byte-for-byte legacy ActionIntent trace spelling.
            trace_id=(
                f"CF:{parent['account_id']}:{int(candidate_direction_v55)}:"
                f"{float(candidate_risk):.2f}:{j}"
            ),
        )
        snap = step["snapshot_t1"]
        snapshot_hashes.append(physics.physics.sha256_obj(snap))
        if j == 0:
            first_step_receipt = {
                "intent": step["intent"],
                "supervisor_decision": step["supervisor_decision"],
                "executable_action": step["executable_action"],
                "execution_metadata": step["execution_metadata"],
            }
        decisions.append(step["supervisor_decision"]["decision"])
        last_index = i
        if snap["termination_state"]["terminated"]:
            terminal_at = j + 1
            break

    if last_index is None:
        raise RuntimeError("R11_H72_EXECUTED_ZERO_STEPS")
    close_price = float(market.ohlcv[int(last_index), 3])
    finalize_receipt: Mapping[str, Any] = {"used": False}
    if terminal_at is None:
        snap, finalize_receipt = physics.evaluation_finalize(snap, close_price)
    wt = physics.equity_at_mark(snap, close_price)
    if not math.isfinite(wt) or wt <= 0:
        return (
            {
                "status": "CENSORED_NONPOSITIVE_TERMINAL_EQUITY",
                "utility": None,
                "w0": w0,
                "wt": wt,
                "terminal_at_step": terminal_at,
                "finalize": finalize_receipt,
                "supervisor_decisions": decisions,
                "first_step": first_step_receipt,
            },
            tuple(snapshot_hashes),
        )
    u = math.log(wt / w0)
    return (
        {
            "status": "MATURED",
            "utility": float(u),
            "w0": float(w0),
            "wt": float(wt),
            "terminal_at_step": terminal_at,
            "finalize": finalize_receipt,
            "supervisor_decisions": decisions,
            "first_step": first_step_receipt,
        },
        tuple(snapshot_hashes),
    )


class TraceRuntimeR11:
    """Campaign-lifetime trace executor with persistent shared-array thread workers."""

    def __init__(
        self,
        *,
        physics: FrozenPhysicsRuntimeR102,
        market_cache: MarketRuntimeCacheR11,
        max_workers: int = 1,
    ):
        workers = int(max_workers)
        if workers <= 0:
            raise ValueError("R11_TRACE_MAX_WORKERS_MUST_BE_POSITIVE")
        self.physics = physics
        self.market_cache = market_cache
        self.max_workers = workers
        # Threads share the read-only market buffers: no pickle/copy of large arrays.
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cb16-r11-trace")
        self._closed = False
        self._lock = RLock()

    def _execute_account_group(
        self, items: Sequence[H72TraceWorkItemR11]
    ) -> list[H72TraceExecutionR11]:
        # A group is one account.  Requests always run in decision-time order,
        # and each H72 request runs its Physics steps sequentially.
        ordered = sorted(
            items, key=lambda x: (int(x.decision_time_ms), int(x.ordinal), x.causal_trace_id)
        )
        if len({x.account_id for x in ordered}) != 1:
            raise RuntimeError("R11_ACCOUNT_GROUP_MIXED_ACCOUNT_IDS")
        out: list[H72TraceExecutionR11] = []
        for item in ordered:
            market = self.market_cache.get(item.symbol)
            branch, snapshots = _simulate_h72_branch_r11(
                self.physics,
                market,
                parent=item.parent_state,
                decision_time_ms=int(item.decision_time_ms),
                candidate_direction_v55=int(item.direction_v55),
                candidate_risk=float(item.requested_risk),
            )
            out.append(
                H72TraceExecutionR11(
                    ordinal=int(item.ordinal),
                    causal_trace_id=item.causal_trace_id,
                    account_id=item.account_id,
                    parent_id=item.parent_id,
                    symbol=item.symbol,
                    branch=branch,
                    snapshot_sha256_sequence=snapshots,
                )
            )
        return out

    def submit(self, items: Sequence[H72TraceWorkItemR11]) -> TraceBatchFutureR11:
        with self._lock:
            if self._closed:
                raise RuntimeError("R11_TRACE_RUNTIME_CLOSED")
            rows = list(items)
            ordinals = [int(x.ordinal) for x in rows]
            if len(ordinals) != len(set(ordinals)):
                raise RuntimeError("R11_DUPLICATE_TRACE_ORDINAL")
            trace_ids = [x.causal_trace_id for x in rows]
            if len(trace_ids) != len(set(trace_ids)):
                raise RuntimeError("R11_DUPLICATE_CAUSAL_TRACE_ID")
            # Deterministic grouping protects per-account chronology. Group dispatch
            # can run concurrently; output identity never depends on completion order.
            grouped: dict[str, list[H72TraceWorkItemR11]] = {}
            for row in rows:
                grouped.setdefault(row.account_id, []).append(row)
            futures: dict[Future, tuple[int, str]] = {}
            for account_id in sorted(grouped):
                group = grouped[account_id]
                first = min(int(x.ordinal) for x in group)
                f = self._executor.submit(self._execute_account_group, tuple(group))
                futures[f] = (first, account_id)
            return TraceBatchFutureR11(futures)

    def run(self, items: Sequence[H72TraceWorkItemR11]) -> list[H72TraceExecutionR11]:
        if not items:
            return []
        return self.submit(items).result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "TraceRuntimeR11":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _choose_on_policy_parents(
    parents: Mapping[str, ParentContextR102], max_groups: int
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
    chosen: list[ParentContextR102] = []
    seen: set[str] = set()
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


def run_real_on_policy_trace_r11(
    *,
    model,
    policy_hash: str,
    generation: int,
    parents: Mapping[str, ParentContextR102],
    parent_states: Mapping[str, Mapping[str, Any]],
    trace_runtime: TraceRuntimeR11,
    lake: ShardedExperienceLake,
    device: str,
    max_groups: int = 24,
) -> dict[str, Any]:
    """Drop-in R11 orchestration equivalent of ``run_real_on_policy_trace``.

    Model inference remains one context at a time so decision numerics stay equal
    to the legacy oracle.  Only the independent H72 Physics traces run concurrently.
    Lake writes are serialized in canonical order after execution.
    """
    chosen = _choose_on_policy_parents(parents, int(max_groups))
    work: list[H72TraceWorkItemR11] = []
    decision_meta: dict[str, tuple[ParentContextR102, int, float, list[float]]] = {}
    model.eval()
    for ordinal, p in enumerate(chosen):
        state = dict(parent_states[p.parent_id])
        state.setdefault("account_id", state["risk_authority"]["account_id"])
        # Load now so a generation never races first decompression with H72 workers.
        trace_runtime.market_cache.get(p.symbol)
        op = torch.tensor([p.operator48], dtype=torch.float32, device=device)
        med = torch.tensor([p.medium48], dtype=torch.float32, device=device)
        acc = torch.tensor([p.account6], dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = model(op, med, acc)
            act = model.compose_action(out)
        direction = int(act["direction"].item())
        requested_risk = float(act["requested_risk"].item())
        trace_id = f"R102TRACE:G{generation}:{p.parent_id}"
        probs = out["direction_probs"].detach().cpu().numpy()[0].tolist()
        decision_meta[trace_id] = (p, direction, requested_risk, probs)
        work.append(
            H72TraceWorkItemR11(
                ordinal=ordinal,
                causal_trace_id=trace_id,
                account_id=str(state["account_id"]),
                parent_id=p.parent_id,
                symbol=p.symbol,
                decision_time_ms=int(p.decision_time_ms),
                parent_state=state,
                # Legacy compose_action is {-1,0,+1}; frozen Physics V5.5 is {0,1,2}.
                direction_v55=direction + 1,
                requested_risk=requested_risk,
            )
        )

    executions = trace_runtime.run(work)
    traces: list[dict[str, Any]] = []
    for execution in executions:
        trace_id = execution.causal_trace_id
        p, direction, requested_risk, probs = decision_meta[trace_id]
        branch = execution.branch
        decision_payload = {
            "schema": "CB16_R10_2_DECISION_EVENT_V1",
            "CausalTraceID": trace_id,
            "generation": generation,
            "policy_hash": policy_hash,
            "parent_id": p.parent_id,
            "dependence_group_id": p.dependence_group_id,
            "symbol": p.symbol,
            "decision_time_ms": p.decision_time_ms,
            "direction": direction,
            "requested_risk": requested_risk,
            "direction_probs": probs,
            "ordered4h30_visible_to_nominal_brain": False,
        }
        d_obj = ExperienceObject(
            object_id=f"DECISION:{trace_id}",
            object_type="DECISION_EVENT",
            generation=generation,
            policy_weight_hash=policy_hash,
            snapshot_hash=p.snapshot_sha256,
            lineage_hash=sha256_obj(decision_payload),
            payload=decision_payload,
        )
        dref, _ = lake.put(d_obj)
        outcome_payload = {
            "schema": "CB16_R10_2_OUTCOME_SAMPLE_V1",
            "CausalTraceID": trace_id,
            "generation": generation,
            "parent_id": p.parent_id,
            "dependence_group_id": p.dependence_group_id,
            "status": branch["status"],
            "realized_utility": branch.get("utility"),
            "w0": branch.get("w0"),
            "wt": branch.get("wt"),
            "terminal_at_step": branch.get("terminal_at_step"),
            "first_step": branch.get("first_step"),
            "evaluation_finalize": branch.get("finalize"),
            "realized_outcome_is_stochastic_sample_not_correct_action_label": True,
        }
        o_obj = ExperienceObject(
            object_id=f"OUTCOME:{trace_id}",
            object_type="OUTCOME_SAMPLE",
            generation=generation,
            policy_weight_hash=policy_hash,
            snapshot_hash=p.snapshot_sha256,
            lineage_hash=sha256_obj(outcome_payload),
            payload=outcome_payload,
        )
        oref, _ = lake.put(o_obj)
        traces.append(
            {
                "CausalTraceID": trace_id,
                "decision_ref": dref.identity_hash,
                "outcome_ref": oref.identity_hash,
                "status": branch["status"],
                "utility": branch.get("utility"),
                "direction": direction,
                "requested_risk": requested_risk,
            }
        )

    matured = sum(x["status"] == "MATURED" for x in traces)
    return {
        "schema": "CB16_R10_2_ON_POLICY_REAL_TRACE_RECEIPT_V1",
        "generation": generation,
        "trace_count": len(traces),
        "matured": matured,
        # Preserve legacy receipt spelling exactly; do not silently change identity semantics here.
        "unique_dependence_groups": len(
            {x["CausalTraceID"].split(":", 2)[-1] for x in traces}
        ),
        "Brain_to_ActionIntent_to_Supervisor_to_exact_Physics_to_H72_Outcome": (
            "PASS" if matured == len(traces) else "PARTIAL"
        ),
        "traces": traces,
    }
