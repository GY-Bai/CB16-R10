from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from .frozen_sensory_stack_r10 import FrozenSensoryStackR10
from .probabilistic_teacher_r5 import CounterfactualBranchSampleR5
from .r102_common import (
    CANONICAL_SYMBOLS_R102, HOUR_MS, TRAIN_END_MS, VALIDATION_END_MS, TRAIN_VALIDATION_PURGE_HOURS,
    atomic_write_json, canonical_json_bytes, sha256_file, sha256_obj, utc_iso_from_ms,
)
from .r102_market import build_symbol_market_cache, load_anchor_frames
from .r102_physics import (
    CANDIDATES_R102, FLAT, LONG, SHORT, FrozenPhysicsRuntimeR102,
    build_parent_scenarios, market_future_lineage_hash,
)
from .r102_parallel_runtime import H72ParentGroupJobR102, run_counterfactual_h72_farm_r102


@dataclass(frozen=True)
class ParentContextR102:
    parent_id: str
    dependence_group_id: str
    symbol: str
    decision_time_ms: int
    split: str
    scenario: str
    operator48: tuple[float, ...]
    medium48: tuple[float, ...]
    account6: tuple[float, ...]
    ordered4h30: tuple[float, ...]
    current_mark: float
    snapshot_sha256: str
    eligible_for_economic_evidence: bool
    market_lineage_hash: str

    @property
    def student_features(self) -> tuple[float, ...]:
        return self.operator48 + self.medium48 + self.account6

    @property
    def student_context_object_id(self) -> str:
        return "CTX:" + sha256_obj({
            "symbol": self.symbol, "decision_time_ms": self.decision_time_ms,
            "scenario": self.scenario, "operator48": self.operator48,
            "medium48": self.medium48, "account6": self.account6,
        })


@dataclass(frozen=True)
class BranchRecordR102:
    parent_id: str
    dependence_group_id: str
    direction: int  # -1 SHORT, 0 FLAT, +1 LONG for Teacher
    requested_risk: float
    status: str
    realized_utility: float | None
    w0: float | None
    wt: float | None
    terminal_at_step: int | None
    evaluation_finalize_used: bool


def _teacher_direction(v55: int) -> int:
    return int(v55) - 1


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=4) as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            n += 1
    return n


def _read_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_real_evidence_cache(
    *,
    package_root: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    device: str = "cuda",
    symbols: Sequence[str] = CANONICAL_SYMBOLS_R102,
    stride_hours: int = 512,
    prehistory_hours: int = 96,
    verify_checksums: bool = False,
    sensory_batch_size: int = 128,
    runtime_parallelism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(package_root).resolve(); out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    source = BinanceUSDMArchiveSourceR10(data_root)
    physics = FrozenPhysicsRuntimeR102.load(root)
    sensory = FrozenSensoryStackR10(root, device=device, verify_hashes=True)

    parents: list[ParentContextR102] = []
    parent_states: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    symbol_manifests = {}
    censored = 0; finalized = 0
    if runtime_parallelism is None:
        from .r102_runtime_authority import load_r102_runtime_parallelism
        runtime_parallelism = load_r102_runtime_parallelism(root, live_environment_check=False).as_dict()
    rp = dict(runtime_parallelism)
    h72_workers = int(rp.get("h72_workers", 1))
    h72_threads = int(rp.get("h72_threads_per_worker", 1))
    h72_max_in_flight = int(rp.get("h72_max_in_flight", max(1, h72_workers)))

    for symbol in symbols:
        cache_dir = out / "market_cache"
        paths = build_symbol_market_cache(
            source=source, symbol=symbol, out_dir=cache_dir, stride_hours=stride_hours,
            prehistory_hours=prehistory_hours, verify_checksums=verify_checksums,
        )
        symbol_manifests[symbol] = json.loads(paths.manifest_json.read_text())
        frames = load_anchor_frames(symbol, paths.frames_npz)
        with np.load(paths.hourly_npz, allow_pickle=False) as hz:
            hourly_ts = hz["open_time_ms"].copy(); hourly = hz["ohlcv"].copy(); funding = hz["funding_rate"].copy()

        # Frozen sensory inference is market-shared: once per market anchor, then broadcast to account scenarios.
        encoded_by_t: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for start in range(0, len(frames), int(sensory_batch_size)):
            chunk = frames[start:start + int(sensory_batch_size)]
            enc = sensory.encode_frames(chunk)
            for j, frame in enumerate(chunk):
                encoded_by_t[int(frame.decision_time_ms)] = (
                    enc.operator48[j].copy(), enc.medium48[j].copy(), enc.ordered4h30[j].copy()
                )

        future_hash_cache = {}
        symbol_h72_jobs: list[H72ParentGroupJobR102] = []
        group_by_parent: dict[str, str] = {}
        for frame in frames:
            t = int(frame.decision_time_ms)
            if t >= VALIDATION_END_MS:
                continue
            if t + (72 + TRAIN_VALIDATION_PURGE_HOURS) * HOUR_MS <= TRAIN_END_MS:
                split = "TRAIN"
            elif t >= TRAIN_END_MS:
                split = "VALIDATION"
            else:
                # Purged boundary zone: never enters Teacher train or validation.
                continue
            group_id = f"FUT:{symbol}:{t}"
            if t not in future_hash_cache:
                future_hash_cache[t] = market_future_lineage_hash(symbol, t, hourly_ts, hourly, funding)
            op, med, riskctx = encoded_by_t[t]
            scenarios = build_parent_scenarios(
                physics, symbol=symbol, decision_time_ms=t, hourly_ts=hourly_ts,
                hourly_ohlcv=hourly, funding=funding, prehistory_hours=prehistory_hours,
            )
            for s in scenarios:
                parent_id = f"P:{symbol}:{t}:{s['scenario']}"
                pc = ParentContextR102(
                    parent_id=parent_id, dependence_group_id=group_id, symbol=symbol,
                    decision_time_ms=t, split=split, scenario=s["scenario"],
                    operator48=tuple(float(x) for x in op), medium48=tuple(float(x) for x in med),
                    account6=tuple(float(x) for x in s["account6"]),
                    ordered4h30=tuple(float(x) for x in riskctx),
                    current_mark=float(s["current_mark"]), snapshot_sha256=s["snapshot_sha256"],
                    eligible_for_economic_evidence=bool(s["eligible_for_economic_evidence"]),
                    market_lineage_hash=future_hash_cache[t],
                )
                parents.append(pc)
                parent_states.append({
                    "parent_id": parent_id, "account_id": s["account_id"], "symbol": symbol, "decision_time_ms": t,
                    "scenario": s["scenario"], "snapshot": s["snapshot"],
                    "risk_authority": s["risk_authority"], "current_mark": float(s["current_mark"]),
                    "snapshot_sha256": s["snapshot_sha256"],
                })
                if not pc.eligible_for_economic_evidence:
                    continue
                # R8.1-qualified execution: one parent group is one bounded CPU work item.
                # Each worker evaluates the complete frozen 9-branch grid in canonical order.
                group_by_parent[parent_id] = group_id
                symbol_h72_jobs.append(H72ParentGroupJobR102(
                    ordinal=len(symbol_h72_jobs), parent_id=parent_id, parent=s, decision_time_ms=t,
                ))

        symbol_h72_results = run_counterfactual_h72_farm_r102(
            package_root=root, symbol=symbol, hourly_ts=hourly_ts, hourly_ohlcv=hourly, funding=funding,
            jobs=symbol_h72_jobs, workers=h72_workers, threads_per_worker=h72_threads,
            max_in_flight=h72_max_in_flight,
        )
        for _, parent_id, candidate_results in symbol_h72_results:
            group_id = group_by_parent[parent_id]
            for d_v55, r, b in candidate_results:
                if b["status"] != "MATURED":
                    censored += 1
                if bool(b.get("finalize", {}).get("used", False)):
                    finalized += 1
                branches.append(asdict(BranchRecordR102(
                    parent_id=parent_id, dependence_group_id=group_id,
                    direction=_teacher_direction(int(d_v55)), requested_risk=float(r),
                    status=str(b["status"]), realized_utility=None if b.get("utility") is None else float(b["utility"]),
                    w0=None if b.get("w0") is None else float(b["w0"]),
                    wt=None if b.get("wt") is None else float(b["wt"]),
                    terminal_at_step=b.get("terminal_at_step"),
                    evaluation_finalize_used=bool(b.get("finalize", {}).get("used", False)),
                )))

    parent_path = out / "PARENT_CONTEXTS_R102.jsonl.gz"
    state_path = out / "PARENT_PHYSICS_STATES_R102.jsonl.gz"
    branch_path = out / "COUNTERFACTUAL_BRANCHES_H72_R102.jsonl.gz"
    _write_jsonl_gz(parent_path, (asdict(p) for p in parents))
    _write_jsonl_gz(state_path, parent_states)
    _write_jsonl_gz(branch_path, branches)

    groups = sorted({p.dependence_group_id for p in parents})
    train_groups = sorted({p.dependence_group_id for p in parents if p.split == "TRAIN"})
    val_groups = sorted({p.dependence_group_id for p in parents if p.split == "VALIDATION"})
    eligible_parents = sum(p.eligible_for_economic_evidence for p in parents)
    matured = sum(b["status"] == "MATURED" for b in branches)
    manifest = {
        "schema": "CB16_R10_2_REAL_EVIDENCE_CACHE_MANIFEST_V1",
        "status": "REAL_HISTORICAL_H72_EVIDENCE_CACHE_READY",
        "scientific_semantics_changed": False,
        "symbols": list(symbols), "stride_hours": int(stride_hours), "prehistory_hours": int(prehistory_hours),
        "parent_contexts": len(parents), "eligible_parents": int(eligible_parents),
        "counterfactual_branches": len(branches), "matured_branches": int(matured),
        "censored_nonpositive_equity_branches": int(censored), "evaluation_finalize_uses": int(finalized),
        "independent_future_groups": len(groups), "train_future_groups": len(train_groups), "validation_future_groups": len(val_groups),
        "dependence_rule": "ALL_ACCOUNT_AND_ACTION_BRANCHES_SHARING_SYMBOL_AND_DECISION_FUTURE_COUNT_AS_ONE_GROUP",
        "train_boundary": "H72 maturity + 128h purge <= 2025-01-01T00:00:00Z",
        "validation_boundary": "2025-01-01 <= decision_time and H72 maturity < 2025-09-01",
        "final_holdout_2025_09_accessed": False,
        "funding_semantics": "EVENT_ONLY__RAW_CALC_TIME_BOUNDED_JITTER_CANONICALIZED_TO_NEAREST_UTC_HOUR__NO_FORWARD_FILL",
        "mark_index_semantics": "1H_CLOSE_PROXY_AS_FROZEN_HISTORICAL_ADAPTER",
        "runtime_parallelism": {
            "authority": "R8_1_MACHINE_SPECIFIC_RUNTIME_PROFILE",
            "h72_workers": h72_workers,
            "h72_threads_per_worker": h72_threads,
            "h72_max_in_flight": h72_max_in_flight,
            "single_cuda_owner": bool(rp.get("single_cuda_owner", True)),
            "scheduling_changes_scientific_semantics": False,
        },
        "parents_file": str(parent_path), "parents_sha256": sha256_file(parent_path),
        "parent_states_file": str(state_path), "parent_states_sha256": sha256_file(state_path),
        "branches_file": str(branch_path), "branches_sha256": sha256_file(branch_path),
        "symbol_market_manifests": symbol_manifests,
    }
    manifest_path = out / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_parent_contexts(path: str | Path) -> dict[str, ParentContextR102]:
    out = {}
    for x in _read_jsonl_gz(path):
        p = ParentContextR102(**{
            **x,
            "operator48": tuple(x["operator48"]), "medium48": tuple(x["medium48"]),
            "account6": tuple(x["account6"]), "ordered4h30": tuple(x["ordered4h30"]),
        })
        out[p.parent_id] = p
    return out


def load_teacher_samples(parent_path: str | Path, branch_path: str | Path) -> tuple[dict[str, ParentContextR102], list[CounterfactualBranchSampleR5]]:
    parents = load_parent_contexts(parent_path)
    samples = []
    incomplete_parents = set()
    branch_counts = {}
    for b in _read_jsonl_gz(branch_path):
        branch_counts[b["parent_id"]] = branch_counts.get(b["parent_id"], 0) + 1
        if b["status"] != "MATURED" or b["realized_utility"] is None:
            incomplete_parents.add(b["parent_id"])
    # A parent enters Teacher only if the complete 9-branch candidate grid matured finitely.
    for b in _read_jsonl_gz(branch_path):
        pid = b["parent_id"]
        p = parents.get(pid)
        if p is None or pid in incomplete_parents or branch_counts.get(pid) != 9 or not p.eligible_for_economic_evidence:
            continue
        samples.append(CounterfactualBranchSampleR5(
            parent_id=pid, student_context_object_id=p.student_context_object_id,
            timestamp=int(p.decision_time_ms), context_features=p.student_features,
            direction=int(b["direction"]), requested_risk=float(b["requested_risk"]),
            realized_utility=float(b["realized_utility"]), dependence_group_id=p.dependence_group_id,
            market_lineage_hash=p.market_lineage_hash,
        ))
    return parents, samples


def load_parent_physics_states(path: str | Path) -> dict[str, dict[str, Any]]:
    return {x["parent_id"]: x for x in _read_jsonl_gz(path)}
