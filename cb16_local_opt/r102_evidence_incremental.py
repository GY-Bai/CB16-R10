from __future__ import annotations

"""Cross-stride incremental historical evidence cache construction.

R10.3 uses a 512h global UTC anchor grid while R10.4 densifies it to 256h.
Every 512h anchor is therefore a strict subset of the 256h grid.  This module
reuses already-qualified parent contexts and H72 counterfactual truth for that
subset, after fail-closed identity checks, and computes only newly introduced
anchors.

Raw market-frame construction still scans chronology so missing-data/gap policy
remains authoritative.  The expensive Frozen Sensory inference and 9-branch H72
physics farm are skipped for reused anchors.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .binance_archive_input_r10 import BinanceUSDMArchiveSourceR10
from .frozen_sensory_stack_r10 import FrozenSensoryStackR10
from .r102_common import (
    ALL_SUPPORTED_SYMBOLS_R102,
    HOUR_MS,
    TRAIN_END_MS,
    TRAIN_VALIDATION_PURGE_HOURS,
    VALIDATION_END_MS,
    atomic_write_json,
    sha256_file,
)
from .r102_evidence_cache import (
    BranchRecordR102,
    ParentContextR102,
    _read_jsonl_gz,
    _teacher_direction,
    _write_jsonl_gz,
    load_parent_contexts,
    load_parent_physics_states,
)
from .r102_market import build_symbol_market_cache, load_anchor_frames
from .r102_parallel_runtime import H72ParentGroupJobR102, run_counterfactual_h72_farm_r102
from .r102_physics import (
    FrozenPhysicsRuntimeR102,
    build_parent_scenarios,
    market_future_lineage_hash,
)


INCREMENTAL_SCHEMA = "CB16_R10_2_INCREMENTAL_EVIDENCE_CACHE_BUILD_V1"


def _expected_split(t: int) -> str | None:
    if t >= VALIDATION_END_MS:
        return None
    if t + (72 + TRAIN_VALIDATION_PURGE_HOURS) * HOUR_MS <= TRAIN_END_MS:
        return "TRAIN"
    if t >= TRAIN_END_MS:
        return "VALIDATION"
    return None


def _frame_equal(a, b) -> bool:
    return (
        int(a.decision_time_ms) == int(b.decision_time_ms)
        and np.array_equal(a.micro_1m_60x5, b.micro_1m_60x5)
        and np.array_equal(a.micro_stamps_60x5, b.micro_stamps_60x5)
        and np.array_equal(a.hourly_64x5, b.hourly_64x5)
        and np.array_equal(a.hourly_stamps_64x5, b.hourly_stamps_64x5)
        and np.array_equal(a.ordered4h30, b.ordered4h30)
    )


def _load_parent_rows_in_order(path: str | Path) -> list[ParentContextR102]:
    # load_parent_contexts preserves insertion order, which is the original file order.
    return list(load_parent_contexts(path).values())


def _validate_parent_manifest(parent: Mapping[str, Any], *, stride_hours: int, prehistory_hours: int) -> None:
    if parent.get("schema") != "CB16_R10_2_REAL_EVIDENCE_CACHE_MANIFEST_V1":
        raise RuntimeError("R102_INCREMENTAL_PARENT_SCHEMA_MISMATCH")
    if parent.get("final_holdout_2025_09_accessed") is not False:
        raise RuntimeError("R102_INCREMENTAL_PARENT_HOLDOUT_BOUNDARY_FAIL")
    parent_stride = int(parent.get("stride_hours", -1))
    desired = int(stride_hours)
    if parent_stride <= desired or parent_stride % desired != 0:
        raise RuntimeError(
            f"R102_INCREMENTAL_STRIDE_NOT_NESTED:parent={parent_stride}:desired={desired}"
        )
    if int(parent.get("prehistory_hours", -1)) != int(prehistory_hours):
        raise RuntimeError("R102_INCREMENTAL_PREHISTORY_MISMATCH")
    for file_key, hash_key in (
        ("parents_file", "parents_sha256"),
        ("parent_states_file", "parent_states_sha256"),
        ("branches_file", "branches_sha256"),
    ):
        p = Path(parent[file_key])
        if not p.is_file() or sha256_file(p) != parent[hash_key]:
            raise RuntimeError(f"R102_INCREMENTAL_PARENT_FILE_HASH_MISMATCH:{file_key}")


def _sensory_reuse_canary(
    *,
    sensory: FrozenSensoryStackR10,
    reusable_frames,
    old_parents_by_time: Mapping[int, list[ParentContextR102]],
) -> int:
    if not reusable_frames:
        return 0
    picks = sorted(set((0, len(reusable_frames) // 2, len(reusable_frames) - 1)))
    frames = [reusable_frames[i] for i in picks]
    enc = sensory.encode_frames(frames)
    checked = 0
    for j, frame in enumerate(frames):
        old_rows = old_parents_by_time[int(frame.decision_time_ms)]
        if not old_rows:
            raise RuntimeError("R102_INCREMENTAL_SENSORY_CANARY_PARENT_MISSING")
        p = old_rows[0]
        if not np.array_equal(enc.operator48[j], np.asarray(p.operator48, dtype=enc.operator48.dtype)):
            raise RuntimeError("R102_INCREMENTAL_OPERATOR48_AUTHORITY_MISMATCH")
        if not np.array_equal(enc.medium48[j], np.asarray(p.medium48, dtype=enc.medium48.dtype)):
            raise RuntimeError("R102_INCREMENTAL_MEDIUM48_AUTHORITY_MISMATCH")
        if not np.array_equal(enc.ordered4h30[j], np.asarray(p.ordered4h30, dtype=enc.ordered4h30.dtype)):
            raise RuntimeError("R102_INCREMENTAL_ORDERED4H30_AUTHORITY_MISMATCH")
        checked += 1
    return checked


def build_real_evidence_cache_incremental(
    *,
    package_root: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    parent_cache_manifest: str | Path,
    device: str = "cuda",
    symbols: Sequence[str] = ALL_SUPPORTED_SYMBOLS_R102,
    stride_hours: int = 256,
    prehistory_hours: int = 96,
    verify_checksums: bool = False,
    sensory_batch_size: int = 128,
    runtime_parallelism: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(package_root).resolve()
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    parent_manifest_path = Path(parent_cache_manifest).resolve()
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    parent_manifest = json.loads(parent_manifest_path.read_text())
    _validate_parent_manifest(
        parent_manifest,
        stride_hours=stride_hours,
        prehistory_hours=prehistory_hours,
    )

    source = BinanceUSDMArchiveSourceR10(data_root)
    physics = FrozenPhysicsRuntimeR102.load(root)
    sensory = FrozenSensoryStackR10(root, device=device, verify_hashes=True)
    if runtime_parallelism is None:
        from .r102_runtime_authority import load_r102_runtime_parallelism
        runtime_parallelism = load_r102_runtime_parallelism(root, live_environment_check=False).as_dict()
    rp = dict(runtime_parallelism)
    h72_workers = int(rp.get("h72_workers", 1))
    h72_threads = int(rp.get("h72_threads_per_worker", 1))
    h72_max_in_flight = int(rp.get("h72_max_in_flight", max(1, h72_workers)))

    old_parent_rows = _load_parent_rows_in_order(parent_manifest["parents_file"])
    old_states = load_parent_physics_states(parent_manifest["parent_states_file"])
    old_branches_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl_gz(parent_manifest["branches_file"]):
        old_branches_by_parent[str(row["parent_id"])].append(row)

    old_parents_by_symbol_time: dict[tuple[str, int], list[ParentContextR102]] = defaultdict(list)
    for p in old_parent_rows:
        old_parents_by_symbol_time[(p.symbol, int(p.decision_time_ms))].append(p)

    parent_cache_root = Path(parent_manifest["parents_file"]).resolve().parent
    parents: list[ParentContextR102] = []
    parent_states: list[dict[str, Any]] = []
    branches: list[dict[str, Any]] = []
    symbol_manifests = {}
    reused_parent_contexts = 0
    reused_branch_rows = 0
    new_parent_contexts = 0
    new_branch_rows = 0
    reusable_anchor_count = 0
    new_anchor_count = 0
    sensory_canaries = 0

    for symbol in symbols:
        cache_dir = out / "market_cache"
        paths = build_symbol_market_cache(
            source=source,
            symbol=symbol,
            out_dir=cache_dir,
            stride_hours=stride_hours,
            prehistory_hours=prehistory_hours,
            verify_checksums=verify_checksums,
        )
        symbol_manifests[symbol] = json.loads(paths.manifest_json.read_text())
        frames = load_anchor_frames(symbol, paths.frames_npz)
        with np.load(paths.hourly_npz, allow_pickle=False) as hz:
            hourly_ts = hz["open_time_ms"].copy()
            hourly = hz["ohlcv"].copy()
            funding = hz["funding_rate"].copy()

        old_frames_path = parent_cache_root / "market_cache" / f"{symbol}.anchors_r102.npz"
        old_hourly_path = parent_cache_root / "market_cache" / f"{symbol}.hourly_r102.npz"
        parent_symbol_manifest = parent_manifest.get("symbol_market_manifests", {}).get(symbol)
        if parent_symbol_manifest is None:
            raise RuntimeError(f"R102_INCREMENTAL_PARENT_SYMBOL_MANIFEST_MISSING:{symbol}")
        if not old_frames_path.is_file() or sha256_file(old_frames_path) != parent_symbol_manifest["frames_sha256"]:
            raise RuntimeError(f"R102_INCREMENTAL_PARENT_FRAMES_HASH_MISMATCH:{symbol}")
        if not old_hourly_path.is_file() or sha256_file(old_hourly_path) != parent_symbol_manifest["hourly_sha256"]:
            raise RuntimeError(f"R102_INCREMENTAL_PARENT_HOURLY_HASH_MISMATCH:{symbol}")
        old_frame_map = {int(f.decision_time_ms): f for f in load_anchor_frames(symbol, old_frames_path)}
        new_frame_map = {int(f.decision_time_ms): f for f in frames}

        # Every parent-grid anchor must still exist on the denser child grid and have
        # byte-identical pre-outcome frame content.
        old_parent_times = sorted(
            t for (sym, t) in old_parents_by_symbol_time if sym == symbol
        )
        for t in old_parent_times:
            if t not in new_frame_map or t not in old_frame_map:
                raise RuntimeError(f"R102_INCREMENTAL_PARENT_ANCHOR_NOT_IN_CHILD_GRID:{symbol}:{t}")
            if not _frame_equal(old_frame_map[t], new_frame_map[t]):
                raise RuntimeError(f"R102_INCREMENTAL_PARENT_FRAME_CHANGED:{symbol}:{t}")

        reusable_frames = [
            new_frame_map[t] for t in old_parent_times if _expected_split(t) is not None
        ]
        old_parents_by_time = {
            t: old_parents_by_symbol_time[(symbol, t)] for t in old_parent_times
        }
        sensory_canaries += _sensory_reuse_canary(
            sensory=sensory,
            reusable_frames=reusable_frames,
            old_parents_by_time=old_parents_by_time,
        )

        # Verify reused future truth points to exactly the same current child-market
        # chronology before accepting any old H72 branch rows.
        future_hash_cache: dict[int, str] = {}
        for t in old_parent_times:
            future_hash_cache[t] = market_future_lineage_hash(
                symbol, t, hourly_ts, hourly, funding
            )
            for p in old_parents_by_symbol_time[(symbol, t)]:
                if p.market_lineage_hash != future_hash_cache[t]:
                    raise RuntimeError(f"R102_INCREMENTAL_FUTURE_LINEAGE_CHANGED:{p.parent_id}")
                expected_split = _expected_split(t)
                if expected_split is None or p.split != expected_split:
                    raise RuntimeError(f"R102_INCREMENTAL_SPLIT_CHANGED:{p.parent_id}")
                if p.eligible_for_economic_evidence:
                    old_rows = old_branches_by_parent.get(p.parent_id, [])
                    if len(old_rows) != 9:
                        raise RuntimeError(f"R102_INCREMENTAL_PARENT_BRANCH_GRID_INCOMPLETE:{p.parent_id}")

        compute_frames = [
            f for f in frames
            if _expected_split(int(f.decision_time_ms)) is not None
            and (symbol, int(f.decision_time_ms)) not in old_parents_by_symbol_time
        ]
        encoded_by_t: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for start in range(0, len(compute_frames), int(sensory_batch_size)):
            chunk = compute_frames[start:start + int(sensory_batch_size)]
            if not chunk:
                continue
            enc = sensory.encode_frames(chunk)
            for j, frame in enumerate(chunk):
                encoded_by_t[int(frame.decision_time_ms)] = (
                    enc.operator48[j].copy(),
                    enc.medium48[j].copy(),
                    enc.ordered4h30[j].copy(),
                )

        symbol_h72_jobs: list[H72ParentGroupJobR102] = []
        group_by_parent: dict[str, str] = {}
        eligible_parent_order: list[str] = []
        branch_by_parent: dict[str, list[dict[str, Any]]] = {}

        for frame in frames:
            t = int(frame.decision_time_ms)
            split = _expected_split(t)
            if split is None:
                continue
            old_group = old_parents_by_symbol_time.get((symbol, t))
            if old_group:
                reusable_anchor_count += 1
                for p in old_group:
                    parents.append(p)
                    state = old_states.get(p.parent_id)
                    if state is None:
                        raise RuntimeError(f"R102_INCREMENTAL_PARENT_STATE_MISSING:{p.parent_id}")
                    parent_states.append(state)
                    reused_parent_contexts += 1
                    if p.eligible_for_economic_evidence:
                        eligible_parent_order.append(p.parent_id)
                        branch_by_parent[p.parent_id] = list(old_branches_by_parent[p.parent_id])
                        reused_branch_rows += len(branch_by_parent[p.parent_id])
                continue

            new_anchor_count += 1
            if t not in encoded_by_t:
                raise RuntimeError(f"R102_INCREMENTAL_NEW_ANCHOR_NOT_ENCODED:{symbol}:{t}")
            if t not in future_hash_cache:
                future_hash_cache[t] = market_future_lineage_hash(
                    symbol, t, hourly_ts, hourly, funding
                )
            op, med, riskctx = encoded_by_t[t]
            scenarios = build_parent_scenarios(
                physics,
                symbol=symbol,
                decision_time_ms=t,
                hourly_ts=hourly_ts,
                hourly_ohlcv=hourly,
                funding=funding,
                prehistory_hours=prehistory_hours,
            )
            group_id = f"FUT:{symbol}:{t}"
            for s in scenarios:
                parent_id = f"P:{symbol}:{t}:{s['scenario']}"
                pc = ParentContextR102(
                    parent_id=parent_id,
                    dependence_group_id=group_id,
                    symbol=symbol,
                    decision_time_ms=t,
                    split=split,
                    scenario=s["scenario"],
                    operator48=tuple(float(x) for x in op),
                    medium48=tuple(float(x) for x in med),
                    account6=tuple(float(x) for x in s["account6"]),
                    ordered4h30=tuple(float(x) for x in riskctx),
                    current_mark=float(s["current_mark"]),
                    snapshot_sha256=s["snapshot_sha256"],
                    eligible_for_economic_evidence=bool(s["eligible_for_economic_evidence"]),
                    market_lineage_hash=future_hash_cache[t],
                )
                parents.append(pc)
                parent_states.append({
                    "parent_id": parent_id,
                    "account_id": s["account_id"],
                    "symbol": symbol,
                    "decision_time_ms": t,
                    "scenario": s["scenario"],
                    "snapshot": s["snapshot"],
                    "risk_authority": s["risk_authority"],
                    "current_mark": float(s["current_mark"]),
                    "snapshot_sha256": s["snapshot_sha256"],
                })
                new_parent_contexts += 1
                if not pc.eligible_for_economic_evidence:
                    continue
                eligible_parent_order.append(parent_id)
                group_by_parent[parent_id] = group_id
                symbol_h72_jobs.append(H72ParentGroupJobR102(
                    ordinal=len(symbol_h72_jobs),
                    parent_id=parent_id,
                    parent=s,
                    decision_time_ms=t,
                ))

        symbol_h72_results = run_counterfactual_h72_farm_r102(
            package_root=root,
            symbol=symbol,
            hourly_ts=hourly_ts,
            hourly_ohlcv=hourly,
            funding=funding,
            jobs=symbol_h72_jobs,
            workers=h72_workers,
            threads_per_worker=h72_threads,
            max_in_flight=h72_max_in_flight,
        ) if symbol_h72_jobs else []

        for _, parent_id, candidate_results in symbol_h72_results:
            group_id = group_by_parent[parent_id]
            rows = []
            for d_v55, r, b in candidate_results:
                rows.append(asdict(BranchRecordR102(
                    parent_id=parent_id,
                    dependence_group_id=group_id,
                    direction=_teacher_direction(int(d_v55)),
                    requested_risk=float(r),
                    status=str(b["status"]),
                    realized_utility=None if b.get("utility") is None else float(b["utility"]),
                    w0=None if b.get("w0") is None else float(b["w0"]),
                    wt=None if b.get("wt") is None else float(b["wt"]),
                    terminal_at_step=b.get("terminal_at_step"),
                    evaluation_finalize_used=bool(b.get("finalize", {}).get("used", False)),
                )))
            branch_by_parent[parent_id] = rows
            new_branch_rows += len(rows)

        for parent_id in eligible_parent_order:
            rows = branch_by_parent.get(parent_id)
            if rows is None or len(rows) != 9:
                raise RuntimeError(f"R102_INCREMENTAL_CHILD_BRANCH_GRID_INCOMPLETE:{parent_id}")
            branches.extend(rows)

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
    censored = sum(b["status"] != "MATURED" for b in branches)
    finalized = sum(bool(b.get("evaluation_finalize_used", False)) for b in branches)
    manifest = {
        "schema": "CB16_R10_2_REAL_EVIDENCE_CACHE_MANIFEST_V1",
        "status": "REAL_HISTORICAL_H72_EVIDENCE_CACHE_READY",
        "scientific_semantics_changed": False,
        "symbols": list(symbols),
        "stride_hours": int(stride_hours),
        "prehistory_hours": int(prehistory_hours),
        "parent_contexts": len(parents),
        "eligible_parents": int(eligible_parents),
        "counterfactual_branches": len(branches),
        "matured_branches": int(matured),
        "censored_nonpositive_equity_branches": int(censored),
        "evaluation_finalize_uses": int(finalized),
        "independent_future_groups": len(groups),
        "train_future_groups": len(train_groups),
        "validation_future_groups": len(val_groups),
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
        "incremental_build": {
            "schema": INCREMENTAL_SCHEMA,
            "parent_manifest": str(parent_manifest_path),
            "parent_manifest_sha256": sha256_file(parent_manifest_path),
            "parent_stride_hours": int(parent_manifest["stride_hours"]),
            "child_stride_hours": int(stride_hours),
            "reused_anchors": int(reusable_anchor_count),
            "new_anchors": int(new_anchor_count),
            "reused_parent_contexts": int(reused_parent_contexts),
            "new_parent_contexts": int(new_parent_contexts),
            "reused_branch_rows": int(reused_branch_rows),
            "new_branch_rows": int(new_branch_rows),
            "sensory_reuse_canaries": int(sensory_canaries),
            "reused_h72_truth_recomputed": False,
            "scheduler_parameters_in_scientific_identity": False,
        },
        "parents_file": str(parent_path),
        "parents_sha256": sha256_file(parent_path),
        "parent_states_file": str(state_path),
        "parent_states_sha256": sha256_file(state_path),
        "branches_file": str(branch_path),
        "branches_sha256": sha256_file(branch_path),
        "symbol_market_manifests": symbol_manifests,
    }
    manifest_path = out / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    atomic_write_json(manifest_path, manifest)
    return manifest


def prepare_r104_incremental_cache(
    *,
    package_root: str | Path,
    data_root: str | Path,
    r103_root: str | Path,
    r104_run_root: str | Path,
    device: str,
    symbols: Sequence[str] = ALL_SUPPORTED_SYMBOLS_R102,
    stride_hours: int = 256,
    prehistory_hours: int = 96,
) -> dict[str, Any]:
    target = Path(r104_run_root).resolve() / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    if target.is_file():
        obj = json.loads(target.read_text())
        if int(obj.get("stride_hours", -1)) != int(stride_hours):
            raise RuntimeError("R104_EXISTING_CACHE_STRIDE_CONFLICT")
        return obj
    parent = Path(r103_root).resolve() / "evidence_cache" / "REAL_EVIDENCE_CACHE_MANIFEST_R102.json"
    return build_real_evidence_cache_incremental(
        package_root=package_root,
        data_root=data_root,
        out_dir=target.parent,
        parent_cache_manifest=parent,
        device=device,
        symbols=symbols,
        stride_hours=stride_hours,
        prehistory_hours=prehistory_hours,
    )
