from __future__ import annotations

"""R11 long-trajectory infra-closure primitives.

This module is infrastructure-only. It binds already-frozen R11 Student,
Supervisor and scalar Account Physics to bounded real-archive qualification
canaries. It creates no market-information verdict.

Important V2 authority correction: the recovered Frozen Physics expresses
max-holding in *bars*. Applying that unchanged bar-count time-stop at 1m cadence
would silently change the historical hourly 72-bar semantics into roughly
72 minutes. Therefore any open-position H72-on-1m Teacher rollout is explicitly
fail-closed here. Minute cadence is qualified only on a flat real-account path;
canonical Teacher->Student minimum training uses the already-qualified hourly
H72 binding. Branch-specific E6 must separately qualify minute feedback semantics.
"""

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .binance_archive_input_r10 import (
    BinanceUSDMArchiveSourceR10,
    KlineRecord,
    MINUTE_MS,
)
from .longtraj_archive_r1 import iter_prefinal_observed_1m_r1
from .r102_physics import FLAT, FrozenPhysicsRuntimeR102


H72_MINUTES_R0 = 72 * 60
CHECKPOINT_SCHEMA_R0 = "CB16_R11_LONGTRAJ_INFRA_CLOSURE_CHECKPOINT_R0_V1"
MINUTE_H72_FORBIDDEN_CODE_R0 = "INFRA_CLOSURE_FORBIDDEN_1M_PHYSICS_H72_SEMANTIC_MISMATCH"


def canonical_json_bytes_r0(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json_r0(obj: Any) -> str:
    return hashlib.sha256(canonical_json_bytes_r0(obj)).hexdigest()


def sha256_file_r0(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def tensor_mapping_sha256_r0(mapping: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(mapping):
        t = mapping[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8") + b"\0")
        h.update(str(t.dtype).encode("ascii") + b"\0")
        h.update(canonical_json_bytes_r0(list(t.shape)) + b"\0")
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def recursive_state_sha256_r0(obj: Any) -> str:
    h = hashlib.sha256()

    def visit(x: Any) -> None:
        if torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(b"T")
            h.update(str(t.dtype).encode("ascii") + b"\0")
            h.update(canonical_json_bytes_r0(list(t.shape)) + b"\0")
            h.update(t.numpy().tobytes(order="C"))
        elif isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(b"N")
            h.update(str(a.dtype).encode("ascii") + b"\0")
            h.update(canonical_json_bytes_r0(list(a.shape)) + b"\0")
            h.update(a.tobytes(order="C"))
        elif isinstance(x, Mapping):
            h.update(b"M")
            for k in sorted(x, key=lambda z: repr(z)):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            h.update(b"L" if isinstance(x, list) else b"U")
            for item in x:
                visit(item)
        elif isinstance(x, bytes):
            h.update(b"B" + x)
        elif isinstance(x, (str, int, float, bool)) or x is None:
            h.update(b"J" + canonical_json_bytes_r0(x))
        else:
            h.update(b"R" + repr(x).encode("utf-8"))

    visit(obj)
    return h.hexdigest()


def recursive_tensor_to_cpu_r0(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: recursive_tensor_to_cpu_r0(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_tensor_to_cpu_r0(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_tensor_to_cpu_r0(v) for v in obj)
    return copy.deepcopy(obj)


def recursive_exact_equal_r0(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, Mapping):
        return set(a) == set(b) and all(recursive_exact_equal_r0(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(recursive_exact_equal_r0(x, y) for x, y in zip(a, b))
    return a == b


@dataclass(frozen=True)
class RNGStateR0:
    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]


def capture_rng_state_r0() -> RNGStateR0:
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(x.detach().cpu().clone() for x in torch.cuda.get_rng_state_all())
    return RNGStateR0(
        copy.deepcopy(random.getstate()),
        copy.deepcopy(np.random.get_state()),
        torch.get_rng_state().detach().cpu().clone(),
        cuda_states,
    )


def restore_rng_state_r0(state: RNGStateR0) -> None:
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("INFRA_CLOSURE_CHECKPOINT_REQUIRES_CUDA_RNG_BUT_CUDA_UNAVAILABLE")
        torch.cuda.set_rng_state_all(list(state.torch_cuda_states))


def save_exact_checkpoint_r0(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    runtime_step_index: int,
    cursor_ms: int,
    account_state: Mapping[str, Any],
    evidence_hash: str,
    run_id: str,
    experiment_id: str,
    spec_sha256: str,
    archive_identity: Mapping[str, Any],
) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA_R0,
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": recursive_tensor_to_cpu_r0(optimizer.state_dict()),
        "rng_state": capture_rng_state_r0(),
        "runtime_step_index": int(runtime_step_index),
        "cursor_ms": int(cursor_ms),
        "account_state": copy.deepcopy(dict(account_state)),
        "evidence_hash": str(evidence_hash),
        "run_id": str(run_id),
        "experiment_id": str(experiment_id),
        "spec_sha256": str(spec_sha256),
        "archive_identity": copy.deepcopy(dict(archive_identity)),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    digest = sha256_file_r0(p)
    p.with_suffix(p.suffix + ".sha256").write_text(digest + "  " + p.name + "\n", encoding="utf-8")
    return {"path": str(p), "sha256": digest, "bytes": p.stat().st_size}


def load_exact_checkpoint_r0(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    expected = p.with_suffix(p.suffix + ".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file_r0(p)
    if actual != expected:
        raise RuntimeError(f"INFRA_CLOSURE_CHECKPOINT_SHA_MISMATCH:{actual}!={expected}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA_R0:
        raise RuntimeError("INFRA_CLOSURE_CHECKPOINT_SCHEMA_MISMATCH")
    return payload


def restore_exact_checkpoint_r0(
    payload: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    runtime: Any,
    device: str | torch.device,
) -> None:
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    optimizer.load_state_dict(payload["optimizer_state"])
    runtime._step_index = int(payload["runtime_step_index"])
    restore_rng_state_r0(payload["rng_state"])


def funding_events_by_minute_r0(
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in source.iter_funding(symbol, verify_checksums=False):
        t = int(row.funding_time)
        minute = int(round(t / MINUTE_MS) * MINUTE_MS)
        if minute < start_ms or minute > end_ms:
            continue
        rate = float(row.funding_rate)
        if not math.isfinite(rate):
            raise RuntimeError(f"INFRA_CLOSURE_NONFINITE_FUNDING:{symbol}:{t}")
        if minute in out and out[minute] != rate:
            raise RuntimeError(f"INFRA_CLOSURE_DUPLICATE_FUNDING_EVENT:{symbol}:{minute}")
        out[minute] = rate
    return out


def find_contiguous_prefinal_run_r0(
    source: BinanceUSDMArchiveSourceR10,
    symbol: str,
    *,
    required_rows: int,
) -> list[KlineRecord]:
    if required_rows < 2:
        raise ValueError("required_rows must be >= 2")
    run: list[KlineRecord] = []
    prev: int | None = None
    for rec in iter_prefinal_observed_1m_r1(source, symbol, verify_checksums=False):
        t = int(rec.open_time)
        if prev is None or t - prev == MINUTE_MS:
            run.append(rec)
        else:
            run = [rec]
        prev = t
        if len(run) >= required_rows:
            return run
    raise RuntimeError(
        f"INFRA_CLOSURE_NO_CONTIGUOUS_PREFINAL_RUN:{symbol}:required={required_rows}:last={len(run)}"
    )


class MinuteFrozenPhysicsAdapterR0:
    """Minute-cadence adapter around unchanged Frozen Supervisor + scalar Physics.

    Qualification authority permits only FLAT/no-position recurrence here. Open
    positions at minute cadence are not a qualified Teacher consequence path.
    """

    def __init__(self, package_root: str | Path):
        self.base = FrozenPhysicsRuntimeR102.load(package_root)
        self.physics_contract = self.base.physics_contract
        self.physics = self.base.physics
        self.supervisor = self.base.supervisor

    def initialize(self, account_id: str, fraction: float = 1.0):
        return self.base.initialize(account_id, fraction)

    def account6(self, snapshot: Mapping[str, Any], mark: float) -> np.ndarray:
        return self.base.account6(snapshot, mark)

    def equity_at_mark(self, snapshot: Mapping[str, Any], mark: float) -> float:
        return self.base.equity_at_mark(snapshot, mark)

    @staticmethod
    def _bar(symbol: str, row: KlineRecord) -> dict[str, Any]:
        return {
            "symbol": str(symbol),
            "bar_start": datetime.fromtimestamp(int(row.open_time) / 1000, tz=timezone.utc).isoformat(),
            "timeframe": "1m",
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "mark_price": float(row.close),
            "index_price": float(row.close),
        }

    def step_intent(
        self,
        snapshot: Mapping[str, Any],
        risk_authority: Mapping[str, Any],
        *,
        direction_v55: int,
        risk: float,
        symbol: str,
        transition_row: KlineRecord,
        funding_rate: float,
        trace_id: str,
    ) -> dict[str, Any]:
        intent = self.base.intent(int(direction_v55), float(risk), trace_id=str(trace_id))
        decision = self.supervisor.supervise(intent, snapshot, risk_authority, self.physics_contract)
        executable = self.supervisor.executable_action(decision, self.physics_contract)
        result = self.supervisor.execute_physics(
            snapshot,
            executable,
            {
                "bar": self._bar(symbol, transition_row),
                "funding_rate": float(funding_rate),
            },
            self.physics_contract,
        )
        return {
            "intent": intent,
            "supervisor_decision": decision,
            "executable_action": executable,
            **result,
        }


def minute_future_lineage_hash_r0(
    *,
    symbol: str,
    parent_time_ms: int,
    future_rows: Sequence[KlineRecord],
    funding_by_minute: Mapping[int, float],
) -> str:
    """Audit hash only; it does not authorize a minute-cadence Teacher rollout."""
    h = hashlib.sha256()
    h.update(b"CB16_R11_LONGTRAJ_MINUTE_FUTURE_V1\0")
    h.update(symbol.encode("utf-8") + b"\0")
    h.update(str(int(parent_time_ms)).encode("ascii") + b"\0")
    for row in future_rows:
        h.update(
            np.asarray(
                [int(row.open_time), int(row.close_time), int(row.number_of_trades)],
                dtype=np.int64,
            ).tobytes()
        )
        h.update(
            np.asarray(
                [
                    row.open,
                    row.high,
                    row.low,
                    row.close,
                    row.volume,
                    row.quote_asset_volume,
                    row.taker_buy_base_asset_volume,
                    row.taker_buy_quote_asset_volume,
                    float(funding_by_minute.get(int(row.open_time), 0.0)),
                ],
                dtype=np.float64,
            ).tobytes()
        )
    return h.hexdigest()


def simulate_h72_minute_branch_r0(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Fail closed: Frozen Physics bar-count semantics are not minute semantics."""
    raise RuntimeError(MINUTE_H72_FORBIDDEN_CODE_R0)


def build_minute_teacher_support_r0(*args: Any, **kwargs: Any):
    """Fail closed until E6 independently qualifies minute feedback binding."""
    raise RuntimeError(MINUTE_H72_FORBIDDEN_CODE_R0)


def flat_real_path_snapshots_r0(
    *,
    adapter: MinuteFrozenPhysicsAdapterR0,
    symbol: str,
    records: Sequence[KlineRecord],
    selected_times: set[int],
    funding_by_minute: Mapping[int, float],
    account_id: str,
) -> tuple[dict[int, Mapping[str, Any]], Mapping[str, Any], dict[str, Any]]:
    snapshot, risk_auth = adapter.initialize(account_id, 1.0)
    snapshots: dict[int, Mapping[str, Any]] = {}
    transitions = 0
    selected_step_indices: list[int] = []
    for i, row in enumerate(records):
        t = int(row.open_time)
        if abs(float(snapshot["kernel_state"]["position"])) >= 1e-12:
            raise RuntimeError("INFRA_CLOSURE_MINUTE_FLAT_CANARY_OPEN_POSITION_FORBIDDEN")
        if t in selected_times:
            snapshots[t] = copy.deepcopy(snapshot)
            selected_step_indices.append(int(snapshot["step_index"]))
        if i + 1 >= len(records):
            break
        nxt = records[i + 1]
        step = adapter.step_intent(
            snapshot,
            risk_auth,
            direction_v55=FLAT,
            risk=0.0,
            symbol=symbol,
            transition_row=nxt,
            funding_rate=float(funding_by_minute.get(int(nxt.open_time), 0.0)),
            trace_id=f"INFRA_CLOSURE_REAL_PATH:{symbol}:{t}",
        )
        snapshot = step["snapshot_t1"]
        transitions += 1
        if bool(snapshot["termination_state"]["terminated"]):
            raise RuntimeError(f"INFRA_CLOSURE_FLAT_REAL_PATH_TERMINATED:{t}")
        if abs(float(snapshot["kernel_state"]["position"])) >= 1e-12:
            raise RuntimeError("INFRA_CLOSURE_MINUTE_FLAT_CANARY_POSITION_MUTATED")
        if len(snapshots) == len(selected_times) and int(nxt.open_time) > max(selected_times):
            break
    if set(snapshots) != selected_times:
        raise RuntimeError(
            f"INFRA_CLOSURE_MISSING_SELECTED_SNAPSHOT:{sorted(selected_times - set(snapshots))[:5]}"
        )
    monotone = all(b > a for a, b in zip(selected_step_indices, selected_step_indices[1:]))
    return snapshots, risk_auth, {
        "transitions": int(transitions),
        "selected_snapshot_count": len(snapshots),
        "selected_step_indices_strictly_increasing": bool(monotone),
        "account_reset_count": 0,
        "open_position_steps": 0,
        "minute_h72_teacher_binding_claimed": False,
    }
