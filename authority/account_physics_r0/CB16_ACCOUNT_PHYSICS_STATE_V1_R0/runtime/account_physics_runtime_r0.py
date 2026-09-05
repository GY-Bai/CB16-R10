"""CB16 Account Physics & Simulator State R0 adapter.

This adapter preserves the recovered V5.5 Phase-1 FrozenTradingKernel numerical
semantics byte-for-byte and adds a typed, serializable simulator-state envelope.
It never treats AccountStatePacketV1 as the simulator ledger.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from v55.kernel import AccountState, CanonicalBar, FrozenTradingKernel, SimConfig, TradeDecision
from upstream_account_state_interface_r0 import encode_account

SNAPSHOT_SCHEMA_VERSION = "SimulatorStateSnapshotV1"
PHYSICS_RUNTIME_VERSION = "CB16_ACCOUNT_PHYSICS_R0_RECOVERED_V55_PHASE1_ADAPTER_V1"
TERMINATION_TYPES = (
    "TERMINATED",
    "TRUNCATED",
    "LIQUIDATED",
    "MAX_HOLDING_FORCED_EXIT",
    "NORMAL_CONTINUE",
)


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def physics_contract_sha256(contract: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(contract))
    payload.pop("contract_sha256", None)
    return sha256_obj(payload)


def validate_physics_contract(contract: Mapping[str, Any]) -> None:
    expected = contract.get("contract_sha256")
    got = physics_contract_sha256(contract)
    if expected != got:
        raise ValueError(f"physics contract SHA mismatch: {expected} != {got}")
    if contract.get("schema") != "CB16_ACCOUNT_PHYSICS_CONTRACT_V1":
        raise ValueError("wrong physics contract schema")
    if contract.get("historical_kernel", {}).get("release") != "V5.5_CLEAN_REBUILD_PHASE1":
        raise ValueError("unexpected historical kernel release")


def _account_state_to_raw_dict(state: AccountState) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(AccountState):
        value = getattr(state, f.name)
        out[f.name] = copy.deepcopy(value)
    return out


def _raw_dict_to_account_state(raw: Mapping[str, Any]) -> AccountState:
    names = {f.name for f in fields(AccountState)}
    missing = sorted(names - set(raw))
    extra = sorted(set(raw) - names)
    if missing or extra:
        raise ValueError(f"AccountState field mismatch missing={missing} extra={extra}")
    kwargs = {name: copy.deepcopy(raw[name]) for name in names}
    return AccountState(**kwargs)


def make_kernel(contract: Mapping[str, Any]) -> FrozenTradingKernel:
    validate_physics_contract(contract)
    cfg = SimConfig(**dict(contract["sim_config"]))
    return FrozenTradingKernel(cfg)


def make_observation_support_state(
    *,
    risk_budget_remaining: float,
    risk_budget_capacity: float,
) -> dict[str, Any]:
    r = float(risk_budget_remaining)
    c = float(risk_budget_capacity)
    if not math.isfinite(r) or not math.isfinite(c) or c <= 0:
        raise ValueError("risk budget values must be finite and capacity > 0")
    return {
        "risk_budget_remaining": r,
        "risk_budget_capacity": c,
        "authority": "EXTERNAL_AUTHORITATIVE_RISK_STATE",
        "physics_mutation_policy": "PRESERVE_UNCHANGED",
    }


def snapshot_kernel(
    kernel: FrozenTradingKernel,
    contract: Mapping[str, Any],
    observation_support_state: Mapping[str, Any],
    *,
    truncated: bool = False,
    truncation_reason: str | None = None,
) -> dict[str, Any]:
    validate_physics_contract(contract)
    support = make_observation_support_state(
        risk_budget_remaining=float(observation_support_state["risk_budget_remaining"]),
        risk_budget_capacity=float(observation_support_state["risk_budget_capacity"]),
    )
    if bool(truncated) and not truncation_reason:
        raise ValueError("truncated snapshot requires truncation_reason")
    return {
        "schema": SNAPSHOT_SCHEMA_VERSION,
        "runtime_version": PHYSICS_RUNTIME_VERSION,
        "physics_contract_sha256": contract["contract_sha256"],
        "historical_kernel_release": "V5.5_CLEAN_REBUILD_PHASE1",
        "step_index": int(kernel.state.steps_survived),
        "timestamp_utc": None if kernel._last_bar_time is None else kernel._last_bar_time.isoformat(),
        "symbol": kernel._symbol,
        "kernel_state": _account_state_to_raw_dict(kernel.state),
        "kernel_hidden_state": {
            "last_bar_time": None if kernel._last_bar_time is None else kernel._last_bar_time.isoformat(),
            "symbol": kernel._symbol,
            "last_carry_cost": float(kernel._last_carry_cost),
        },
        "observation_support_state": support,
        "termination_state": {
            "terminated": bool(kernel.state.terminal),
            "terminal_reason": kernel.state.terminal_reason,
            "truncated": bool(truncated),
            "truncation_reason": truncation_reason,
        },
        "rng_state": None,
        "rng_contract": "NO_STOCHASTIC_PHYSICS_RNG",
    }


def restore_kernel(snapshot: Mapping[str, Any], contract: Mapping[str, Any]) -> FrozenTradingKernel:
    validate_physics_contract(contract)
    if snapshot.get("schema") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("wrong snapshot schema")
    if snapshot.get("physics_contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("snapshot/physics contract lineage mismatch")
    kernel = make_kernel(contract)
    kernel.state = _raw_dict_to_account_state(snapshot["kernel_state"])
    hidden = snapshot["kernel_hidden_state"]
    from datetime import datetime
    kernel._last_bar_time = None if hidden["last_bar_time"] is None else datetime.fromisoformat(hidden["last_bar_time"])
    kernel._symbol = hidden["symbol"]
    kernel._last_carry_cost = float(hidden["last_carry_cost"])
    return kernel


def initialize_snapshot(
    contract: Mapping[str, Any],
    *,
    risk_budget_remaining: float,
    risk_budget_capacity: float,
) -> dict[str, Any]:
    kernel = make_kernel(contract)
    support = make_observation_support_state(
        risk_budget_remaining=risk_budget_remaining,
        risk_budget_capacity=risk_budget_capacity,
    )
    return snapshot_kernel(kernel, contract, support)


def _state_at_mark(snapshot: Mapping[str, Any], current_market_mark: float) -> AccountState:
    mark = float(current_market_mark)
    if not math.isfinite(mark) or mark <= 0:
        raise ValueError("current_market_mark must be finite and positive")
    state = _raw_dict_to_account_state(snapshot["kernel_state"])
    state.last_mark_price = mark
    return state


def _margin_capacity_from_recovered_kernel(state: AccountState) -> float:
    # Adapter definition from the recovered kernel's own authoritative ledger:
    # total current margin capacity = currently locked margin + currently available margin.
    return float(state.margin_used + state.available_margin())


def project_observation(
    snapshot: Mapping[str, Any],
    current_market_mark: float,
    frozen_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure projection: SimulatorStateSnapshotV1 -> frozen AccountStatePacketV1."""
    before = sha256_obj(snapshot)
    validate_physics_contract(frozen_contract)
    if snapshot.get("physics_contract_sha256") != frozen_contract.get("contract_sha256"):
        raise ValueError("snapshot/contract mismatch")
    state = _state_at_mark(snapshot, current_market_mark)
    equity = state.equity()
    signed_notional = state.position * float(current_market_mark)
    support = snapshot["observation_support_state"]
    margin_capacity = _margin_capacity_from_recovered_kernel(state)
    raw = {
        "equity": equity,
        "peak_equity": state.peak_equity,
        "signed_position_notional": signed_notional,
        "max_gross_leverage_contract": float(frozen_contract["sim_config"]["max_leverage"]),
        "current_price": float(current_market_mark),
        "entry_price": None if abs(state.position) < 1e-12 else float(state.avg_entry_price),
        "holding_bars": float(state.position_age_bars),
        "max_holding_bars_contract": float(frozen_contract["sim_config"]["max_holding_bars"]),
        "risk_budget_remaining": float(support["risk_budget_remaining"]),
        "risk_budget_capacity": float(support["risk_budget_capacity"]),
        "margin_used": float(state.margin_used),
        "margin_capacity": margin_capacity,
    }
    packet = encode_account(raw)
    after = sha256_obj(snapshot)
    if before != after:
        raise RuntimeError("observation projection mutated simulator snapshot")
    return packet


def _canonical_market_execution_input(inp: Mapping[str, Any]) -> dict[str, Any]:
    bar = inp.get("bar")
    if isinstance(bar, CanonicalBar):
        bar_dict = bar.to_dict()
    elif isinstance(bar, Mapping):
        bar_dict = CanonicalBar.from_dict(dict(bar)).to_dict()
    else:
        raise TypeError("market_execution_input.bar must be CanonicalBar or dict")
    funding_rate = float(inp.get("funding_rate", 0.0))
    if not math.isfinite(funding_rate):
        raise ValueError("funding_rate must be finite")
    return {
        "bar": bar_dict,
        "funding_rate": funding_rate,
        "truncate_after_step": bool(inp.get("truncate_after_step", False)),
        "truncation_reason": inp.get("truncation_reason"),
    }


def _termination_type(before: AccountState, after: AccountState, *, truncate_after_step: bool) -> str:
    if after.liquidation_count > before.liquidation_count or (after.terminal and after.terminal_reason == "liquidation"):
        return "LIQUIDATED"
    if after.time_stop_count > before.time_stop_count:
        return "MAX_HOLDING_FORCED_EXIT"
    if after.terminal:
        return "TERMINATED"
    if truncate_after_step:
        return "TRUNCATED"
    return "NORMAL_CONTINUE"


def step_account(
    snapshot_t: Mapping[str, Any],
    action: Mapping[str, Any],
    market_execution_input: Mapping[str, Any],
    physics_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministically advance the authoritative simulator state by one bar."""
    term_state = snapshot_t.get("termination_state", {})
    if term_state.get("terminated"):
        raise RuntimeError("TERMINATED_REQUIRES_EXPLICIT_RESET")
    if term_state.get("truncated"):
        raise RuntimeError("TRUNCATED_REQUIRES_EXPLICIT_RESET")
    kernel = restore_kernel(snapshot_t, physics_contract)
    before_state = copy.deepcopy(kernel.state)
    support = copy.deepcopy(snapshot_t["observation_support_state"])
    m = _canonical_market_execution_input(market_execution_input)
    bar = CanonicalBar.from_dict(m["bar"])
    result = kernel.step(bar, dict(action), float(m["funding_rate"]))
    termination_type = _termination_type(before_state, kernel.state, truncate_after_step=bool(m["truncate_after_step"]))
    trunc = termination_type == "TRUNCATED"
    trunc_reason = None
    if trunc:
        trunc_reason = str(m.get("truncation_reason") or "EXTERNAL_TIME_LIMIT")
    snapshot_t1 = snapshot_kernel(
        kernel,
        physics_contract,
        support,
        truncated=trunc,
        truncation_reason=trunc_reason,
    )
    obs_t1 = project_observation(snapshot_t1, float(bar.mark_price), physics_contract)
    return {
        "snapshot_t1": snapshot_t1,
        "account_observation_t1": obs_t1,
        "termination_type": termination_type,
        "execution_metadata": {
            "step_reward": float(result.step_reward),
            "carry_cost": float(result.carry_cost),
            "kernel_done": bool(result.done),
            "kernel_info": copy.deepcopy(result.info),
            "market_execution_input_sha256": sha256_obj(m),
            "action_sha256": sha256_obj(dict(action)),
        },
    }


def explicit_reset(
    physics_contract: Mapping[str, Any],
    *,
    risk_budget_remaining: float,
    risk_budget_capacity: float,
) -> dict[str, Any]:
    return initialize_snapshot(
        physics_contract,
        risk_budget_remaining=risk_budget_remaining,
        risk_budget_capacity=risk_budget_capacity,
    )


def snapshot_ref(snapshot: Mapping[str, Any]) -> str:
    return "sha256:" + sha256_obj(snapshot)


class ContentAddressedSnapshotStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, snapshot: Mapping[str, Any]) -> str:
        data = canonical_json_bytes(snapshot)
        digest = sha256_bytes(data)
        path = self.root / f"{digest}.json"
        if path.exists() and path.read_bytes() != data:
            raise RuntimeError("content-address collision")
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return "sha256:" + digest

    def get(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("sha256:") or len(ref) != 71:
            raise ValueError("invalid snapshot ref")
        digest = ref.split(":", 1)[1]
        data = (self.root / f"{digest}.json").read_bytes()
        if sha256_bytes(data) != digest:
            raise RuntimeError("snapshot store corruption")
        return json.loads(data)


def snapshot_to_portable_npz_bytes(snapshot: Mapping[str, Any]) -> bytes:
    raw = canonical_json_bytes(snapshot)
    bio = io.BytesIO()
    np.savez_compressed(
        bio,
        canonical_json_utf8=np.frombuffer(raw, dtype=np.uint8),
        snapshot_sha256_utf8=np.frombuffer(sha256_bytes(raw).encode("ascii"), dtype=np.uint8),
        schema_utf8=np.frombuffer(SNAPSHOT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8),
    )
    return bio.getvalue()


def snapshot_from_portable_npz_bytes(data: bytes) -> dict[str, Any]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        raw = bytes(z["canonical_json_utf8"].tolist())
        expected = bytes(z["snapshot_sha256_utf8"].tolist()).decode("ascii")
        schema = bytes(z["schema_utf8"].tolist()).decode("utf-8")
    if schema != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("portable snapshot schema mismatch")
    if sha256_bytes(raw) != expected:
        raise ValueError("portable snapshot checksum mismatch")
    return json.loads(raw)


def transition_record_v2(
    *,
    market_packet_key_t: Mapping[str, Any],
    snapshot_ref_t: str,
    account_observation_t: Mapping[str, Any],
    action: Mapping[str, Any],
    execution_metadata: Mapping[str, Any],
    snapshot_ref_t1: str,
    account_observation_t1: Mapping[str, Any],
    termination_type: str,
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    if termination_type not in TERMINATION_TYPES:
        raise ValueError("invalid termination_type")
    return {
        "schema": "TransitionRecordV2",
        "market_packet_key_t": copy.deepcopy(dict(market_packet_key_t)),
        "simulator_snapshot_ref_t": snapshot_ref_t,
        "account_observation_t": copy.deepcopy(dict(account_observation_t)),
        "action": copy.deepcopy(dict(action)),
        "execution_metadata": copy.deepcopy(dict(execution_metadata)),
        "simulator_snapshot_ref_t1": snapshot_ref_t1,
        "account_observation_t1": copy.deepcopy(dict(account_observation_t1)),
        "termination_type": termination_type,
        "lineage": copy.deepcopy(dict(lineage)),
    }
