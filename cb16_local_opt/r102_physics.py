from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .r102_common import H72, HOUR_MS, PHYSICS_CONTRACT_SHA256, RISK_SUPERVISOR_SHA256, sha256_file, sha256_obj

SHORT, FLAT, LONG = 0, 1, 2


@dataclass(frozen=True)
class FrozenPhysicsRuntimeR102:
    package_root: Path
    physics_root: Path
    control_root: Path
    physics_contract: Mapping[str, Any]
    physics: Any
    supervisor: Any

    @classmethod
    def load(cls, package_root: str | Path) -> "FrozenPhysicsRuntimeR102":
        root = Path(package_root).resolve()
        pr = root / "authority/account_physics_r0/CB16_ACCOUNT_PHYSICS_STATE_V1_R0"
        cr = root / "authority/control_plane_r1"
        if not pr.is_dir(): raise FileNotFoundError(pr)
        if not cr.is_dir(): raise FileNotFoundError(cr)
        if str(pr / "runtime") not in sys.path: sys.path.insert(0, str(pr / "runtime"))
        if str(cr) not in sys.path: sys.path.insert(0, str(cr))
        # Import after sys.path is bound to exact vendored authority.
        physics = importlib.import_module("account_physics_runtime_r0")
        supervisor = importlib.import_module("risk_supervisor_r1")
        contract = json.loads((pr / "ACCOUNT_PHYSICS_CONTRACT_V1.json").read_text())
        physics.validate_physics_contract(contract)
        if contract.get("contract_sha256") != PHYSICS_CONTRACT_SHA256:
            raise RuntimeError("PHYSICS_CONTRACT_SHA_MISMATCH")
        if sha256_file(cr / "risk_supervisor_r1.py") != RISK_SUPERVISOR_SHA256:
            raise RuntimeError("RISK_SUPERVISOR_SHA_MISMATCH")
        return cls(root, pr, cr, contract, physics, supervisor)

    def risk_authority(self, account_id: str, fraction: float = 1.0) -> dict[str, Any]:
        f = float(fraction)
        if not (0.0 <= f <= 1.0): raise ValueError("risk authority fraction")
        return {
            "schema": "EXTERNAL_RISK_BUDGET_AUTHORITY_V1",
            "authority": "CONTROL_PLANE_EXTERNAL_STATE",
            "account_id": account_id,
            "risk_budget_remaining": f,
            "risk_budget_capacity": 1.0,
            "update_rule_status": "UPDATE_RULE_UNRESOLVED",
            "physics_mutation_policy": "PRESERVE_UNCHANGED",
        }

    def initialize(self, account_id: str, fraction: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
        r = self.risk_authority(account_id, fraction)
        s = self.physics.initialize_snapshot(
            self.physics_contract,
            risk_budget_remaining=r["risk_budget_remaining"],
            risk_budget_capacity=r["risk_budget_capacity"],
        )
        self.supervisor.validate_snapshot_risk_sync(s, r)
        return s, r

    @staticmethod
    def bar_dict(symbol: str, open_time_ms: int, ohlcv: Sequence[float]) -> dict[str, Any]:
        o, h, l, c, v = [float(x) for x in ohlcv]
        return {
            "symbol": symbol,
            "bar_start": datetime.fromtimestamp(int(open_time_ms) / 1000, tz=timezone.utc).isoformat(),
            "timeframe": "1h", "open": o, "high": h, "low": l, "close": c, "volume": v,
            # Historical adapter authority: close is explicit mark/index proxy when separate series are absent.
            "mark_price": c, "index_price": c,
        }

    @staticmethod
    def intent(direction_v55: int, risk: float, *, trace_id: str) -> dict[str, Any]:
        return {
            "schema": "ActionIntentV1", "direction": int(direction_v55),
            "requested_risk_multiplier": float(risk),
            "intent_source": "CB16_R10_2_CENTRAL_BRAIN_OR_COUNTERFACTUAL",
            "trace_id": trace_id,
        }

    def step_intent(
        self, snapshot: Mapping[str, Any], risk_auth: Mapping[str, Any],
        *, direction_v55: int, risk: float, symbol: str, open_time_ms: int,
        ohlcv: Sequence[float], funding_rate: float, trace_id: str,
    ) -> dict[str, Any]:
        intent = self.intent(direction_v55, risk, trace_id=trace_id)
        decision = self.supervisor.supervise(intent, snapshot, risk_auth, self.physics_contract)
        executable = self.supervisor.executable_action(decision, self.physics_contract)
        result = self.supervisor.execute_physics(
            snapshot, executable,
            {"bar": self.bar_dict(symbol, open_time_ms, ohlcv), "funding_rate": float(funding_rate)},
            self.physics_contract,
        )
        return {"intent": intent, "supervisor_decision": decision, "executable_action": executable, **result}

    def equity_at_mark(self, snapshot: Mapping[str, Any], mark: float) -> float:
        k = self.physics.restore_kernel(snapshot, self.physics_contract)
        k.state.last_mark_price = float(mark)
        return float(k.state.equity())

    def account6(self, snapshot: Mapping[str, Any], mark: float) -> np.ndarray:
        p = self.physics.project_observation(snapshot, float(mark), self.physics_contract)
        x = np.asarray(p["payload"], dtype=np.float32)
        if x.shape != (6,) or not np.all(np.isfinite(x)):
            raise RuntimeError("ACCOUNT6_INVALID")
        return x

    def evaluation_finalize(self, snapshot: Mapping[str, Any], close_price: float) -> tuple[dict[str, Any], dict[str, Any]]:
        """H72 evaluation-only close if exact max-holding has not already closed.

        This uses FrozenTradingKernel.finalize() from the byte-exact recovered authority. It is not a new
        Physics transition rule and is recorded explicitly in the branch receipt.
        """
        k = self.physics.restore_kernel(snapshot, self.physics_contract)
        before_pos = float(k.state.position)
        if abs(before_pos) < 1e-12:
            return dict(snapshot), {"used": False, "position_before": before_pos, "reward": 0.0}
        reward = float(k.finalize(float(close_price)))
        support = copy.deepcopy(snapshot["observation_support_state"])
        out = self.physics.snapshot_kernel(k, self.physics_contract, support)
        return out, {"used": True, "position_before": before_pos, "reward": reward, "reason": "H72_EVALUATION_FINALIZE"}


SCENARIOS_R102 = (
    ("CLEAN_FLAT_FULL", None, 0.0, 1.0),
    ("CLEAN_FLAT_LOW_ENVELOPE", None, 0.0, 0.25),
    ("PRIOR_LONG_R025", LONG, 0.25, 1.0),
    ("PRIOR_SHORT_R025", SHORT, 0.25, 1.0),
    ("PRIOR_LONG_R075", LONG, 0.75, 1.0),
    ("PRIOR_SHORT_R075", SHORT, 0.75, 1.0),
)

CANDIDATES_R102 = ((FLAT, 0.0),) + tuple((d, r) for d in (SHORT, LONG) for r in (0.25, 0.50, 0.75, 1.0))


def build_parent_scenarios(
    runtime: FrozenPhysicsRuntimeR102, *, symbol: str, decision_time_ms: int,
    hourly_ts: np.ndarray, hourly_ohlcv: np.ndarray, funding: np.ndarray,
    prehistory_hours: int = 96,
) -> list[dict[str, Any]]:
    idx = {int(t): i for i, t in enumerate(hourly_ts)}
    start = int(decision_time_ms - prehistory_hours * HOUR_MS)
    pre_times = [start + i * HOUR_MS for i in range(prehistory_hours)]
    if any(t not in idx for t in pre_times):
        raise RuntimeError(f"PREHISTORY_MISSING:{symbol}:{decision_time_ms}")
    out = []
    # Warm first 16h so ATR/state machinery is not entry-triggered from a cold genesis.
    prior_entry_index = 16
    for scenario_name, prior_direction, prior_risk, risk_fraction in SCENARIOS_R102:
        account_id = f"{symbol}:{decision_time_ms}:{scenario_name}"
        snap, ra = runtime.initialize(account_id, risk_fraction)
        trace = []
        for j, t in enumerate(pre_times):
            direction = prior_direction if (prior_direction is not None and j == prior_entry_index) else FLAT
            risk = prior_risk if direction != FLAT else 0.0
            i = idx[t]
            step = runtime.step_intent(
                snap, ra, direction_v55=direction, risk=risk, symbol=symbol,
                open_time_ms=t, ohlcv=hourly_ohlcv[i], funding_rate=float(funding[i]),
                trace_id=f"PRE:{account_id}:{j}",
            )
            snap = step["snapshot_t1"]
            trace.append(step["supervisor_decision"]["decision"])
            if snap["termination_state"]["terminated"]:
                break
        mark = float(hourly_ohlcv[idx[decision_time_ms - HOUR_MS], 3])
        obs = runtime.account6(snap, mark)
        state = snap["kernel_state"]
        eligible = (
            not snap["termination_state"]["terminated"] and
            not snap["termination_state"]["truncated"] and
            abs(float(state["position"])) < 1e-12
        )
        out.append({
            "scenario": scenario_name, "account_id": account_id, "snapshot": snap,
            "risk_authority": ra, "account6": obs, "current_mark": mark,
            "eligible_for_economic_evidence": bool(eligible),
            "prehistory_supervisor_decisions": trace,
            "snapshot_sha256": runtime.physics.sha256_obj(snap),
        })
    return out


def simulate_h72_branch(
    runtime: FrozenPhysicsRuntimeR102, *, parent: Mapping[str, Any], symbol: str,
    decision_time_ms: int, candidate_direction_v55: int, candidate_risk: float,
    hourly_ts: np.ndarray, hourly_ohlcv: np.ndarray, funding: np.ndarray,
) -> dict[str, Any]:
    idx = {int(t): i for i, t in enumerate(hourly_ts)}
    future_times = [int(decision_time_ms + i * HOUR_MS) for i in range(H72)]
    if any(t not in idx for t in future_times):
        raise RuntimeError(f"H72_FUTURE_MISSING:{symbol}:{decision_time_ms}")
    snap = copy.deepcopy(parent["snapshot"])
    ra = copy.deepcopy(parent["risk_authority"])
    w0 = runtime.equity_at_mark(snap, float(parent["current_mark"]))
    if not math.isfinite(w0) or w0 <= 0:
        return {"status": "CENSORED_NONPOSITIVE_INITIAL_EQUITY", "utility": None, "w0": w0}
    decisions = []
    first_step_receipt = None
    terminal_at = None
    for j, t in enumerate(future_times):
        d, r = (candidate_direction_v55, candidate_risk) if j == 0 else (FLAT, 0.0)
        i = idx[t]
        step = runtime.step_intent(
            snap, ra, direction_v55=d, risk=r, symbol=symbol, open_time_ms=t,
            ohlcv=hourly_ohlcv[i], funding_rate=float(funding[i]),
            trace_id=f"CF:{parent['account_id']}:{candidate_direction_v55}:{candidate_risk:.2f}:{j}",
        )
        snap = step["snapshot_t1"]
        if j == 0:
            first_step_receipt = {
                "intent": step["intent"], "supervisor_decision": step["supervisor_decision"],
                "executable_action": step["executable_action"],
                "execution_metadata": step["execution_metadata"],
            }
        decisions.append(step["supervisor_decision"]["decision"])
        if snap["termination_state"]["terminated"]:
            terminal_at = j + 1
            break
    last_t = future_times[min(len(decisions), H72) - 1]
    close_price = float(hourly_ohlcv[idx[last_t], 3])
    finalize_receipt = {"used": False}
    if terminal_at is None:
        snap, finalize_receipt = runtime.evaluation_finalize(snap, close_price)
    wt = runtime.equity_at_mark(snap, close_price)
    if not math.isfinite(wt) or wt <= 0:
        return {
            "status": "CENSORED_NONPOSITIVE_TERMINAL_EQUITY", "utility": None,
            "w0": w0, "wt": wt, "terminal_at_step": terminal_at,
            "finalize": finalize_receipt, "supervisor_decisions": decisions,
            "first_step": first_step_receipt,
        }
    u = math.log(wt / w0)
    return {
        "status": "MATURED", "utility": float(u), "w0": float(w0), "wt": float(wt),
        "terminal_at_step": terminal_at, "finalize": finalize_receipt,
        "supervisor_decisions": decisions, "first_step": first_step_receipt,
    }


def market_future_lineage_hash(symbol: str, decision_time_ms: int, hourly_ts: np.ndarray, hourly_ohlcv: np.ndarray, funding: np.ndarray) -> str:
    idx = {int(t): i for i, t in enumerate(hourly_ts)}
    h = hashlib.sha256()
    h.update(symbol.encode()); h.update(str(int(decision_time_ms)).encode())
    for j in range(H72):
        t = int(decision_time_ms + j * HOUR_MS); i = idx[t]
        h.update(np.asarray([t], dtype=np.int64).tobytes())
        h.update(np.asarray(hourly_ohlcv[i], dtype=np.float64).tobytes())
        h.update(np.asarray([funding[i]], dtype=np.float64).tobytes())
    return h.hexdigest()
