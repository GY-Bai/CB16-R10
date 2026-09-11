from __future__ import annotations

import copy
import math
from typing import Any, Sequence

import numpy as np

from .minute_physics_binding_r2 import MinutePhysicsSessionR2 as _MinutePhysicsSessionR2Base


class MinutePhysicsSessionR2(_MinutePhysicsSessionR2Base):
    """R2 session with observation projection from the exact in-memory ledger.

    The frozen historical projection helper reconstructs AccountState through its
    dataclass constructor.  That constructor has genesis-only peak normalization
    which is not idempotent for an evolved account.  R2 therefore projects from a
    detached copy of the already-authoritative kernel ledger.  Transition equations,
    risk/supervisor semantics and the AccountStatePacketV1 encoder remain unchanged.
    """

    def _state_at_mark_exact(self, mark: float) -> Any:
        px = float(mark)
        if not math.isfinite(px) or px <= 0.0:
            raise ValueError("R2_PROJECTION_MARK_MUST_BE_FINITE_POSITIVE")
        st = copy.deepcopy(self.kernel.state)
        st.last_mark_price = px
        return st

    def equity_at_mark(self, mark: float) -> float:
        return float(self._state_at_mark_exact(mark).equity())

    def account6(self, mark: float) -> np.ndarray:
        px = float(mark)
        st = self._state_at_mark_exact(px)
        equity = float(st.equity())
        support = self.support_state
        margin_capacity = float(st.margin_used + st.available_margin())
        raw = {
            "equity": equity,
            "peak_equity": float(st.peak_equity),
            "signed_position_notional": float(st.position) * px,
            "max_gross_leverage_contract": float(self.runtime.physics_contract["sim_config"]["max_leverage"]),
            "current_price": px,
            "entry_price": None if abs(float(st.position)) < 1e-12 else float(st.avg_entry_price),
            "holding_bars": float(st.position_age_bars),
            "max_holding_bars_contract": float(self.runtime.physics_contract["sim_config"]["max_holding_bars"]),
            "risk_budget_remaining": float(support["risk_budget_remaining"]),
            "risk_budget_capacity": float(support["risk_budget_capacity"]),
            "margin_used": float(st.margin_used),
            "margin_capacity": margin_capacity,
        }
        packet = self.runtime.physics.encode_account(raw)
        x = np.asarray(packet["payload"], dtype=np.float32)
        if x.shape != (6,) or not np.all(np.isfinite(x)):
            raise RuntimeError("R2_EXACT_ACCOUNT6_INVALID")
        return x

    def step_intent(
        self,
        *,
        direction_v55: int,
        risk: float,
        symbol: str,
        open_time_ms: int,
        ohlcv: Sequence[float],
        funding_rate: float,
        trace_id: str,
    ) -> dict[str, Any]:
        out = super().step_intent(
            direction_v55=direction_v55,
            risk=risk,
            symbol=symbol,
            open_time_ms=open_time_ms,
            ohlcv=ohlcv,
            funding_rate=funding_rate,
            trace_id=trace_id,
        )
        # Replace the legacy deserialize-based projection with the exact-ledger view.
        out["account_observation_t1"] = self.account6(float(ohlcv[3])).tolist()
        return out
