"""V5 single-asset simulation configuration.

One explicit schema, no compatibility flags. Anything that changes the
sandbox rules must be expressed here and serialised into every checkpoint.
"""
# V5.5 FROZEN KERNEL: numerical semantics must remain parity-locked to audited V5.4.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    # execution / account
    fee_rate: float = 0.0004
    slippage_bps: float = 2.0
    initial_cash: float = 100_000.0
    max_leverage: float = 3.0
    max_drawdown: float = 1.0

    # margin
    maintenance_margin_rate: float = 0.1
    initial_margin_rate: Optional[float] = None
    margin_type: str = "cross"

    # optional Binance-style filters (None = disabled)
    tick_size: Optional[float] = None
    lot_step_size: Optional[float] = None
    lot_min_qty: Optional[float] = None
    lot_max_qty: Optional[float] = None
    min_notional: Optional[float] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    percent_price_multiplier_up: Optional[float] = None
    percent_price_multiplier_down: Optional[float] = None

    # SL/TP: fixed from entry, ATR-sized, never re-anchored while holding.
    atr_period: int = 14
    stop_atr_mult: float = 1.0
    min_risk_reward: float = 3.0
    risk_fraction_per_trade: float = 0.01
    stop_cooldown_bars: int = 1
    max_holding_bars: int = 72

    def __post_init__(self) -> None:
        numeric = (
            self.fee_rate,
            self.slippage_bps,
            self.initial_cash,
            self.max_leverage,
            self.max_drawdown,
            self.maintenance_margin_rate,
            self.stop_atr_mult,
            self.min_risk_reward,
            self.risk_fraction_per_trade,
        )
        if not all(math.isfinite(float(v)) for v in numeric):
            raise ValueError("sim parameters must be finite")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if self.fee_rate < 0 or self.slippage_bps < 0:
            raise ValueError("fee_rate and slippage_bps must be non-negative")
        if not 0 < self.maintenance_margin_rate < 1:
            raise ValueError("maintenance_margin_rate must be in (0, 1)")
        if self.initial_margin_rate is not None and not 0 < float(self.initial_margin_rate) <= 1:
            raise ValueError("initial_margin_rate must be in (0, 1]")
        if self.margin_type not in ("cross", "isolated"):
            raise ValueError("margin_type must be cross or isolated")
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be in (0, 1]")
        if not isinstance(self.atr_period, int) or isinstance(self.atr_period, bool) or self.atr_period <= 0:
            raise ValueError("atr_period must be a positive integer")
        if self.stop_atr_mult <= 0:
            raise ValueError("stop_atr_mult must be positive")
        if self.min_risk_reward < 1:
            raise ValueError("min_risk_reward must be at least 1")
        if not 0 < self.risk_fraction_per_trade < 1:
            raise ValueError("risk_fraction_per_trade must be in (0, 1)")
        if not isinstance(self.stop_cooldown_bars, int) or isinstance(self.stop_cooldown_bars, bool):
            raise ValueError("stop_cooldown_bars must be an integer")
        if self.stop_cooldown_bars < 0:
            raise ValueError("stop_cooldown_bars must be non-negative")
        if not isinstance(self.max_holding_bars, int) or isinstance(self.max_holding_bars, bool):
            raise ValueError("max_holding_bars must be an integer")
        if self.max_holding_bars <= 0:
            raise ValueError("max_holding_bars must be positive")
        for name, value in (
            ("tick_size", self.tick_size),
            ("lot_step_size", self.lot_step_size),
            ("lot_min_qty", self.lot_min_qty),
            ("lot_max_qty", self.lot_max_qty),
            ("price_min", self.price_min),
            ("price_max", self.price_max),
            ("percent_price_multiplier_up", self.percent_price_multiplier_up),
            ("percent_price_multiplier_down", self.percent_price_multiplier_down),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.price_min is not None
            and self.price_max is not None
            and self.price_min > self.price_max
        ):
            raise ValueError("price_min must not exceed price_max")
        if self.min_notional is not None and (
            not math.isfinite(float(self.min_notional)) or float(self.min_notional) < 0
        ):
            raise ValueError("min_notional must be finite and non-negative")
        if (
            self.percent_price_multiplier_down is not None
            and self.percent_price_multiplier_up is not None
            and self.percent_price_multiplier_down > self.percent_price_multiplier_up
        ):
            raise ValueError("percent_price_multiplier_down must not exceed up")

    def as_dict(self) -> dict:
        out = {}
        for field_name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            out[field_name] = getattr(self, field_name)
        return out
