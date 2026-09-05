"""Frozen V5.5 market/accounting kernel.

Numerical semantics are intentionally locked to the audited V5.4 contracts.
Learning code must treat this package as an immutable boundary.
"""
from .actions import TradeDecision, NUM_DECISIONS, decision_sign, validate_decision
from .bar import CanonicalBar
from .engine import FrozenTradingKernel, StepResult
from .reward import log_equity_reward
from .sim_config import SimConfig
from .state import AccountState

__all__ = [
    "TradeDecision", "NUM_DECISIONS", "decision_sign", "validate_decision",
    "CanonicalBar", "FrozenTradingKernel", "StepResult", "log_equity_reward",
    "SimConfig", "AccountState",
]
