from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn


MINUTE_MS = 60_000
FINAL_HOLDOUT_START_MS = 1_756_684_800_000  # 2025-09-01T00:00:00Z
BRAIN_PARAMETER_TARGET = 16_116_420
DIRECTION_ORDER = ("SHORT", "FLAT", "LONG")

# Preserve the R10.1/R11 gradient-owner naming contract exactly.
LOGICAL_PARAMETER_GROUPS = {
    "operator_brain_stem": ("operator_encoder.",),
    "medium_brain_stem": ("medium_encoder.",),
    "account_brain_stem": ("account_encoder.",),
    "shared_decision_core": ("shared_core.",),
    "direction_head": ("direction_body.", "direction_out."),
    "requested_risk_head": ("direction_embedding.", "sizing_body.", "sizing_out."),
}


@dataclass(frozen=True)
class MinuteBar:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class RollingMinuteSample:
    """One causal decision point plus the realized next transition.

    `market_window` is the only market payload that may be passed to policy/sensory
    code. `next_bar` exists solely on the environment/feedback side after action.
    """

    symbol: str
    segment_id: int
    market_window: tuple[Any, ...]
    next_bar: Any

    @property
    def decision_time_ms(self) -> int:
        return int(self.market_window[-1].open_time)

    @property
    def dependence_key(self) -> tuple[str, int]:
        # Multiple action/account replicas at this same clock are one market future.
        return (self.symbol, self.decision_time_ms)

    def causal_observation(self, account_state: Any, sensory: Any = None) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision_time_ms": self.decision_time_ms,
            "market_window": self.market_window,
            "account_state": account_state,
            "sensory": sensory,
        }


def _timestamp(x: Any) -> int:
    if not hasattr(x, "open_time"):
        raise TypeError("minute record must expose open_time")
    return int(x.open_time)


def iter_rolling_minute_samples(
    symbol: str,
    records: Iterable[Any],
    *,
    lookback_minutes: int,
    final_holdout_start_ms: int = FINAL_HOLDOUT_START_MS,
) -> Iterator[RollingMinuteSample]:
    """Stream every legal one-minute rolling endpoint without crossing real gaps.

    Exactly one new decision sample is emitted per contiguous input minute after
    `lookback_minutes` causal rows exist and a next realized transition is available.
    Rows at/after FINAL are never buffered, so the holdout cannot become feedback.
    """

    if lookback_minutes < 1:
        raise ValueError("lookback_minutes must be >= 1")
    buf: deque[Any] = deque(maxlen=lookback_minutes + 1)
    prev_ts: int | None = None
    segment_id = 0

    for rec in records:
        ts = _timestamp(rec)
        if ts >= final_holdout_start_ms:
            break
        if prev_ts is not None:
            delta = ts - prev_ts
            if delta <= 0:
                raise ValueError(f"NON_INCREASING_1M:{symbol}:{prev_ts}->{ts}")
            if delta != MINUTE_MS:
                segment_id += 1
                buf.clear()
        buf.append(rec)
        prev_ts = ts
        if len(buf) == lookback_minutes + 1:
            rows = tuple(buf)
            market_window = rows[:-1]
            next_bar = rows[-1]
            if _timestamp(next_bar) - _timestamp(market_window[-1]) != MINUTE_MS:
                raise AssertionError("internal gap guard failed")
            yield RollingMinuteSample(
                symbol=symbol,
                segment_id=segment_id,
                market_window=market_window,
                next_bar=next_bar,
            )


@dataclass(frozen=True)
class ActionIntent:
    direction: str
    requested_risk: float

    def __post_init__(self) -> None:
        if self.direction not in DIRECTION_ORDER:
            raise ValueError(f"invalid direction {self.direction!r}")
        if not (0.0 <= float(self.requested_risk) <= 1.0):
            raise ValueError("requested_risk must be in [0,1]")
        if self.direction == "FLAT" and float(self.requested_risk) != 0.0:
            raise ValueError("FLAT requires requested_risk=0")


@dataclass(frozen=True)
class ReplayTransition:
    replica_id: str
    symbol: str
    segment_id: int
    decision_time_ms: int
    dependence_key: tuple[str, int]
    account_before: Any
    action_intent: Any
    executable_action: Any
    account_after: Any


class PathDependentMinuteReplay:
    """Environment-side coordinator; it does not implement economic Physics.

    The caller supplies the existing Supervisor and Frozen Physics adapters. Account
    state is initialized once per contiguous episode/replica and then recursively
    propagated. It is never reconstructed from the rolling market window.
    """

    def __init__(
        self,
        *,
        policy: Callable[[Mapping[str, Any]], Any],
        supervisor: Callable[[Any, Any], Any],
        physics_step: Callable[[Any, Any, Any, Any], Any],
        initial_account_factory: Callable[[str, RollingMinuteSample], Any],
        sensory_provider: Callable[[str, Sequence[Any]], Any] | None = None,
    ) -> None:
        self.policy = policy
        self.supervisor = supervisor
        self.physics_step = physics_step
        self.initial_account_factory = initial_account_factory
        self.sensory_provider = sensory_provider

    def run(
        self,
        samples: Iterable[RollingMinuteSample],
        *,
        replica_id: str,
    ) -> Iterator[ReplayTransition]:
        account: Any = None
        active_segment: int | None = None
        for sample in samples:
            if active_segment != sample.segment_id:
                account = self.initial_account_factory(replica_id, sample)
                active_segment = sample.segment_id
            before = deepcopy(account)
            sensory = None
            if self.sensory_provider is not None:
                sensory = self.sensory_provider(sample.symbol, sample.market_window)
            observation = sample.causal_observation(account, sensory)
            # Policy receives no next_bar/future object.
            intent = self.policy(observation)
            executable = self.supervisor(intent, account)
            account = self.physics_step(
                account,
                executable,
                sample.market_window[-1],
                sample.next_bar,
            )
            yield ReplayTransition(
                replica_id=replica_id,
                symbol=sample.symbol,
                segment_id=sample.segment_id,
                decision_time_ms=sample.decision_time_ms,
                dependence_key=sample.dependence_key,
                account_before=before,
                action_intent=intent,
                executable_action=executable,
                account_after=deepcopy(account),
            )


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + y


class ReplayCentralBrain16M(nn.Module):
    """Pre-registered FP32 Central Brain capacity candidate for replay R0.

    This is a candidate Student surface only. Frozen sensory organs, Supervisor,
    Teacher/Evidence, and Physics are deliberately absent from this module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.operator_encoder = nn.Sequential(
            nn.Linear(48, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.medium_encoder = nn.Sequential(
            nn.Linear(48, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.account_encoder = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU()
        )
        self.shared_core = nn.Sequential(
            nn.Linear(640, 1024),
            nn.GELU(),
            *[ResidualMLPBlock(1024) for _ in range(7)],
            nn.LayerNorm(1024),
        )
        self.direction_body = nn.Sequential(nn.Linear(1024, 256), nn.GELU())
        self.direction_out = nn.Linear(256, 3)
        self.direction_embedding = nn.Embedding(3, 64)
        self.sizing_body = nn.Sequential(
            nn.Linear(1024 + 64, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.sizing_out = nn.Linear(128, 1)

        # R0 is explicitly FP32. Do not silently inherit a different default dtype.
        self.float()
        report = brain_parameter_report(self)
        if report["total"] != BRAIN_PARAMETER_TARGET:
            raise RuntimeError(
                f"BRAIN_PARAMETER_TARGET_MISMATCH:{report['total']}!={BRAIN_PARAMETER_TARGET}"
            )
        if report["unknown_parameter_names"]:
            raise RuntimeError("UNKNOWN_PARAMETER_NAMES:" + ",".join(report["unknown_parameter_names"]))

    def forward(
        self,
        operator48: torch.Tensor,
        medium48: torch.Tensor,
        account6: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Typed external observations are frozen authorities, not gradient owners.
        operator48 = operator48.detach().float()
        medium48 = medium48.detach().float()
        account6 = account6.detach().float()

        op = self.operator_encoder(operator48)
        med = self.medium_encoder(medium48)
        acc = self.account_encoder(account6)
        shared = self.shared_core(torch.cat((op, med, acc), dim=-1))

        direction_hidden = self.direction_body(shared)
        direction_logits = self.direction_out(direction_hidden)
        direction_index = torch.argmax(direction_logits.detach(), dim=-1)
        direction_context = self.direction_embedding(direction_index)
        sizing_hidden = self.sizing_body(torch.cat((shared, direction_context), dim=-1))
        requested_risk_raw = torch.sigmoid(self.sizing_out(sizing_hidden).squeeze(-1))
        requested_risk = torch.where(
            direction_index == 1,
            torch.zeros_like(requested_risk_raw),
            requested_risk_raw,
        )
        return {
            "direction_logits": direction_logits,
            "direction_index": direction_index,
            "requested_risk_raw": requested_risk_raw,
            "requested_risk": requested_risk,
        }


def brain_parameter_report(model: nn.Module) -> dict[str, Any]:
    out: dict[str, Any] = {k: 0 for k in LOGICAL_PARAMETER_GROUPS}
    out["total"] = 0
    out["trainable"] = 0
    unknown: list[str] = []
    for name, p in model.named_parameters():
        n = int(p.numel())
        out["total"] += n
        if p.requires_grad:
            out["trainable"] += n
        matches = [
            group
            for group, prefixes in LOGICAL_PARAMETER_GROUPS.items()
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if len(matches) != 1:
            unknown.append(name)
        else:
            out[matches[0]] += n
    out["unknown_parameter_names"] = unknown
    return out


def gradient_boundary_canary(model: nn.Module, *, batch: int = 8) -> dict[str, Any]:
    torch.manual_seed(110917)
    op = torch.randn(batch, 48, requires_grad=True)
    med = torch.randn(batch, 48, requires_grad=True)
    acc = torch.randn(batch, 6, requires_grad=True)
    out = model(op, med, acc)
    loss = out["direction_logits"].square().mean() + out["requested_risk_raw"].mean()
    loss.backward()

    group_grad = {k: 0.0 for k in LOGICAL_PARAMETER_GROUPS}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        for group, prefixes in LOGICAL_PARAMETER_GROUPS.items():
            if any(name.startswith(prefix) for prefix in prefixes):
                group_grad[group] += float(p.grad.detach().abs().sum().item())
                break
    external_blocked = all(x.grad is None for x in (op, med, acc))
    return {
        "external_input_gradients_blocked": external_blocked,
        "all_six_groups_receive_gradient": all(v > 0.0 for v in group_grad.values()),
        "group_gradient_l1": group_grad,
    }
