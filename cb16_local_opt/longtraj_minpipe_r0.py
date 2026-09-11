from __future__ import annotations

"""R11 long-trajectory minimum causal pipeline.

Execution-only adapter. Time remains strictly sequential because Account[t+1]
depends on Account[t] and Action[t]. Independent lanes that share decision clock
t are batched through sensory/policy first; only after the complete action batch is
frozen may environment/Frozen Physics consume the t+1 minute.

The module intentionally does not change Frozen Physics numerical semantics or the
R1 Student-visible market surface. Market timestamps used by causality assertions
remain scheduler-private audit metadata; policy receives the frozen OHLCV window,
decision_time, Account[t], and optional causal sensory only.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .binance_archive_input_r10 import KlineRecord, MINUTE_MS
from .full_minute_historical_replay_r0 import FINAL_HOLDOUT_START_MS
from .full_minute_long_trajectory_r1 import (
    MinuteEnvelopeR1,
    expand_internal_halts_r1,
    student_visible_market_tuple,
)


@dataclass(frozen=True)
class LaneTrajectoryR0:
    lane_id: str
    minutes: tuple[MinuteEnvelopeR1, ...]
    initial_account_state: Any


@dataclass(frozen=True)
class CausalLaneObservationR0:
    lane_id: str
    decision_time_ms: int
    market_window: tuple[tuple[float, ...], ...]
    market_times_ms: tuple[int, ...]
    account_state_t: Any
    sensory: Any = None

    def policy_payload(self) -> dict[str, Any]:
        # Deliberately closed Student surface. market_times_ms is scheduler-private
        # audit metadata and is NOT exported as an extra model feature.
        return {
            "lane_id": self.lane_id,
            "decision_time_ms": self.decision_time_ms,
            "market_window": self.market_window,
            "account_state_t": deepcopy(self.account_state_t),
            "sensory": deepcopy(self.sensory),
        }


@dataclass(frozen=True)
class MinpipeTransitionEventR0:
    lane_id: str
    decision_time_ms: int
    observation_max_time_ms: int
    transition_time_ms: int
    current_halt_imputed: bool
    next_halt_imputed: bool
    account_state_t: Any
    frozen_action_t: Any
    account_state_t1: Any
    policy_batch_ordinal: int
    policy_batch_size: int


@dataclass(frozen=True)
class _CandidateR0:
    lane_id: str
    decision_time_ms: int
    visible: tuple[MinuteEnvelopeR1, ...]
    current: MinuteEnvelopeR1
    nxt: MinuteEnvelopeR1


def prepare_lane_from_observed_r0(
    lane_id: str,
    records: Iterable[KlineRecord],
    initial_account_state: Any,
    *,
    final_holdout_start_ms: int = FINAL_HOLDOUT_START_MS,
) -> LaneTrajectoryR0:
    """Expand internal exchange halts prefix-only and seal FINAL before replay."""
    minutes = tuple(
        x for x in expand_internal_halts_r1(records)
        if int(x.record.open_time) < int(final_holdout_start_ms)
    )
    for a, b in zip(minutes, minutes[1:]):
        if int(b.record.open_time) - int(a.record.open_time) != MINUTE_MS:
            raise RuntimeError(
                f"MINPIPE_NONCONTIGUOUS_AFTER_HALT_EXPANSION:{lane_id}:"
                f"{a.record.open_time}->{b.record.open_time}"
            )
    return LaneTrajectoryR0(
        lane_id=str(lane_id),
        minutes=minutes,
        initial_account_state=deepcopy(initial_account_state),
    )


def _candidate_schedule_r0(
    lanes: Sequence[LaneTrajectoryR0],
    lookback_minutes: int,
    final_holdout_start_ms: int,
) -> dict[int, list[_CandidateR0]]:
    if lookback_minutes < 1:
        raise ValueError("lookback_minutes must be >= 1")
    seen: set[str] = set()
    by_clock: dict[int, list[_CandidateR0]] = defaultdict(list)
    for lane in lanes:
        if lane.lane_id in seen:
            raise RuntimeError(f"MINPIPE_DUPLICATE_LANE:{lane.lane_id}")
        seen.add(lane.lane_id)
        m = lane.minutes
        for i in range(lookback_minutes - 1, len(m) - 1):
            visible = tuple(m[i - lookback_minutes + 1 : i + 1])
            current = m[i]
            nxt = m[i + 1]
            t = int(current.record.open_time)
            nt = int(nxt.record.open_time)
            if t >= final_holdout_start_ms or nt >= final_holdout_start_ms:
                continue
            if nt - t != MINUTE_MS:
                raise RuntimeError(f"MINPIPE_TRANSITION_NOT_ONE_MINUTE:{lane.lane_id}:{t}->{nt}")
            visible_times = tuple(int(x.record.open_time) for x in visible)
            if not visible_times or max(visible_times) != t:
                raise RuntimeError("MINPIPE_VISIBLE_WINDOW_CLOCK_DRIFT")
            if any(vt > t for vt in visible_times):
                raise RuntimeError("MINPIPE_FUTURE_MARKET_ENTERED_SCHEDULE")
            by_clock[t].append(_CandidateR0(lane.lane_id, t, visible, current, nxt))
    return dict(by_clock)


def time_major_batch_scan_r0(
    lanes: Sequence[LaneTrajectoryR0],
    *,
    lookback_minutes: int,
    policy_batch: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]],
    scalar_transition: Callable[[Any, Any, MinuteEnvelopeR1, MinuteEnvelopeR1, str], Any],
    sensory_batch_provider: Callable[[Sequence[Mapping[str, Any]]], Sequence[Any]] | None = None,
    final_holdout_start_ms: int = FINAL_HOLDOUT_START_MS,
    audit_hook: Callable[[str, int, Sequence[str]], None] | None = None,
) -> Iterator[MinpipeTransitionEventR0]:
    """Causal time scan with batch parallelism only across independent lanes.

    Ordering invariant at every clock t:
      1. construct observations from market<=t and Account[t]
      2. optionally compute sensory for the full active-lane batch
      3. run policy once for the full batch and deep-freeze *all* actions
      4. only now expose t+1 to the scalar environment/Physics, lane by lane
      5. store Account[t+1] for the next clock

    Therefore time itself is never vectorized across a recurrent account boundary.
    """
    if not lanes:
        return
    schedule = _candidate_schedule_r0(lanes, lookback_minutes, final_holdout_start_ms)
    accounts = {x.lane_id: deepcopy(x.initial_account_state) for x in lanes}
    started: set[str] = set()
    last_decision: dict[str, int] = {}
    batch_ordinal = 0

    for t in sorted(schedule):
        candidates = sorted(schedule[t], key=lambda x: x.lane_id)
        base_payloads: list[dict[str, Any]] = []
        observations: list[CausalLaneObservationR0] = []
        active: list[_CandidateR0] = []

        for c in candidates:
            if c.lane_id in started:
                expected = last_decision[c.lane_id] + MINUTE_MS
                if t != expected:
                    raise RuntimeError(
                        f"MINPIPE_ACCOUNT_CLOCK_GAP:{c.lane_id}:{last_decision[c.lane_id]}->{t}"
                    )
            market_times = tuple(int(x.record.open_time) for x in c.visible)
            if any(vt > t for vt in market_times) or max(market_times) != t:
                raise RuntimeError("MINPIPE_FUTURE_MARKET_ENTERED_OBSERVATION")
            # Preserve R1 Student-visible OHLCV surface exactly; no audit flag or
            # timestamp is injected as a new market feature.
            market_window = student_visible_market_tuple(c.visible)
            obs = CausalLaneObservationR0(
                lane_id=c.lane_id,
                decision_time_ms=t,
                market_window=market_window,
                market_times_ms=market_times,
                account_state_t=deepcopy(accounts[c.lane_id]),
            )
            observations.append(obs)
            base_payloads.append(obs.policy_payload())
            active.append(c)

        if not active:
            continue

        if audit_hook is not None:
            audit_hook("OBSERVATIONS_READY", t, tuple(c.lane_id for c in active))

        if sensory_batch_provider is not None:
            sensory = tuple(sensory_batch_provider(tuple(deepcopy(base_payloads))))
            if len(sensory) != len(active):
                raise RuntimeError("MINPIPE_SENSORY_BATCH_SIZE_MISMATCH")
            observations = [replace(o, sensory=deepcopy(s)) for o, s in zip(observations, sensory)]

        policy_payloads = tuple(o.policy_payload() for o in observations)
        if audit_hook is not None:
            audit_hook("POLICY_BATCH_BEGIN", t, tuple(c.lane_id for c in active))
        actions = tuple(policy_batch(policy_payloads))
        if len(actions) != len(active):
            raise RuntimeError("MINPIPE_POLICY_BATCH_SIZE_MISMATCH")
        frozen_actions = tuple(deepcopy(a) for a in actions)
        if audit_hook is not None:
            audit_hook("ACTIONS_FROZEN", t, tuple(c.lane_id for c in active))

        # t+1 exists only on this environment-private side of ACTIONS_FROZEN.
        pending: list[tuple[_CandidateR0, CausalLaneObservationR0, Any, Any]] = []
        for c, obs, action in zip(active, observations, frozen_actions):
            if audit_hook is not None:
                audit_hook("TRANSITION_BEGIN", t, (c.lane_id,))
            account_t1 = scalar_transition(
                deepcopy(accounts[c.lane_id]),
                deepcopy(action),
                c.current,
                c.nxt,
                c.lane_id,
            )
            pending.append((c, obs, action, deepcopy(account_t1)))

        # Commit all lane states together after the clock's transitions finish.
        for c, obs, action, account_t1 in pending:
            accounts[c.lane_id] = deepcopy(account_t1)
            started.add(c.lane_id)
            last_decision[c.lane_id] = t
            yield MinpipeTransitionEventR0(
                lane_id=c.lane_id,
                decision_time_ms=t,
                observation_max_time_ms=max(obs.market_times_ms),
                transition_time_ms=int(c.nxt.record.open_time),
                current_halt_imputed=bool(c.current.halt_imputed),
                next_halt_imputed=bool(c.nxt.halt_imputed),
                account_state_t=deepcopy(obs.account_state_t),
                frozen_action_t=deepcopy(action),
                account_state_t1=deepcopy(account_t1),
                policy_batch_ordinal=batch_ordinal,
                policy_batch_size=len(active),
            )
        batch_ordinal += 1


def make_frozen_physics_scalar_transition_r0(
    *,
    physics_contract: Mapping[str, Any],
    market_execution_input_factory: Callable[..., Mapping[str, Any]],
    step_account_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> Callable[[Any, Any, MinuteEnvelopeR1, MinuteEnvelopeR1, str], Any]:
    """Bridge the batch scheduler to existing scalar Account Physics authority.

    The market execution factory and ``step_account`` are called only inside the
    post-action environment boundary. This prevents next-minute execution input
    from being constructed before Action[t] is frozen.
    """
    if step_account_fn is None:
        from .account_physics_r0 import step_account as step_account_fn

    def transition(
        snapshot_t: Any,
        action_t: Any,
        current: MinuteEnvelopeR1,
        nxt: MinuteEnvelopeR1,
        lane_id: str,
    ) -> Any:
        market_execution_input = market_execution_input_factory(
            lane_id=lane_id,
            decision_minute=current.record,
            transition_minute=nxt.record,
        )
        result = step_account_fn(
            snapshot_t,
            action_t,
            market_execution_input,
            physics_contract,
        )
        if "snapshot_t1" not in result:
            raise RuntimeError("MINPIPE_PHYSICS_RESULT_MISSING_SNAPSHOT_T1")
        return deepcopy(result["snapshot_t1"])

    return transition
