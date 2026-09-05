# External Simulator State Survey — CB16 Account Physics R0

Scope: architecture principles only. No robotics schema is copied into CB16.

## 1. MuJoCo: compact integration state vs complete simulation state

MuJoCo explicitly distinguishes the state needed to continue dynamics from the larger `mjData` workspace. Its API exposes `mj_getState` / `mj_setState` with state specifications, and the programming documentation describes an **integration state** as the full set of forward-dynamics inputs. The central architectural lesson is not the specific qpos/qvel layout; it is that reproducible continuation requires every hidden accumulator/input that can affect the next transition, not merely what an agent observes. MuJoCo also documents stateful details such as warm-start accelerations and sleeping/island state that can matter for bit-level replay, which is a warning against assuming a compact observation is a sufficient snapshot.

Sources:
- https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
- https://mujoco.readthedocs.io/en/stable/programming/simulation.html

**CB16 adoption:** snapshot the entire recovered account ledger plus hidden kernel continuation state (`last_bar_time`, symbol binding, carry accumulator), and bind the immutable physics config by hash. Do not use the 6D observation as restore state.

## 2. Brax: `pipeline_state` is not `obs`

Brax environment `State` carries `pipeline_state` separately from `obs`, reward, done, metrics, and info. Training wrappers vectorize the environment state, and autoreset logic explicitly preserves/replaces pipeline and observation state as separate objects. This is the exact conceptual split CB16 needs: internal simulator state can be high-dimensional and implementation-oriented while the policy observation remains a stable typed contract.

Sources:
- https://github.com/google/brax/blob/main/brax/envs/base.py
- https://github.com/google/brax/blob/main/brax/envs/wrappers/training.py

**CB16 adoption:** `SimulatorStateSnapshotV1` and `AccountStatePacketV1` are different types with a one-way pure projection.

## 3. Gymnasium: termination, truncation, and final state preservation

Gymnasium's modern Step API separates `terminated` from `truncated` because a true task terminal and an external time limit have different learning/replay meaning. Vector autoreset conventions additionally require care about the observation associated with the final transition: same-step autoreset can replace the terminal observation unless the final observation/info is preserved explicitly.

Sources:
- https://gymnasium.farama.org/api/env/
- https://gymnasium.farama.org/api/vector/
- https://farama.org/Vector-Autoreset-Mode

**CB16 adoption:** a single `done` boolean is insufficient. `step_account` returns an explicit termination class and the final pre-reset snapshot/observation; reset is a separate operation.

## 4. Vectorized simulators: independent mutable state, shared immutable inputs

Both Brax-style vectorization and mature environment APIs treat per-environment state as independent values while sharing code and immutable parameters. Reset masks operate per environment, not by mutating a global state shared across vector members.

**CB16 adoption:** one MarketSensory packet/key may be broadcast to many accounts, while every account owns a separate snapshot reference. Branching from one content-addressed snapshot must create isolated descendants.

## 5. Event sourcing / deterministic financial ledger design

Event-sourced systems separate the durable authority (ordered state-affecting events and snapshots) from read projections. NautilusTrader's event-sourcing design uses sequence order as replay authority, binds snapshots to replay anchors/high-watermarks, and records run identity such as config/binary/schema/seed. Martin Fowler's classic description likewise treats state as a projection of an ordered event log and notes periodic snapshots as a recovery accelerator.

Sources:
- https://nautilustrader.io/docs/latest/concepts/event_sourcing/
- https://martinfowler.com/eaaDev/EventSourcing.html

**CB16 adoption:** replay records store content-addressed simulator snapshot refs plus immutable market/input lineage. Sequence/step identity, config hash, code lineage, funding input, action, and termination class must all be replay-visible. A cached account observation is a projection, never the book of record.

## 6. Hidden-state checklist distilled for CB16

The most common deterministic-replay omissions are: previous timestamp/order constraints; rolling numerical accumulators (ATR/previous close); cost/funding accumulators; margin and liquidation flags; trade-age/deadline state; peak/drawdown path state; cooldown state; terminal/truncation flags; symbol/instrument binding; mutable RNG state when stochasticity exists; and external state-affecting inputs that were not captured. The recovered V5.5 kernel has no stochastic physics RNG, so R0 records `rng_state=null` rather than fabricating one.

## 7. Architecture conclusion

The mature pattern is consistent across physics engines, RL environments, and financial ledgers:

`authoritative internal state -> pure observation projection -> agent`

and independently:

`authoritative state + ordered external input + action + immutable contract -> next authoritative state`.

CB16 should therefore preserve AccountStatePacketV1 exactly as a Brain-visible 6D observation while freezing a larger simulator-only snapshot sufficient for exact continuation, branching, and replay.
