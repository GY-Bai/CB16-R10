# CC Thread A — Runtime / Account / Continuous Interaction

**Thread:** A  
**Frozen base:** `89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Branch:** `ai/r11-cc-thread-a-runtime-r0`  
**Independence rule:** no imports/cherry-picks/dependencies from CC Threads B/C/D  
**Primary evidence target:** COMPONENT → thread-local CLOSED_LOOP of runtime/account consequences  

Read first:

1. `docs/R11_CC_PARALLEL_HARD_CUTOVER_TODO.md`
2. `docs/VISION.md`
3. `docs/PRINCIPLE_ALIGNMENT.md`
4. `docs/COMPONENT_REQUIREMENTS.md`
5. current R1 execution/account files on the frozen base.

Thread A builds the **semantic reference runtime**. It is intentionally not responsible for the final high-throughput topology; Thread D owns that. A must be simple enough to audit and strong enough to serve as the semantic oracle later.

It may reuse the landed R1 execution/account surfaces already on main, including the current R1 action contract, Supervisor, target exposure, execution feasibility, reversal, signed account economics, terminal responsibility, account lineage and observation projections. It must not edit those shared baseline modules unless an unavoidable defect is demonstrated; preferred CC work is additive under the `cc_runtime_*`, `cc_clock_*`, `cc_environment_*`, `cc_account_*` path families.

---

## A-01 — Freeze Thread-A runtime input inventory

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_A_BASELINE_V1.json`
- `[NEW] tests/test_cc_thread_a_baseline_r0.py`

Record exact base SHA and blob hashes for the R1 execution/account modules A consumes. Bind the CC master wire-contract document blob. Record that B/C/D branches are forbidden dependencies.

**PASS:** changed input blob or sibling dependency fails closed.

---

## A-02 — Thread-local W-01 / W-02 typed fixtures

**Files**
- `[NEW] cb16_local_opt/cc_runtime_wire_r0.py`
- `[NEW] tests/test_cc_runtime_wire_r0.py`

Implement local typed representations of `CCPolicyDecisionV1` and `CCEnvironmentTransitionV1` exactly as documented by the CC master. These are local A types, not the final shared integration type.

Must validate:

- policy/account/science identities non-empty;
- nominal direction/risk legal;
- `log_mu` finite where required;
- FLAT risk exactly zero;
- monotone decision/environment indices;
- pre/post account hashes exact length/format;
- ordered execution legs;
- explicit boundary type.

**Negative tests:** missing `log_mu`, invalid risk, account-ID mismatch, nonmonotone clock, malformed hash.

---

## A-03 — Four-clock contract

**Files**
- `[NEW] cb16_local_opt/cc_clock_r0.py`
- `[NEW] tests/test_cc_clock_r0.py`

Represent separately:

- environment clock;
- policy decision clock;
- storage/computation chunk clock;
- economic objective horizon identity.

No generic overloaded `done`. No assumption that every environment tick calls Actor. No assumption that chunk boundary or process end kills the account.

**PASS:** changing one clock does not silently mutate another.

---

## A-04 — Decision schedule contract

**Files**
- `[NEW] cb16_local_opt/cc_runtime_decision_schedule_r0.py`
- `[NEW] tests/test_cc_runtime_decision_schedule_r0.py`

Define exact decision instants from causal environment time. Detect skipped, duplicated and reordered policy calls.

The schedule must permit environment advancement between decisions.

**Known cases:** every-bar decision, every-N-bar decision, resumed schedule, end-of-data truncation.

---

## A-05 — Policy-neutral environment advance primitive

**Files**
- `[NEW] cb16_local_opt/cc_environment_advance_r0.py`
- `[NEW] tests/test_cc_environment_advance_r0.py`

Advance an existing account through one environment interval without requiring a new order. Preserve current R1 economic meaning:

- mark-to-market consequence;
- funding/fees where mechanically applicable;
- legal liquidation/settlement mechanics;
- liability/equity updates;
- no fixed SL/TP/max-hold/cooldown strategy preference.

Do not call historical policy-owning exit logic simply for convenience.

**PASS:** held position changes economically under market movement even when no policy order is submitted.

---

## A-06 — REJECT / NOOP / abstention world-continuation tests

**Files**
- `[MODIFY] tests/test_cc_environment_advance_r0.py`

Construct separate cases for:

- Supervisor REJECT;
- same executable target after live sizing;
- below-minimum order delta;
- explicit FLAT while already flat;
- no decision scheduled at this environment tick.

In every case environment time advances and pre-existing exposure continues to incur its real consequences.

**FAIL condition:** any no-order state freezes PnL/funding/liquidation merely because no trade executed.

---

## A-07 — CC bar lifecycle

**Files**
- `[NEW] cb16_local_opt/cc_environment_lifecycle_r0.py`
- `[NEW] tests/test_cc_environment_lifecycle_r0.py`

Define the ordering of:

1. causal observation availability;
2. policy decision if scheduled;
3. permission/sizing/feasibility;
4. ordered execution legs;
5. environment/account mechanical progression;
6. post-state publication.

The exact ordering must be versioned and tests must protect it. Do not inherit historical SL/TP/max-hold ordering.

---

## A-08 — Single-account continuous state machine

**Files**
- `[NEW] cb16_local_opt/cc_runtime_account_loop_r0.py`
- `[NEW] tests/test_cc_runtime_account_loop_r0.py`

Implement repeated:

```text
actual account truth
→ policy-decision fixture callback
→ R1 permission / target sizing / feasibility / execution
→ environment advance
→ next actual account truth
→ next scheduled decision
```

The callback returns A's local W-01 policy-decision type, so no dependency on Thread B exists.

**PASS:** second/third decisions observe actual consequences of prior actions.

---

## A-09 — Strict within-account serialization

**Files**
- `[MODIFY] cb16_local_opt/cc_runtime_account_loop_r0.py`
- `[MODIFY] tests/test_cc_runtime_account_loop_r0.py`

Give every transition a predecessor token/index. Reject concurrent/reordered mutation of one logical account.

**Hostile fixture:** attempt to commit decision `n+1` before `n`; attempt duplicate commit of `n`; both fail closed.

---

## A-10 — Full runtime recovery bundle

**Files**
- `[NEW] cb16_local_opt/cc_account_recovery_r0.py`
- `[NEW] tests/test_cc_account_recovery_r0.py`

Seal/restore:

- complete signed account truth;
- positions/cost basis;
- liabilities;
- account lineage;
- all clocks;
- decision index;
- policy generation/identity reference;
- pending settlement/execution state;
- optional policy-memory opaque token if enabled by external caller.

Thread A does not own policy RNG internals; it stores the opaque provenance/token provided in W-01.

---

## A-11 — Exact pause/resume continuation

**Files**
- `[MODIFY] tests/test_cc_account_recovery_r0.py`

Compare uninterrupted execution against pauses at:

- immediately before a decision;
- after decision before execution;
- after execution before environment advance;
- after environment advance;
- arbitrary computation chunk boundary.

No duplicate/skipped action, fee, funding or environment step.

**PASS:** final account truth and ordered transition semantics match the uninterrupted reference.

---

## A-12 — Runtime boundary taxonomy

**Files**
- `[NEW] cb16_local_opt/cc_runtime_boundary_r0.py`
- `[NEW] tests/test_cc_runtime_boundary_r0.py`

At minimum represent:

- `ECONOMIC_TERMINAL`;
- `TRADING_DISABLED_PENDING_SETTLEMENT`;
- `OBJECTIVE_HORIZON_REACHED`;
- `DATA_END_TRUNCATION`;
- `COMPUTE_CHUNK`;
- `PAUSE`;
- `PROCESS_FAILURE`.

Do not encode them as one boolean.

---

## A-13 — Objective horizon vs physical account continuity

**Files**
- `[MODIFY] cb16_local_opt/cc_runtime_boundary_r0.py`
- `[MODIFY] tests/test_cc_runtime_boundary_r0.py`

Prove reaching an evaluation horizon does not itself force a trade or kill the physical account. Prove data end is not silently converted to a realized terminal outcome.

Thread A only records the boundary; Thread C later owns economic interpretation.

---

## A-14 — Generation-switch account continuity

**Files**
- `[NEW] cb16_local_opt/cc_runtime_generation_switch_r0.py`
- `[NEW] tests/test_cc_runtime_generation_switch_r0.py`

Allow policy identity/generation to change at an explicit boundary without resetting:

- account lineage;
- capital;
- positions;
- liabilities;
- environment time;
- decision index.

Record which transition belongs to which behavior policy.

Thread A uses opaque policy hashes; it does not import Thread B checkpoints.

---

## A-15 — Multi-account reference scheduler

**Files**
- `[NEW] cb16_local_opt/cc_runtime_reference_scheduler_r0.py`
- `[NEW] tests/test_cc_runtime_reference_scheduler_r0.py`

Implement a correctness-first multi-account scheduler for controlled qualification. Accounts may interleave, but each account remains locally serialized.

This is **not** the production performance scheduler. Keep it simple and deterministic enough to serve as an integration oracle against Thread D.

**PASS:** different inter-account scheduling orders produce the same per-account semantic sequence when inputs are fixed.

---

## A-16 — Account/economics hostile matrix

**Files**
- `[NEW] tests/test_cc_runtime_hostile_matrix_r0.py`

Must include:

- negative equity/liability;
- liquidation then pending settlement;
- partial reduce;
- full close;
- two-leg reversal with second leg rejected;
- dynamic same-risk rebalance after price/equity change;
- external capital-flow event with new lineage/account identity rules;
- pause at every lifecycle phase;
- process-failure recovery.

No scenario may silently clamp an economic liability away.

---

## A-17 — Thread-A semantic qualification compiler

**Files**
- `[NEW] cb16_local_opt/cc_runtime_qualification_r0.py`
- `[NEW] tests/test_cc_runtime_qualification_r0.py`

Compile A-01..A-16 into a machine-readable result.

Minimum claims:

- W-01/W-02 local contract valid;
- four clocks distinct;
- policy-each-decision continuous runtime works with callback decisions;
- no-order environment progression works;
- account serialization works;
- pause/resume exact at semantic level;
- generation switch preserves account truth;
- hostile economics matrix passes.

Do **not** claim learner, storage, economic improvement or high-throughput evidence.

---

## A-18 — Thread-A receipt

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_A_RECEIPT_V1.json`

Record:

- frozen base and head SHA;
- exact test commands/results;
- owned files;
- R1 source blobs consumed;
- strongest evidence level;
- FINAL/fresh-data firewall;
- unresolved defects;
- exact W-01/W-02 semantic version names.

### Thread-A DONE

Thread A is complete when its branch can be merged without any Thread B/C/D code present and produces a deterministic semantic reference runtime suitable for later comparison against the Thread-D fast path.