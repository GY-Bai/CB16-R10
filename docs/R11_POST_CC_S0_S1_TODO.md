# CB16 R11 Post-CC — S0/S1 Execution TODO

**Status:** S0 PASS / S1 ACTIVE  
**Frozen planning baseline:** `main@fc7102442e91a1c27cf705487c6d06bd64b8ea09`  
**Planning date:** 2026-09-12  
**Execution order:** `S0 -> S1`  
**CC status:** CLOSED / do not reopen A/B/C/D  
**FINAL:** SEALED  
**Fresh data:** FORBIDDEN unless separately authorized  

This document is the current task-routing authority for the first two post-CC stages.

Detailed packages:

- S0: `docs/post_cc/S0_CONTRACT_MIGRATION_TODO.md`
- S1: `docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`

Authoring/review rules: [S-series TODO authoring principles](S_SERIES_TODO_AUTHORING_PRINCIPLES.md). The [2026-09-13 executability review](reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md) maps concrete gaps to the existing task IDs. S0 PASS stands and S1-A/B development continues. Before formal S1 qualification, bind executable task/control definitions, scoring predicates, budgets and seed roles in an explicit compatible supplement or successor version as appropriate. Do not overwrite frozen registry/run-spec/receipt bytes or renumber the tasks. Implement and check S1-032 predicates before consuming formal results; final compilation and receipt remain in S1-E.

Read together with:

- `docs/ECONOMIC_ORDERING_AND_PROMOTION.md`
- `docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`
- `docs/CURRENT_STATE.md`
- `docs/CC_INTEGRATION_HANDOFF.md`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_SPEC_V1.json`
- `authority/rearchitecture_r11/CB16_R11_CC_INTEGRATION_RECEIPT_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`

---

# 1. Current live stage

S0 is complete and qualified.

Authoritative identities:

```text
S0 frozen implementation base:
fc7102442e91a1c27cf705487c6d06bd64b8ea09

S0 qualified implementation identity:
5760061d6c274e9f8796e6608e86bb173018148f

S0 receipt-bearing handoff head:
ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f
```

S1 is therefore **authorized and active**.

Current S1 working branch:

`ai/r11-post-cc-s1-end-to-end-learnability-r0`

Correct S1 working base:

`ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`

Do not reset S1 to `5760061d...` merely because that is the qualified implementation identity. The receipt-bearing head is the correct working handoff because it contains the formal S0 receipt and frozen S1 preregistration artifacts.

---

# 2. Why S0/S1 exist

CC proved the joined architecture can execute a synthetic closed loop with correct provenance, recovery, account continuity, V-trace update, generation switch and a semantically equivalent high-throughput path.

Its strongest evidence remains:

`INTEGRATED_SYNTHETIC_CLOSED_LOOP_KNOWN_ANSWER_PLUS_SHANXI_PERFORMANCE`

CC did **not** prove economic edge, transfer, or robust learnability under repeated optimization.

Two post-CC gaps were defined:

## A. Economic ordering / baseline / promotion semantics

Closed by S0.

Current canonical semantics:

1. model-vs-model ordering is one question;
2. B&H comparison is an independent benchmark component;
3. FLAT comparison is an independent benchmark component;
4. promotion is a separate versioned decision rule;
5. **B&H and FLAT have no master precedence and no master winner.**

Historical CC code/receipts remain historical evidence and are not rewritten.

## B. Integrated execution is not yet end-to-end learnability evidence

This is the active S1 program.

Known gaps include:

- live rollout-memory learner side channel;
- incomplete durable Brain-ready observation materialization;
- categorical-only generic batch versus canonical joint direction+risk Actor;
- absence of a full persistent replay -> joint learner -> child policy repeated-learning proof;
- component toys that do not by themselves establish end-to-end learnability;
- insufficient negative-control evidence for a final S1 claim.

S1 must implement and qualify these missing pieces.

---

# 3. Frozen authority hierarchy

For active S1, precedence is:

1. current owner-confirmed principles in `VISION`, `DECISIONS`, `ECONOMIC_ORDERING_AND_PROMOTION` and `POST_CC_SCIENTIFIC_PROGRAM_R0`;
2. frozen CC science identity and W-01..W-05 semantics;
3. exact CC integration spec/receipt;
4. S0 successor authority and S0 PASS receipt;
5. S0-frozen S1 task registry/run spec;
6. S1 implementation and qualification evidence.

Historical receipts describe historical qualification states. New successor routing may supersede future behavior without rewriting historical receipts.

If an implementation requires changing frozen scientific meaning rather than implementing the authorized successor, classify `CONTRACT_MISMATCH`.

---

# 4. Non-negotiable scientific semantics

S1 must preserve:

- `Truth != Belief != Decision != Permission != Execution`;
- requested target risk is not confidence;
- logical account continuity across chunk/pause/recovery/generation boundaries where the task requires it;
- signed account economics and liabilities;
- nominal sampled action distinct from permitted/executed action;
- true behavior likelihood `log_mu` retained at behavior-decision time and never reconstructed from execution;
- stochastic RNG / policy / generation provenance;
- failures and terminal facts in raw experience;
- historical experience not expired merely because it is old;
- no handcrafted cycle/regime/resonance activation subsystem;
- no hidden strategic SL/TP/max-hold/cooldown in the policy-neutral environment;
- arithmetic expected return as the current economic orientation;
- high-bankruptcy/higher-arithmetic-EV behavior allowed to win the relevant known-answer task;
- FINAL sealed and no fresh market data consumed.

---

# 5. S0 — CLOSED / PASS

S0 performed successor contract migration and preregistration.

It established, among other things:

- separate model ordering;
- separate B&H benchmark component;
- separate FLAT benchmark component;
- separate versioned promotion rule;
- no master baseline precedence/winner;
- preservation of historical CC receipts/code provenance;
- frozen S1 observation/replay successor contracts;
- frozen S1 task registry, run spec, model identity, seeds, evidence rules and firewall.

S0 strongest justified evidence:

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

Do not reopen S0 unless a concrete integrity failure is found.

---

# 6. S1 — ACTIVE END-TO-END LEARNABILITY PROGRAM

S1 is not a status-review exercise.

Its purpose is to implement and then qualify:

```text
synthetic environment
 -> canonical Brain observation
 -> stochastic nominal joint action
 -> true log_mu persisted
 -> authoritative account/runtime consequence
 -> immutable durable experience
 -> durable observation fact
 -> restart-safe replay materialization
 -> joint direction+risk batch
 -> target log_pi + Critic + V-trace + Actor update
 -> exactly-once child checkpoint
 -> account-preserving generation switch where required
 -> child acts
 -> repeated known-answer improvement
 -> negative controls stay negative
```

The detailed execution authority is:

`docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md`

That document now defines five sequential execution waves:

```text
S1-A Durable Data Plane
S1-B Joint Learning Plane
S1-C Scientific Known-Answer Tasks
S1-D Qualification Runtime
S1-E Gate Compiler + Final Receipt
```

Missing later-wave functionality is **not** a reason to stop. It is the work to implement.

---

# 7. S1 execution discipline

An Agent working S1 must:

1. verify branch lineage and S0 receipt integrity;
2. identify the first unfinished S1 task;
3. implement it;
4. add tests;
5. run tests;
6. commit/push actual S1 code;
7. continue to the next task/wave if execution capacity remains.

A normal S1 execution session must leave durable commits beyond `ec30185...`, unless it stops under a legitimate classified blocker.

A report that only says:

> S1 cannot yet receive its final receipt because durable replay / joint learner / known-answer tasks remain unfinished

is **not progress**. Those items are exactly what S1 is authorized to implement.

---

# 8. S1 hard qualification invariants

These are fail-closed qualification rules, not implementation blockers.

- executed action cannot replace nominal action;
- `log_mu` cannot be reconstructed after the fact;
- observation hash mismatch fails closed;
- FLAT implies exact target risk `0`;
- qualification learner may not train from live collector memory;
- restart after destroying live rollout objects must reconstruct the same durable replay identity;
- child generation cannot fake learning by resetting to a fresh flat account where continuity is required;
- loss decrease / parameter change / gradient / checkpoint existence alone cannot produce S1 PASS;
- `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION` uses complete-sample arithmetic EV without survivor filtering, log wealth, Sharpe or hidden risk penalty;
- A -> B -> A may not use a handcrafted regime detector or manual resonance activation system.

Violation prevents PASS; absence of implementation means continue implementing until the gate can be tested.

---

# 9. Required known-answer tasks and controls

Mandatory positive task classes:

1. `ACCOUNT_DEPENDENT_ACTION`
2. `DELAYED_CONSEQUENCE_CREDIT`
3. `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
4. `OFF_POLICY_VTRACE_CORRECTION`
5. `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

Mandatory control families include:

- no-signal / zero-reward;
- shuffled-credit;
- random/impossible target where applicable;
- account ablation where required;
- behavior-likelihood corruption;
- observation/replay integrity corruption.

Frozen seeds and qualification rules come from the S0 S1 run spec/registry.

No best-seed cherry-picking or threshold rescue after qualification begins.

---

# 10. Evidence ladder

S0 maximum evidence:

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

S1 maximum evidence if every mandatory gate passes:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

This means the canonical post-CC loop learned preregistered synthetic tasks end-to-end.

It does **not** mean:

- historical profitability;
- market edge;
- transfer;
- production readiness.

`ECONOMIC` and `TRANSFER` remain later claims.

---

# 11. Failure classification

Use one of:

- `PASS`;
- `SCIENTIFIC_FAIL` — valid experiment, frozen scientific criterion not met;
- `CONTRACT_MISMATCH` — proposed implementation violates frozen semantics;
- `EXECUTION_BLOCKED` — software/infrastructure prevents a valid implementation or run;
- `HARDWARE_LIMIT` — frozen valid workload cannot run within declared resources;
- `EVIDENCE_INSUFFICIENT` — execution completes but frozen evidence requirements remain unmet.

Do not use `EXECUTION_BLOCKED` because ordinary S1 tasks remain unimplemented.

Do not rescue `SCIENTIFIC_FAIL` by changing seeds, thresholds, task distributions or objectives under the same run identity.

---

# 12. Immediate Agent handoff

The next Agent must **not** re-run the old S0/S1 decomposition.

It should:

1. read S0 receipt;
2. read the updated `S1_END_TO_END_LEARNABILITY_TODO.md`;
3. compare `ai/r11-post-cc-s1-end-to-end-learnability-r0` against `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`;
4. find the first unfinished S1 task;
5. start with S1-A durable data-plane implementation if no S1 implementation commits exist;
6. proceed through S1-B, S1-C, S1-D and S1-E;
7. stop only after final S1 verdict/receipt or a concrete classified blocker;
8. do **not** automatically start S2, historical economic qualification, FINAL work or capacity scaling.

The post-S1 program must be authorized from actual S1 evidence, not presumed in advance.
