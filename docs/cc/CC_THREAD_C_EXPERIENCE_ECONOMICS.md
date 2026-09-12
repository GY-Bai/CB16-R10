# CC Thread C — Experience / Replay / Economic Evaluation

**Thread:** C  
**Frozen base:** `89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Branch:** `ai/r11-cc-thread-c-experience-r0`  
**Independence rule:** no imports/cherry-picks/dependencies from CC Threads A/B/D  
**Primary evidence target:** COMPONENT + economic known-answer correctness  

Thread C owns semantic experience durability, replay admission/support metadata and economic evaluation. It does not own the live collector, learner math or high-throughput transport.

Use synthetic W-02/W-03/W-05 fixtures. Existing `sharded_experience_lake.py`, `demonstration_shard_store_r11.py`, checkpoint/storage infrastructure and Stage-4 persistence may be reused where their semantics fit, but C must not silently inherit historical Teacher/H72/log-utility meaning.

---

## C-01 — Freeze Thread-C input inventory

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_C_BASELINE_V1.json`
- `[NEW] tests/test_cc_thread_c_baseline_r0.py`

Bind frozen base, relevant storage/evidence modules, R1 science identity and the CC master W-02/W-03/W-05 definitions.

Any sibling branch dependency fails closed.

---

## C-02 — Thread-local W-02 / W-03 / W-05 typed fixtures

**Files**
- `[NEW] cb16_local_opt/cc_experience_wire_r0.py`
- `[NEW] tests/test_cc_experience_wire_r0.py`

Implement local C representations of:

- `CCEnvironmentTransitionV1`;
- `CCExperienceSequenceV1`;
- `CCEconomicResultV1`.

Validate exact field meaning, hashes, account lineage, ordering and failure classifications.

---

## C-03 — Immutable transition schema

**Files**
- `[NEW] cb16_local_opt/cc_experience_transition_r0.py`
- `[NEW] tests/test_cc_experience_transition_r0.py`

A transition must be self-auditable and include at least:

- account/generation/policy/science IDs;
- observation identity/hash;
- nominal action + `log_mu` + RNG provenance;
- permission/sizing/feasibility;
- ordered execution legs;
- pre/post account truth hashes;
- equity/cost/liability deltas;
- environment and decision clocks;
- boundary;
- market lineage;
- source classification.

No mutable runtime object reference may be required to interpret the fact later.

---

## C-04 — Ordered sequence schema

**Files**
- `[NEW] cb16_local_opt/cc_experience_sequence_r0.py`
- `[NEW] tests/test_cc_experience_sequence_r0.py`

Sequence invariants:

- one account lineage unless an explicitly represented lineage event says otherwise;
- strictly ordered decision indices;
- no duplicate transition IDs;
- continuity of pre/post account hashes;
- chunk boundary typed separately from terminal;
- mixed policy generations allowed with per-transition ownership.

Reorder/drop/duplicate/cross-account splice fails closed.

---

## C-05 — Content-addressed raw fact store

**Files**
- `[NEW] cb16_local_opt/cc_experience_store_r0.py`
- `[NEW] tests/test_cc_experience_store_r0.py`

Build semantic raw-fact persistence, reusing CAS/lake infrastructure where useful.

Requirements:

- immutable content identity;
- exactly-once semantic object identity;
- same ID/different content hard conflict;
- successful/ordinary/failed/terminal facts accepted equally;
- no learner/evaluator filtering in the ingest layer.

Physical high-throughput writer/tiering is Thread D's responsibility; C defines durable semantic behavior.

---

## C-06 — Failure and signed-terminal persistence

**Files**
- `[NEW] tests/test_cc_experience_failure_persistence_r0.py`

Explicitly persist and round-trip:

- liquidation;
- negative equity/liability;
- reversal second-leg failure;
- rejected action with subsequent market loss;
- pending settlement;
- economic terminal;
- data-end truncation.

No failure may disappear because an account observation cannot be generated after death.

---

## C-07 — Four-view separation

**Files**
- `[NEW] cb16_local_opt/cc_experience_views_r0.py`
- `[NEW] tests/test_cc_experience_views_r0.py`

Implement separate semantic views:

1. raw facts;
2. replay-admissible facts;
3. optional demonstration/elite view;
4. economic evaluation cohort.

Rules:

- one object may belong to multiple views, but membership is explicit;
- demonstration selection cannot delete raw failures;
- replay sampling weight cannot silently redefine evaluation weight;
- evaluation cohort cannot be survivor-filtered.

---

## C-08 — Historical source classification

**Files**
- `[NEW] cb16_local_opt/cc_experience_source_r0.py`
- `[NEW] tests/test_cc_experience_source_r0.py`

Classify at minimum:

- CC stochastic trajectory;
- BC/R1 component fixture;
- legacy H72/Teacher demonstration;
- diagnostic/unknown source.

Historical records lacking true behavior likelihood remain readable as facts/demonstrations/diagnostics but do not become V-trace replay by fabricated `log_mu`.

---

## C-09 — Replay compatibility gate

**Files**
- `[NEW] cb16_local_opt/cc_replay_compatibility_r0.py`
- `[NEW] tests/test_cc_replay_compatibility_r0.py`

Check compatibility axes such as:

- science semantic version;
- observation/normalizer identity;
- action/distribution semantics;
- true behavior likelihood availability;
- execution/environment semantics;
- boundary/bootstrap interpretability.

Reject incompatible samples explicitly with reason codes.

---

## C-10 — Off-policy support health

**Files**
- `[NEW] cb16_local_opt/cc_replay_support_r0.py`
- `[NEW] tests/test_cc_replay_support_r0.py`

Given stored `log_mu` and supplied target `log_pi`, report:

- ratio distribution;
- clipping fraction;
- finite/nonfinite checks;
- low-support frequency;
- effective-support diagnostic;
- generation/source contribution.

This is a health/admission/diagnostic contract; it does not invent new objective weights after outcomes are seen.

---

## C-11 — No age-expiration historical policy

**Files**
- `[MODIFY] tests/test_cc_replay_support_r0.py`

Prove record age alone does not make a compatible historical fact invalid.

Allowed reasons to exclude from a particular learner include semantic incompatibility or inadequate behavior-policy support, not a hand-written "older than N days" recurrence rule.

---

## C-12 — Replay selection metadata contract

**Files**
- `[NEW] cb16_local_opt/cc_replay_selection_r0.py`
- `[NEW] tests/test_cc_replay_selection_r0.py`

Persist, for every selected sequence:

- selection RNG identity;
- sampling probability/weight;
- source/generation class;
- support-health summary;
- selection-policy version.

Thread C does not perform gradient updates; it makes the training sample auditable.

---

## C-13 — Mixed-generation transition attribution

**Files**
- `[MODIFY] cb16_local_opt/cc_experience_sequence_r0.py`
- `[MODIFY] tests/test_cc_experience_sequence_r0.py`

A long account sequence may contain G3→G4→G5. Preserve exact transition ownership and switch locations.

Do not retroactively label the whole sequence as G5.

---

## C-14 — Raw fact restart / idempotent persistence

**Files**
- `[NEW] cb16_local_opt/cc_experience_transaction_r0.py`
- `[NEW] tests/test_cc_experience_transaction_r0.py`

Crash cases:

- before write;
- partial temporary write;
- durable object written before index/receipt acknowledgement;
- retry of same object.

Result: one durable fact identity, no duplicate semantic insertion, no silent loss.

This is semantic persistence; Thread D may later replace physical IO layout.

---

## C-15 — Economic cohort contract

**Files**
- `[NEW] cb16_local_opt/cc_economic_cohort_r0.py`
- `[NEW] tests/test_cc_economic_cohort_r0.py`

Represent preregistered:

- account/start-state population;
- weighting;
- capital denominator;
- common objective horizon identity;
- policy object under evaluation;
- realized vs truncated/estimated status.

If the formal owner decision for horizon/master ranking is absent, component construction remains possible but formal promotion must return unresolved rather than inventing a choice.

---

## C-16 — B&H and FLAT baseline engine

**Files**
- `[NEW] cb16_local_opt/cc_economic_baselines_r0.py`
- `[NEW] tests/test_cc_economic_baselines_r0.py`

Compute both baselines on identical market/cohort/capital/horizon assumptions.

A Trader account dying early does not shorten the baseline's common horizon where the evaluation contract requires the baseline to continue.

---

## C-17 — Capital denominator / restart fairness

**Files**
- `[NEW] cb16_local_opt/cc_economic_capital_r0.py`
- `[NEW] tests/test_cc_economic_capital_r0.py`

Track:

- initial allocated capital;
- every external deposit/withdrawal;
- every new-account capital allocation;
- account lineage restart.

A dead account reopened with new money is not free replenishment of the old account and cannot disappear from the denominator.

---

## C-18 — Arithmetic-return evaluator

**Files**
- `[NEW] cb16_local_opt/cc_economic_evaluator_r0.py`
- `[NEW] tests/test_cc_economic_evaluator_r0.py`

Primary statistic: declared arithmetic expected return on the fixed cohort/capital/horizon.

Also report, without turning them into hidden objectives:

- median;
- quantiles;
- liquidation/debt frequency;
- survival;
- tails;
- B&H delta;
- FLAT delta;
- truncation counts.

Deleting one failed account from a protected fixture must change the expected-return result and fail the integrity test.

---

## C-19 — Mixed-generation economic identity

**Files**
- `[NEW] cb16_local_opt/cc_economic_policy_identity_r0.py`
- `[NEW] tests/test_cc_economic_policy_identity_r0.py`

Distinguish:

- frozen checkpoint evaluation;
- generation-chain/deployment-strategy evaluation;
- baseline evaluation.

A G3→G4→G5 account cannot be called a pure G5 result.

---

## C-20 — Promotion boundary contract

**Files**
- `[NEW] cb16_local_opt/cc_economic_promotion_r0.py`
- `[NEW] tests/test_cc_economic_promotion_r0.py`

Report expected arithmetic return and baseline deltas.

If the exact owner rule for conflicting B&H/FLAT outcomes remains unresolved, return:

```text
UNRESOLVED_OWNER_DECISION
```

Do not substitute Sharpe/log growth/drawdown/bankruptcy ranking.

---

## C-21 — Evaluation freeze / no-rescue guard

**Files**
- `[NEW] cb16_local_opt/cc_economic_freeze_r0.py`
- `[NEW] tests/test_cc_economic_freeze_r0.py`

Once scoring starts, freeze:

- cohort;
- horizon;
- weights;
- capital denominator;
- baseline definitions;
- policy identity;
- data lineage.

Changing any after observing results creates a new evaluation version, never an in-place rescue.

---

## C-22 — Economic known-answer fixtures

**Files**
- `[NEW] cb16_local_opt/cc_economic_toys_r0.py`
- `[NEW] tests/test_cc_economic_toys_r0.py`

Must include:

1. high-bankruptcy/higher-arithmetic-expectation strategy wins;
2. survivor-only mean gives the wrong answer and is rejected;
3. new capital injection changes denominator/identity correctly;
4. mixed-generation chain cannot be credited to final checkpoint;
5. B&H and FLAT can disagree without forcing a fabricated master winner.

---

## C-23 — Thread-C qualification compiler

**Files**
- `[NEW] cb16_local_opt/cc_experience_economic_qualification_r0.py`
- `[NEW] tests/test_cc_experience_economic_qualification_r0.py`

Compile:

- immutable facts/sequences;
- failure retention;
- four-view separation;
- historical-source classification;
- replay compatibility/support health;
- no age expiration;
- generation attribution;
- idempotent persistence;
- cohort/capital/baselines/evaluator/promotion-boundary known answers.

Do not claim learner improvement or real-market profitability.

---

## C-24 — Thread-C receipt

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_C_RECEIPT_V1.json`

Record base/head SHAs, exact tests, owned files, evidence level, FINAL/fresh-data firewall, unresolved formal economic owner decisions and W-02/W-03/W-05 versions.

### Thread-C DONE

Thread C is complete when its persistent semantics and economic known-answer tests pass entirely against synthetic/local fixtures and frozen-base infrastructure, with no Thread A/B/D code present.