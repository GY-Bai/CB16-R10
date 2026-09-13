# CB16 R11 Post-CC S0 — Contract Closure & Economic Semantic Migration TODO

**Stage:** S0  
**Status:** CLOSED / PASS\
**Frozen implementation base:** `fc7102442e91a1c27cf705487c6d06bd64b8ea09`  
**Parent plan:** `docs/R11_POST_CC_S0_S1_TODO.md`  
**Primary purpose:** close successor contracts before S1 learnability qualification  
**Scientific execution:** no market training / no economic qualification  
**FINAL:** SEALED  

Status verified 2026-09-13: S0 qualified implementation is `5760061d6c274e9f8796e6608e86bb173018148f`; receipt-bearing handoff and S1 working base is `ec30185bf9b815f37d6dda99f6cd7ad6cad0c19f`. Evidence: `authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`, workflow `34721332483`. The task body below preserves the original S0 requirements; it is not an instruction to repeat the closed stage. For remaining S1 concretization, see the [executability review](../reviews/S0_S1_EXECUTABILITY_REVIEW_2026-09-13.md).

Suggested branch:

`ai/r11-post-cc-s0-contract-migration-r0`

S0 is a migration/contract qualification stage. It explicitly resolves the implementation mismatch created by the historical B&H/FLAT precedence ambiguity while preserving the original CC receipts unchanged.

---

## 0. Frozen owner decision — do not ask again

The following is settled and is not an S0 owner question:

- B&H and FLAT are **parallel benchmark components**;
- neither has master precedence;
- there is no master baseline winner;
- model-vs-model ordering is distinct from baseline comparison;
- promotion is distinct from both ordering and baseline comparison;
- disagreement between B&H and FLAT components is a valid result vector, not an unresolved precedence problem.

Canonical explanatory authority:

`docs/ECONOMIC_ORDERING_AND_PROMOTION.md`

S0's job is to make the code/API/routing obey that decision prospectively.

---

## 1. Baseline audit facts that S0 must treat as inputs

At the frozen base:

- `cb16_local_opt/cc_economic_evaluator_r0.py` already reports arithmetic economic results and can preserve B&H/FLAT as separate deltas;
- `cb16_local_opt/cc_economic_promotion_r0.py` still contains historical promotion semantics that can map B&H/FLAT disagreement to an unresolved-owner state;
- `tests/test_cc_economic_promotion_r0.py` preserves that historical behavior;
- `CB16_R11_CC_THREAD_C_RECEIPT_V1.json` and `CB16_R11_CC_INTEGRATION_RECEIPT_V1.json` record the old unresolved decision as part of their historical qualification state;
- those receipts must remain byte-for-byte unchanged by S0;
- current documentation has superseded the unresolved precedence question for future work.

This is a **version migration**, not a retroactive correction of historical evidence.

---

## 2. S0 target state

After S0, the canonical post-CC economic decision surface must be able to express independently:

```text
model_ordering_result
buy_and_hold_component_result
flat_component_result
promotion_rule_id
promotion_result
promotion_reasons[]
```

It must never require:

```text
master_baseline
master_baseline_winner
baseline_precedence
B&H_overrides_FLAT
FLAT_overrides_B&H
UNRESOLVED_OWNER_DECISION solely because the two baseline components disagree
```

A future promotion rule may require one component, both components, neither component, or another preregistered criterion. That rule must be explicitly versioned. It must not be inferred from the names B&H/FLAT.

---

# 3. Mandatory tasks

## S0-001 — Freeze exact S0 baseline inventory

Create a machine-readable inventory of:

- exact base SHA;
- CC integration spec/receipt identities and hashes;
- current economic evaluator module;
- current old promotion module;
- old promotion tests;
- current `ECONOMIC_ORDERING_AND_PROMOTION.md` blob identity;
- current `POST_CC_SCIENTIFIC_PROGRAM_R0.md` blob identity;
- FINAL/fresh-data firewall state.

Expected authority artifact:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S0_BASELINE_V1.json`

Gate: inventory must prove the two historical CC receipts are inputs, not edit targets.

---

## S0-002 — Define successor economic ordering contract

Create a new versioned post-CC contract. Recommended implementation surface:

`cb16_local_opt/post_cc_economic_ordering_v1.py`

It must define at least:

- policy/model identity;
- cohort/common horizon/capital denominator identity;
- arithmetic expected return used for model-vs-model ordering;
- deterministic tie handling;
- result scope;
- no implicit baseline precedence.

The ordering API answers only:

> Under the same declared comparison population and denominator, which policy/model has the higher preregistered primary economic statistic?

It must not answer promotion by itself.

Tests must include:

- A > B;
- B > A;
- tie;
- invalid mixed cohort/horizon/denominator -> fail closed;
- baseline values changed while model ordering remains unchanged.

---

## S0-003 — Define parallel baseline-component contract

Create a versioned result type for independent benchmark components. Recommended surface:

`cb16_local_opt/post_cc_baseline_components_v1.py`

Required semantics:

```text
buy_and_hold.status      # PASS / FAIL / NOT_APPLICABLE / INSUFFICIENT_EVIDENCE
buy_and_hold.delta
flat.status
flat.delta
```

Exact status vocabulary may be versioned, but B&H and FLAT must remain siblings.

Tests must explicitly cover all four combinations:

```text
B&H PASS / FLAT PASS
B&H PASS / FLAT FAIL
B&H FAIL / FLAT PASS
B&H FAIL / FLAT FAIL
```

The middle two combinations are ordinary valid outputs, not owner-decision errors.

---

## S0-004 — Define promotion as a separate rule engine

Create successor promotion logic. Recommended surface:

`cb16_local_opt/post_cc_promotion_v1.py`

A promotion assessment must consume an explicit `promotion_rule_id` or equivalent versioned rule object.

Minimum supported test rules for qualification:

- `REPORT_ONLY_NO_PROMOTION` — produces component reports but never promotes;
- `REQUIRE_BOTH_BASELINE_COMPONENTS` — may be used as a test rule, but its strictness comes from the rule ID, not from hidden precedence;
- `MODEL_ORDERING_ONLY` — promotion depends on preregistered model ordering and intentionally ignores component baselines;
- at least one synthetic rule showing that a B&H/FLAT disagreement can still lead to a deterministic rule result without choosing a master baseline.

Important: these are qualification rules demonstrating separation of concerns. They do not become the universal market promotion rule merely by existing.

---

## S0-005 — Migrate the old unresolved disagreement behavior

Do **not** rewrite `cc_economic_promotion_r0.py` as if it had always meant the new semantics.

Instead, create a post-CC routing/migration layer that classifies the old module as historical for current post-CC execution.

Recommended surface:

`cb16_local_opt/post_cc_economic_migration_v1.py`

It must provide:

- legacy contract ID;
- successor contract ID;
- explicit reason for supersession;
- mapping from historical unresolved disagreement to a valid pair of component outcomes plus `promotion_rule_required`;
- fail-closed behavior if old unresolved state is accidentally presented as current owner uncertainty.

The migration must preserve source provenance; it must not fabricate a new historical verdict.

---

## S0-006 — Add explicit adapter from `CCEconomicResultV1`

Implement a deterministic adapter from the frozen W-05 economic result into the successor post-CC ordering/component surface.

Requirements:

- preserve policy identity;
- preserve cohort/common horizon/capital denominator identity;
- preserve mean arithmetic return;
- preserve `buy_hold_delta` and `flat_delta` independently;
- preserve failure counts/tail diagnostics as diagnostics;
- do not infer a master baseline winner;
- do not infer promotion without a promotion rule.

Recommended surface:

`cb16_local_opt/post_cc_economic_adapter_v1.py`

---

## S0-007 — Preserve historical code and tests as historical evidence

The following must remain auditable:

- `cc_economic_promotion_r0.py`;
- its original test semantics;
- Thread-C receipt;
- integration receipt.

If old tests become incompatible with a new default router, isolate them under explicit historical behavior rather than deleting their assertions.

Add a successor test that proves:

1. historical module still reproduces the historical unresolved state;
2. successor router does not treat that state as a current owner question;
3. no historical receipt bytes changed.

---

## S0-008 — Add legacy-path firewall for post-CC routing

Any post-CC canonical entrypoint must refuse to use the R0 legacy promotion decision as current authority.

This can be a router/test guard rather than physical deletion.

Required negative test:

- intentionally configure the post-CC run with legacy promotion authority;
- expected result: `CONTRACT_MISMATCH` or equivalent fail-closed status;
- never silently fall back to the old unresolved behavior.

---

## S0-009 — Freeze successor science contract identity

Create:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S0_SPEC_V1.json`

It must identify at least:

- parent CC science identity;
- successor post-CC science/contract identity;
- economic ordering contract version;
- baseline component contract version;
- promotion contract version;
- migration contract version;
- unchanged W-01..W-05 meanings;
- historical receipt preservation rule;
- FINAL/fresh firewall;
- evidence ceiling.

Recommended successor identity:

`CB16_R11_POST_CC_SCIENCE_SEMANTIC_V1`

Exact final identifier may differ, but it must be immutable once S0 qualification begins.

---

## S0-010 — Freeze S1 observation-materialization extension rule

S1 requires durable Brain-ready observations, while frozen W-01/W-03 identify observations primarily by identity/hash.

S0 must decide and freeze the **extension mechanism**, not implement the entire S1 store.

Preferred rule:

- W-01..W-05 remain unchanged;
- S1 adds a content-addressed post-CC observation fact/sidecar linked by `observation_hash` and science identity;
- observation material contains separated `market`, `account`, `execution` payloads required by `CCCentralBrain`;
- no future market data may enter the observation fact;
- normalizer/source/version identities are included;
- hash mismatch fails closed.

Recommended contract name:

`PostCCObservationFactV1`

S0 qualification must include schema known-answer examples and corruption rejection.

---

## S0-011 — Freeze S1 joint replay-sample contract

Define, at contract level, the sample S1 learner must receive.

Recommended identity:

`PostCCJointReplaySampleV1`

Required information includes:

- sequence ID / transition ref;
- account lineage / decision index / environment time;
- materialized market/account/execution observations;
- nominal direction;
- nominal target risk;
- risk measure kind;
- true behavior `log_mu`;
- behavior policy/generation identity;
- reward;
- discount / boundary / bootstrap semantics;
- replay sampling probability/weight;
- target policy identity at training time;
- source hashes linking back to durable facts.

Executed direction/risk must not replace nominal action in this sample.

---

## S0-012 — Freeze S1 known-answer task registry

Create a machine-readable preregistration for S1 task identities. Recommended file:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json`

Minimum mandatory tasks:

1. `ACCOUNT_DEPENDENT_ACTION`
2. `DELAYED_CONSEQUENCE_CREDIT`
3. `HIGH_BANKRUPTCY_HIGHER_ARITHMETIC_EXPECTATION`
4. `OFF_POLICY_VTRACE_CORRECTION`
5. `A_B_A_RETENTION_WITHOUT_HANDCRAFTED_REGIME_ACTIVATION`

Each task entry must freeze:

- environment generator/version;
- observation/action semantics;
- reward semantics;
- horizon/boundary semantics;
- known-answer criterion;
- allowed seeds or seed-generation rule;
- training budget;
- evaluation schedule;
- success metric;
- negative controls;
- no-rescue rule.

Do not freeze thresholds after seeing S1 results.

---

## S0-013 — Freeze S1 model identity / parameter counting

S1 must report the actual trainable Actor and Critic identities rather than a colloquial model size.

Freeze required fields:

```text
P_actor_trainable
P_critic_trainable
P_frozen
P_replica
Actor architecture identity
Critic architecture identity
optimizer identity
normalizer identity
policy distribution identity
```

The S1 qualification model should remain deliberately bounded. Capacity scaling belongs to a later stage; S1 is about learnability, not finding the largest model.

---

## S0-014 — Freeze S1 run manifest and seed policy

Create a run-spec schema requiring:

- exact code SHA;
- S0 receipt hash;
- task registry hash;
- model config hash;
- initial checkpoint hash;
- RNG roots/seed identities;
- task/episode count;
- collection/update schedule;
- replay selection rule;
- evaluation policy identity;
- resource limits;
- expected artifact paths;
- FINAL/fresh firewall state.

At least one independent repeat/multi-seed rule must be frozen before execution.

---

## S0-015 — Freeze what counts as S1 “learning”

S1 PASS cannot be based only on:

- loss decreased;
- gradient was non-zero;
- parameter checksum changed;
- checkpoint was written;
- child policy produced a different action;
- a single lucky seed passed.

S0 must freeze that S1 requires **preregistered behavioral/return success** on each mandatory task after actual repeated learner updates through durable replay.

For tasks with a known optimal action/policy, score action probability/behavior and expected task return or another preregistered known-answer statistic.

---

## S0-016 — Freeze negative controls

Mandatory S1 controls must include at least:

- `NO_SIGNAL_OR_ZERO_REWARD_CONTROL` — learner must not manufacture systematic success;
- `SHUFFLED_CREDIT_CONTROL` — breaking action/consequence alignment must materially damage the target learning signal;
- one impossible/random-label control where applicable.

The exact expected relation must be preregistered. A control failure blocks learnability PASS even if positive tasks look good.

---

## S0-017 — Freeze reward/discount/boundary use in S1

The current arithmetic reward contract and finite-horizon telescoping semantics must not be silently replaced by an arbitrary canary discount.

S0 must record, per task:

- `E_ref` definition;
- reward equation;
- gamma/discount choice and why it is compatible with that known-answer task;
- bootstrap behavior at computational truncation vs mechanical terminal;
- whether a task intentionally tests a non-1 discount as an algorithmic control.

The high-bankruptcy/higher-expectation task must preserve arithmetic expected return as the winning criterion.

---

## S0-018 — Freeze failure classifications and no-rescue discipline

S0/S1 statuses:

- `PASS`
- `SCIENTIFIC_FAIL`
- `CONTRACT_MISMATCH`
- `EXECUTION_BLOCKED`
- `HARDWARE_LIMIT`
- `EVIDENCE_INSUFFICIENT`

No in-place rescue by changing seeds, thresholds, reward, model size, task difficulty or evaluation population after observing results.

Any such change opens a new R/version identity.

---

## S0-019 — Build S0 qualification tests

Required qualification test families:

- economic ordering independent from baselines;
- all four B&H/FLAT component combinations;
- promotion-rule separation;
- no-master-baseline schema sentinel;
- legacy R0 promotion remains historically reproducible;
- post-CC router rejects legacy authority;
- W-05 adapter preserves independent deltas;
- old receipts unchanged;
- observation-fact schema/hash known-answer;
- joint replay sample schema known-answer;
- S1 registry/run-spec completeness;
- FINAL/fresh firewall.

Tests may be synthetic; no market training is authorized in S0.

---

## S0-020 — Compile migration report

Create a durable report summarizing each old-to-new semantic change.

Recommended path:

`artifacts/post_cc_s0/CONTRACT_MIGRATION_REPORT.json`

For each migration record:

```text
old_surface
old_meaning
historical_authority
successor_surface
successor_meaning
reason
historical_bytes_modified=false
compatibility_behavior
fail_closed_behavior
```

---

## S0-021 — Emit S0 machine-readable receipt

Required final receipt:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S0_RECEIPT_V1.json`

Minimum fields:

```text
status
frozen_base_sha
qualified_head_sha
spec_path + hash
parent_cc_spec/receipt hashes
economic_ordering_contract
baseline_component_contract
promotion_contract
migration_contract
historical_cc_receipts_unchanged
legacy_post_cc_route_disabled
s1_task_registry hash
s1_run_spec identity
s1_model_identity rule
s1_data_scope=SYNTHETIC_ONLY
s1_evidence_ceiling
qualification tests
FINAL_opened=false
fresh_data_used=false
unresolved_items[]
```

Target S0 evidence:

`POST_CC_CONTRACT_MIGRATION_QUALIFIED`

No ECONOMIC/TRANSFER claim.

---

## S0-022 — Update canonical navigation only after PASS

After S0 PASS, update:

- `docs/CURRENT_STATE.md`
- `docs/README.md`
- `AGENTS.md`
- `docs/POST_CC_SCIENTIFIC_PROGRAM_R0.md`

They must point S1 executors to the S0 receipt and exact qualified S1 base.

Do not mark S1 started merely because S0 docs were created.

---

# 4. S0 file ownership guidance

Prefer new post-CC prefixes to avoid rewriting CC history:

```text
cb16_local_opt/post_cc_economic_*.py
cb16_local_opt/post_cc_promotion_*.py
cb16_local_opt/post_cc_contract_*.py
tests/test_post_cc_s0_*.py
authority/rearchitecture_r11/CB16_R11_POST_CC_S0_*.json
authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json
```

Do not rename/delete the CC R0 modules merely to make the repository look cleaner.

---

# 5. S0 hard gates

S0 PASS requires all of the following:

```text
S0_BASELINE_IDENTITY_PASS
NO_MASTER_BASELINE_PRECEDENCE_PASS
MODEL_ORDERING_SEPARATION_PASS
PARALLEL_BASELINE_COMPONENTS_PASS
PROMOTION_RULE_SEPARATION_PASS
LEGACY_MIGRATION_PASS
HISTORICAL_RECEIPT_IMMUTABILITY_PASS
W05_ADAPTER_PASS
S1_OBSERVATION_EXTENSION_CONTRACT_PASS
S1_JOINT_REPLAY_CONTRACT_PASS
S1_TASK_REGISTRY_FROZEN_PASS
S1_RUN_SPEC_FROZEN_PASS
S1_LEARNING_EVIDENCE_RULE_FROZEN_PASS
NEGATIVE_CONTROL_PREREGISTRATION_PASS
FINAL_FRESH_FIREWALL_PASS
```

Any missing gate prevents S1 canonical qualification from starting.

---

# 6. Definition of Done

S0 is DONE when:

1. the old ambiguity is migrated in code/API routing without modifying historical receipts;
2. B&H and FLAT disagreements are representable as ordinary parallel component results;
3. model ordering and promotion are separately versioned;
4. post-CC code cannot silently invoke legacy precedence semantics;
5. S1 observation/replay extension and joint-action batch semantics are frozen;
6. S1 task registry, run identity, seed rule, evidence threshold rule and negative controls are frozen before S1 results exist;
7. S0 receipt is PASS;
8. FINAL/fresh data remain untouched.

Then—and only then—freeze the S1 branch from the S0 qualified head and execute `S1_END_TO_END_LEARNABILITY_TODO.md`.
