# CB16 R11 S1 — Post-S0v2 Base Handoff

**Status:** S1 BASE FROZEN / IMPLEMENTATION NOT STARTED  
**Authoritative baseline:** `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`  
**Working branch:** `ai/r11-post-cc-s1-end-to-end-learnability-r1`  
**Exact parent main SHA:** `fc80b472236e7a4df8094563f8adac826fc42231`  
**Parent evidence:** `POST_CC_DURABLE_LEARNING_FOUNDATION_QUALIFIED`  
**FINAL:** SEALED  
**Fresh data:** FORBIDDEN

## 1. Why this handoff exists

`docs/post_cc/S1_END_TO_END_LEARNABILITY_TODO.md` and `CB16_R11_POST_CC_S1_RUN_SPEC_V1.json` were frozen before S0-v2 was inserted and qualified. Their task definitions, seeds, model, optimizer, budget, reward, controls, thresholds and no-rescue rules remain authoritative.

Their old S1 working-base text does **not** remain authoritative. S0-v2 closed the durable observation/replay/joint-learning/exactly-once/generation-continuity foundation that the legacy S1 TODO still lists as missing.

The exact post-S0v2 base binding is now defined by:

`authority/rearchitecture_r11/CB16_R11_POST_CC_S1_BASELINE_V1.json`

This is a base-binding migration only. It is not a result-driven scientific change.

## 2. Exact lineage

```text
main@fc80b472236e7a4df8094563f8adac826fc42231
  = Sol R2 accepted S0-v2 merge state
        |
        v
ai/r11-post-cc-s1-end-to-end-learnability-r1
        |
        +-- CB16_R11_POST_CC_S1_BASELINE_V1.json
        +-- this handoff
```

Do not use the old branch `ai/r11-post-cc-s1-end-to-end-learnability-r0@9d991143...` as the S1 base. It diverged before S0-v2. Preserve it as history; do not force-reset it or silently import its seven divergent commits.

## 3. What S1 inherits as already qualified foundation

Reuse rather than reimplement the accepted S0-v2 surfaces:

- durable canonical observation fact and immutable observation store;
- observation-to-runtime collection binding;
- durable joint replay records and restart-safe materializer;
- canonical joint direction+risk batch;
- target joint `log_pi` evaluated on nominal behavior action;
- decision-time persisted true `log_mu` integrity;
- separate Critic, explicit boundary/bootstrap semantics and V-trace path;
- durable-replay-only learner path and gradient ownership;
- exactly-once durable update journal and recovery;
- committed child checkpoint authority and same-account generation continuity.

S1 may extend these only where the frozen S1 scientific tasks require it. Do not replace them with alternate algorithms merely because a toy task would be easier.

## 4. What remains for S1 to prove

S0-v2 component/foundation qualification is not S1 learnability evidence. S1 must still execute the preregistered repeated-learning program across all mandatory positive tasks and negative controls, including multi-seed qualification, repeated durable replay learning, behavioral/arithmetic-return improvement, failure retention, and the unified end-to-end qualification runtime.

The only S1 PASS evidence ceiling is:

`INTEGRATED_SYNTHETIC_END_TO_END_LEARNABILITY_KNOWN_ANSWER`

Loss reduction, nonzero gradients, checkpoint change, one optimizer step, or one child action are insufficient.

## 5. Frozen preregistration remains unchanged

Continue to read:

- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_TASK_REGISTRY_V1.json`
- `authority/rearchitecture_r11/CB16_R11_POST_CC_S1_RUN_SPEC_V1.json`

Keep unchanged unless a genuinely new, versioned scientific program is opened before observing qualification outcomes:

- seeds `1701..1705`;
- minimum `4/5` positive seeds passing;
- `CCCentralBrain` + separate Critic model family and frozen dimensions;
- SGD learning rate `0.01` for Actor and Critic;
- maximum `100000` policy decisions per task/seed;
- arithmetic reward orientation and complete-failure denominator;
- negative controls and no-rescue rules.

## 6. Start rule

Before any S1 implementation or qualification:

1. verify current branch descends from exact parent `fc80b472236e7a4df8094563f8adac826fc42231`;
2. verify `CB16_R11_S0V2_RECEIPT_V1.json` remains `QUALIFIED` / Sol R2 `PASS`;
3. verify the S1 baseline, task registry and run spec are present and unchanged;
4. verify FINAL/fresh-data firewall remains closed;
5. adopt the qualified S0-v2 foundation and begin only the remaining S1 scientific learnability work.

No S1 scientific execution was performed as part of this handoff.
