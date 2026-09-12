# CC Thread B — Policy / Critic / Learner / Retention

**Thread:** B  
**Frozen base:** `89d62bf966f476e598f0e2f5c5e8e03c15a8db51`  
**Branch:** `ai/r11-cc-thread-b-learning-r0`  
**Independence rule:** no imports/cherry-picks/dependencies from CC Threads A/C/D  
**Primary evidence target:** COMPONENT + KNOWN_ANSWER  

Thread B finishes the learning side using synthetic, in-memory CC wire fixtures. It does not need a real collector, lake, economic evaluator or high-throughput runtime.

It may reuse current landed R1 observation/state/Actor work already present on the frozen base. It should prefer additive `cc_policy_*`, `cc_critic_*`, `cc_vtrace_*`, `cc_learner_*` modules and tests. Existing `actor_policy_r1.py` is an implementation input, not proof that B is complete.

---

## B-01 — Freeze Thread-B baseline and Actor audit

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_B_BASELINE_V1.json`
- `[NEW] tests/test_cc_thread_b_baseline_r0.py`

Bind base SHA and blob hashes for:

- Actor/Critic observation contracts;
- observation normalization;
- policy memory;
- gradient ownership;
- state sufficiency;
- `actor_policy_r1.py`;
- R1 science contract.

Record exactly which BC-034..037 semantics are already present and which are still absent.

---

## B-02 — Thread-local W-01 / W-03 / W-04 fixtures

**Files**
- `[NEW] cb16_local_opt/cc_policy_wire_r0.py`
- `[NEW] tests/test_cc_policy_wire_r0.py`

Implement thread-local typed fixtures for:

- `CCPolicyDecisionV1`;
- `CCExperienceSequenceV1`;
- `CCLearningUpdateV1`.

These are local B types only. They must preserve the master field meanings exactly.

---

## B-03 — Complete joint nominal-action likelihood

**Files**
- `[NEW or ADAPT] cb16_local_opt/cc_policy_distribution_r0.py`
- `[NEW] tests/test_cc_policy_distribution_r0.py`

Using the current Actor R1 semantics, provide exact joint scoring:

```text
log p(direction) + log p(risk | direction)
```

with:

- FLAT exact zero-risk point mass;
- no extra continuous density term for FLAT;
- non-FLAT continuous interior risk;
- exact disallow/probability semantics at risk 0/1 according to current R1 endpoint policy;
- stable finite direction log-probs for extreme logits.

**PASS:** sampled action can be rescored exactly under an unchanged policy identity.

---

## B-04 — Isolated serializable policy RNG

**Files**
- `[NEW] cb16_local_opt/cc_policy_rng_r0.py`
- `[NEW] tests/test_cc_policy_rng_r0.py`

Provide per-policy/per-account or otherwise explicitly isolated RNG identity suitable for later batching without order-dependent corruption.

Requirements:

- deterministic save/restore;
- stream/counter provenance exported into W-01;
- evaluation calls do not consume training/collection RNG;
- inter-account batching order cannot silently change an account's intended RNG stream.

**PASS:** restore reproduces exact next action sequence.

---

## B-05 — Deterministic evaluation policy identity

**Files**
- `[NEW] cb16_local_opt/cc_policy_evaluation_r0.py`
- `[NEW] tests/test_cc_policy_evaluation_r0.py`

Define deterministic evaluation as an explicit policy object/identity, not as an undocumented runtime flag on the stochastic behavior policy.

Evaluation must not consume stochastic RNG.

---

## B-06 — CC Central Brain composition

**Files**
- `[NEW] cb16_local_opt/cc_policy_brain_r0.py`
- `[NEW] tests/test_cc_policy_brain_r0.py`

Compose:

- frozen market-organ outputs;
- account observation;
- execution/legal observation;
- optional causal policy memory;
- trainable account/fusion path;
- stochastic Actor heads.

Bind exact observation/normalizer/distribution/science hashes.

Do not modify or rename the historical Brain in place.

---

## B-07 — Gradient ownership qualification

**Files**
- `[NEW] tests/test_cc_policy_gradient_ownership_r0.py`

Positive proof:

- frozen market organs do not receive gradient/mutation;
- declared trainable account/fusion/Actor paths do receive gradients in a solvable fixture.

Negative proof:

- deliberately enabling a frozen parameter causes qualification failure.

---

## B-08 — Nominal-action / execution firewall in learner inputs

**Files**
- `[NEW] cb16_local_opt/cc_policy_probability_firewall_r0.py`
- `[NEW] tests/test_cc_policy_probability_firewall_r0.py`

Learner importance ratios must use nominal sampled action + true behavior likelihood from W-01.

Tests include synthetic clamp/reject/partial execution/reversal failure. Executed action may differ but cannot replace the behavior sample in the probability term.

---

## B-09 — Arithmetic-equity reward

**Files**
- `[NEW] cb16_local_opt/cc_policy_reward_r0.py`
- `[NEW] tests/test_cc_policy_reward_r0.py`

Implement:

```text
r_t = (E_{t+1} - E_t) / E_ref
```

with fixed declared `E_ref` per objective definition.

No log/sign/clipping/drawdown replacement. Synthetic signed-equity/liability fixtures must work.

---

## B-10 — Reward telescoping qualification

**Files**
- `[MODIFY] tests/test_cc_policy_reward_r0.py`

For complete finite synthetic paths verify:

```text
sum(r_t) == (E_T - E_0) / E_ref
```

including fees, funding, partial close, liquidation and negative terminal equity.

---

## B-11 — Boundary-aware bootstrap calculator

**Files**
- `[NEW] cb16_local_opt/cc_critic_bootstrap_r0.py`
- `[NEW] tests/test_cc_critic_bootstrap_r0.py`

Consume synthetic boundary labels and implement:

- zero bootstrap only when the objective/economic contract truly terminates future value;
- bootstrap at compute chunk/pause;
- do not treat unknown data end as realized terminal;
- pending settlement is not zeroed early.

No Thread-A import is allowed; use local boundary fixtures.

---

## B-12 — Separate Critic model

**Files**
- `[NEW] cb16_local_opt/cc_critic_value_r0.py`
- `[NEW] tests/test_cc_critic_value_r0.py`

Build a distinct value network for expected future arithmetic return.

Critic may see allowed causal `tau`; Actor remains tau-free.

No Teacher semantics, no quantile/twin-Q pessimistic substitution.

---

## B-13 — Critic mean-regression objective

**Files**
- `[MODIFY] cb16_local_opt/cc_critic_value_r0.py`
- `[MODIFY] tests/test_cc_critic_value_r0.py`

Use mean-value regression consistent with arithmetic expectation. Prove exact-value convergence on a small deterministic fixture under preregistered budget/tolerance.

---

## B-14 — V-trace recurrence

**Files**
- `[NEW] cb16_local_opt/cc_vtrace_r0.py`
- `[NEW] tests/test_cc_vtrace_r0.py`

Implement exact:

- `log_pi - log_mu` ratio;
- rho/c clipping;
- masks;
- backward recurrence;
- same-policy reduction;
- explicit finite-horizon gamma semantics.

Must pass hand-computed short sequences and clipping-boundary tests.

---

## B-15 — Actor policy-gradient loss

**Files**
- `[NEW] cb16_local_opt/cc_policy_loss_r0.py`
- `[NEW] tests/test_cc_policy_loss_r0.py`

Compute policy-gradient loss from frozen V-trace advantage/targets.

No silent PPO/GAE stacking. No permanent entropy preference unless separately versioned; if an entropy diagnostic exists it must not change the objective by default.

**PASS:** gradient sign matches hand-built action-preference cases.

---

## B-16 — In-memory sequence batch adapter

**Files**
- `[NEW] cb16_local_opt/cc_learner_batch_r0.py`
- `[NEW] tests/test_cc_learner_batch_r0.py`

Convert synthetic W-03 sequences into learner tensors. This is **not** the persistent replay implementation; Thread C owns that.

Requirements:

- deterministic sequence ordering;
- masks/boundaries explicit;
- behavior likelihood retained;
- no survivor filtering;
- no implicit data-source reweighting.

---

## B-17 — Actor-Critic learner runtime

**Files**
- `[NEW] cb16_local_opt/cc_learner_r0.py`
- `[NEW] tests/test_cc_learner_r0.py`

Implement one logical learner with:

- Actor and Critic optimizers;
- explicit update counter;
- B-07 gradient ownership;
- B-14 V-trace;
- B-15 policy loss;
- metrics for support/clipping/value/policy behavior;
- no mutation of the running behavior policy object.

**PASS:** fixed seed produces deterministic parameter delta on a fixed synthetic batch.

---

## B-18 — Exactly-once learner transaction

**Files**
- `[NEW] cb16_local_opt/cc_learner_transaction_r0.py`
- `[NEW] tests/test_cc_learner_transaction_r0.py`

Bind update ID to:

- parent checkpoint;
- sampled sequence IDs;
- optimizer counters;
- W-04 update metadata.

Crash cases:

- before gradient;
- after gradient calculation before durable commit;
- after durable commit before receipt acknowledgement.

Exactly one update may commit.

---

## B-19 — Full learner checkpoint/recovery

**Files**
- `[NEW] cb16_local_opt/cc_learner_checkpoint_r0.py`
- `[NEW] tests/test_cc_learner_checkpoint_r0.py`

Bundle:

- Actor;
- Critic;
- both optimizers;
- policy RNG;
- learner RNG;
- normalizer/science identities;
- update transaction state;
- policy generation identity;
- optional policy-memory configuration/state contract.

**PASS:** restart gives the same next sampled action and same next committed gradient transaction.

---

## B-20 — Fixed behavior generation contract

**Files**
- `[NEW] cb16_local_opt/cc_policy_generation_r0.py`
- `[NEW] tests/test_cc_policy_generation_r0.py`

A registered collection policy is immutable during its collection unit. Learner trains a separate child copy.

This task uses synthetic collection-unit metadata only; it does not depend on Thread A runtime.

---

## B-21 — A→B→A retention fixture

**Files**
- `[NEW] cb16_local_opt/cc_continual_retention_r0.py`
- `[NEW] tests/test_cc_continual_retention_r0.py`

Build analytically solvable tasks/regimes A and B:

1. learn A;
2. adapt on B;
3. revisit A;
4. compare no historical replay vs compatible historical replay under frozen budgets.

Report both B adaptation and A retention. Do not add handcrafted regime switches/cycle rules.

---

## B-22 — Account-dependent action toy

**Files**
- `[NEW] cb16_local_opt/cc_learning_toys_r0.py`
- `[NEW] tests/test_cc_learning_toys_r0.py`

Same market observation, different account observations, analytically different optimal actions. If observation aliasing is deliberately introduced, classify representation gap rather than optimization failure.

---

## B-23 — Delayed-consequence/horizon toy

**Files**
- `[MODIFY] cb16_local_opt/cc_learning_toys_r0.py`
- `[MODIFY] tests/test_cc_learning_toys_r0.py`

Construct an action with short-term gain followed by delayed larger loss and a separate case where a longer objective horizon genuinely reverses ranking.

**PASS:** learner follows the analytical long-horizon answer under preregistered budget.

---

## B-24 — High-bankruptcy/higher-expectation toy

**Files**
- `[MODIFY] cb16_local_opt/cc_learning_toys_r0.py`
- `[MODIFY] tests/test_cc_learning_toys_r0.py`

Exact distribution where the risky strategy has higher bankruptcy frequency but higher arithmetic expectation.

**PASS:** learned/ranked preference follows expected arithmetic return; no survival override.

---

## B-25 — Off-policy correction toy

**Files**
- `[MODIFY] cb16_local_opt/cc_learning_toys_r0.py`
- `[MODIFY] tests/test_cc_learning_toys_r0.py`

Known behavior `mu` and target `pi`. Compare:

- correct V-trace;
- same-policy path;
- deliberately uncorrected replay.

Verify expected recurrence and clipping/support diagnostics.

---

## B-26 — No handcrafted recurrence sentinel

**Files**
- `[NEW] tests/test_cc_no_handcrafted_regime_activation_r0.py`

Reject newly introduced rule tables/calendar switches/manual regime activators used to force B-21 to pass.

Learned weights/replay/causal memory are allowed; handcrafted activation logic is not.

---

## B-27 — Thread-B qualification compiler

**Files**
- `[NEW] cb16_local_opt/cc_learning_qualification_r0.py`
- `[NEW] tests/test_cc_learning_qualification_r0.py`

Compile:

- joint probability;
- RNG/recovery;
- Actor/Critic firewall;
- reward telescoping;
- V-trace math;
- gradient ownership;
- exactly-once learner update;
- checkpoint recovery;
- A→B→A retention;
- account-dependent, delayed, high-risk and off-policy known-answer toys.

Do not claim real-market economic edge.

---

## B-28 — Thread-B receipt

**Files**
- `[NEW] authority/rearchitecture_r11/CB16_R11_CC_THREAD_B_RECEIPT_V1.json`

Record base/head SHAs, exact tests, owned files, strongest evidence, no sibling dependency, FINAL/fresh firewall and known-answer results.

### Thread-B DONE

Thread B is complete when all learning math and synthetic known-answer qualifications pass using only frozen-base code plus B-owned modules. It must be mergeable without Threads A/C/D.