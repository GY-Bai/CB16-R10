# S0-v2 Sol Code Review R1 — CHANGES_REQUIRED

**Review date:** 2026-09-13  
**Stage:** `CB16_R11_S0V2_DURABLE_LEARNABILITY_FOUNDATION_R0`  
**Branch:** `ai/r11-s0v2-durable-learnability-foundation-r0`  
**Task handoff/base:** `134e9cc128a8cd664d9ae2539b06f78d994e064d`  
**Runtime/test/workflow candidate reviewed:** `1b795f9fd1cadf6134e6b963ebc8e28e980f2ad0`  
**Reviewed runtime tree:** `2bd170b6501a8a635b9e35956f2466cbcd288e8e`  
**Evidence-binding branch tip reviewed separately:** `ee3d24029389b2bf841e9dd17528e54c431aae26`  
**Reviewer verdict:** `CHANGES_REQUIRED`  
**Merge authorization:** `NO`  
**Final S0-v2 receipt:** NOT AUTHORIZED YET

This review follows `docs/STAGE_REVIEW_AND_HANDOFF_PROTOCOL.md`, `docs/ROLES_AND_REVIEW_PROTOCOL.md`, and `docs/post_cc/S0_V2_DURABLE_LEARNABILITY_FOUNDATION_TODO.md`.

The candidate is substantial and the exact-SHA Shanxi evidence is real, but green CI does not override semantic review. Three blocking issues must be corrected on the same stage branch before acceptance.

---

## 1. Evidence independently verified before code verdict

The following evidence was independently checked rather than accepted from the implementer summary:

- Candidate lineage: branch is ahead of task handoff without historical receipt edits.
- Qualified runtime/test/workflow candidate: `1b795f9fd1cadf6134e6b963ebc8e28e980f2ad0`.
- Candidate tree: `2bd170b6501a8a635b9e35956f2466cbcd288e8e`.
- Evidence-only follow-up from runtime candidate to `ee3d2402...` changes only `CB16_R11_S0V2_REVIEW_CANDIDATE_V1.json`.
- Shanxi workflow run: `34737871474`, attempt 1, conclusion SUCCESS.
- Shanxi business job: `103672395901`, runner `shanxi-docker-r11`, labels `[self-hosted, shanxi-docker-r11]`.
- Actual checkout SHA in the job: `1b795f9fd1cadf6134e6b963ebc8e28e980f2ad0`.
- Verified environment: Python `3.10.21`, torch `2.8.0+cu126`, GTX 1060 6GB.
- Focused + regression suite: `109 passed in 60.44s`.
- Foundation qualification runner: PASS, 16/16 candidate gates green.
- Artifact: `10312243690`, digest `sha256:40b39dfc5ec57e892635fb2606992ae1c5c8fb22530f393cd6c7ee9ed0d3be38`.
- FINAL/fresh-data firewall remained closed in the reviewed run.

These facts remain valid evidence for the reviewed candidate, but they do not make the candidate acceptable while the blocking semantic holes below remain.

---

# 2. BLOCKER B1 — mechanical terminal provenance is dropped in the actual Critic/V-trace path

**Owning task:** `S0V2-009`  
**Primary file:** `cb16_local_opt/post_cc_critic_vtrace_v1.py`  
**Primary symbol:** `joint_actor_critic_losses_v1`

The durable batch correctly carries and validates `mechanical_terminals`, and the boundary helper can distinguish:

- economic terminal;
- mechanical terminal;
- task horizon;
- compute truncation;
- dataset truncation / other truncation classes.

However the actual Actor/Critic/V-trace path currently does:

```python
sequence_boundary = batch.samples[end - 1].boundary_type
sequence_mechanical = False
...
bootstrap_decision_v1(
    sequence_boundary,
    mechanical_terminal=sequence_mechanical,
    ...
)
```

This hard-codes away the persisted `mechanical_terminal` truth.

For the present zero-bootstrap arithmetic, both economic and mechanical terminal happen to produce terminal/zero bootstrap, so ordinary numerical tests can remain green. That does **not** make the implementation semantically valid: S0V2-009 explicitly requires the actual learning path to distinguish mechanical/economic terminal and other boundary classes rather than collapsing them.

The current qualification gate only proves the helper can distinguish a mechanical terminal; it does not prove `joint_actor_critic_losses_v1` propagates the batch's mechanical-terminal truth. This is exactly the kind of green-test / wrong-branch-routing error that Sol review is required to catch.

### Required correction

- Derive the sequence terminal flag from the durable final sample/batch, not a constant.
- Preserve that value in `BoundaryBootstrapDecisionV1` and any relevant diagnostics/provenance.
- Keep the existing fail-closed mismatch checks.
- Add an end-to-end loss-path test where final `boundary_type == ECONOMIC_TERMINAL` and `mechanical_terminal == True`, then prove:
  - the loss path receives `mechanical_terminal=True`;
  - the bootstrap is zero;
  - mechanical terminal remains distinguishable from ordinary economic terminal in the returned decision/provenance.
- Extend the S0-v2 qualification gate so the **actual loss path**, not just the helper, covers this case.

---

# 3. BLOCKER B2 — generation switch does not fail closed on missing committed-child authority

**Owning task:** `S0V2-012`  
**Primary file:** `cb16_local_opt/post_cc_generation_continuity_v1.py`  
**Primary symbol:** `commit_child_generation_v1`

S0V2-012 requires the generation switch to bind a **committed child checkpoint** through the existing generation-switch authority.

The current API makes the proof optional:

```python
committed_update_record: DurableLearningUpdateV1 | None = None
```

and checks `STATUS_COMMITTED` only when a record is supplied.

Therefore an authorized boundary can call this wrapper without a committed learner-update record and still reach `switch_generation_r0`. That is not fail-closed. It leaves a path where an arbitrary 64-hex child identity can be promoted without proving that it is the committed child of the durable learner transaction.

There is a second attribution gap in the same surface: `child_checkpoint_sha256` and `new_policy_sha256` are independent parameters. The happy-path tests set them equal, but the implementation does not enforce an explicit binding between the runtime policy identity and the committed child checkpoint.

### Required correction

Use one of these fail-closed forms:

1. make committed-update authority mandatory and require `commit_status == COMMITTED`; or
2. resolve the committed update from a durable authority/store inside the switch wrapper.

In either case:

- `child_checkpoint_sha256` must match the committed update's child checkpoint;
- the runtime `new_policy_sha256` must be explicitly bound to that committed child identity. If policy SHA is intentionally a different artifact identity, define and validate a deterministic/versioned mapping instead of accepting an unrelated hash;
- missing authority must fail;
- PREPARED must fail;
- STAGED must fail;
- committed-child hash mismatch must fail;
- runtime policy SHA / child identity mismatch must fail unless an explicit canonical mapping proves equivalence.

Add direct negative tests for all of the above and include them in the S0-v2 qualification gate.

---

# 4. BLOCKER B3 — same-process retry after post-gradient/pre-stage failure can apply a second gradient

**Owning task:** `S0V2-011`  
**Primary file:** `cb16_local_opt/post_cc_learner_v1.py`  
**Primary symbol:** `PostCCDurableReplayLearnerV1.apply_durable_update_v1`

The candidate does good restart testing for:

- PREPARED;
- STAGED;
- COMMITTED;
- process reconstruction from the original parent checkpoint.

But the `after_gradient_before_stage` test hook raises **after** `actor_opt.step()` / `critic_opt.step()` mutate the live learner and before the child checkpoint becomes STAGED.

At that point the durable journal still contains the original PREPARED update. If a caller catches the exception and retries on the **same mutated learner instance**, `parent_checkpoint_sha256()` now represents the already-mutated model. The deterministic update ID therefore changes, allowing a second logical update/gradient application rather than forcing recovery of the original transaction.

The current tests only restart from the saved pre-update parent payload, so they do not exercise this retry path.

S0V2-011 states that the same logical update must never be applied twice and explicitly includes restart/retry hostile handling. A retryable public API must therefore fail closed here.

### Required correction

Choose a fail-closed transaction rule and test it. Acceptable examples:

- after a post-gradient/pre-stage failure, mark the learner instance `POISONED_RESTART_REQUIRED` and reject any further update until reconstructed from the durable parent checkpoint; or
- retain the original parent/update identity and guarantee retry cannot execute a second optimizer step; or
- redesign staging so the durable transaction can deterministically recover the already-computed child without permitting live-state double application.

Required test:

1. trigger the post-gradient/pre-stage fault;
2. retry on the same learner instance;
3. prove the retry cannot perform a second gradient application or create a second authoritative logical update;
4. then prove the approved recovery path reaches one COMMITTED child with optimizer step exactly `before + 1`.

Extend the qualification gate with this hostile counterexample.

---

# 5. Reviewed areas with no blocking defect found in R1

R1 found no blocking formula/sign/index issue in the following reviewed surfaces:

- canonical joint `log_pi` uses the persisted **nominal** direction + target risk, not execution outcome;
- FLAT is a point mass at exact risk 0;
- LONG/SHORT risk density follows the frozen transformed-Normal form and log-Jacobian;
- off-policy ratio order is `log_pi - log_mu`;
- persisted decision-time `log_mu` is carried through durable replay and fabricated/reconstructed values are rejected by the qualified path;
- observation content store is content-addressed, immutable by normal API, and verifies stored bytes/hash on read;
- collection persists decision observation before linking the replay transition and keeps no-decision advances raw-only;
- replay materialization reads durable observation/transition/sequence stores and the current qualification verifies restart reconstruction identity;
- joint batch validates observation reconstruction, nominal action, log_mu, time order, source refs and bootstrap requirements;
- behavior checkpoint immutability and child-account continuation are exercised by the current test suite;
- historical receipts are unchanged; FINAL/fresh-data remained closed.

These are positive review findings, not a final PASS.

---

# 6. Non-blocking hardening note for successor work

The current replay API carries `restart_verified` as caller-provided provenance, while the qualification runner separately demonstrates a fresh durable reconstruction. Because the materializer itself reads from durable stores and does not accept collector-private records, this is **not an R1 merge blocker**. However S1 should avoid treating a bare boolean as independent scientific proof. Prefer an explicit restart-attestation/manifest comparison path when S1 freezes its scientific qualification harness.

---

# 7. Required repair / re-review flow

Do not create a sibling branch.

Repair the three blockers on:

`ai/r11-s0v2-durable-learnability-foundation-r0`

Then:

1. update/add focused counterexample tests;
2. update the S0-v2 qualification compiler gates so each blocker is exercised through the real path;
3. update `CB16_R11_S0V2_REVIEW_CANDIDATE_V1.json` to the new exact runtime/test/workflow candidate SHA/tree;
4. run `.github/workflows/cb16-r11-s0v2-foundation-qualification.yml` on the new exact candidate SHA via Shanxi Docker;
5. preserve the new workflow/run/job/artifact identities and exact checkout SHA;
6. return the branch to `READY_FOR_SOL_REVIEW`.

Any code/test/workflow change invalidates run `34737871474` as merge evidence for the changed scope. It remains historical evidence for R1 only.

Sol will then re-review the affected code plus the exact-SHA CI evidence. No final `CB16_R11_S0V2_RECEIPT_V1.json` and no merge are authorized until that re-review passes.
