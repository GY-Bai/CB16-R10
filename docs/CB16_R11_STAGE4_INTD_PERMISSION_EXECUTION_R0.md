# CB16 R11 Stage-4 INTD — Permission / Execution Closure R0

## Scope

INTD integrates the already-qualified S4E Permission boundary with live canonical
Stage-4 runtime authority. It is an additive integration only. It does not modify
Frozen Supervisor, Frozen Physics, S4E, S4G, S4B, or S4D.

The production ingress is closed to:

`Brain/policy -> ActionIntent -> canonical runtime identity + S4G eligibility + live fence -> Frozen Supervisor -> ExecutableAction -> live fence -> Frozen Physics -> account transition`

No caller-facing API accepts a Supervisor decision, `allowed_action`,
`ExecutableAction`, Permission, account transition, post-outcome authorization,
or recovery-carried Permission.

## Authority composition

`CanonicalExecutionClosureR11` consumes:

1. an S4B `RuntimeContext`, bound exactly to the configured authority id,
   generation, Semantic Freeze identity, recovered state id, and scientific
   history id;
2. an injected `LiveFenceAssertionBoundaryR11`, whose implementation must retain
   the qualified S4D live-fence binding and ultimately assert the current live
   fence;
3. an S4G `RuntimeIdentity` and admission tokens for
   `GRANT_PERMISSION` and `EXECUTE_ACCOUNT_PHYSICS_TRANSITION`;
4. the already-qualified S4E `FrozenPermissionBoundaryR11`;
5. the existing frozen runtime supplying Supervisor, Physics, and physics
   contract.

S4G is used only as a negative eligibility guard. The integration verifies that
S4G decisions have `authority_granted == false` and still require independent
runtime authority. A retirement admission token is never interpreted as
Permission or as a fencing token.

The opaque S4B lease string is never decoded or trusted by INTD. The injected
fence provider is responsible for proving that the lease still corresponds to a
live S4D fence.

## Stale-owner closure

There are two positive authority assertions per execution:

- immediately before entering the frozen S4E Supervisor chain;
- immediately after `ExecutableAction` construction and immediately before
  `execute_physics`.

The second assertion is installed through a private representation-only
Supervisor proxy. It does not alter Supervisor decisions, ExecutableAction
contents, Physics inputs, or Physics outputs. If authority becomes stale after
Supervisor adjudication, Physics is not invoked.

## Semantic invariants

- Truth != Belief != Decision != Permission.
- `requested_risk` / `requested_risk_multiplier` remains a risk request. It is
  not confidence, Permission, or Supervisor authorization.
- Supervisor exclusively produces the decision.
- S4E exclusively converts the Supervisor decision into `ExecutableAction`.
- Frozen Physics owns the account transition.
- Compatibility and recovery conversion remain S4E representation-only
  conversions and cannot carry privileged fields.
- No Stage-4 Permission mechanism is minted.
- Replay is not new Evidence.
- No scientific verdict is created.

Scientific status remains:

`DISTRIBUTIONAL_MARKET_INFORMATION_NOT_QUALIFIED__TRUE_WORSE_THAN_SHUFFLE`

## Hostile coverage

The INTD tests fail closed on forged direct executable input, legacy
`allowed_action`, caller-supplied Supervisor decision, direct Physics-shaped
input, requested-risk-as-confidence/Permission fields, stale fence before
Supervisor, stale fence after Supervisor but before Physics, legacy/replay/test/
diagnostic runtime identity escalation, S4G subject mismatch, wrong generation
or runtime identity, stale recovery Permission, malformed ActionIntent, missing
Supervisor, missing Physics, and privileged-looking compatibility objects.

## Final consolidation boundary

INTD does **not** implement the S4D-to-S4B lease adapter; INTA/final
consolidation must bind `LiveFenceAssertionBoundaryR11.assert_current` to the
actual live S4D fencing proof, not to a copied opaque string.

Final consolidation must also:

- route every production Brain/compatibility/recovery action ingress through
  this closure (or prove the exact equivalent);
- ensure the underlying frozen runtime is not separately exposed as a
  caller-facing bypass API;
- run S4H H14 forged Permission and H15 direct Physics bypass against the real
  consolidated runtime;
- re-run the S4A writer/authority inventory at final head and require
  `UNKNOWN_AUTHORITY = 0`;
- prove the final holdout and Semantic Freeze remain untouched.

INTD PASS is only an integration-fork qualification. It is not
`R11_CANONICAL_AUTHORITY_CUTOVER_QUALIFIED`.
