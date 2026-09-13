# CB16 Stage Review, Merge, Receipt, and Successor Handoff Protocol

**Status:** ACTIVE / applies prospectively beginning with S0-v2  
**Purpose:** make every S-stage a closed pipeline loop with one implementation branch, independent Sol review, exact-SHA Shanxi CI evidence, reviewer receipt, merge, and an unambiguous successor base.

This protocol does not rewrite historical receipts or retroactively change completed stages. It applies to S0-v2 and successor S1/S2/... stages unless a later owner-authorized document explicitly supersedes it.

## 1. Canonical stage lifecycle

Every new S-stage follows this lifecycle:

```text
main authority / frozen stage input
  -> one designated stage branch
  -> one implementer executes the complete bounded stage package
  -> implementer-side tests + GitHub Actions -> Shanxi Docker on exact candidate SHA
  -> READY_FOR_SOL_REVIEW
  -> Sol reviews actual diff, formulas, branches, masks, boundaries, recovery and CI evidence
  -> changes requested and repaired on the same stage branch, or reviewer acceptance
  -> mandatory reviewer receipt binds the accepted candidate and evidence
  -> merge to main
  -> verify merged tree is exactly the reviewed tree, or invalidate acceptance and re-review/re-test the changed integration
  -> reviewer receipt is finalized with merge identity and successor_working_base_sha
  -> only then may the next S-stage be routed from that recorded successor base
```

An implementer does not self-merge and does not self-sign the reviewer acceptance.

## 2. Mandatory reviewer receipt

For S0-v2 the reviewer-side receipt path is:

`authority/rearchitecture_r11/CB16_R11_S0V2_RECEIPT_V1.json`

Successor stages must define an analogous versioned receipt path in their TODO before implementation begins.

The reviewer receipt is mandatory for a stage to become qualified and routable to the next stage. It must bind at least:

- stage identity and evidence ceiling;
- scientific/input baseline identity;
- designated stage branch;
- exact implementer candidate commit SHA;
- exact reviewed candidate tree SHA;
- candidate record identity/hash;
- reviewed implementation/test/workflow surfaces;
- Shanxi workflow path/version, run ID, attempt, job ID/runner, actual checkout SHA, artifact ID/hash where applicable;
- focused/regression/hostile test inventory and gate verdicts;
- formula/control-flow/boundary review verdict;
- FINAL/fresh-data firewall state;
- reviewer decision: ACCEPTED / CHANGES_REQUIRED / REJECTED / classified blocker as applicable;
- merge commit SHA once merged;
- merged tree SHA;
- reviewed-tree-versus-merged-tree equality verdict;
- `successor_working_base_sha`;
- unresolved issues and evidence limitations.

The receipt must not claim reviewer acceptance before the exact candidate SHA and its relevant CI evidence have actually been reviewed.

## 3. Reviewed tree must equal merged tree

Reviewer acceptance is attached to the reviewed Git tree, not merely to a branch name or an ancestor commit.

Before a stage is closed:

```text
reviewed_candidate_tree_sha == merged_main_tree_sha
```

must hold for the accepted stage content.

If merge conflict resolution, manual edits, extra commits, rebasing, generated-file changes, workflow changes, or any other integration step changes the reviewed tree, the prior acceptance does not automatically cover the new tree. The affected diff must be re-reviewed and the relevant Shanxi CI rerun on the new exact SHA before final qualification.

A successful ancestor run may not be used as evidence for a changed descendant tree.

## 4. Single successor working base

The next stage must never infer its base from phrases such as “qualified implementation head”, “candidate head”, “receipt-bearing head”, or a moving branch name.

The final reviewer receipt explicitly records exactly one:

`successor_working_base_sha`

For S0-v2 -> S1 this value is the accepted S0-v2 merged-main handoff commit that contains, or is explicitly paired with, the finalized reviewer receipt and whose tree has passed the reviewed-tree/merged-tree equality check.

S1 must start from that exact SHA. It must not independently choose the S0-v2 candidate SHA, an earlier code-only SHA, or a later unrelated moving `main` head.

The same rule applies to S1 -> S2 and all later transitions.

## 5. Candidate SHA, reviewed SHA, merge SHA, and successor base are distinct identities

A stage may legitimately have several SHAs. They must never be conflated:

- **candidate SHA:** implementer branch head offered for review;
- **reviewed SHA/tree:** exact code and configuration Sol reviewed;
- **merge SHA/tree:** result actually merged into main;
- **successor working base SHA:** single handoff identity recorded by the finalized reviewer receipt.

If two identities happen to be equal, record that fact explicitly rather than assuming equivalence by convention.

## 6. Repair loop stays on the same branch

If Sol finds a defect, the implementer repairs it on the same designated stage branch unless a concrete repository integrity reason requires otherwise. Each repair that changes relevant code/tests/workflows invalidates superseded exact-SHA runtime evidence for the affected scope and requires appropriate re-review/re-test.

Do not create sibling implementation branches merely to evade failed review or failed CI.

## 7. Stage boundaries remain scientific boundaries

The lifecycle standardizes engineering governance; it does not collapse scientific evidence levels.

Examples:

- S0-v2 may at most establish the durable learning foundation evidence defined by its TODO;
- S1 may qualify synthetic known-answer learnability if its frozen scientific gates pass;
- S2 may only execute the scope explicitly authorized after S1 evidence is accepted.

A stage being merged does not automatically authorize the next stage's science. The next stage becomes executable only after its TODO is issued against the prior receipt's `successor_working_base_sha`.

## 8. Stop and handoff rule

At stage completion the implementer stops at `READY_FOR_SOL_REVIEW`. Sol owns acceptance, receipt finalization and merge. After merge, the current stage stops. The next stage starts only from its separately issued task package.
