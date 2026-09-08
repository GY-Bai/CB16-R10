# CB16 Shanxi Gate Control V1

## Invariant

No GitHub Actions job may execute CB16 business/qualification steps on either Shanxi self-hosted runner unless the same workflow run has already completed, in order:

1. **A Repo Guard** on a GitHub-hosted runner;
2. **B Node24 Guard** on a GitHub-hosted runner;
3. only then **C**, the Shanxi self-hosted job.

A or B failure means C is not eligible to start.

This applies independently of the event source (`push`, `pull_request`, `workflow_dispatch`, or another supported trigger).

## Canonical workflow shape

Every workflow that can target a self-hosted/Shanxi runner must contain:

```yaml
jobs:
  shanxi-preflight:
    uses: ./.github/workflows/_cb16-shanxi-preflight.yml

  business-job:
    needs: shanxi-preflight
    permissions:
      contents: read
    runs-on: [self-hosted, shanxi, <runner-profile>]
```

The reusable preflight owns A -> B ordering. Business workflows do not reimplement either gate.

## Repository-side fail-closed policy

`ci/check_shanxi_workflow_policy.py` rejects workflows that:

- target `self-hosted`, `shanxi`, `cb16-r10-canonical`, or `cb16-wss-qualification` without the canonical preflight;
- omit `needs: shanxi-preflight`;
- do not declare job-level `permissions.contents: read`;
- use checkout without `persist-credentials: false` on a self-hosted job;
- contain direct repository mutation commands in a self-hosted job;
- use dynamic `runs-on` expressions, which could hide a self-hosted target.

The normal repo guard runs this topology policy on every push/PR.

The canonical preflight pins the git blob identities of both the repository policy and the Shanxi topology policy before executing them. A task branch cannot silently weaken the gate implementation and still pass A.

## Host-side non-bypass boundary

Repository checks are not sufficient by themselves: a newly added naked workflow could otherwise be scheduled in parallel with a failing repo guard.

For that reason, both Shanxi runner services must install a copy of:

`ci/shanxi_runner_pre_job_gate.py`

outside any repository checkout and configure it with:

`ACTIONS_RUNNER_HOOK_JOB_STARTED=/absolute/path/to/shanxi_runner_pre_job_gate.py`

The hook is synchronous and runs after GitHub assigns a job but before workflow steps execute. It fails closed unless all of the following are true:

- the current direct caller job declares `needs: shanxi-preflight`;
- `shanxi-preflight` calls the canonical reusable workflow;
- the reusable preflight matches the pinned git blob identity;
- exactly one `A Repo Guard` job completed successfully in the same run;
- exactly one `B Node24 Guard` job completed successfully in the same run;
- B started only after A completed.

The installed copy is host authority. It must not be executed from the mutable repository checkout.

### One-time installation on each Shanxi runner service

Use the actual runner installation directory for each of the two runner services.

```bash
set -euo pipefail

RUNNER_ROOT=/absolute/path/to/actions-runner
HOOK_DIR="$HOME/.cb16-runner-hooks"
HOOK="$HOOK_DIR/shanxi_runner_pre_job_gate.py"

mkdir -p "$HOOK_DIR"
cp /path/to/CB16-R10/ci/shanxi_runner_pre_job_gate.py "$HOOK"
chmod 0555 "$HOOK"

ENV_FILE="$RUNNER_ROOT/.env"
touch "$ENV_FILE"
grep -v '^ACTIONS_RUNNER_HOOK_JOB_STARTED=' "$ENV_FILE" > "$ENV_FILE.tmp"
printf '%s\n' "ACTIONS_RUNNER_HOOK_JOB_STARTED=$HOOK" >> "$ENV_FILE.tmp"
mv "$ENV_FILE.tmp" "$ENV_FILE"

cd "$RUNNER_ROOT"
sudo ./svc.sh stop
sudo ./svc.sh start
```

Repeat for both `cb16-r10-canonical` and `cb16-wss-qualification` runner services.

The runner host needs only public/read access to GitHub metadata. It does not need repository write permission.

## Self-hosted runner is execution-only

For Shanxi jobs:

- `contents: read` is mandatory;
- checkout credentials must not persist;
- GitHub repository push/mutation is forbidden;
- qualification evidence should be emitted as GitHub Actions artifacts or job results;
- any later authoritative repository mutation must occur outside the Shanxi execution plane.

## Multi-sub-agent authority

Task/sub-agents do not own CI orchestration. They may add implementation code, tests, scripts, receipts, and documentation, but must not create an alternative self-hosted execution path.

A task requiring a new runner behavior must extend the central CI authority rather than bypassing A/B.

## Stage-4 receipt preservation

Existing qualified Stage-4 integration branches must not receive CI-only commits after their finalized receipt commit. The Stage-4 receipt compiler requires the receipt commit to remain the final branch tip. Therefore this control plane is introduced through the default/future CI authority rather than rewriting already-qualified INTA-INTH history.

Once the host-side hook is enabled, any attempted rerun of a legacy naked Shanxi workflow will fail before business steps. This is intentional fail-closed behavior; new/future execution must use the gated workflow shape.
