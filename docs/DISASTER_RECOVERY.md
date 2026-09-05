# CB16 Remote CI Relay – Disaster Recovery

1. Relay DB corruption:
   - Stop `cb16-ci-relay.service`.
   - Backup `/home/bgy/cb16-ci/state/relay.db*`.
   - Restore from backup or reset DB (jobs can be re-delivered from GitHub webhook).
2. Bundle loss:
   - Re-create from bare mirror with `git archive <sha>` and re-sha256.
3. Cloudflare tunnel outage:
   - Verify `cloudflared` process; restart user service.
   - Tunnel credentials never leave OCI.
4. Worker crash:
   - `systemctl --user restart cb16-ci-worker.service`.
   - Workspaces are isolated per job; stale dirs can be removed.
5. GitHub result push failure:
   - Check OCI `GITHUB_RESULT_TOKEN` scope; retry by running relay publish.
