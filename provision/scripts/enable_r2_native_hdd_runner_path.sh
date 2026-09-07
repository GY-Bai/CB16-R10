#!/usr/bin/env bash
set -euo pipefail

# Host-side, root-only repair for the Shanxi canonical GitHub runner mount namespace.
# This does NOT remount /data globally and does NOT modify canonical R10.4 data.
# It only grants the runner service RW access to one disposable R2 qualification root.

UNIT="cb16-github-canonical-runner.service"
R2_RW_ROOT="/data/cb16_hdd/cb16_diagnostics/r2_native"
CANONICAL_R104="/data/cb16_hdd/cb16_runtime/R10_4"
DROPIN_DIR="/etc/systemd/system/${UNIT}.d"
DROPIN="${DROPIN_DIR}/r2-native-hdd-rw.conf"
RUNNER_USER="cb16-ci"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ "$(id -u)" != "0" ]]; then
  fail "RUN_AS_ROOT_REQUIRED"
fi

getent passwd "$RUNNER_USER" >/dev/null || fail "RUNNER_USER_NOT_FOUND:${RUNNER_USER}"
systemctl cat "$UNIT" >/dev/null || fail "RUNNER_UNIT_NOT_FOUND:${UNIT}"

# Restarting the runner while canonical R10.4 is active could kill its process tree.
# Refuse rather than trying to infer whether it is safe.
active="$(ps -eo pid=,comm=,args= | awk '$2 ~ /^python/ && $0 ~ /scripts\/run_r104_long_research\.py/ {print}' || true)"
if [[ -n "$active" ]]; then
  echo "$active" >&2
  exit 75
fi

# Create only the dedicated noncanonical qualification path in the host namespace.
install -d -o "$RUNNER_USER" -g "$RUNNER_USER" -m 2775 "$R2_RW_ROOT"

install -d -m 0755 "$DROPIN_DIR"
cat > "$DROPIN" <<EOF
[Service]
# Keep /data read-only by default; expose only this disposable R2 qualification subtree.
ReadWritePaths=${R2_RW_ROOT}
EOF
chmod 0644 "$DROPIN"

systemctl daemon-reload
systemctl restart "$UNIT"

main_pid="$(systemctl show -p MainPID --value "$UNIT")"
[[ "$main_pid" =~ ^[1-9][0-9]*$ ]] || fail "RUNNER_MAINPID_INVALID:${main_pid}"

# Verify from INSIDE the runner service mount namespace, not from the host shell.
findmnt_ns() {
  nsenter -t "$main_pid" -m -- findmnt -n -o TARGET,SOURCE,FSTYPE,OPTIONS -T "$1"
}

opts_for() {
  nsenter -t "$main_pid" -m -- findmnt -n -o OPTIONS -T "$1"
}

has_opt() {
  local opts="$1" want="$2"
  [[ ",${opts}," == *",${want},"* ]]
}

echo '=== runner namespace mounts ==='
findmnt_ns /data
findmnt_ns /data/cb16_ci
findmnt_ns "$CANONICAL_R104"
findmnt_ns "$R2_RW_ROOT"

root_data_opts="$(opts_for /data)"
r2_opts="$(opts_for "$R2_RW_ROOT")"
canonical_opts="$(opts_for "$CANONICAL_R104")"

has_opt "$root_data_opts" ro || fail "RUNNER_DATA_ROOT_NOT_READ_ONLY:${root_data_opts}"
has_opt "$r2_opts" rw || fail "R2_RW_BIND_NOT_ACTIVE:${r2_opts}"
has_opt "$canonical_opts" rw || fail "CANONICAL_R104_BIND_UNEXPECTEDLY_NOT_RW:${canonical_opts}"

# One disposable canary only in the new R2 root. Never write to canonical R10.4.
nsenter -t "$main_pid" -m -- runuser -u "$RUNNER_USER" -- \
  bash -c 'set -euo pipefail; root="$1"; f="$root/.cb16_runner_rw_canary_$$"; : > "$f"; sync "$f" 2>/dev/null || true; rm -f "$f"' _ "$R2_RW_ROOT"

echo 'R2_RUNNER_STORAGE_BINDING=PASS'
echo "UNIT=${UNIT}"
echo "R2_RW_ROOT=${R2_RW_ROOT}"
echo 'DATA_ROOT_REMAINS_READ_ONLY=TRUE'
echo 'CANONICAL_R104_MODIFIED=FALSE'
echo 'SCIENTIFIC_SEMANTICS_CHANGED=FALSE'

echo
echo 'Rollback (only when no runner job is active):'
echo "  sudo rm -f '$DROPIN'"
echo "  sudo systemctl daemon-reload && sudo systemctl restart '$UNIT'"
