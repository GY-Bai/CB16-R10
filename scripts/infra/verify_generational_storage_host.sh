#!/usr/bin/env bash
set -Eeuo pipefail
umask 027

EXPECTED_MIGRATION_TS="20260908T193313Z"
EXPECTED_INVENTORY_SHA="4eae7ee407ab05dc8fce809f8bee6bcc3660e23e6c8f9c2167ebbb777a67f07a"
EXPECTED_RAW_TARGET="/data/cb16_hdd/cb16_raw_view"
STORE="/data/cb16_hdd/cb16_store/annex-repo"
STAGING="/data/cb16_hdd/cb16_store/staging"
LEGACY_INDEX="/data/cb16_hdd/cb16_store/legacy-index"
RUNTIME="/var/tmp/cb16_runtime/R11"
REPORT_ROOT="/var/tmp/cb16_meta/migration/${EXPECTED_MIGRATION_TS}"

fail() {
  echo "FAIL_CLOSED: $*" >&2
  exit 1
}

pass() {
  echo "PASS: $*"
}

require_dir() {
  [ -d "$1" ] || fail "missing directory: $1"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

echo '=== CB16 generational storage host qualification ==='
id
uname -a

for c in git git-annex sha256sum findmnt lsblk readlink mktemp; do
  require_cmd "$c"
done

for d in "$STORE" "$STAGING" "$LEGACY_INDEX" "$RUNTIME" "$REPORT_ROOT"; do
  require_dir "$d"
done
pass "required storage/runtime/report directories exist"

# Storage topology: /var/tmp must resolve to non-rotational backing; /data to rotational backing.
root_src="$(findmnt -no SOURCE -T /var/tmp)"
data_src="$(findmnt -no SOURCE -T /data)"
[ -n "$root_src" ] || fail "cannot resolve /var/tmp backing source"
[ -n "$data_src" ] || fail "cannot resolve /data backing source"
[ "$root_src" != "$data_src" ] || fail "/var/tmp and /data unexpectedly share the same mounted source"
root_real="$(readlink -f "$root_src" 2>/dev/null || printf '%s' "$root_src")"
data_real="$(readlink -f "$data_src" 2>/dev/null || printf '%s' "$data_src")"

if ! lsblk -s -n -o ROTA "$root_real" 2>/dev/null | awk '$1==1{bad=1} END{exit bad?0:1}'; then
  pass "/var/tmp backing chain contains no rotational device"
else
  fail "/var/tmp backing chain contains rotational storage: $root_src"
fi

if lsblk -s -n -o ROTA "$data_real" 2>/dev/null | awk '$1==1{ok=1} END{exit ok?0:1}'; then
  pass "/data backing chain contains rotational device"
else
  fail "/data backing chain does not contain rotational storage: $data_src"
fi

# Namespace and permissions.
[ -L /cb16/raw ] || fail "/cb16/raw is not a symlink"
raw_target="$(readlink -f /cb16/raw)"
[ "$raw_target" = "$EXPECTED_RAW_TARGET" ] || fail "/cb16/raw target mismatch: $raw_target"
[ ! -w /cb16/raw ] || fail "/cb16/raw is writable by qualification runner"
pass "/cb16/raw is the expected read-only view"

[ -w "$RUNTIME" ] || fail "R11 runtime is not writable by qualification runner"
[ -w "$STAGING" ] || fail "staging is not writable by qualification runner"
[ -w "$STORE" ] || fail "artifact store is not writable by qualification runner"
pass "current runtime/staging/store write permissions are available"

legacy_seen=0
for p in \
  /data/cb16_ci/r2-authority/r103 \
  /data/cb16_ci/r2-authority/package_r102 \
  /data/cb16_hdd/cb16_diagnostics/r2_native
do
  if [ -e "$p" ]; then
    legacy_seen=$((legacy_seen + 1))
    [ ! -w "$p" ] || fail "legacy authority path is writable: $p"
  fi
done
[ "$legacy_seen" -ge 1 ] || fail "no known legacy authority roots visible to qualification runner"
pass "visible legacy authority roots are read-only"

# Installed production tools must exist, but this gate deliberately does not invoke production ingest.
for t in cb16_artifact_put cb16_artifact_link cb16_artifact_verify cb16_legacy_register cb16_runtime_lock_verify; do
  require_cmd "$t"
done
pass "production artifact management entry points are installed"

# Production git-annex policy and integrity.
backend="$(git -C "$STORE" config --get annex.backend || true)"
secure="$(git -C "$STORE" config --get annex.securehashesonly || true)"
[ "$backend" = "SHA256" ] || fail "production annex backend is not SHA256: ${backend:-<unset>}"
case "$secure" in
  true|1|yes|on) ;;
  *) fail "annex.securehashesonly is not enabled: ${secure:-<unset>}" ;;
esac
grep -Eq 'annex\.backend=SHA256' "$STORE/.gitattributes" || fail ".gitattributes does not force SHA256 backend"
git -C "$STORE" annex version
git -C "$STORE" annex fsck
pass "production annex policy and deep fsck"

# The sealed legacy inventory must still exist somewhere in the dedicated legacy-index namespace.
inventory_match=""
while IFS= read -r -d '' f; do
  got="$(sha256sum "$f" | awk '{print $1}')"
  if [ "$got" = "$EXPECTED_INVENTORY_SHA" ]; then
    inventory_match="$f"
    break
  fi
done < <(find "$LEGACY_INDEX" -maxdepth 2 -type f -print0)
[ -n "$inventory_match" ] || fail "sealed legacy inventory SHA not found in $LEGACY_INDEX"
pass "sealed legacy inventory found: $inventory_match"

# Rollback receipt must remain available.
[ -f "$REPORT_ROOT/rollback.sh" ] || fail "rollback.sh missing from migration report"
pass "rollback entry point retained"

# Isolated HDD canary: validate the underlying git-annex behavior without polluting production CAS.
CANARY="$(mktemp -d "$STAGING/.gha-storage-qual.XXXXXX")"
cleanup() {
  chmod -R u+w "$CANARY" 2>/dev/null || true
  rm -rf "$CANARY"
}
trap cleanup EXIT
cd "$CANARY"
git init -q
git config user.name "CB16 Storage Qualification"
git config user.email "cb16-storage-qualification@localhost"
git config annex.backend SHA256
git config annex.securehashesonly true
git config annex.addunlocked false
printf '* annex.backend=SHA256\n' > .gitattributes
git add .gitattributes
git commit -qm 'init qualification repo'
git annex init -q "cb16-storage-qualification-${GITHUB_RUN_ID:-manual}"

printf 'cb16-generational-storage-canary-v1\n' > object-a.bin
cp object-a.bin object-b.bin
git annex add --force-large --backend=SHA256 object-a.bin object-b.bin >/dev/null
key_a="$(git annex lookupkey object-a.bin)"
key_b="$(git annex lookupkey object-b.bin)"
[ -n "$key_a" ] || fail "canary object-a has no annex key"
[ "$key_a" = "$key_b" ] || fail "identical content produced different annex keys"
case "$key_a" in
  SHA256-*) ;;
  *) fail "canary key is not SHA256: $key_a" ;;
esac
pass "identical bytes deduplicate to one SHA256 key"

# A new generation/reference must reuse the existing key without a payload copy.
git annex fromkey "$key_a" generation-b.bin >/dev/null
key_gen_b="$(git annex lookupkey generation-b.bin)"
[ "$key_gen_b" = "$key_a" ] || fail "fromkey reference does not preserve object identity"
pass "cross-generation reference reuses exact immutable key"

# Locked annex content must not present as a normal writable file.
if [ -L object-a.bin ]; then
  pass "annex content is represented by locked symlink"
elif [ ! -w object-a.bin ]; then
  pass "annex content is non-writable"
else
  fail "annexed content appears directly writable"
fi

git annex fsck >/dev/null
pass "isolated annex canary fsck"

# A partial/uncommitted staging file must not become an annex object merely by existing.
printf 'incomplete\n' > partial.part
if git annex lookupkey partial.part >/dev/null 2>&1; then
  fail "unannexed partial staging file unexpectedly has an annex key"
fi
pass "partial staging file has no object identity/authority"

echo '=== SAFETY RECEIPT ==='
echo 'fresh_market_data_accessed=false'
echo 'final_holdout_payload_opened=false'
echo 'scientific_training_started=false'
echo 'scientific_verdict_changed=false'
echo 'production_ingest_invoked=false'
echo 'legacy_objects_modified=false'
echo 'CB16_GENERATIONAL_STORAGE_HOST_QUALIFICATION=PASS'
