#!/usr/bin/env bash
set -Eeuo pipefail

DIST_ROOT="/opt/actions-runner-dist"
STATE_ROOT="${RUNNER_STATE_ROOT:-/cb16/worker/actions-runner}"
WORK_ROOT="${RUNNER_WORK_DIR:-/cb16/worker/_work}"
HOME_ROOT="${RUNNER_HOME:-/cb16/worker/home}"
TOKEN_FILE="${RUNNER_TOKEN_FILE:-/run/secrets/github-runner-token}"
RUNNER_URL="${RUNNER_URL:-https://github.com/GY-Bai/CB16-R10}"
RUNNER_NAME="${RUNNER_NAME:-shanxi-docker-r11}"
RUNNER_LABELS="${RUNNER_LABELS:-shanxi,shanxi-docker-r11}"

mkdir -p "$STATE_ROOT" "$WORK_ROOT" "$HOME_ROOT" "${UV_CACHE_DIR:-/cb16/uv-cache}"
export HOME="$HOME_ROOT"

if [[ ! -x "$STATE_ROOT/run.sh" ]]; then
  cp -a "$DIST_ROOT/." "$STATE_ROOT/"
fi

cd "$STATE_ROOT"

if [[ ! -f .runner ]]; then
  token="${RUNNER_TOKEN:-}"
  if [[ -z "$token" && -r "$TOKEN_FILE" ]]; then
    token="$(<"$TOKEN_FILE")"
  fi
  if [[ -z "$token" ]]; then
    echo "CB16_DOCKER_RUNNER=FAIL RUNNER_REGISTRATION_TOKEN_MISSING" >&2
    exit 78
  fi

  config_args=(
    --unattended
    --replace
    --url "$RUNNER_URL"
    --token "$token"
    --name "$RUNNER_NAME"
    --labels "$RUNNER_LABELS"
    --work "$WORK_ROOT"
  )
  if [[ "${RUNNER_DISABLE_UPDATE:-true}" == "true" ]]; then
    config_args+=(--disableupdate)
  fi
  ./config.sh "${config_args[@]}"
  unset token RUNNER_TOKEN
fi

if [[ ! -x /opt/cb16/ci/shanxi_runner_pre_job_gate.py ]]; then
  echo "CB16_DOCKER_RUNNER=FAIL PRE_JOB_GATE_MISSING" >&2
  exit 78
fi

if [[ ! -d "${CB16_RAW_ROOT:-/cb16/raw}" ]]; then
  echo "CB16_DOCKER_RUNNER=FAIL RAW_VOLUME_MISSING" >&2
  exit 78
fi
if [[ ! -d "${CB16_G0_ROOT:-/cb16/g0}" ]]; then
  echo "CB16_DOCKER_RUNNER=FAIL G0_VOLUME_MISSING" >&2
  exit 78
fi
if [[ ! -x "${CB16_VERIFIED_VENV:-/cb16/venv}/bin/python" ]]; then
  echo "CB16_DOCKER_RUNNER=FAIL VERIFIED_VENV_MISSING" >&2
  exit 78
fi

echo "CB16_DOCKER_RUNNER=READY name=$RUNNER_NAME labels=$RUNNER_LABELS"
exec ./run.sh
