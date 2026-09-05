#!/usr/bin/env bash
# Bootstrap a Python venv for CB16 static services (relay/worker).
# Uses uv when available; falls back to python3 -m venv + pip.
# Usage: bootstrap_venv.sh <venv_dir> [requirements_file ...]
set -euo pipefail
VENV_DIR="$1"; shift || true
PYTHON_BIN="${VENV_DIR}/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Creating venv with uv: $VENV_DIR"
    uv venv "$VENV_DIR"
  else
    echo "Creating venv with python3 -m venv: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
fi

if [ "$#" -gt 0 ]; then
  if command -v uv >/dev/null 2>&1; then
    echo "Installing requirements with uv"
    uv pip install --python "$PYTHON_BIN" --index-strategy unsafe-best-match "$@"
  else
    echo "Installing requirements with pip"
    "$PYTHON_BIN" -m pip install "$@"
  fi
fi
