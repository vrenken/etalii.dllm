#!/usr/bin/env bash
# Prepares Claude Code cloud sessions: a Python virtual environment with the package installed in editable mode,
# which also compiles the C++ kernels (needs cmake and a C++ compiler, both preinstalled in the cloud image).
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"
PYTHON=$(command -v python3.12 || command -v python3)
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q scikit-build-core nanobind
.venv/bin/pip install -q --no-build-isolation -e ".[dev]"

if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo "export PATH=\"$PWD/.venv/bin:\$PATH\"" >> "$CLAUDE_ENV_FILE"
fi
