#!/usr/bin/env bash
# Installs the .NET 10 SDK in Claude Code cloud sessions, where it is not preinstalled.
# builds.dotnet.microsoft.com is blocked by the default network policy there, so use the Ubuntu package.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if ! command -v dotnet >/dev/null 2>&1 || ! dotnet --list-sdks | grep -q '^10\.'; then
  apt-get update -qq >/dev/null 2>&1 || true
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq dotnet-sdk-10.0 >/dev/null
fi

cd "${CLAUDE_PROJECT_DIR:-.}"
dotnet restore --verbosity quiet
