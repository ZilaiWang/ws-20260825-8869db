#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "ERROR: environment.yml must be exported on Linux x86_64" >&2
  exit 2
fi

ENV_NAME="${1:-detector}"
OUTPUT="${2:-submission/docker/environment.yml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
conda env export -n "$ENV_NAME" --no-builds | sed '/^prefix:/d' > "$OUTPUT"

if grep -Eq 'win-64|pywin32|^[[:space:]]*prefix:' "$OUTPUT"; then
  echo "ERROR: exported environment contains forbidden platform/path fields" >&2
  exit 2
fi

echo "Exported Linux x86_64 environment: $OUTPUT"
sha256sum "$OUTPUT"
