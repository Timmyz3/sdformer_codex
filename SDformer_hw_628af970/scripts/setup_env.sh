#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -f environment.yml
elif command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml || conda env update -f environment.yml
else
  echo "conda or micromamba is required to create the environment." >&2
  exit 1
fi

for tool in python iverilog yosys; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "missing tool: ${tool}" >&2
  fi
done

echo "environment setup finished"

