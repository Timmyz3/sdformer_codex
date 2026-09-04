#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINTER="/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py"
cd "$ROOT"
for source in rtl_hitflow/*.sv; do
  python "$LINTER" "$source"
done
