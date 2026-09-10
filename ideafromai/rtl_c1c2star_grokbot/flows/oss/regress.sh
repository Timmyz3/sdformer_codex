#!/usr/bin/env bash
# GROKBOT NEW FILE -- iscas_ssh
# Thin wrapper: run full regression; exit non-zero on any sim fail.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FLOW="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
bash "$FLOW/run_all.sh"
if [[ -f out/REGRESSION_REPORT.md ]]; then
  if grep -E '\*\*FAIL\*\*' out/REGRESSION_REPORT.md >/dev/null; then
    echo "regress: FAIL entries in REGRESSION_REPORT.md" >&2
    exit 1
  fi
fi
echo "regress: OK"
