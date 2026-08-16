#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFIER="$ROOT/scripts/verify_local5_phase_summary_contract_v2.py"
TEST="$ROOT/scripts/test_verify_local5_phase_summary_contract_v2.py"
CACHE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/local5-phase-preflight-v2.XXXXXX")"
trap 'rm -rf "$CACHE_ROOT"' EXIT

cd "$ROOT"

PYTHONPYCACHEPREFIX="$CACHE_ROOT" python3 -m py_compile \
  "$VERIFIER" \
  "$TEST"

PYTHONPYCACHEPREFIX="$CACHE_ROOT" python3 -m unittest -v "$TEST"

PYTHONPYCACHEPREFIX="$CACHE_ROOT" python3 "$VERIFIER" --root "$ROOT"

printf 'PASS Local5 phase summary v2 static preflight; no RTL compile, H24 run, or GPU work performed\n'
