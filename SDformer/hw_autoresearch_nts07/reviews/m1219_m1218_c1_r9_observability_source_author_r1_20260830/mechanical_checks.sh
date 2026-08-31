#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V="$ROOT/verif_m1219r9_c1_common_charge_protocol"
C="$ROOT/contracts/m1219_m1218_m1213_c1_r9_observability_source_contract_r1_20260830.json"

(cd "$ROOT/contracts" && sha256sum -c "$(basename "$C").sha256" >/dev/null && \
    sha256sum -c "$(basename "$C").sha256.seal.sha256" >/dev/null)
python3 -m py_compile "$V/check_m1219r9_source.py" "$V/test_m1219r9_source.py"
(cd "$V" && python3 -m unittest -q test_m1219r9_source.py >/dev/null)
python3 "$V/check_m1219r9_source.py" >/dev/null
test "$(sha256sum "$ROOT/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
printf '%s\n' 'M1219_AUTHOR_CHECKS_PASS source_only=true tests=7 vcs=false eda=false fresh_hammer_required=true'

