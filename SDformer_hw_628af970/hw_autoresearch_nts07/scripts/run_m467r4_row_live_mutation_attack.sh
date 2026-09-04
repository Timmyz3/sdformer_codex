#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m467r4_row_live_premature_mutation_attack_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M467R4 mutation attack exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m467r4_row_shared_live_invariant_vcs_contract_r1_20260826.json ea6cc8169a84692190b19a9f32bd69c7897dbe96e5620734ac2221732957b3b6
check_sha scripts/m467r4_row_live_premature_mutation_attack.py cc80aab849bb5f0c0e47e5a64e6e29999672431ed15fda91220f890a96af562d
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

test ! -e "$RUN_DIR" || { echo "M467R4 mutation attack result directory exists: $RUN_DIR" >&2; exit 3; }
mkdir -p "$RUN_DIR"
cp contracts/m467r4_row_shared_live_invariant_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
python3 scripts/m467r4_row_live_premature_mutation_attack.py | tee "$RUN_DIR/m467r4_mutation_attack.json"
grep -q '"status": "COUNTEREXAMPLE_REPRODUCED"' "$RUN_DIR/m467r4_mutation_attack.json"
echo PASS_M467R4_PREMATURE_ROW_LIVE_SET_AND_CLEAR_COUNTEREXAMPLES > "$RUN_DIR/RUN_COMPLETE.txt"
(cd "$RUN_DIR" && sha256sum contract.json m467r4_mutation_attack.json RUN_COMPLETE.txt > SHA256SUMS)
(cd "$RUN_DIR" && sha256sum SHA256SUMS > SHA256SUMS.seal.sha256)
