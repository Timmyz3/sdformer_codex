#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
Q="$ROOT/results/m1213_m1210r8_m1162_c1_common_charge_protocol_unit_delay_vcs_r8_20260830.failed_or_incomplete.284863.quarantine"
A="$ROOT/results/.m1213_m1210r8_m1162_c1_common_charge_protocol_vcs_r8_attempt_consumed/identity.txt"
TB="$ROOT/verif_m1210r8_c1_common_charge_protocol/tb_m1210r8_m1162_common_charge_protocol_unit_delay_r8.sv"
R7="$ROOT/verif_m1193r6_c1_common_charge_protocol/tb_m1193r6_m1162_common_charge_protocol_unit_delay_r6.sv"
RUNNER="$ROOT/dc_handoff/scripts/run_vcs_m1213_m1210r8_m1162_c1_common_charge_protocol_exact_sha_r8.sh"

test -f "$A"
test -f "$Q/RUN_FAILED_OR_INCOMPLETE.txt"
(cd "$Q" && sha256sum -c SHA256SUMS >/dev/null && sha256sum -c SHA256SUMS.seal.sha256 >/dev/null)
grep -qx 'exit_code=124' "$Q/RUN_FAILED_OR_INCOMPLETE.txt"
grep -qx 'functional_vcs_verified=false' "$Q/RUN_FAILED_OR_INCOMPLETE.txt"
grep -qx 'automatic_retry=false' "$Q/RUN_FAILED_OR_INCOMPLETE.txt"
grep -q 'CPU time: .* to compile .* to elab .* to link' "$Q/compile.log"
test -x "$Q/simv"
grep -q 'SVART-AMAXINT' "$Q/sim.log"
grep -q 'Exceeding INT_MAX assertion attempts' "$Q/sim.log"
! grep -q '^PASS_M1210R8_' "$Q/sim.log"
grep -q 'timeout --signal=TERM --kill-after=30s 1800s ./simv -no_save' "$RUNNER"
grep -q 'while (!prep_ready) @(negedge clk_core);' "$TB"
test "$(grep -Ec 'wait \(' "$TB")" -eq 3
test "$(grep -Ec 'while \(!prep_ready\)' "$TB")" -eq 1
test "$(sha256sum "$TB" | awk '{print $1}')" = 060ec9d5ae6085a0dd013160d22f63e21615730384ddaef342eb3fa77e17947b
test "$(sha256sum "$R7" | awk '{print $1}')" = 0fcc2138ef5d716735eea01dee25a148a5223b1d6adf1e3b2fa464341fbf1345
test "$(sha256sum "$ROOT/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4
printf '%s\n' 'M1218_MECHANICAL_CHECKS_PASS failure_only=true compile_pass=true sim_timeout=true recursive_seal=true dut_fault_unproven=true vcs_rerun=false'
