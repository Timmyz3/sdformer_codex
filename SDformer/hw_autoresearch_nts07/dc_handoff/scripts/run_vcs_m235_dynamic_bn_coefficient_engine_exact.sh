#!/usr/bin/env bash
set -euo pipefail
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "$dc_root/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
run="$hw_root/results/m235_dynamic_bn_coefficient_engine_directed_vcs_r1_exact_20260825"
vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$run" ]] || exit 2
mkdir -p "$(dirname "$run")"
mkdir "$run"
complete=0
trap 'rc=$?; if [[ $complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$rc" >"$run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$hw_root"
declare -A expected=(
 ["rtl_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv"]="933b0cab1e63a06ae4ef3a13334806b3167b917c603e63b088c2d03ea9ac6fb0"
 ["verif_m235/m235_dynamic_bn_segmented_lut_newton_coefficient_engine_assertions.sv"]="2c138fc3902a14d0d53a709e5533e11b686bd724bc986e059e4e841137ac8a45"
 ["tb_m235/tb_m235_dynamic_bn_segmented_lut_newton_coefficient_engine.sv"]="557cda41e4ce5a55ecb2a67475479e4743890f30346cbf26c6ff9c6595c1a0ab"
 ["dc_handoff/filelists/date_m235_dynamic_bn_coefficient_engine_directed_vcs.f"]="1edbc3f71aeb2fa9b92f0c58b5f1649f10c7281bcc73641ab381ec4463aca9db"
 ["contracts/m235_dynamic_bn_segmented_lut_newton_coefficient_engine_contract_r1_20260825.json"]="fb661071ab72650f744c0501e712b31c6a0b29cb88c7bada817238e9ad103d35"
 ["contracts/m234_h67_dynamic_bn_lut_newton_coefficient_dse_contract_r1_20260825.json"]="31e6de148b249799edfe3d03bd4ec3004aebcf9d5758a40229af4fb558195d4c"
 ["results/m234_h67_dynamic_bn_lut_newton_coefficient_dse_r1_20260825/manifest.sha256"]="c7fc0ba9495a821e634644023cbbbd4477385f6b4126a61b668f0458b4bcec71"
 ["results/m233_independent_hammer_review_r1_20260825/SHA256SUMS"]="375dc78b7f979fbe1120dde5f2ed5245e8d0fb018fe21a6f7325c17b0d6d4473"
 ["results/m232_r1_storage_and_first_latency_correction_overlay_r1_20260825/manifest.sha256"]="b4159d3beeb4ec91acc95e1b34bede0da3ff2eb7fd936c82015af2c4ab185cc3"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"$run/preflight_sha_checks.txt"
for path in "${!expected[@]}"; do
 observed="$(sha256sum "$path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$path" "${expected[$path]}" "$observed" >>"$run/preflight_sha_checks.txt"
 [[ "$observed" == "${expected[$path]}" ]] || exit 10
done
sha256sum "${!expected[@]}" >"$run/input_sha256.txt"
export VCS_HOME="$vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$vcs/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
 -cm assert -Mdir="$run/csrc" \
 -f dc_handoff/filelists/date_m235_dynamic_bn_coefficient_engine_directed_vcs.f \
 -top tb_m235_dynamic_bn_segmented_lut_newton_coefficient_engine \
 -o "$run/simv" >"$run/compile.log" 2>&1
rc=$?
set -e
echo "$rc" >"$run/compile.rc"
[[ $rc -eq 0 && -x "$run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$run/compile.log" && exit 21 || true
set +e
"$run/simv" +ntb_random_seed=23520260825 -no_save -cm assert \
 -assert report="$run/assert.report" >"$run/sim.log" 2>&1
rc=$?
set -e
echo "$rc" >"$run/sim.rc"
[[ $rc -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
 "$run/sim.log" "$run/assert.report" && exit 23 || true
grep -Eq '^PASS M235 checkpoint vectors=1024 mismatches=0 max_latency=[0-9]+ max_accept_ii=[0-9]+ result_stalls=[1-9][0-9]* protocol_attacks=1 shared_multiplier_slots=1 lut_entries=64 moment_finalizer=false event_equivalence=false system_speedup=false headline=false$' "$run/sim.log" || exit 30
latency="$(sed -n 's/^PASS M235 .*max_latency=\([0-9][0-9]*\).*/\1/p' "$run/sim.log")"
interval="$(sed -n 's/^PASS M235 .*max_accept_ii=\([0-9][0-9]*\).*/\1/p' "$run/sim.log")"
[[ -n "$latency" && -n "$interval" && "$latency" -le 16 && "$interval" -le 16 ]] || exit 31
for cover in cp_result cp_result_stall cp_fault_with_pending_result; do
 grep -Eq "$cover, .* [1-9][0-9]* match" "$run/assert.report" || exit 32
done
{
 echo status=PASS_M235_DYNAMIC_BN_COEFFICIENT_ENGINE_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo checkpoint_vectors=1024
 echo integer_output_mismatches=0
 echo maximum_first_result_latency_cycles="$latency"
 echo maximum_observed_accept_interval_cycles="$interval"
 echo result_backpressure=true
 echo protocol_attacks=1
 echo shared_multiplier_slots=1
 echo lut_entries=64
 echo moment_finalizer=false
 echo event_equivalence=false
 echo system_speedup=false
 echo headline=false
} >"$run/m235_vcs_receipt_r1.txt"
sha256sum "$runner" >"$run/runner_sha256.txt"
find "$run" -type f ! -name simv ! -path '*/csrc/*' ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$run/SHA256SUMS"
echo PASS_M235_DYNAMIC_BN_COEFFICIENT_ENGINE_EXACT_VCS >"$run/RUN_COMPLETE.txt"
complete=1
echo "PASS M235 exact VCS sealed at $run"
