#!/usr/bin/env bash
set -euo pipefail
dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hw_root="$(cd "$dc_root/.." && pwd)"
runner="$(realpath "${BASH_SOURCE[0]}")"
run="$hw_root/results/m236_dynamic_bn_lut16_newton2_full220800_vcs_r1_exact_20260825"
vectors_dir="$hw_root/results/m236_h67_lut16_newton2_full_vectors_r1_20260825"
vectors="$vectors_dir/m236_h67_lut16_newton2_full220800_vectors.csv"
vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$run" ]] || exit 2
mkdir -p "$(dirname "$run")"
mkdir "$run"
complete=0
trap 'rc=$?; if [[ $complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$rc" >"$run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$hw_root"
declare -A expected=(
 ["rtl_m236/m236_dynamic_bn_lut16_newton2_coefficient_engine.sv"]="a342b6dd825851cfb8d16282019b063d31b66fc56148783cc4b4c7c9b30c5cfe"
 ["verif_m236/m236_dynamic_bn_lut16_newton2_coefficient_engine_assertions.sv"]="78963911f163e663642e590d960f4c8678f4c4ad44692ba527b67ebe772dd9db"
 ["tb_m236/tb_m236_dynamic_bn_lut16_newton2_coefficient_engine.sv"]="f09b0a1e1a7eb6957b6fee70fa9f430710e22f596dc9f9df1512d6c47a2568e0"
 ["dc_handoff/filelists/date_m236_dynamic_bn_lut16_newton2_coefficient_engine_directed_vcs.f"]="a5d0411b1b08262d462aba28e1c92e2b0ffc7c3c2d38553c68f6268159d92851"
 ["contracts/m236_dynamic_bn_lut16_newton2_coefficient_engine_contract_r1_20260825.json"]="06298f2be91ecefcd08e90da2e2b06f95cba76114dec2ae26fa446e01d63df6e"
 ["system_simulator/scripts/generate_m236_h67_lut16_newton2_full_vectors.py"]="f5a40ead4ef9a13a030e96db4ca8cb9bb8e04f93920633478d3f95a8eebe5374"
 ["results/m236_h67_lut16_newton2_full_vectors_r1_20260825/manifest.sha256"]="30432441ff07bd30e40766f7792055d17ca697bfd431caa64077386279a04004"
 ["results/m234_independent_hammer_review_r1_20260825/SHA256SUMS"]="18a57e92dde575c680646ae020fb0d4ae5f8a0d6a4bac7cadf07eaf13dd32404"
 ["results/m235r2_synthesis_safe_directed_vcs_r1_exact_20260825/SHA256SUMS"]="b813ac5f8fcb5b3273f580db9a70b230df72d18a9c646964dd0b8bee7927fff5"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"$run/preflight_sha_checks.txt"
for path in "${!expected[@]}"; do
 observed="$(sha256sum "$path" | awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$path" "${expected[$path]}" "$observed" >>"$run/preflight_sha_checks.txt"
 [[ "$observed" == "${expected[$path]}" ]] || exit 10
done
(cd "$vectors_dir" && sha256sum -c manifest.sha256) >"$run/vector_manifest_check.txt"
[[ "$(wc -l <"$vectors")" -eq 220801 ]] || exit 11
for source_index in 175162 175604 176110 182167 190728 219956; do
 awk -F, -v target="$source_index" '$2==target {found=1;exit} END{exit !found}' "$vectors" || exit 12
done
sha256sum "${!expected[@]}" >"$run/input_sha256.txt"
export VCS_HOME="$vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
compile_start="$(date +%s)"
set +e
"$vcs/bin/vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
 -cm assert -Mdir="$run/csrc" \
 -f dc_handoff/filelists/date_m236_dynamic_bn_lut16_newton2_coefficient_engine_directed_vcs.f \
 -top tb_m236_dynamic_bn_lut16_newton2_coefficient_engine \
 -o "$run/simv" >"$run/compile.log" 2>&1
rc=$?
set -e
compile_end="$(date +%s)"
echo "$rc" >"$run/compile.rc"
[[ $rc -eq 0 && -x "$run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$run/compile.log" && exit 21 || true
sim_start="$(date +%s)"
set +e
"$run/simv" +ntb_random_seed=23620260825 -no_save -cm assert \
 -assert report="$run/assert.report" >"$run/sim.log" 2>&1
rc=$?
set -e
sim_end="$(date +%s)"
echo "$rc" >"$run/sim.rc"
[[ $rc -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
 "$run/sim.log" "$run/assert.report" && exit 23 || true
grep -Eq '^PASS M236 checkpoint vectors=220800 mismatches=0 max_latency=[0-9]+ max_unstalled_accept_ii=[0-9]+ result_stalls=[1-9][0-9]* protocol_attacks=1 shared_multiplier_slots=1 multiply_ops_per_pair=8 lut_entries=16 newton_steps=2 tail_extrema_included=6 moment_finalizer=false event_equivalence=false system_speedup=false headline=false$' "$run/sim.log" || exit 30
latency="$(sed -n 's/^PASS M236 .*max_latency=\([0-9][0-9]*\).*/\1/p' "$run/sim.log")"
interval="$(sed -n 's/^PASS M236 .*max_unstalled_accept_ii=\([0-9][0-9]*\).*/\1/p' "$run/sim.log")"
[[ -n "$latency" && -n "$interval" && "$latency" -le 16 && "$interval" -le 16 ]] || exit 31
for cover in cp_first_newton cp_second_newton cp_result cp_result_stall cp_fault_with_pending_result; do
 grep -Eq "$cover, .* [1-9][0-9]* match" "$run/assert.report" || exit 32
done
source_multiply_operators="$(rg -o 'multiplier_a \* multiplier_b' rtl_m236/m236_dynamic_bn_lut16_newton2_coefficient_engine.sv | wc -l)"
[[ "$source_multiply_operators" -eq 1 ]] || exit 33
{
 echo status=PASS_M236_FULL220800_EXACT_VCS
 echo exact_sha=true
 echo tool=Synopsys_VCS_V-2023.12-SP1
 echo checkpoint_vectors=220800
 echo integer_output_mismatches=0
 echo source_index_min=0
 echo source_index_max=220799
 echo previously_missing_tail_extrema_included=6
 echo maximum_first_result_latency_cycles="$latency"
 echo maximum_unstalled_accept_interval_cycles="$interval"
 echo source_shared_multiply_operators="$source_multiply_operators"
 echo multiply_operations_per_pair=8
 echo lut_entries=16
 echo newton_steps=2
 echo compile_wall_seconds="$((compile_end-compile_start))"
 echo simulation_wall_seconds="$((sim_end-sim_start))"
 echo moment_finalizer=false
 echo event_equivalence=false
 echo system_speedup=false
 echo headline=false
} >"$run/m236_vcs_receipt_r1.txt"
sha256sum "$runner" >"$run/runner_sha256.txt"
find "$run" -maxdepth 1 -type f ! -name simv ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum >"$run/SHA256SUMS"
echo PASS_M236_FULL220800_EXACT_VCS >"$run/RUN_COMPLETE.txt"
complete=1
echo "PASS M236 full220800 exact VCS sealed at $run"
