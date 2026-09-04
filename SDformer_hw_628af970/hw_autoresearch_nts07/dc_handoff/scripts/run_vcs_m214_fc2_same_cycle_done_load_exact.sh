#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_hw_root/results/m214_fc2_same_cycle_done_load_vcs_calibration_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M214 sealed VCS run" >&2; exit 2; }
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
 ["rtl_m214/m214_fc2_descriptor4_same_done_load_frontend.sv"]="e9384a4825d6d0fde11679e74ec5e3973d17da325e6c8df40d7491ce203c0317"
 ["rtl_m214/m214_fc2_raw4_to_same_done_load_frontend.sv"]="d5caa7f3431761bacde2190412215ef84346a64b3b0559e7cff3116c63f97862"
 ["verif_m214/m214_fc2_raw4_to_same_done_load_frontend_assertions.sv"]="fbe9792478c91efbe433e25ff86d521cfa8c622b0bb4beffb9ee1009edb28b08"
 ["tb_m214/tb_m214_fc2_raw4_to_same_done_load_frontend.sv"]="d8d64c0aef21213c3e7a139abba58f6897bb4c0738eec999bc5ba9204ba22b4e"
 ["tb_m214/tb_m214_fc2_same_done_load_tail_sweep.sv"]="8ce6dd421123b163689f9c9854d351d7a7767e6da98e0bff96d64d2828457c3e"
 ["tb_m214/tb_m214_fc2_same_done_load_stall.sv"]="ad7a5404a8af0e2e66c50993f1182c92e9c60233d69d1ce821c614eea4a1196f"
 ["tb_m214/tb_m214_fc2_bank48_adversarial.sv"]="bd63a3be610bf80ce44c5a657cc403e7c9b2f2a23abb222db44a63be1a42a98f"
 ["tb_m214/tb_m214_fc2_stage0_handoff_prefetch.sv"]="28e5aac865597384bfc660db6bb57a00ee5221a0ed439fe5e4e2979ed3d4af6e"
 ["dc_handoff/filelists/date_m214_fc2_raw4_to_same_done_load_directed_vcs.f"]="642d23156c35de14d5d7d60b87d8d331b03100eb9b4fae929f24443db54b01c2"
 ["dc_handoff/filelists/date_m214_fc2_same_done_load_tail_sweep_vcs.f"]="b2ee40e53caf43cb0833334cbf59aac48ac00c097cb4d4fa0d4297dd393320e2"
 ["dc_handoff/filelists/date_m214_fc2_same_done_load_stall_vcs.f"]="7bd0c27f6420238c7a0c2fd84bfc9e5292c25dc266d4d01e82468dfc184307bb"
 ["dc_handoff/filelists/date_m214_fc2_bank48_adversarial_vcs.f"]="868084b0ccac65369ddd906db3e80197518c7c27b09cf8aeaeb14068efc5546d"
 ["dc_handoff/filelists/date_m214_fc2_stage0_handoff_prefetch_vcs.f"]="20730806a00175ae08cee1032915ab49a825497bb653bfb54e9de67698508595"
 ["system_simulator/scripts/explore_m214_fc2_same_cycle_done_load_recurrence.py"]="01a870f85ae62208d9d9c145021a476a59b26cb2fc0fad343d2ed51006517b5e"
 ["system_simulator/scripts/analyze_m214_m212_tail_cycle_ab.py"]="b650289bc1b2ee30d628c5f61c42f03d6b0805dc7dd87ffde4f75de0160bc7e6"
 ["results/m212_fc2_terminal_close_vcs_calibration_r2_exact_20260825/tail_sweep_vcs_sim.log"]="1d86060cca88c7543158c5d63f07900f0b286e47ef7a9bbbc340521bfba9a7c6"
 ["results/m212_fc2_terminal_close_vcs_calibration_r2_exact_20260825/SHA256SUMS"]="1ad3828cdc0c591db370c907b60fc100e49a5a021282d10e74c8fc78bcbe350f"
 ["contracts/m214_fc2_same_cycle_done_load_vcs_contract_r1_20260825.json"]="f2d10e68fb6ccc768f0c888dec21aafe2bad6e0c550241809c4860e9794435ea"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
run_vcs() {
    local task_name="$1" task_filelist="$2" task_top="$3"
    local task_dir="$task_run/$task_name"
    mkdir "$task_dir"
    set +e
    "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="$task_dir/csrc" -f "$task_filelist" -top "$task_top" \
        -o "$task_dir/simv" > "$task_dir/compile.log" 2>&1
    local task_rc=$?
    set -e
    echo "$task_rc" > "$task_dir/compile.rc"
    [[ "$task_rc" -eq 0 && -x "$task_dir/simv" ]] || exit 20
    grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_dir/compile.log" && exit 21 || true
    set +e
    "$task_dir/simv" +ntb_random_seed=214025 -no_save \
        -assert report="$task_dir/assert.report" -cm assert \
        > "$task_dir/sim.log" 2>&1
    task_rc=$?
    set -e
    echo "$task_rc" > "$task_dir/sim.rc"
    [[ "$task_rc" -eq 0 ]] || exit 22
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "$task_dir/sim.log" "$task_dir/assert.report" && exit 23 || true
}

run_vcs directed dc_handoff/filelists/date_m214_fc2_raw4_to_same_done_load_directed_vcs.f tb_m214_fc2_raw4_to_same_done_load_frontend
run_vcs tail dc_handoff/filelists/date_m214_fc2_same_done_load_tail_sweep_vcs.f tb_m214_fc2_same_done_load_tail_sweep
run_vcs same_done_stall dc_handoff/filelists/date_m214_fc2_same_done_load_stall_vcs.f tb_m214_fc2_same_done_load_stall
run_vcs bank48 dc_handoff/filelists/date_m214_fc2_bank48_adversarial_vcs.f tb_m214_fc2_bank48_adversarial
run_vcs stage0 dc_handoff/filelists/date_m214_fc2_stage0_handoff_prefetch_vcs.f tb_m214_fc2_stage0_handoff_prefetch

grep -Fq 'PASS M214 terminal-hint-to-terminal-close cycle co-sim VCS legal_headers=6 raw_packets=17 groups=104 done=5' "$task_run/directed/sim.log" || exit 30
grep -Fxq 'PASS M214 terminal-collapse tail sweep cases=256' "$task_run/tail/sim.log" || exit 31
grep -Fq 'PASS M214 same-cycle done-fence load stall VCS same_done_loads=1 groups=2 done=1 group_stalls=3 identity_mismatches=0' "$task_run/same_done_stall/sim.log" || exit 32
grep -Fq 'PASS M214 bank48 adversarial VCS groups=192 done=1 bank48_accepts=2 header_to_done_cycles=195' "$task_run/bank48/sim.log" || exit 33
grep -Fq 'PASS M214 stage0 handoff prefetch groups=2 done=1 handoffs=1 stalls=1 header_to_done_cycles=5' "$task_run/stage0/sim.log" || exit 34
[[ "$(grep -c '^M214TAIL ' "$task_run/tail/sim.log")" -eq 256 ]] || exit 35
grep -Eq 'cp_same_cycle_done_load, .* 47 match' "$task_run/tail/assert.report" || exit 36
grep -Eq 'cp_same_cycle_done_load, .* 1 match' "$task_run/same_done_stall/assert.report" || exit 37
grep -Eq 'cp_group_stall, .* 3 match' "$task_run/same_done_stall/assert.report" || exit 38
grep -Eq 'cp_descriptor_bank_sum_48, .* 2 match' "$task_run/bank48/assert.report" || exit 39
grep -Eq 'cp_stage0_handoff, .* 1 match' "$task_run/stage0/assert.report" || exit 40

python3 system_simulator/scripts/explore_m214_fc2_same_cycle_done_load_recurrence.py \
    --sweep-log "$task_run/tail/sim.log" \
    --output "$task_run/model/m214_rtl_control_recurrence.json" \
    > "$task_run/model_stdout.log"
python3 system_simulator/scripts/analyze_m214_m212_tail_cycle_ab.py \
    --m212-log results/m212_fc2_terminal_close_vcs_calibration_r2_exact_20260825/tail_sweep_vcs_sim.log \
    --m214-log "$task_run/tail/sim.log" \
    --output "$task_run/ab/m214_m212_exact_vcs_tail_cycle_ab.json" \
    > "$task_run/ab_stdout.log"
python3 - "$task_run" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
model = json.loads((root / "model/m214_rtl_control_recurrence.json").read_text())
ab = json.loads((root / "ab/m214_m212_exact_vcs_tail_cycle_ab.json").read_text())
assert model["status"] == "PASS_EXACT_256_CASE_VCS"
assert model["cases"] == 256 and model["mismatches"] == 0
assert sum(item["same_cycle_done_loads"] for item in model["all_records"]) == 47
assert ab["status"] == "PASS_NO_REGRESSION" and ab["cases"] == 256
assert ab["improved_cases"] == 47 and ab["unchanged_cases"] == 209
assert ab["regressed_cases"] == 0 and ab["total_cycles_saved_across_sweep"] == 47
PY

{
    echo status=PASS_M214_FC2_SAME_CYCLE_DONE_LOAD_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo exact_model_cases=256
    echo exact_model_mismatches=0
    echo improved_cases_vs_m212=47
    echo unchanged_cases_vs_m212=209
    echo regressed_cases_vs_m212=0
    echo causal_same_done_load_cover=47
    echo causal_stall_identity_mismatches=0
    echo bank48_regression_cycles=195
    echo stage0_regression_cycles=5
    echo frozen_h67_cycles=false
    echo complete_fc2=false
    echo complete_ffn=false
    echo physical_speedup=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/RUN_COMPLETE.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$task_run/SHA256SUMS"
sha256sum dc_handoff/scripts/run_vcs_m214_fc2_same_cycle_done_load_exact.sh \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M214 exact VCS sealed at $task_run"
