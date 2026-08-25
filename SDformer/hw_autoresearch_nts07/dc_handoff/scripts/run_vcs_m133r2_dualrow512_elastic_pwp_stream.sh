#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M133r2 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
on_exit() {
    local task_rc="$?"
    if [[ "$task_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$task_rc"
        } > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$task_hw_root"
task_rtl="rtl_m133/m133_dualrow512_elastic_pwp_stream.sv"
task_sva="verif_m133/m133_dualrow512_elastic_pwp_stream_assertions.sv"
task_tb="tb_m133/tb_m133_dualrow512_elastic_pwp_stream.sv"
task_files="dc_handoff/filelists/date_m133_dualrow512_elastic_pwp_stream_directed_vcs.f"
task_contract="contracts/m133r2_dualrow512_elastic_pwp_stream_vcs_contract_r1_20260824.json"
task_r1_correction="contracts/m133_r1_stall_fault_composition_correction_r1_20260824.json"
task_m132_correction="contracts/m132_r1_independent_review_correction_overlay_r1_20260824.json"

declare -A task_expected=(
    ["$task_rtl"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["$task_sva"]="564fc8184977f352d4d841164583f0dc694ce8ba33fd3d2d6f871a3c2cbc6cea"
    ["$task_tb"]="3b73c0ea7d572382521e112a7962febe9c9733899b3a1ca30fa282b97708a742"
    ["$task_files"]="575a3171e12b701f58709a68703a18eb0a4d111e215e7e4393921c2a4f347c31"
    ["$task_contract"]="75d827342d36a82318a29f3efe7149a87b73eeacd576f94c9c533d9cb4c2020f"
    ["$task_r1_correction"]="a32d3bab8faddf0a318c6ba6a3a1b36cb2ac579c08b3d327b56f4e599f59feff"
    ["$task_m132_correction"]="82ca925af73a7fecb55c4a47d6d95fbba5eb5c22698a2c27695b6a68fbda36a9"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M133r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" \
    -top tb_m133_dualrow512_elastic_pwp_stream \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
task_pass='PASS M133r2 dualrow512 elastic PWP stream VCS vectors=105 outputs=105 beats=236 lanes=9696 escapes=4 ii_checks=63 stalls=43 long_stall=23 boundaries=5 protocol_attacks=2 stall_fault_overlap=1 reset_attacks=1 idle_payload=1 cycles_8_9_10_11=2_2_2_3 input_bits=512 bank_mapper=false macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_width8, .* [1-9][0-9]* match' \
        'cp_width9, .* [1-9][0-9]* match' \
        'cp_width10, .* [1-9][0-9]* match' \
        'cp_width11, .* [1-9][0-9]* match' \
        'cp_escape, .* [1-9][0-9]* match' \
        'cp_output_stall_release, .* [1-9][0-9]* match' \
        'cp_last_to_next_start, .* [1-9][0-9]* match' \
        'cp_same_cycle_fault_quarantine, .* [1-9][0-9]* match' \
        'cp_stall_to_fault_quarantine, .* [1-9][0-9]* match' \
        'cp_reset_quiesce, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M133R2_DUALROW512_ELASTIC_PWP_STREAM_VCS_SVA"
    echo "exact_sha=true"
    echo "positive_vectors=105"
    echo "positive_outputs=105"
    echo "positive_accepted_beats=233"
    echo "total_accepted_beats_including_attack_setup=236"
    echo "numeric_lane_checks=9696"
    echo "exact_start_interval_checks=63"
    echo "stalled_output_fault_overlap_attacks=1"
    echo "cycles_8_9_10_11=2_2_2_3"
    echo "service_port_bits=512"
    echo "rtl_unchanged_from_m133_r1=true"
    echo "bank_mapper_implemented=false"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m133r2_dualrow512_elastic_pwp_stream.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M133r2 dualrow512 elastic PWP stream VCS sealed at $task_run"
