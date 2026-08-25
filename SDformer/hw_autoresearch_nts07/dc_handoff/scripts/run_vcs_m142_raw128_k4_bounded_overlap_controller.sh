#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m142_raw128_k4_bounded_overlap_controller_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M142 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m142/m142_sparse_mask_k4_three_bank_overlap_controller.sv"
task_sva="verif_m142/m142_sparse_mask_k4_bounded_overlap_controller_assertions.sv"
task_tb="tb_m142/tb_m142_sparse_mask_k4_bounded_overlap_controller.sv"
task_files="dc_handoff/filelists/date_m142_sparse_mask_k4_bounded_overlap_controller_directed_vcs.f"
task_contract="contracts/m142_raw128_k4_bounded_overlap_controller_vcs_contract_r1_20260824.json"
task_m143_contract="contracts/m143r2_raw128_full_materialized_overlap_dse_contract_r1_20260824.json"
task_m143_result="results/m143r2_raw128_full_materialized_overlap_dse_r1_20260824/m143_raw128_full_materialized_overlap_dse.json"
task_m141_overlay="contracts/m141r3_independent_review_correction_overlay_r1_20260824.json"

declare -A task_expected=(
    ["$task_rtl"]="da80d61a4fe95bfd97ea50af388b48d924dcc0466836aa72f3809552d6c1915d"
    ["$task_sva"]="a6b9a153ce67244bb0fb8b7d5258f293e368fa23b4347271b0fbf71e35fde5e2"
    ["$task_tb"]="4b0a1615aeff4a4bb822864c5c136339d9f29a8682b10c8d1e7ff4d918ce2d01"
    ["$task_files"]="bcdc3e4bf5b2968e7e0c55312d015b56f8e6f3426b0e6f103e1a43ecfb21ebd9"
    ["$task_contract"]="8996fd7d15ce91a76ea0bcefe515b8c9abe6dcfda42631cadaa6f258010efa9e"
    ["$task_m143_contract"]="288f03c77556c3e9ea26bfeb18e457423e8f8d8c3dfac9bef070769436051413"
    ["$task_m143_result"]="8b5821d747e653ac9053a4cfe94fe9eb40c78ce0eaaca4c9af4fdf8073b5bd19"
    ["$task_m141_overlay"]="309ac23757ed743a7731b018a4a94aec0802af6ba81289f514897102042ce3d3"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M142 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m142_sparse_mask_k4_bounded_overlap_controller \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" -no_save -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 30; fi
task_pass='PASS M142 bounded overlap VCS banks=4 windows=32 rows=96 zero_rows=32 descriptors=1184 sources=4576 pwp=32 correction=32 completed=32 descriptor_ii1=1038 descriptor_stalls=120 pwp_stalls=3 correction_stalls=2 bank_reuses=28 early_pwp=0 raw_zero_rows_accepted=true protocol_attacks=3 pwp_correction_overlap=1 all_banks_owned=1 engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_k4_descriptor, .* [1-9][0-9]* match' \
        'cp_descriptor_stall, .* [1-9][0-9]* match' \
        'cp_pwp_correction_overlap, .* [1-9][0-9]* match' \
        'cp_all_banks_owned, .* [1-9][0-9]* match' \
        'cp_materialized_before_pwp, .* [1-9][0-9]* match' \
        'cp_correction_release, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M142_RAW128_K4_BOUNDED_OVERLAP_CONTROLLER_VCS_SVA"
    echo "exact_sha=true"
    echo "banks=4"
    echo "raw_row_bits=128"
    echo "windows=32"
    echo "accepted_rows=96"
    echo "accepted_zero_rows=32"
    echo "accepted_descriptors=1184"
    echo "checked_sources=4576"
    echo "pwp_launches=32"
    echo "correction_launches=32"
    echo "correction_completions=32"
    echo "descriptor_ii1_intervals=1038"
    echo "descriptor_stall_cycles=120"
    echo "pwp_stall_cycles=3"
    echo "correction_stall_cycles=2"
    echo "bank_reuses=28"
    echo "early_pwp_launches=0"
    echo "protocol_attacks=3"
    echo "pwp_correction_overlap_covered=true"
    echo "all_banks_owned_covered=true"
    echo "engine_arithmetic=false"
    echo "descriptor_result_sram_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m142_raw128_k4_bounded_overlap_controller.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M142 raw128 K4 bounded overlap controller VCS sealed at $task_run"
