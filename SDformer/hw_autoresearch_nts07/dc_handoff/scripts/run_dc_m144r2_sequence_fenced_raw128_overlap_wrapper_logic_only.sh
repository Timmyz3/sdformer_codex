#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m144r2_sequence_fenced_raw128_overlap_wrapper_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M144r2 sealed DC run: $task_run" >&2
    exit 2
fi
if [[ ! -x "$task_dc_shell" || ! -s "$task_lib" || ! -s "$task_min_lib" ]]; then
    echo "M144r2 Synopsys executable or TSMC28 library missing" >&2
    exit 3
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
task_files="dc_handoff/filelists/date_m144_sequence_fenced_raw128_overlap_wrapper_logic_only_dc.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m142/m142_sparse_mask_k4_three_bank_overlap_controller.sv"]="da80d61a4fe95bfd97ea50af388b48d924dcc0466836aa72f3809552d6c1915d"
    ["rtl_m144/m144_sequence_fenced_raw128_overlap_wrapper.sv"]="74a15a781c098a2d9a2a522fa97c93aeeb5db1d6eb9f8851882d22a26a18a6de"
    ["$task_files"]="d06d0d97805c2b08f720a4f79d91af4338f64316bb6a7dcbf0a15f62935cbf9d"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m144r2_sequence_fenced_raw128_overlap_wrapper_logic_only_dc_contract_r1_20260824.json"]="891be1fff7a76dc22a0e16da81433dcf234734e8afb9c0cffef1bb62e6cc2883"
    ["contracts/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_contract_r1_20260824.json"]="d6d807fe0f71da20bbb87d21975ffc1147dc59f6c9987ab80aa64ee79b34c40f"
    ["dc_handoff/runs/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="a99295aa36f847a75ddef753fa66d3f3c08920bda3a6d22a9ce8ff15d187218b"
    ["contracts/m142_independent_review_correction_overlay_r1_20260824.json"]="9667c026b0dddd6eabfe6743087938d3855cdae98c6cfe16ef3a71ecb73ee929"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M144r2 DC exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME="m144_sequence_fenced_raw128_overlap_wrapper"
export HW_ROOT="$task_hw_root"
export RTL_FILELIST="$task_hw_root/$task_files"
export LIB_DB="$task_lib"
export MIN_LIB_DB="$task_min_lib"
export SDC_FILE="$task_hw_root/$task_sdc"
export OUTPUT_DIR="$task_run"
export OPERATING_CONDITION="ssg0p9v125c"

set +e
"$task_dc_shell" -f "$task_hw_root/$task_tcl" > "$task_run/dc.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_run/dc.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 20; fi
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:|^Fatal:' "$task_run/dc.log"; then exit 21; fi
if ! grep -Fq 'Thank you...' "$task_run/dc.log"; then exit 22; fi

for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt \
        check_timing_postcompile.rpt; do
    if [[ ! -s "$task_run/reports/$task_report" ]]; then exit 30; fi
done
if [[ ! -s "$task_run/netlist/${DESIGN_NAME}_mapped.v" \
      || ! -s "$task_run/netlist/${DESIGN_NAME}_mapped.sdc" \
      || ! -s "$task_run/netlist/${DESIGN_NAME}.ddc" ]]; then exit 31; fi
if grep -Fq 'slack (VIOLATED)' "$task_run/reports/timing_setup.rpt" \
        "$task_run/reports/timing_hold.rpt"; then exit 32; fi
if ! grep -Fq 'slack (MET)' "$task_run/reports/timing_setup.rpt" \
        || ! grep -Fq 'slack (MET)' "$task_run/reports/timing_hold.rpt"; then exit 33; fi
if [[ "$(grep -Fc 'This design has no violated constraints.' \
        "$task_run/reports/constraint_violators.rpt")" -ne 5 ]]; then exit 34; fi
if [[ "$(tr -d '[:space:]' < "$task_run/reports/check_design_postcompile.rpt")" != "1" ]]; then exit 35; fi
if [[ "$(tail -n 1 "$task_run/reports/check_timing_postcompile.rpt" | tr -d '[:space:]')" != "1" ]]; then exit 36; fi
if ! grep -Fq 'Number of macros/black boxes:               0' "$task_run/reports/area.rpt"; then exit 37; fi

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
if ! awk -v task_candidate="$task_area" 'BEGIN {exit !(task_candidate < 4200)}'; then exit 38; fi
if ! awk -v task_candidate="$task_seq" 'BEGIN {exit !(task_candidate < 900)}'; then exit 39; fi
{
    echo "status=PASS_M144R2_SEQUENCE_FENCED_RAW128_OVERLAP_WRAPPER_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "hierarchy=flattened_before_mapping"
    echo "banks=4"
    echo "raw_row_bits=128"
    echo "sequence_bits=32"
    echo "cell_area_um2=$task_area"
    echo "cell_count=$task_cells"
    echo "sequential_cells=$task_seq"
    echo "setup_worst_slack_ns=$task_setup"
    echo "hold_worst_slack_ns=$task_hold"
    echo "macro_count=0"
    echo "engine_arithmetic=false"
    echo "descriptor_result_sram_macro=false"
    echo "paper_ppa_ready=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m144r2_sequence_fenced_raw128_overlap_wrapper_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M144r2 sequence-fenced raw128 overlap wrapper logic-only DC sealed at $task_run"
