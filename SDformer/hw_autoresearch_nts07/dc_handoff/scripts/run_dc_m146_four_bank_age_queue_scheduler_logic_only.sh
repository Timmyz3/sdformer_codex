#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m146_four_bank_age_queue_scheduler_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M146 sealed DC run: $task_run" >&2
    exit 2
fi
if [[ ! -x "$task_dc_shell" || ! -s "$task_lib" || ! -s "$task_min_lib" ]]; then
    echo "M146 Synopsys executable or TSMC28 library missing" >&2
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
task_files="dc_handoff/filelists/date_m146_four_bank_age_queue_scheduler_logic_only_dc.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
declare -A task_expected=(
    ["rtl_m146/m146_four_bank_age_queue_scheduler.sv"]="0780a406f480ffd0b22ec81be288ef95384963ee03548277634634dbb68627ed"
    ["$task_files"]="7fa1dd0ce7196497ab6ea861700e12e9ad72712466ba0aa48f3f55a161122fb0"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["contracts/m146_four_bank_age_queue_scheduler_logic_only_dc_contract_r1_20260824.json"]="85d4411c75cddaf171179f6cb90c918805bf0dd3ed4f987b484800beeedec6c5"
    ["contracts/m146_four_bank_age_queue_scheduler_vcs_contract_r1_20260824.json"]="c0418cf1eae3488be825f2c3036f6e68e12e6cade05de9e798a3656bd1b4cf3b"
    ["dc_handoff/runs/m146_four_bank_age_queue_scheduler_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="1af5335adff4335773a12b7931c5c9844c8ec9d853be9db33f22fae71bbc7858"
    ["contracts/m142_raw128_k4_bounded_overlap_controller_logic_only_dc_contract_r1_20260824.json"]="cc88801bb7e6d6f0e2969775dda1b68712af6e5a50ba186e9527157c5eecdf4d"
    ["dc_handoff/runs/m142_raw128_k4_bounded_overlap_controller_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="3e6f6fabc2b4fdd686f54a57a6a724e451e402ede138486a3ed2ff87a6f0fef6"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M146 DC exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME="m146_four_bank_age_queue_scheduler"
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
task_levels="$(awk '/Levels of Logic:/ {print $4; exit}' "$task_run/reports/qor.rpt")"
task_path_length="$(awk '/Critical Path Length:/ {print $5; exit}' "$task_run/reports/qor.rpt")"
if ! awk -v task_candidate="$task_area" 'BEGIN {exit !(task_candidate < 1700)}'; then exit 38; fi
if ! awk -v task_candidate="$task_seq" 'BEGIN {exit !(task_candidate < 400)}'; then exit 39; fi
if ! awk -v task_candidate="$task_setup" 'BEGIN {exit !(task_candidate >= 1.5)}'; then exit 40; fi
if ! awk -v task_candidate="$task_levels" 'BEGIN {exit !(task_candidate <= 20)}'; then exit 41; fi
{
    echo "status=PASS_M146_FOUR_BANK_AGE_QUEUE_SCHEDULER_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "hierarchy=flattened_before_mapping"
    echo "banks=4"
    echo "sequence_bits=32"
    echo "sequence_age_comparators=0"
    echo "completion_identity_equality=true"
    echo "cell_area_um2=$task_area"
    echo "cell_count=$task_cells"
    echo "sequential_cells=$task_seq"
    echo "logic_levels=$task_levels"
    echo "critical_path_length_ns=$task_path_length"
    echo "setup_worst_slack_ns=$task_setup"
    echo "hold_worst_slack_ns=$task_hold"
    echo "macro_count=0"
    echo "m142_structural_reference_logic_levels=68"
    echo "m142_structural_reference_path_length_ns=2.52"
    echo "drop_in_m142_equivalence=false"
    echo "engine_arithmetic=false"
    echo "sram_macro=false"
    echo "paper_ppa_ready=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m146_four_bank_age_queue_scheduler_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M146 four-bank age-queue scheduler logic-only DC sealed at $task_run"
