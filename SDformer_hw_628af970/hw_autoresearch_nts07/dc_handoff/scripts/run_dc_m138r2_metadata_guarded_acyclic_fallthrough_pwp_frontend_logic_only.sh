#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_logic_only_dc_3p000ns_r1_sealed_20260824"
task_dc_shell="${DC_SHELL:-/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell}"
task_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"
task_min_lib="/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M138r2 sealed DC run: $task_run" >&2
    exit 2
fi
if [[ ! -x "$task_dc_shell" || ! -s "$task_lib" || ! -s "$task_min_lib" ]]; then
    echo "M138r2 Synopsys executable or TSMC28 library missing" >&2
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
task_m133="rtl_m133/m133_dualrow512_elastic_pwp_stream.sv"
task_m137="rtl_m137/m137_fallthrough_tagged_16bank_response_bridge.sv"
task_m138="rtl_m138/m138_metadata_guarded_fallthrough_pwp_frontend.sv"
task_files="dc_handoff/filelists/date_m138_metadata_guarded_fallthrough_pwp_frontend_logic_only_dc.f"
task_sdc="dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
task_tcl="dc_handoff/scripts/run_dc_m135r3_flattened_logic_only.tcl"
task_contract="contracts/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_logic_only_dc_contract_r1_20260824.json"
task_vcs_contract="contracts/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_vcs_contract_r1_20260824.json"
task_vcs_receipt="dc_handoff/runs/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_correction="contracts/m138_r1_dc_timing_loop_failure_correction_r1_20260824.json"
task_failed_log="dc_handoff/runs/m138_metadata_guarded_fallthrough_pwp_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/dc.log"
task_failed_receipt="dc_handoff/runs/m138_metadata_guarded_fallthrough_pwp_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_FAILED_OR_INCOMPLETE.txt"
task_m133_dc_receipt="dc_handoff/runs/m133_dualrow512_elastic_pwp_stream_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m137_dc_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m133"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["$task_m137"]="e2b0a271728dc8c0f79ba3361f76df554ad61e6d6efaf11ae09ff89be9384af2"
    ["$task_m138"]="f12066770e1f8dd445bf48b60115380423c0d34ce420390321948ce88197709c"
    ["$task_files"]="d312ee2a62693fb3c7d31f999525786aa60ac1357d711668d1375b3579dedd26"
    ["$task_sdc"]="808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5"
    ["$task_tcl"]="837b50d1c1700ef83edebd070ca3ebb6a6de166a2d649272f7b597c13e56dbe9"
    ["$task_contract"]="c5a5838728b6adcb462e12dbc44db5d8a620f8e61d0c3485139e446009459d18"
    ["$task_vcs_contract"]="392fcf64f2193fdbf657bf3a46516ea26d5881402d57e6f328c092e984848e3a"
    ["$task_vcs_receipt"]="5a1eff41f9461afa2990253be2fb740b24c9fa050d1f084a15ab4573244ef87b"
    ["$task_correction"]="49ed0de76c3aad5597573ab2a1260c91b1a558f2719526d7749440ffbe1e76f7"
    ["$task_failed_log"]="536d205bf5a9bd5283b6c18edc303dec45f79aa6433b6df6a597eb904e0a5c6c"
    ["$task_failed_receipt"]="0b71bda8d02ca4d69532170e5592679f03858fb0b3a4464fc4a68ca839e2ee23"
    ["$task_m133_dc_receipt"]="4cf52cb502b516b129e8345ea941c933c01e1f9ca10d361225a9844133f5481e"
    ["$task_m137_dc_receipt"]="dd77ebb075825d501e20d6a726abf1858c0789e92a0f99d1daed37e8a20dd3c6"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M138r2 DC exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export DESIGN_NAME="m138_metadata_guarded_fallthrough_pwp_frontend"
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
if grep -Eq 'ELAB-312|TIM-209|OPT-150|^Error:' "$task_run/dc.log"; then exit 21; fi
if ! grep -Fq 'Thank you...' "$task_run/dc.log"; then exit 22; fi

for task_report in area.rpt qor.rpt timing_setup.rpt timing_hold.rpt \
        constraint_violators.rpt check_design_postcompile.rpt check_timing_postcompile.rpt; do
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
if [[ "$(tail -n 1 "$task_run/reports/check_timing_postcompile.rpt" \
        | tr -d '[:space:]')" != "1" ]]; then exit 36; fi
if ! grep -Fq 'Number of macros/black boxes:               0' \
        "$task_run/reports/area.rpt"; then exit 37; fi

task_area="$(awk '/Total cell area:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_setup="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_setup.rpt")"
task_hold="$(awk '/slack \(MET\)/ {print $3; exit}' "$task_run/reports/timing_hold.rpt")"
task_cells="$(awk '/Number of cells:/ {print $4; exit}' "$task_run/reports/area.rpt")"
task_seq="$(awk '/Number of sequential cells:/ {print $5; exit}' "$task_run/reports/area.rpt")"
task_sum_area="15582.924048"
task_integration_area_delta_pct="$(awk -v task_candidate="$task_area" -v task_sum="$task_sum_area" \
    'BEGIN {printf "%.6f", 100.0*(task_candidate-task_sum)/task_sum}')"
{
    echo "status=PASS_M138R2_METADATA_GUARDED_ACYCLIC_FALLTHROUGH_PWP_FRONTEND_LOGIC_ONLY_DC_3NS"
    echo "exact_sha=true"
    echo "tool=Synopsys_DC_V-2023.12-SP3"
    echo "clock_period_ns=3.000"
    echo "hierarchy=flattened_before_mapping"
    echo "acyclic_timing_graph=true"
    echo "cell_area_um2=$task_area"
    echo "cell_count=$task_cells"
    echo "sequential_cells=$task_seq"
    echo "standalone_m133_plus_m137_area_um2=$task_sum_area"
    echo "integration_area_delta_vs_standalone_sum_pct=$task_integration_area_delta_pct"
    echo "setup_worst_slack_ns=$task_setup"
    echo "hold_worst_slack_ns=$task_hold"
    echo "macro_count=0"
    echo "banks=16"
    echo "service_bits=512"
    echo "fixed_behavioral_macro_latency_cycles=1"
    echo "foundry_macro=false"
    echo "paper_ppa_ready=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/dc.log "$task_run"/reports/*.rpt \
    "$task_run"/netlist/* "$task_run"/RUN_COMPLETE.txt > "$task_run/evidence_manifest.sha256"
sha256sum "dc_handoff/scripts/run_dc_m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_logic_only.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M138r2 metadata-guarded acyclic fallthrough PWP frontend logic-only DC sealed at $task_run"
