#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M138r2 sealed VCS run: $task_run" >&2
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
task_m133="rtl_m133/m133_dualrow512_elastic_pwp_stream.sv"
task_m137="rtl_m137/m137_fallthrough_tagged_16bank_response_bridge.sv"
task_m138="rtl_m138/m138_metadata_guarded_fallthrough_pwp_frontend.sv"
task_sva="verif_m138/m138_metadata_guarded_fallthrough_pwp_frontend_assertions.sv"
task_tb="tb_m138/tb_m138_metadata_guarded_fallthrough_pwp_frontend.sv"
task_files="dc_handoff/filelists/date_m138_metadata_guarded_fallthrough_pwp_frontend_directed_vcs.f"
task_contract="contracts/m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend_vcs_contract_r1_20260824.json"
task_correction="contracts/m138_r1_dc_timing_loop_failure_correction_r1_20260824.json"
task_m135_overlay="contracts/m135r3_independent_review_and_r2_failure_identity_overlay_r1_20260824.json"
task_m136_overlay="contracts/m136_independent_review_latency_scope_overlay_r1_20260824.json"
task_m137_vcs_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m137_dc_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m133"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["$task_m137"]="e2b0a271728dc8c0f79ba3361f76df554ad61e6d6efaf11ae09ff89be9384af2"
    ["$task_m138"]="f12066770e1f8dd445bf48b60115380423c0d34ce420390321948ce88197709c"
    ["$task_sva"]="e3ce5da0df3694dbb2137195dffa7d80d88dd5e8c8e4966007322d7bf7fab6c3"
    ["$task_tb"]="b3f28d44f49a370481660ef5369fc3d2f5e2d8c3049eb141a79a107c5f2180e0"
    ["$task_files"]="db790c5f9ff54140bc5653d45a89bc0914d94100db5ee3a11a98879646825159"
    ["$task_contract"]="392fcf64f2193fdbf657bf3a46516ea26d5881402d57e6f328c092e984848e3a"
    ["$task_correction"]="49ed0de76c3aad5597573ab2a1260c91b1a558f2719526d7749440ffbe1e76f7"
    ["$task_m135_overlay"]="2ad920d745871b11b5b2336ec9a93231cda5a8bc2bbb41a8b61562b1754642da"
    ["$task_m136_overlay"]="3f6608a404fcd98e5fcb74d85bcb1ace8dfae8dca3da2ddff4e0a5eac8c97f8d"
    ["$task_m137_vcs_receipt"]="faf463a052213cc878fbce9786321d090ec9ccf057ae2ddab4a1a1b869254f4e"
    ["$task_m137_dc_receipt"]="dd77ebb075825d501e20d6a726abf1858c0789e92a0f99d1daed37e8a20dd3c6"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M138r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m138_metadata_guarded_fallthrough_pwp_frontend \
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
task_pass='PASS M138r2 metadata-guarded acyclic fallthrough PWP frontend VCS vectors=96 outputs=96 beats=212 macro_requests=208 lanes=8832 escapes=4 ii_checks=63 stalls=13 row_crossings=194 base_offsets=16 metadata_attacks=4 suppressed_reads=4 data_padding_attacks=1 invalid_base_attacks=1 reset_attacks=6 cycles_8_9_10_11=2_2_2_3 macro_latency=1 delivery_latency=1 banks=16 service_bits=512 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_macro_request_cross_row, .* [1-9][0-9]* match' \
        'cp_escape_without_macro, .* [1-9][0-9]* match' \
        'cp_output_stall_release, .* [1-9][0-9]* match' \
        'cp_metadata_suppressed_read, .* [4-9][0-9]* match|cp_metadata_suppressed_read, .* [4-9] match' \
        'cp_restart_suppressed, .* [1-9][0-9]* match' \
        'cp_data_fault_registered, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M138R2_METADATA_GUARDED_ACYCLIC_FALLTHROUGH_PWP_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "vectors=96"
    echo "outputs=96"
    echo "accepted_input_beats=212"
    echo "positive_macro_requests=208"
    echo "numeric_lane_checks=8832"
    echo "escapes=4"
    echo "exact_start_interval_checks=63"
    echo "output_stall_cycles=13"
    echo "metadata_decidable_attacks=4"
    echo "metadata_suppressed_bank_reads=4"
    echo "data_dependent_padding_attacks=1"
    echo "illegal_base_attacks=1"
    echo "reset_recoveries=6"
    echo "cycles_8_9_10_11=2_2_2_3"
    echo "macro_latency_cycles=1"
    echo "delivery_latency_cycles=1"
    echo "dc_acyclic_timing_graph=false"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m138r2_metadata_guarded_acyclic_fallthrough_pwp_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M138r2 metadata-guarded acyclic fallthrough PWP frontend VCS sealed at $task_run"
