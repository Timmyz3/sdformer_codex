#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m138_metadata_guarded_fallthrough_pwp_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M138 sealed VCS run: $task_run" >&2
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
task_contract="contracts/m138_metadata_guarded_fallthrough_pwp_frontend_vcs_contract_r1_20260824.json"
task_m135_overlay="contracts/m135r3_independent_review_and_r2_failure_identity_overlay_r1_20260824.json"
task_m136_overlay="contracts/m136_independent_review_latency_scope_overlay_r1_20260824.json"
task_m133_receipt="dc_handoff/runs/m133r2_dualrow512_elastic_pwp_stream_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m137_vcs_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m137_dc_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m133"]="84f1b6f6e8d085f14bbe8abe7b2fbfd9dbac586d178ce7e3eb2dff55db92f6de"
    ["$task_m137"]="e2b0a271728dc8c0f79ba3361f76df554ad61e6d6efaf11ae09ff89be9384af2"
    ["$task_m138"]="bf69ec0e296d2ab11a60ab3fc6f13644659275e3da160678a71d03e46351b58b"
    ["$task_sva"]="127f07bf54814e73ae56af4672d4df696238b466d7c29f44180e1582bf6b778d"
    ["$task_tb"]="5bd6e8d23658d76fa6df29ce3df3a16a62e26d79b743ad1411f341176dd83747"
    ["$task_files"]="db790c5f9ff54140bc5653d45a89bc0914d94100db5ee3a11a98879646825159"
    ["$task_contract"]="7d0f48816241b8fe9f28221a034c04829c39bc83602c76ee47b9160d97413ebf"
    ["$task_m135_overlay"]="2ad920d745871b11b5b2336ec9a93231cda5a8bc2bbb41a8b61562b1754642da"
    ["$task_m136_overlay"]="3f6608a404fcd98e5fcb74d85bcb1ace8dfae8dca3da2ddff4e0a5eac8c97f8d"
    ["$task_m133_receipt"]="e8981a5fb623f76df044225513d8334b03b65b3fcd73620eeee57d6707b2dc49"
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
        echo "M138 exact-SHA preflight mismatch: $task_path" >&2
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
task_pass='PASS M138 metadata-guarded fallthrough PWP frontend VCS vectors=96 outputs=96 beats=212 macro_requests=208 lanes=8832 escapes=4 ii_checks=63 stalls=13 row_crossings=194 base_offsets=16 metadata_attacks=4 suppressed_reads=4 invalid_base_attacks=1 reset_attacks=5 cycles_8_9_10_11=2_2_2_3 macro_latency=1 delivery_latency=1 banks=16 service_bits=512 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_macro_request_cross_row, .* [1-9][0-9]* match' \
        'cp_escape_without_macro, .* [1-9][0-9]* match' \
        'cp_output_stall_release, .* [1-9][0-9]* match' \
        'cp_metadata_suppressed_read, .* [4-9][0-9]* match|cp_metadata_suppressed_read, .* [4-9] match' \
        'cp_restart_suppressed, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M138_METADATA_GUARDED_FALLTHROUGH_PWP_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "vectors=96"
    echo "outputs=96"
    echo "accepted_input_beats=212"
    echo "macro_requests=208"
    echo "numeric_lane_checks=8832"
    echo "escapes=4"
    echo "exact_start_interval_checks=63"
    echo "output_stall_cycles=13"
    echo "row_crossing_beats=194"
    echo "metadata_decidable_attacks=4"
    echo "suppressed_bank_reads=4"
    echo "invalid_base_attacks=1"
    echo "reset_recoveries=5"
    echo "cycles_8_9_10_11=2_2_2_3"
    echo "macro_latency_cycles=1"
    echo "delivery_latency_cycles=1"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m138_metadata_guarded_fallthrough_pwp_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M138 metadata-guarded fallthrough PWP frontend VCS sealed at $task_run"
