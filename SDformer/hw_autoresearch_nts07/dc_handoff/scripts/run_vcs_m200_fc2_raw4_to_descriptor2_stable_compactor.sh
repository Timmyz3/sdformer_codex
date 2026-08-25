#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m200_fc2_raw4_to_descriptor2_stable_compactor_vcs_r1_sealed_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then echo "refusing to overwrite M200 sealed VCS run" >&2; exit 2; fi
mkdir -p "$(dirname "$task_run")"; mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m200/m200_fc2_raw4_to_descriptor2_stable_compactor.sv"]="d6aa9cf2e485fd53b1776b80d9e94d17ace0165e25c5a49a534561ae9e2a2027"
    ["verif_m200/m200_fc2_raw4_to_descriptor2_stable_compactor_assertions.sv"]="0fc6c93a5a331ba455063bee48a4adbb2e3a0088314509bb0f09d74bab833628"
    ["tb_m200/tb_m200_fc2_raw4_to_descriptor2_stable_compactor.sv"]="cfd5a0a75be50303d8b23a83cb89bd09d07f677059ade73173b2f66fe13d6f4b"
    ["dc_handoff/filelists/date_m200_fc2_raw4_to_descriptor2_stable_compactor_directed_vcs.f"]="7a35b250ff323860e0ffc9d119240d2c6f155bba46ea8612f982e0c1ee8f040d"
    ["contracts/m200_fc2_raw4_to_descriptor2_stable_compactor_vcs_contract_r1_20260825.json"]="0df527d4f0d6c145632a27b126acc055e6a3f51d3e6b7c41cdd229c442cd34e8"
    ["results/m199_h67_fc2_decoupled_scanner_compactor_dse_r1_20260825/manifest.sha256"]="e23a72e2a59e4119a3d54eb78bcbf56dd768a165c1e883b364cf6cb4075c0c08"
    ["results/m199_independent_hammer_review_r1_20260825/SHA256SUMS"]="ae383f764c40899579dbdc2b8592d80ef1c5079e5b7149662803dab48a341c00"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m200_fc2_raw4_to_descriptor2_stable_compactor_directed_vcs.f \
    -top tb_m200_fc2_raw4_to_descriptor2_stable_compactor \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then exit 21; fi

set +e
"$task_run/simv" +ntb_random_seed=200025 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass='PASS M200 raw4-to-descriptor2 stable compactor VCS tokens=241 raw_packets=911 raw_beats=3643 descriptors=2305 descriptor_packets=1281 descriptor_stalls=324 raw_backpressure=326 simultaneous_push_pop=670 full4=69 zero_tokens=1 protocol_attacks=4 queue_depth=8 physical_speedup=false complete_fc2=false system_speedup=false headline=false'
grep -Fxq "$task_pass" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 32; fi
for task_cover in cp_raw4_all_nonzero cp_raw4_all_zero cp_descriptor2 \
        cp_window_boundary cp_descriptor_stall cp_raw_backpressure \
        cp_simultaneous_push_pop cp_zero_token_done \
        cp_bad_header_attack cp_bad_raw_attack; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M200_FC2_RAW4_TO_DESCRIPTOR2_STABLE_COMPACTOR_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=200025"
    echo "tokens=241"
    echo "raw_packets=911"
    echo "raw_beats=3643"
    echo "descriptors=2305"
    echo "descriptor_packets=1281"
    echo "descriptor_stalls=324"
    echo "raw_backpressure_cycles=326"
    echo "simultaneous_push_pop=670"
    echo "full4_nonzero_packets=69"
    echo "zero_tokens=1"
    echo "protocol_attacks=4"
    echo "queue_depth=8"
    echo "m199_stage_aware_abstract_speed=1.2413530123517351"
    echo "standalone_compactor=true"
    echo "integrated_window_frontend=false"
    echo "weight_sram_response=false"
    echo "complete_fc2=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m200_fc2_raw4_to_descriptor2_stable_compactor.sh" > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M200 FC2 raw4-to-descriptor2 stable compactor VCS sealed at $task_run"
