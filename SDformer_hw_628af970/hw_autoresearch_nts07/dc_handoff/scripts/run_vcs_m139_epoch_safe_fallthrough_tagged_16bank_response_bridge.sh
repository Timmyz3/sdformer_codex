#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M139 sealed VCS run: $task_run" >&2
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
task_m137="rtl_m137/m137_fallthrough_tagged_16bank_response_bridge.sv"
task_m139="rtl_m139/m139_epoch_safe_fallthrough_tagged_16bank_response_bridge.sv"
task_sva="verif_m139/m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_assertions.sv"
task_tb="tb_m139/tb_m139_epoch_safe_fallthrough_tagged_16bank_response_bridge.sv"
task_files="dc_handoff/filelists/date_m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_directed_vcs.f"
task_contract="contracts/m139_epoch_safe_fallthrough_tagged_16bank_response_bridge_vcs_contract_r1_20260824.json"
task_m137_overlay="contracts/m137_independent_review_reset_epoch_overlay_r1_20260824.json"
task_spec="results/m139_epoch_safe_protocol_design_review_r1_20260824/m139_epoch_safe_protocol_spec_r1.md"
task_matrix="results/m139_epoch_safe_protocol_design_review_r1_20260824/m139_directed_vcs_attack_matrix_r1.json"
task_review="results/m139_epoch_safe_protocol_design_review_r1_20260824/m139_epoch_safe_protocol_design_review_r1.json"
task_review_manifest="results/m139_epoch_safe_protocol_design_review_r1_20260824/immutable_manifest.sha256"
task_m137_vcs_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
task_m137_dc_receipt="dc_handoff/runs/m137_fallthrough_tagged_16bank_response_bridge_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"

declare -A task_expected=(
    ["$task_m137"]="e2b0a271728dc8c0f79ba3361f76df554ad61e6d6efaf11ae09ff89be9384af2"
    ["$task_m139"]="e1d4b1acd99d054137d43863802058f08c980a3559a3e2e55276aee9b2208c32"
    ["$task_sva"]="3871983c4200d73e34be435fce894c877d45fbb4273d07550f5dcb512dd69b17"
    ["$task_tb"]="a1ff333e3cd3890f2f6190f71d99a45d129def78763c56b433ba372555f8a542"
    ["$task_files"]="66463f63632be7aad8ddeb361a39f8212a8aa23374d4733affc40d50dcc7656b"
    ["$task_contract"]="42545bc1538cfd3c3c334f6256dfe58e8e502b4848dbaf354d4992d53bd7c54f"
    ["$task_m137_overlay"]="c1ba046da2ff3d0f38e17579ccfc476110a718237ab7177c4c6de5ed42e3d623"
    ["$task_spec"]="8fbf544d91698f1ba33d8c470098abd266311485e8e96cb1044418e25fd04746"
    ["$task_matrix"]="3774fb23849f7be10e2f2741eedc717bed1b7a06efb2311059e4894725f6ac12"
    ["$task_review"]="f98e74d7a71b769e78ad7d083f3923012e0c2cef0d01ed38fbf5013b5ee76ec8"
    ["$task_review_manifest"]="efe088a0b5e0158d758ed6673d1ba9898cda6927abe316eb8c50dd5e28ec51c6"
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
        echo "M139 exact-SHA preflight mismatch: $task_path" >&2
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
    -top tb_m139_epoch_safe_fallthrough_tagged_16bank_response_bridge \
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
task_pass='PASS M139 epoch-safe fallthrough tagged 16-bank bridge VCS requests=65667 outputs=65667 words=1050672 wrap_requests=65538 wrap_crossings=1 ii1=65657 flushes=7 initial_high_ack=1 stale_drain=1 completion_collision=1 post_completion=1 wrong_token=1 reset_pending=1 reset_skid=1 stalls=16 skid_cycles=15 flush_fsm_bits=2 normal_ii=1 delivery_latency=1 macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_minimum_flush, .* [1-9][0-9]* match' \
        'cp_initial_high_ack_rejected, .* [1-9][0-9]* match' \
        'cp_drain_response_dropped, .* [1-9][0-9]* match' \
        'cp_completion_collision, .* [1-9][0-9]* match' \
        'cp_post_completion_response, .* [1-9][0-9]* match' \
        'cp_first_token_zero, .* [1-9][0-9]* match' \
        'cp_contiguous_eight_requests, .* [1-9][0-9]* match' \
        'cp_skid_capture_release, .* [1-9][0-9]* match' \
        'cp_token_wrap, .* [1-9][0-9]* match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M139_EPOCH_SAFE_FALLTHROUGH_TAGGED_16BANK_RESPONSE_BRIDGE_VCS_SVA"
    echo "exact_sha=true"
    echo "requests=65667"
    echo "outputs=65667"
    echo "logical_word_checks=1050672"
    echo "natural_wrap_requests=65538"
    echo "token_wrap_crossings=1"
    echo "ii1_intervals=65657"
    echo "flush_handshakes=7"
    echo "initial_high_ack_rejections=1"
    echo "stale_response_drains=1"
    echo "completion_collision_attacks=1"
    echo "post_completion_response_attacks=1"
    echo "wrong_token_attacks=1"
    echo "reset_with_pending_attacks=1"
    echo "reset_with_skid_attacks=1"
    echo "flush_fsm_bits=2"
    echo "normal_ii=1"
    echo "delivery_latency_cycles=1"
    echo "actual_macro_flush_wrapper=false"
    echo "foundry_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m139_epoch_safe_fallthrough_tagged_16bank_response_bridge.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M139 epoch-safe fallthrough tagged 16-bank bridge VCS sealed at $task_run"
