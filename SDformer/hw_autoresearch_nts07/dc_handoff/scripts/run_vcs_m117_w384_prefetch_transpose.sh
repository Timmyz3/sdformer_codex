#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m117_w384_prefetch_transpose_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M117 sealed VCS run: $task_run" >&2
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
task_rtl="rtl_m117/m117_w384_prefetch_transpose_scheduler.sv"
task_sva="verif_m117/m117_w384_prefetch_transpose_assertions.sv"
task_tb="tb_m117/tb_m117_w384_prefetch_transpose.sv"
task_files="dc_handoff/filelists/date_m117_w384_prefetch_transpose_directed_vcs.f"
task_contract="contracts/m117_w384_prefetch_transpose_vcs_contract_r1_20260824.json"
task_m113_review="reviews/m113_minimal_integrated_replay_arch_hammer_r1_20260824/m113_minimal_integrated_replay_arch_hammer_review.json"
task_m113_manifest="reviews/m113_minimal_integrated_replay_arch_hammer_r1_20260824/manifest.sha256"

declare -A task_expected=(
    ["$task_rtl"]="4e640770349fa2d95ac09731efe7f8587d8bb108bd89169c204200cf41f3983a"
    ["$task_sva"]="3778e478655cf19e56c4c23f577766324f461d20dcaa1f735add882579383d7d"
    ["$task_tb"]="8304144ed0e6b9673c59da8942a940d41b56776c145782db0fb9efef59e1cff8"
    ["$task_files"]="5396892fa91475fc605ba9f525419f4da95b59c93dbcf6a750a277b04a24eb37"
    ["$task_contract"]="b327f0e14d83ecf1df18fcbedb2d5986a1b53971b54a972892f6552b44ca1fef"
    ["$task_m113_review"]="5709290a3abdbe9f01cab32bd0eb82ce54b2316fa5f6f47f76433c28bd096945"
    ["$task_m113_manifest"]="95516b631a9384cc813d9cf5d2a02ca41d412e230d048fa25dfec39bbafa36b0"
)

: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M117 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f "$task_files" -top tb_m117_w384_prefetch_transpose \
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
grep -qx 'PASS M117 W384 prefetch transpose VCS windows=2 ingress_events=98304 active_keys=256 rows_per_key=384 weight_prefetches=256 zero_bubble_key_transitions=254 load_tokens=768 event_tokens=98304 service_tokens=99072 ii1_pairs=98302 service_stalls=9953 prefetch_stall_attack_cycles=3 descriptor_done=2 overlap_cycles=49152 close_grace=2 protocol_attacks=1 win_rows=384 bitmap_payload_bits=196608 accumulator_contract_bits=20 accumulator_implemented=false weight_payload_memory=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in \
        'cp_dispatch_prefetch, .* 3 match' \
        'cp_zero_bubble_next_key, .* 229 match' \
        'cp_descriptor_done, .* 3 match' \
        'cp_pingpong_overlap, .* 49152 match' \
        'cp_last_row, .* 256 match' \
        'cp_fault, .* 1 match'; do
    grep -Eq "$task_cover" "$task_run/assert.report"
done

{
    echo "status=PASS_M117_W384_PREFETCH_TRANSPOSE_DIRECTED_VCS_SVA"
    echo "exact_sha=true"
    echo "window_rows=384"
    echo "active_keys=256"
    echo "weight_prefetches=256"
    echo "zero_bubble_key_transitions=254"
    echo "prefetch_stall_attack_cycles=3"
    echo "descriptor_done_full_windows=2"
    echo "conditional_on_always_ready_prefetch=true"
    echo "weight_payload_memory=false"
    echo "shared_payload_arbiter=false"
    echo "numeric_mapper=false"
    echo "accumulator_integrated=false"
    echo "exact_heldout_integrated_replay=false"
    echo "m109_r2_projected_ratio=2.53546204172554"
    echo "scheduled_cycle_ratio=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m117_w384_prefetch_transpose.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M117 W384 prefetch transpose VCS sealed at $task_run"
