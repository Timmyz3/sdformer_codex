#!/usr/bin/env bash
set -euo pipefail

task_review="reviews/m123_w384_signed19_forwarding_accumulator_independent_hammer_r1_20260824"
task_sealed_run="$task_review/sealed_vcs_rerun"
task_hammer_run="$task_review/independent_vcs"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_sealed_run" || -e "$task_hammer_run" ]]; then
    echo "refusing to overwrite M123 independent evidence" >&2
    exit 2
fi
if [[ ! -x "$task_vcs/bin/vcs" ]]; then
    echo "M123 independent hammer requires commercial VCS" >&2
    exit 3
fi

declare -A task_expected=(
    ["rtl_m123/m123_w384_signed19_forwarding_accumulator_frontend.sv"]="7729848c8172b9f3f768cac1b6ce3bf310b9f9b1a1e8def8ea3725c4b7356adc"
    ["rtl_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_adapter.sv"]="a040675cb03f69edeb24e321ea3e163f49c9c9eadebb08f7c0c94ce1dbd963e7"
    ["verif_m123/m123_w384_signed19_forwarding_lane_sliced_accumulator_assertions.sv"]="2e4333d7a19f1adfa11f28d0a5ee1ee49efccd32711ea83b845c76032b45137f"
    ["tb_m123/tb_m123_w384_signed19_forwarding_lane_sliced_accumulator.sv"]="7a198caed3e0cb90eb9a07db2fe5168826681795d4fd5717f071a506917a4a58"
    ["dc_handoff/filelists/date_m123_w384_signed19_forwarding_lane_accumulator_directed_vcs.f"]="7072f0a32a2efe78d9690adef462fdd70f7c3e07c1aaa55253f0d2e8e2eaaacb"
    ["contracts/m123_w384_signed19_forwarding_accumulator_vcs_contract_r1_20260824.json"]="63432933d974b277453545118ac02f5d8a803987f8102982e56ee70177eb3f87"
    ["reviews/m120_integrated_pwp_tail_mapper_signed19_accumulator_independent_hammer_r1_20260824/manifest.sha256"]="51ad53084fd73b64c3e7bf902ea72313bf0f4df660adaf4124c08cb2cb8116f1"
    ["$task_review/tb_m123_independent_hammer.sv"]="b3a723e52714de99c2d7bd35a12941cb9fb24715bc7de39a9973bf5f3b9c90c3"
    ["$task_review/m123_independent.f"]="f96f8a3439f238a3a8ad5b986e457cd37091ef4ab8494e011c031724cfbebbd2"
)
mkdir "$task_sealed_run" "$task_hammer_run"
: > "$task_review/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_actual="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s actual=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_actual" \
        >> "$task_review/preflight_sha_checks.txt"
    if [[ "$task_actual" != "${task_expected[$task_path]}" ]]; then
        echo "M123 independent frozen SHA mismatch: $task_path" >&2
        exit 10
    fi
done

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_sealed_run/csrc" \
    -f dc_handoff/filelists/date_m123_w384_signed19_forwarding_lane_accumulator_directed_vcs.f \
    -top tb_m123_w384_signed19_forwarding_lane_sliced_accumulator \
    -o "$task_sealed_run/simv" > "$task_sealed_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_sealed_run/simv" ]]; then exit 20; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_sealed_run/compile.raw.log"; then exit 21; fi

set +e
"$task_sealed_run/simv" -no_save \
    -assert report="$task_sealed_run/assert.report" \
    -cm line+cond+tgl+fsm+assert -cm_dir "$task_sealed_run/simv.vdb" \
    > "$task_sealed_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_sealed_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 22; fi
grep -qx 'PASS M123 W384 forwarding lane-sliced accumulator VCS windows=3 updates=1072 vector_lane_checks=884736 commits=9216 lazy_valid_clears=3 positive_memory_writes=1072 ii1_pairs=1069 same_address_accept_pairs=15 same_address_forward_read_suppressed=15 same_address_chain=16 read_write_overlap=1054 commit_stalls=1101 overflow_attacks=2 lanes=96 vector_bits=1824 accumulator_bytes=700416 valid_bits=3072 lane_macros=96 macro_depth=3072 macro_width=19 behavioral_macro=true same_address_rdw_mode_independent=true overflow_guard=true reset_recovery=false scheduled_cycle_ratio=false physical_speedup=false system_speedup=false headline=false' "$task_sealed_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_sealed_run/sim.raw.log" "$task_sealed_run/assert.report"; then exit 23; fi
for task_cover in \
        'cp_update_ii1, .* 1073 match' \
        'cp_same_address_forward_chain, .* 14 match' \
        'cp_read_write_overlap, .* 1058 match' \
        'cp_commit_stall, .* 1050 match' \
        'cp_full_commit, .* 3 match' \
        'cp_fault, .* 2 match'; do
    grep -Eq "$task_cover" "$task_sealed_run/assert.report"
done

set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_hammer_run/csrc" \
    -f "$task_review/m123_independent.f" \
    -top tb_m123_independent_hammer \
    -o "$task_hammer_run/simv" > "$task_hammer_run/compile.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer_run/compile.rc"
if [[ "$task_rc" -ne 0 || ! -x "$task_hammer_run/simv" ]]; then exit 30; fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_hammer_run/compile.raw.log"; then exit 31; fi

set +e
"$task_hammer_run/simv" -no_save \
    -assert report="$task_hammer_run/assert.report" \
    -cm line+cond+tgl+fsm+assert -cm_dir "$task_hammer_run/simv.vdb" \
    > "$task_hammer_run/sim.raw.log" 2>&1
task_rc="$?"
set -e
printf '%s\n' "$task_rc" > "$task_hammer_run/sim.rc"
if [[ "$task_rc" -ne 0 ]]; then exit 32; fi
grep -Eq '^PASS M123 independent hammer commercial_vcs=true positive_windows=2 positive_updates=16 positive_writes=16 positive_write_lane_checks=1536 commits=6144 commit_lane_checks=589824 same_address_pairs=6 same_address_reads_suppressed=6 original_m120_two_event_closed=true aaa_chains=1 aba_chains=1 new_invalid_row=1 existing_row=1 different_bank_same_row=1 mixed_sign_delta=1 signed19_nonoverflow_boundaries=2 forwarded_overflow_attacks=2 invalid_row_attacks=1 one_cycle_sync_macro_poisoned_no_read=true pending_sum_data_exact=true end_commit_full_numeric=true commit_stalls=[1-9][0-9]* reset_edge_write_enable=1 reset_edge_accept=1 reset_physical_writes=1 reset_quiescence=false reset_recovery=false foundry_macro=false physical_speedup=false system_speedup=false headline=false$' "$task_hammer_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog' \
        "$task_hammer_run/sim.raw.log" "$task_hammer_run/assert.report"; then exit 33; fi
for task_cover in \
        'cp_update_ii1, .* [1-9][0-9]* match' \
        'cp_same_address_forward_chain, .* [1-9][0-9]* match' \
        'cp_read_write_overlap, .* [1-9][0-9]* match' \
        'cp_commit_stall, .* [1-9][0-9]* match' \
        'cp_full_commit, .* 2 match' \
        'cp_fault, .* 3 match'; do
    grep -Eq "$task_cover" "$task_hammer_run/assert.report"
done

# Negative boundary: the interface has no response-valid/tag and therefore is
# explicitly fixed to a one-cycle synchronous macro. A two-cycle model must be
# detected by the independent pending-sum scoreboard and must not print PASS.
set +e
"$task_hammer_run/simv" +MACRO_DELAY2 -no_save \
    -assert report="$task_hammer_run/macro_latency2_negative.assert.report" \
    -cm line+cond+tgl+fsm+assert \
    -cm_dir "$task_hammer_run/macro_latency2_negative.vdb" \
    > "$task_hammer_run/macro_latency2_negative.raw.log" 2>&1
task_delay_rc="$?"
set -e
printf '%s\n' "$task_delay_rc" > "$task_hammer_run/macro_latency2_negative.rc"
if grep -q '^PASS M123 independent hammer' \
        "$task_hammer_run/macro_latency2_negative.raw.log"; then exit 40; fi
grep -q '^Fatal:' "$task_hammer_run/macro_latency2_negative.raw.log"
grep -q 'M123 hammer forwarded/pending sum mismatch' \
    "$task_hammer_run/macro_latency2_negative.raw.log"

{
    echo 'status=PASS_M123_INDEPENDENT_FORWARDING_AND_RESET_BOUNDARY_HAMMER'
    echo 'sealed_rerun=true'
    echo 'original_m120_two_event_closed=true'
    echo 'positive_updates=16'
    echo 'positive_writes=16'
    echo 'positive_write_lane_checks=1536'
    echo 'aaa_chains=1'
    echo 'aba_chains=1'
    echo 'new_and_existing_row=true'
    echo 'different_bank_same_row=true'
    echo 'forwarded_overflow_attacks=2'
    echo 'invalid_row_attacks=1'
    echo 'commit_lane_checks=589824'
    echo 'macro_one_cycle_required=true'
    echo 'macro_two_cycle_negative_detected=true'
    echo 'reset_edge_write_enable=1'
    echo 'reset_edge_accept=1'
    echo 'reset_quiescence=false'
    echo 'reset_recovery=false'
    echo 'foundry_macro=false'
    echo 'physical_speedup=false'
    echo 'system_speedup=false'
    echo 'headline=false'
} > "$task_review/RUN_COMPLETE.txt"

sha256sum \
    "$task_review/preflight_sha_checks.txt" \
    "$task_sealed_run/compile.raw.log" \
    "$task_sealed_run/sim.raw.log" \
    "$task_sealed_run/assert.report" \
    "$task_hammer_run/compile.raw.log" \
    "$task_hammer_run/sim.raw.log" \
    "$task_hammer_run/assert.report" \
    "$task_hammer_run/macro_latency2_negative.raw.log" \
    "$task_hammer_run/macro_latency2_negative.assert.report" \
    "$task_review/RUN_COMPLETE.txt" \
    > "$task_review/vcs_output.sha256"
echo "PASS M123 independent hammer and boundary characterization"
