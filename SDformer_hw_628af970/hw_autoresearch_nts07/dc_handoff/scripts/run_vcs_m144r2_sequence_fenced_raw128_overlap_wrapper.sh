#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M144r2 sealed VCS run: $task_run" >&2
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
declare -A task_expected=(
    ["rtl_m142/m142_sparse_mask_k4_three_bank_overlap_controller.sv"]="da80d61a4fe95bfd97ea50af388b48d924dcc0466836aa72f3809552d6c1915d"
    ["rtl_m144/m144_sequence_fenced_raw128_overlap_wrapper.sv"]="74a15a781c098a2d9a2a522fa97c93aeeb5db1d6eb9f8851882d22a26a18a6de"
    ["verif_m144/m144_sequence_fenced_raw128_overlap_wrapper_assertions.sv"]="c06be045fa0eb91350e9c4e4332cb4a2293c9e4cd13b2f27d13f2a8074f36209"
    ["tb_m144/tb_m144_sequence_fenced_raw128_overlap_wrapper.sv"]="b1922e7598d041ec3a447676797364912831109f3336226f653afb77d7f1bd95"
    ["dc_handoff/filelists/date_m144_sequence_fenced_raw128_overlap_wrapper_directed_vcs.f"]="f24c3a789bfdf3794595f545c5d747bffa5200813b2d7a91bb92f9f4487c9b78"
    ["contracts/m144r2_sequence_fenced_raw128_overlap_wrapper_vcs_contract_r1_20260824.json"]="d6d807fe0f71da20bbb87d21975ffc1147dc59f6c9987ab80aa64ee79b34c40f"
    ["contracts/m142_independent_review_correction_overlay_r1_20260824.json"]="9667c026b0dddd6eabfe6743087938d3855cdae98c6cfe16ef3a71ecb73ee929"
    ["contracts/m144_r1_dc_timing_loop_supersession_r1_20260824.json"]="b059b8664a1128a8f748e657c88cdc84225ae5a1e8641b3bc77c7fbc11782699"
    ["results/m142_independent_hammer_review_r1_20260824/manifest.sha256"]="336b8b205e81344bb692948201565da9fe1e327b855fd652045e6f29ff756679"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M144r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m144_sequence_fenced_raw128_overlap_wrapper_directed_vcs.f \
    -top tb_m144_sequence_fenced_raw128_overlap_wrapper \
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
task_pass='PASS M144 sequence-fenced wrapper VCS banks=4 jobs=5 rows=5 descriptors=8 pwp=5 correction=5 completed=5 barriers=1 commits=1 protocol_attacks=3 sequence_bits=32 exact_relative_rows=true post_fence_lookahead=true zero_work_endpoint_floor=true engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in cp_four_bank_lookahead cp_fence_blocks_post_sequence \
        cp_commit_then_post_fence_pwp cp_pwp_correction_overlap \
        cp_minimum_one_cycle_endpoint; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"
done

{
    echo "status=PASS_M144R2_SEQUENCE_FENCED_RAW128_OVERLAP_WRAPPER_VCS_SVA"
    echo "exact_sha=true"
    echo "banks=4"
    echo "raw_row_bits=128"
    echo "sequence_bits=32"
    echo "jobs=5"
    echo "accepted_rows=5"
    echo "accepted_descriptors=8"
    echo "pwp_launches=5"
    echo "correction_launches=5"
    echo "correction_completions=5"
    echo "barrier_accepts=1"
    echo "commit_accepts=1"
    echo "protocol_attacks=3"
    echo "exact_relative_rows=true"
    echo "post_fence_lookahead=true"
    echo "one_bit_fence_classification=true"
    echo "zero_work_endpoint_floor=true"
    echo "engine_arithmetic=false"
    echo "descriptor_result_sram_macro=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m144r2_sequence_fenced_raw128_overlap_wrapper.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M144r2 sequence-fenced raw128 overlap wrapper VCS sealed at $task_run"
