#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m146r2_four_bank_age_queue_scheduler_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"

if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M146r2 sealed VCS run: $task_run" >&2
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
    ["rtl_m146/m146_four_bank_age_queue_scheduler.sv"]="5853460af465bae704c2ed5dca4e365211af31d8595ecda0c9fb9bf031d9ef75"
    ["verif_m146/m146_four_bank_age_queue_scheduler_assertions.sv"]="59b619eaf316c30836f9fab9154102a0e18c0e9748eeb5367b0e77fd3868268e"
    ["tb_m146/tb_m146_four_bank_age_queue_scheduler.sv"]="95660fed1b6827242e4304ead991d57b3d17dad44aaf0f1936dfe7c9d258aeec"
    ["dc_handoff/filelists/date_m146_four_bank_age_queue_scheduler_directed_vcs.f"]="9e4c3696963c909dd215d3c3917d7d48ab19a6fb9a77f1079fad0809b4d5a0a7"
    ["contracts/m146r2_four_bank_age_queue_scheduler_vcs_contract_r1_20260824.json"]="11324ba7e92ae0273c8faac64745da920b8e5757a6df94cf5af9c141c05b8e0c"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$task_path" "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    if [[ "$task_observed" != "${task_expected[$task_path]}" ]]; then
        echo "M146r2 exact-SHA preflight mismatch: $task_path" >&2
        exit 10
    fi
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m146_four_bank_age_queue_scheduler_directed_vcs.f \
    -top tb_m146_four_bank_age_queue_scheduler \
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
task_pass='PASS M146r2 age-queue scheduler VCS banks=4 jobs=40 fills=40 pwp=40 correction=40 releases=40 pwp_stalls=11 correction_stalls=5 bank_reuses=36 protocol_attacks=4 reset_release_guard=true sequence_age_comparators=0 completion_identity_equality=true sequence_bits=32 pwp_correction_overlap=1 engine_arithmetic=false sram_macro=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then exit 31; fi
for task_cover in cp_all_banks_live cp_engines_overlap cp_pwp_queue_full \
        cp_pwp_to_correction_handoff cp_correction_release; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" "$task_run/assert.report"
done

{
    echo "status=PASS_M146R2_FOUR_BANK_AGE_QUEUE_SCHEDULER_VCS_SVA"
    echo "exact_sha=true"
    echo "banks=4"
    echo "sequence_bits=32"
    echo "jobs=40"
    echo "fills=40"
    echo "pwp_launches=40"
    echo "correction_launches=40"
    echo "releases=40"
    echo "bank_reuses=36"
    echo "protocol_attacks=4"
    echo "reset_release_guard=true"
    echo "maximum_allocations_per_identity_epoch=4294967295"
    echo "external_reset_flush_required=true"
    echo "full_identity_reuse_aba_unconditionally_closed=false"
    echo "sequence_age_comparators=0"
    echo "completion_identity_equality=true"
    echo "pwp_correction_overlap=true"
    echo "engine_arithmetic=false"
    echo "sram_macro=false"
    echo "drop_in_m142_equivalence=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m146r2_four_bank_age_queue_scheduler.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M146r2 four-bank age-queue scheduler VCS sealed at $task_run"
