#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m227_fc1_k8_masked_held_weight_slice_directed_vcs_r2_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M227 sealed VCS run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m227/m227_fc1_k8_masked_held_weight_slice.sv"]="939e3dc4dcdb20d0962fde84d0c8a8f576886b6f9a259f8a702130149e9bb1b0"
 ["verif_m227/m227_fc1_k8_masked_held_weight_slice_assertions.sv"]="d6feb5e77eb292009de5bb9331d8d7911b778c1047da2e7d99ea2cc6b404e13f"
 ["tb_m227/tb_m227_fc1_k8_masked_held_weight_slice.sv"]="0c6336d88b4a52f48b09e93943f92db4c1692889df8c9e0dd389baa7d1c9b8d8"
 ["dc_handoff/filelists/date_m227_fc1_k8_masked_held_weight_slice_rtl.f"]="faacf0204c3b05bb7b2b92d85e7eb7eb14e8d5e46053e75a440ea15371480ff1"
 ["dc_handoff/filelists/date_m227_fc1_k8_masked_held_weight_slice_directed_vcs.f"]="84d5787015a23ceb97a2ce27f3de96a6cc3e9db22900b78758daabcacf614047"
 ["contracts/m227_fc1_k8_masked_held_weight_slice_synopsys_contract_r1_20260825.json"]="2537092faf1bf46f9dc5632b2c67bb6b34aa0cfb3e825e784da8bb0ba71d06f5"
 ["results/m226_m225_capacity_matched_reference_correction_r1_20260825/SHA256SUMS"]="09f43c9ec47f9ae8276aeb814f881521aebcaa6af778c5844e66fa2b205c3568"
 ["results/m226_independent_hammer_review_r1_20260825/SHA256SUMS"]="6c77665dfbae822186fe656b9d746e4e94b9aaaf5f1c3311389389275fe3c898"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
for task_fanout in 1 2 4; do
    task_variant="$task_run/f${task_fanout}"
    mkdir "$task_variant"
    set +e
    "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
        +define+M227_FANOUT="$task_fanout" \
        -timescale=1ns/1ps -cm assert \
        -Mdir="$task_variant/csrc" \
        -f dc_handoff/filelists/date_m227_fc1_k8_masked_held_weight_slice_directed_vcs.f \
        -top tb_m227_fc1_k8_masked_held_weight_slice \
        -o "$task_variant/simv" > "$task_variant/compile.log" 2>&1
    task_rc=$?
    set -e
    echo "$task_rc" > "$task_variant/compile.rc"
    [[ "$task_rc" -eq 0 && -x "$task_variant/simv" ]] || exit 20
    grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_variant/compile.log" \
        && exit 21 || true

    set +e
    "$task_variant/simv" +ntb_random_seed="22702${task_fanout}" \
        -no_save -assert report="$task_variant/assert.report" -cm assert \
        > "$task_variant/sim.log" 2>&1
    task_rc=$?
    set -e
    echo "$task_rc" > "$task_variant/sim.rc"
    [[ "$task_rc" -eq 0 ]] || exit 22
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "$task_variant/sim.log" "$task_variant/assert.report" && exit 23 || true
    grep -Eq "PASS M227 F=${task_fanout} clean_groups=3 sources=38 updates=119 results=24 protocol_attacks=2 empty=1 signed=2 tail=2 full8=2 request_stalls=[1-9][0-9]* result_stalls=[1-9][0-9]* cycles=" \
        "$task_variant/sim.log" || exit 30
    for task_cover in cp_signed_scan cp_tail_source383 cp_request_stall \
            cp_result_stall cp_full_fanout cp_empty_group \
            cp_protocol_attack cp_done; do
        grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
            "$task_variant/assert.report" || exit 31
    done
done

{
    echo status=PASS_M227_FC1_K8_MASKED_HELD_WEIGHT_SLICE_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo variants=F1_F2_F4
    echo clean_groups_per_variant=3
    echo unique_sources_per_variant=38
    echo context_updates_per_variant=119
    echo result_beats_per_variant=24
    echo protocol_attacks_per_variant=2
    echo numeric_mismatches=0
    echo conservation_mismatches=0
    echo assertion_failures=0
    echo scanner_included=true
    echo mask_sign_storage_included=true
    echo held_weight_replay_included=true
    echo tagged_weight_backpressure_included=true
    echo complete_fc1=false
    echo complete_ffn=false
    echo macro_aware_ppa=false
    echo physical_speedup=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/m227_fc1_k8_masked_held_weight_slice_vcs_receipt_r1.txt"
sha256sum "$task_runner" > "$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name simv ! -path '*/csrc/*' \
    ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum \
    > "$task_run/SHA256SUMS"
printf 'PASS_M227_FC1_K8_MASKED_HELD_WEIGHT_SLICE_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M227 exact VCS sealed at $task_run"
