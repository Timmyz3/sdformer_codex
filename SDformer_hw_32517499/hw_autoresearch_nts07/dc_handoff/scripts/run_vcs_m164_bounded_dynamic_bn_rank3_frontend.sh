#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m164_bounded_dynamic_bn_rank3_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M164 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m164/m164_q8_bounded_dynamic_bn_rank3_frontend.sv"]="fd103faaebeee2a99535edf11402525e0b2bcdcb17e78d8451d9d9bf1dc2d5ae"
    ["verif_m164/m164_q8_bounded_dynamic_bn_rank3_frontend_assertions.sv"]="bbffb1b152ce917001dd90e5bc8579794050012d5935edc6b6667c4800745916"
    ["tb_m164/tb_m164_q8_bounded_dynamic_bn_rank3_frontend.sv"]="d49249ccadeccbf1536c7e49f48b2eb2a4997799d2c7e839f22b13c4d386a404"
    ["dc_handoff/filelists/date_m164_q8_bounded_dynamic_bn_rank3_frontend_directed_vcs.f"]="7c9857d94389e35f1315c1e58f1cccbb4af73700e5b1d5c071b29290d4f1f56e"
    ["contracts/m164_bounded_dynamic_bn_rank3_frontend_vcs_contract_r1_20260824.json"]="12452ebdbb9fe85b2142ed661147d67b5f76cf6bbb6af065db5b538351de808a"
    ["results/m163r2_independent_hammer_review_r1_20260824/README.md"]="f945d2d8b4624b1dcae22d7ba6b897127e5b90eff3516e6378fc638b3f809c6c"
    ["results/m163r2_independent_hammer_review_r1_20260824/source_manifest.sha256"]="63d4f22f6b84acc760d6732230d982fc3dd3becbdb464402734dbe4c27c61793"
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
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m164_q8_bounded_dynamic_bn_rank3_frontend_directed_vcs.f \
    -top tb_m164_q8_bounded_dynamic_bn_rank3_frontend \
    -o "$task_run/simv" > "$task_run/compile.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.raw.log"; then
    exit 21
fi

set +e
"$task_run/simv" +ntb_random_seed=1 -no_save \
    -assert report="$task_run/assert.report" \
    -cm line+cond+tgl+fsm+assert > "$task_run/sim.raw.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 30
task_pass_regex='^PASS M164 bounded-width per-hidden-lane dynamic-BN rank3 frontend VCS channels=25 tiles=19264 input_beats=96320 q8_samples=3082240 signed_products=9246720 squares=3082240 rank_results=19264 moment_results=25 moment_state_lanes=16 max_samples_per_lane=192000 max_population_exercised=true exact_max_negative_sum=-24576000 exact_max_sumsq=3145728000 sum_bits=26 sumsq_bits=32 count_bits=18 projection_bits=19 moment_samples_per_lane_total=192650 explicit_rne_half_even_checks=12 explicit_saturation_checks=6 explicit_shift23_checks=3 rank_stall_cycles=[1-9][0-9]* moment_stall_cycles=[1-9][0-9]* input_gap_cycles=[0-9]+ protocol_attacks=1 product_slots=96 square_issue_lanes=32 requant_lanes=16 input_tile_ii_accepted_cycles=5 coefficient_generation=false atlif=false left_projection=false fc2=false network_accuracy=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_five_beat_tile cp_rank_stall_then_accept \
        cp_moment_stall_then_accept cp_negative_128_input \
        cp_positive_127_input cp_channel_last_tile \
        cp_distinct_hidden_lane_moments \
        cp_positive_and_negative_half_ties \
        cp_positive_and_negative_saturation \
        cp_shift23_rounds_to_zero \
        cp_exact_h67_max_population_and_worst_q8_moments \
        cp_fault_with_pending_outputs; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_rank_stalls="$(sed -n 's/.* rank_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_moment_stalls="$(sed -n 's/.* moment_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_input_gaps="$(sed -n 's/.* input_gap_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M164_BOUNDED_DYNAMIC_BN_RANK3_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "channels=25"
    echo "tiles=19264"
    echo "input_beats=96320"
    echo "q8_samples=3082240"
    echo "per_hidden_lane_samples_total=192650"
    echo "signed_products=9246720"
    echo "squares=3082240"
    echo "rank_results=19264"
    echo "moment_results=25"
    echo "maximum_samples_per_hidden_lane=192000"
    echo "maximum_population_exercised=true"
    echo "maximum_population_sum=-24576000"
    echo "maximum_population_sumsq=3145728000"
    echo "sum_bits=26"
    echo "sumsq_bits=32"
    echo "count_bits=18"
    echo "projection_bits=19"
    echo "explicit_rne_half_even_checks=12"
    echo "explicit_saturation_checks=6"
    echo "explicit_shift23_checks=3"
    echo "rank_stall_cycles=$task_rank_stalls"
    echo "moment_stall_cycles=$task_moment_stalls"
    echo "input_gap_cycles=$task_input_gaps"
    echo "protocol_attacks=1"
    echo "signed_int8_product_slots=96"
    echo "square_issue_lanes=32"
    echo "shared_rne_saturating_requant_lanes=16"
    echo "input_tile_ii_accepted_cycles=5"
    echo "tile_channel_start_on_moment_add_path=false"
    echo "accepted_outputs_survive_younger_fault=true"
    echo "fc1_to_q8_early_requant_implemented=false"
    echo "checkpoint_factors_and_scales=false"
    echo "full_dynamic_bn_barrier=false"
    echo "dynamic_bn_coefficient_generation=false"
    echo "atlif=false"
    echo "rank3_left_projection=false"
    echo "fc2=false"
    echo "paft_valid825=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m164_bounded_dynamic_bn_rank3_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M164 bounded dynamic-BN rank3 frontend VCS sealed at $task_run"
