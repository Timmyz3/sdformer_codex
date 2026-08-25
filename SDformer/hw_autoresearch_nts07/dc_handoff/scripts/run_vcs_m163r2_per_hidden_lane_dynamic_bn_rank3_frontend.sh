#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m163r2_per_hidden_lane_dynamic_bn_rank3_frontend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M163r2 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m163/m163_q8_dynamic_bn_rank3_frontend.sv"]="f5ca30a590524ce8d185ff6b672514286b8915fe390052fa45bc58d1cc0e4fac"
    ["verif_m163/m163_q8_dynamic_bn_rank3_frontend_assertions.sv"]="e93e46d04710ff7df687760dd27fafb1e76cdaa86fa07e563e439c7254a8b9cc"
    ["tb_m163/tb_m163_q8_dynamic_bn_rank3_frontend.sv"]="2e220ba414c3682868798672506dd484ed371cbc194f19d924ef682b5e695117"
    ["dc_handoff/filelists/date_m163_q8_dynamic_bn_rank3_frontend_directed_vcs.f"]="aedef7a5cd1b5c3016e7c487e53fb9e404f5bbe50fc4f9ac164fe393becdf6e2"
    ["contracts/m163r2_per_hidden_lane_dynamic_bn_rank3_frontend_vcs_contract_r1_20260824.json"]="79a78997e6d16b6502ee8bd63a529800519aed763cb19f13570791322ef3f56b"
    ["contracts/m163_r1_cross_hidden_lane_moment_correction_overlay_r1_20260824.json"]="22f2696ab95e7fc4b746e7fe58335cbdb0d80f465e3acb4e204407d530dfc4cc"
    ["contracts/m163_r1_dc_worst_path_parser_correction_overlay_r1_20260824.json"]="813ce1fcb028335aab1b1613adc0ca647eaf5f6457202624dfae94c507c29408"
    ["contracts/m161_r2_independent_review_admission_overlay_r1_20260824.json"]="458e5eb7f2332ba040110afad423bbbd32701dfdd71564bca8ec381d20580502"
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
    -f dc_handoff/filelists/date_m163_q8_dynamic_bn_rank3_frontend_directed_vcs.f \
    -top tb_m163_q8_dynamic_bn_rank3_frontend \
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
task_pass='PASS M163r2 per-hidden-lane dynamic-BN rank3 frontend VCS channels=21 tiles=61 input_beats=305 q8_samples=9760 signed_products=29280 squares=9760 rank_results=61 moment_results=21 moment_state_lanes=16 moment_samples_per_lane_total=610 rank_stall_cycles=7 moment_stall_cycles=9 input_gap_cycles=306 protocol_attacks=1 product_slots=96 square_issue_lanes=32 requant_lanes=16 input_tile_ii_accepted_cycles=5 coefficient_generation=false atlif=false left_projection=false fc2=false network_accuracy=false physical_speedup=false system_speedup=false headline=false'
grep -Fqx "$task_pass" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_five_beat_tile cp_rank_stall_then_accept \
        cp_moment_stall_then_accept cp_negative_128_input \
        cp_positive_127_input cp_channel_last_tile \
        cp_distinct_hidden_lane_moments \
        cp_fault_with_pending_outputs; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done

{
    echo "status=PASS_M163R2_PER_HIDDEN_LANE_DYNAMIC_BN_RANK3_FRONTEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "channels=21"
    echo "tiles=61"
    echo "input_beats=305"
    echo "q8_samples=9760"
    echo "per_hidden_lane_samples_total=610"
    echo "signed_products=29280"
    echo "squares=9760"
    echo "rank_results=61"
    echo "moment_results=21"
    echo "independent_hidden_channel_moment_states=16"
    echo "cross_hidden_lane_moment_reduction=false"
    echo "rank_stall_cycles=7"
    echo "moment_stall_cycles=9"
    echo "input_gap_cycles=306"
    echo "protocol_attacks=1"
    echo "signed_int8_product_slots=96"
    echo "square_issue_lanes=32"
    echo "shared_rne_saturating_requant_lanes=16"
    echo "input_tile_ii_accepted_cycles=5"
    echo "factor_row_sum_forwarded=true"
    echo "accepted_outputs_survive_younger_fault=true"
    echo "q8_early_requant_network_accuracy=false"
    echo "dynamic_bn_coefficient_generation=false"
    echo "atlif=false"
    echo "rank3_left_projection=false"
    echo "fc2=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m163r2_per_hidden_lane_dynamic_bn_rank3_frontend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M163r2 per-hidden-lane dynamic-BN rank3 frontend VCS sealed at $task_run"
