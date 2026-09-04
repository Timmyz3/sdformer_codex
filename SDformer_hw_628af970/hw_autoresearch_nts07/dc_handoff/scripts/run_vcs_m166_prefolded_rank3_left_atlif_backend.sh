#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_dc_root/runs/m166r2_prefolded_rank3_left_atlif_backend_vcs_r1_sealed_20260824"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
if [[ -e "$task_run" ]]; then
    echo "refusing to overwrite M166r2 sealed VCS run: $task_run" >&2
    exit 2
fi
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
    ["rtl_m166/m166_q8_prefolded_rank3_left_atlif_backend.sv"]="9afaf28c92f344c8a1cc0126226579b842420bda4d48f8ddcc26458c86f2d646"
    ["verif_m166/m166_q8_prefolded_rank3_left_atlif_backend_assertions.sv"]="fd350b062a39bb0e5a40d988c1632b420afe908de48175bcbc1611221c58a1bb"
    ["tb_m166/tb_m166_q8_prefolded_rank3_left_atlif_backend.sv"]="9743562d2bde4cc1f82331b7865686dedf8871b72975dd586faf48b0c8c0c1db"
    ["dc_handoff/filelists/date_m166_q8_prefolded_rank3_left_atlif_backend_directed_vcs.f"]="777b71e902d117a37d5738b2c18f1582100707836fda0e8a14c850bdd51ec9b2"
    ["contracts/m166_prefolded_rank3_left_atlif_backend_vcs_contract_r1_20260824.json"]="e53e2a741d0784f6f6bd924b35392a44445cc24d439876c1fd2fa4c951daa0bf"
    ["dc_handoff/runs/m166_prefolded_rank3_left_atlif_backend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="8b6ceb4b41bfef5fd0da4aead76e81c31368bfb07eb09a201cf54d03a928d9be"
    ["dc_handoff/runs/m165r2_owned_raw_bank_dynamic_bn_rank3_frontend_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"]="3ed9d9dabfab1ea9f5f88ada2219f0572b483fb7ba7e343c53642702269efdc7"
    ["dc_handoff/runs/m165_owned_raw_bank_dynamic_bn_rank3_frontend_logic_only_dc_3p000ns_r1_sealed_20260824/RUN_COMPLETE.txt"]="e08705a7f7d5cb2f80292471a6b5cd41821ee03cca5766f359639832ed3ed9fd"
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
    -f dc_handoff/filelists/date_m166_q8_prefolded_rank3_left_atlif_backend_directed_vcs.f \
    -top tb_m166_q8_prefolded_rank3_left_atlif_backend \
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
task_pass_regex='^PASS M166 prefolded rank3-left ATLIF backend VCS tiles=241 output_beats=1205 signed_products=115680 product_slots=96 service_cycles_per_tile=5 steady_ii5_hits=6[0-9] input_push_release_overlap_cycles=[1-9][0-9]* output_stall_cycles=[1-9][0-9]* mixed_event_words=1205 protocol_attacks=1 folded_left_int8=true folded_bias_q24=true threshold_q24=true dense_reconstruction_materialized=false dynamic_bn_coefficient_generation=false epoch_rank_buffer=false fc2=false paft_valid825=false physical_speedup=false system_speedup=false headline=false$'
grep -Eq "$task_pass_regex" "$task_run/sim.raw.log" || exit 31
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$task_run/sim.raw.log" "$task_run/assert.report"; then
    exit 32
fi
for task_cover in cp_unstalled_five_cycle_tile \
        cp_back_to_back_five_cycle_tiles cp_input_push_release_same_cycle \
        cp_full_owned_input_fifo cp_event_stall_then_accept \
        cp_mixed_event_word cp_fault_after_configuration; do
    grep -Eq "$task_cover, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 33
done
task_pass="$(grep -E "$task_pass_regex" "$task_run/sim.raw.log")"
task_ii_hits="$(sed -n 's/.* steady_ii5_hits=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_overlap="$(sed -n 's/.* input_push_release_overlap_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"
task_stalls="$(sed -n 's/.* output_stall_cycles=\([0-9][0-9]*\) .*/\1/p' <<<"$task_pass")"

{
    echo "status=PASS_M166R2_CENTER_CORRECTED_PREFOLDED_RANK3_LEFT_ATLIF_BACKEND_VCS_SVA"
    echo "exact_sha=true"
    echo "random_seed=1"
    echo "tiles=241"
    echo "output_beats=1205"
    echo "signed_products=115680"
    echo "signed_int8_product_slots=96"
    echo "time_steps=10"
    echo "rank=3"
    echo "hidden_lanes=16"
    echo "time_rows_per_cycle=2"
    echo "service_cycles_per_tile=5"
    echo "steady_five_cycle_ii_hits=$task_ii_hits"
    echo "input_push_release_overlap_cycles=$task_overlap"
    echo "output_stall_cycles=$task_stalls"
    echo "configuration_bits_per_hidden_group=7704"
    echo "configuration_bytes_per_hidden_group=963"
    echo "rank_state_bits_per_tile=384"
    echo "event_bits_per_tile=160"
    echo "dense_reconstruction_materialized=false"
    echo "prefold_real_algebra=true"
    echo "folded_left_int8=true"
    echo "folded_bias_q24=true"
    echo "threshold_q24=true"
    echo "backend_cycle_defined=true"
    echo "backend_five_cycle_ii_is_not_end_to_end_ffn_ii=true"
    echo "m165_plus_m166_two_pass_arithmetic_cycles_per_tile=10"
    echo "dynamic_bn_coefficient_generation=false"
    echo "epoch_rank_buffer=false"
    echo "fc1_to_q8_early_requant_implemented=false"
    echo "fc2=false"
    echo "bn2=false"
    echo "residual_commit=false"
    echo "paft_valid825=false"
    echo "physical_speedup=false"
    echo "system_speedup=false"
    echo "paper_ppa_ready=false"
    echo "headline=false"
} > "$task_run/RUN_COMPLETE.txt"
sha256sum "$task_run"/compile.raw.log "$task_run"/sim.raw.log \
    "$task_run"/assert.report "$task_run"/RUN_COMPLETE.txt \
    > "$task_run/output_sha256.txt"
sha256sum "dc_handoff/scripts/run_vcs_m166_prefolded_rank3_left_atlif_backend.sh" \
    > "$task_run/runner_sha256.txt"
task_complete=1
echo "PASS M166r2 center-corrected prefolded rank3-left ATLIF backend VCS sealed at $task_run"
