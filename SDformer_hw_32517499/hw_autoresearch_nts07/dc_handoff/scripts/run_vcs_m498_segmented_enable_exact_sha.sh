#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m498_segmented_enable_vcs_r1_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || { echo "refusing to overwrite M498 sealed VCS run" >&2; exit 2; }
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m498/m498_segmented_enable_parent_queue_pipeline.sv"]="95967b386e31427e48b4a8cae81af244ee7b68b47316915d95fa6e1e92978fba"
 ["rtl_m498/m498_segmented_enable_backpressure_safe_parent_queue_pipeline.sv"]="3f7c188df7325984dbf536faa6c09362a74d562ddbe5ffe13a80a042616954d7"
 ["verif_m476/m476_dual_slot_parent_queue_assertions.sv"]="a4a30988c0321624caaf5776995a783378a00c6a49ac3babc2dc4191afb9e0f0"
 ["verif_m476r2/m476r2_backpressure_safe_assertions.sv"]="ea8327e07b2793cad36324d52b064b5e079b8dec3a07ad0339fb5534d87fa5e8"
 ["tb_m498/tb_m498_segmented_enable_full_regression.sv"]="85e883e4d779cc43b0ecfb6b37bc671187d238264c8b4eda7f50435da7d311d8"
 ["tb_m498/tb_m498_segmented_enable_backpressure_targeted.sv"]="7e1f31b8e8775df590f8905e68ff0d0e99fa95c06fee5faca27abcadee16e63e"
 ["dc_handoff/filelists/date_m498_segmented_enable_full_regression_vcs.f"]="ba21860709ec4945bd6d4104881ca09b32b3af5afb3c64d0cb48ef1fd4dd8817"
 ["dc_handoff/filelists/date_m498_segmented_enable_targeted_vcs.f"]="47e06d7e1e2b137dd0f78e2aa16e42fc9d9cbe66053cf77246d50b93296e854e"
 ["contracts/m498_segmented_enable_parent_queue_logic_only_dc_contract_r1_20260827.json"]="87d77361232c637ac2b92d9ce75dfc9d1c632fbd0ba0a37f9e5719473cdc6600"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"
sha256sum "$task_runner" > "$task_run/runner_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
task_compile_and_run() {
    local task_name="$1" task_filelist="$2" task_top="$3"
    local task_subrun="$task_run/$task_name"
    mkdir "$task_subrun"
    set +e
    "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="$task_subrun/csrc" -f "$task_filelist" -top "$task_top" \
        -o "$task_subrun/simv" > "$task_subrun/compile.log" 2>&1
    local task_rc=$?
    set -e
    echo "$task_rc" > "$task_subrun/compile.rc"
    [[ "$task_rc" -eq 0 && -x "$task_subrun/simv" ]] || exit 20
    ! grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_subrun/compile.log" || exit 21
    set +e
    "$task_subrun/simv" +ntb_random_seed=498027 -no_save \
        -assert report="$task_subrun/assert.report" -cm assert \
        > "$task_subrun/sim.log" 2>&1
    task_rc=$?
    set -e
    echo "$task_rc" > "$task_subrun/sim.rc"
    [[ "$task_rc" -eq 0 ]] || exit 22
    ! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "$task_subrun/sim.log" "$task_subrun/assert.report" || exit 23
}

task_compile_and_run full \
    dc_handoff/filelists/date_m498_segmented_enable_full_regression_vcs.f \
    tb_m498_segmented_enable_full_regression
task_compile_and_run targeted \
    dc_handoff/filelists/date_m498_segmented_enable_targeted_vcs.f \
    tb_m498_segmented_enable_backpressure_targeted

grep -Fq 'PASS M498 segmented-enable full issues=6 rows=5 forward=1 reads=4 responses=4 dual_enqueue=1 full=2 fullconsume=2 stalls=9 b2b=2 exact=2 partialbeats=2 id_attacks=1 overflow_attacks=1' "$task_run/full/sim.log" || exit 30
grep -Fq 'PASS M498 segmented-enable stalled_raw_guard stalled=3 reads=0 forward=1 writes=2 child_checks=96 stale_mismatches=0 old=5 new=1' "$task_run/targeted/sim.log" || exit 31
for task_cover in cp_forward cp_macro_read cp_read_response cp_dual_enqueue \
        cp_queue_full cp_full_consume_no_prefetch_credit \
        cp_back_to_back_completion cp_output_stall cp_overflow_atomic_block; do
    grep -Eq "sva\.base\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/full/assert.report" || exit 40
done
for task_cover in cp_stalled_same_address_prefetch cp_release_to_new_value_forward; do
    grep -Eq "sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/targeted/assert.report" || exit 41
done

python3 - "$task_run" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
receipt = {
    "schema": "m498_segmented_enable_exact_vcs_receipt_v1",
    "status": "PASS_M498_SEGMENTED_ENABLE_EXACT_VCS",
    "exact_sha": True,
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 498027,
    "full_regression": {
        "issues": 6, "rows": 5, "forward": 1, "reads": 4,
        "responses": 4, "dual_enqueue": 1, "queue_full": 2,
        "full_consume": 2, "stalls": 9, "back_to_back": 2,
        "identity_attacks": 1, "overflow_attacks": 1,
    },
    "stale_raw_targeted": {
        "stalled_cycles": 3, "reads": 0, "forward": 1,
        "writes": 2, "child_checks": 96, "stale_mismatches": 0,
    },
    "rtl_mechanism": "zero-cycle 12x8-lane branch plus per-lane row/psum synthesis-only preserved BUFFD1 leaves",
    "functional_delta_vs_m476r2": False,
    "claim_boundary": {
        "directed_integrated_vcs": True,
        "zero_cycle_semantics": True,
        "explicit_physical_tree_after_dc": False,
        "timing": False, "area": False, "power": False,
        "system_speedup": False, "paper_ppa_ready": False,
        "headline": False,
    },
}
(root / "m498_segmented_enable_exact_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(root / "README.md").write_text(
    "# M498 segmented-enable exact VCS\n\n"
    "Exact-SHA Synopsys VCS passes the inherited full parent-queue regression "
    "and the stale-RAW/backpressure attack with all required SVA covers. VCS "
    "sees zero-latency Boolean buffers; DC must still prove the explicit "
    "TSMC28 BUFFD1 hierarchy survives and all five electrical reports pass.\n")
PY
printf 'PASS_M498_SEGMENTED_ENABLE_EXACT_VCS\n' > "$task_run/RUN_COMPLETE.txt"
(
    cd "$task_run"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M498 exact VCS sealed at $task_run"
