#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUNNER_SELF="$(realpath "${BASH_SOURCE[0]}")"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m86_r3_phase_fsm_actual_records_vcs_r1_sealed_20260823}"
M83_RECORDS="${M83_RECORDS:-/tmp/m85_inputs/m83_cap11_phase_records.bin}"
M83_OFFSETS="${M83_OFFSETS:-/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86-R3 VCS run: $RUN_DIR" >&2
    exit 2
fi
for external in "$M83_RECORDS" "$M83_OFFSETS"; do
    if [[ ! -f "$external" ]]; then
        echo "missing M86-R3 external input: $external" >&2
        exit 3
    fi
done
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
run_complete=0
on_exit() {
    local rc="$?"
    if [[ "$run_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$rc"
        } > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"
M82="rtl_m82/zero_bubble_elastic_pwp_stream.sv"
M85="rtl_m85/guarded_wordpacked_pwp_stream.sv"
M86="rtl_m86/sync_banked_guarded_pwp_frontend.sv"
RTL="rtl_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend.sv"
SVA="verif_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend_assertions.sv"
TB_DIRECTED="tb_m86_r3/tb_phase_fsm_sync_banked_guarded_pwp_frontend.sv"
TB_ACTUAL="tb_m86_r3/tb_phase_fsm_sync_bank_actual_records_diff.sv"
FL_DIRECTED="dc_handoff/filelists/date_m86_r3_phase_fsm_sync_bank_vcs.f"
FL_ACTUAL="dc_handoff/filelists/date_m86_r3_phase_fsm_actual_records_diff_vcs.f"
CONTRACT="contracts/m86_r3_phase_fsm_actual_records_vcs_contract_r1_20260823.json"
METADATA="results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$M85"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["$M86"]="edb06b7f4e891d4b00c8b49ace547efdf8daf84dc19716c710a6a343dc97f781"
    ["$RTL"]="bd3d9ea0e4e2a2a98c0403442b9ff589af5f818756528249eec01d3c16333986"
    ["$SVA"]="e4befbfeff9c9e2b30c02f0bd2f48fd260848e4ca0c4c23fa8316f968f4b78c1"
    ["$TB_DIRECTED"]="6b8314dbca77f5ff2f8b43b87451b13419f94284fde8e4ad21b6ecfedb593e8b"
    ["$TB_ACTUAL"]="383b9cd59e5cee9056d1929a2700fda112b276ac826184435024ced9756c0e30"
    ["$FL_DIRECTED"]="0799d78b26f5ada92c4a59c4f9d77d057cd07fc57cfc19bc4b760abb8f63af95"
    ["$FL_ACTUAL"]="cf6f22da0522eda0098d735d2562637e09b2108540113182d28878ce39d2c082"
    ["$CONTRACT"]="519fdf647d1016a17cf51e6daeea73d4648e65735966c968bf57eb8ce0689e5f"
    ["$METADATA"]="52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$M85" "$M86" "$RTL" "$SVA" "$TB_DIRECTED" \
        "$TB_ACTUAL" "$FL_DIRECTED" "$FL_ACTUAL" "$CONTRACT" "$METADATA"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M86-R3 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
records_sha="$(sha256sum "$M83_RECORDS" | awk '{print $1}')"
offsets_sha="$(sha256sum "$M83_OFFSETS" | awk '{print $1}')"
[[ "$records_sha" == "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d" ]] \
    || { echo "M86-R3 records SHA mismatch" >&2; exit 11; }
[[ "$offsets_sha" == "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c" ]] \
    || { echo "M86-R3 offsets SHA mismatch" >&2; exit 12; }
sha256sum "$RUNNER_SELF" "$M82" "$M85" "$M86" "$RTL" "$SVA" \
    "$TB_DIRECTED" "$TB_ACTUAL" "$FL_DIRECTED" "$FL_ACTUAL" \
    "$CONTRACT" "$METADATA" > "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' \
    "$M83_RECORDS" "$records_sha" "$(stat -c %s "$M83_RECORDS")" \
    >> "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' \
    "$M83_OFFSETS" "$offsets_sha" "$(stat -c %s "$M83_OFFSETS")" \
    >> "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
compile_one() {
    local label="$1" filelist="$2" top="$3"
    set +e
    "$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
        -Mdir="$RUN_DIR/csrc_$label" -f "$filelist" -top "$top" \
        -o "$RUN_DIR/simv_$label" > "$RUN_DIR/compile_${label}.raw.log" 2>&1
    local rc="$?"
    set -e
    printf '%s\n' "$rc" > "$RUN_DIR/compile_${label}.rc"
    [[ "$rc" -eq 0 && -x "$RUN_DIR/simv_$label" ]] || return 1
    ! grep -Eiq 'Warning-\[|Error-\[|^Error' \
        "$RUN_DIR/compile_${label}.raw.log"
}
compile_one directed "$FL_DIRECTED" \
    tb_phase_fsm_sync_banked_guarded_pwp_frontend || exit 20
compile_one actual "$FL_ACTUAL" \
    tb_phase_fsm_sync_bank_actual_records_diff || exit 21

set +e
"$RUN_DIR/simv_directed" -no_save > "$RUN_DIR/sim_directed.raw.log" 2>&1
directed_rc="$?"
set -e
printf '%s\n' "$directed_rc" > "$RUN_DIR/sim_directed.rc"
[[ "$directed_rc" -eq 0 ]] || exit 30
grep -qx 'PASS M86-R3 phase-fsm triple_contention=3 payload_accepts=461 phase_accepts=1 descriptor_accepts=128 outputs=128 bank_issues=384 bank_responses=384 bounded_loader_wait=8 silent_deadlock=0' \
    "$RUN_DIR/sim_directed.raw.log"
for cover in cp_load_triple cp_commit_triple cp_execute_triple \
        cp_descriptor_128 cp_return_to_load; do
    grep -Eq "$cover.* [1-9][0-9]* match" "$RUN_DIR/sim_directed.raw.log"
done

set +e
"$RUN_DIR/simv_actual" -no_save \
    +RECORDS_BIN="$M83_RECORDS" +OFFSETS_BIN="$M83_OFFSETS" \
    +METADATA_BIN="$METADATA" > "$RUN_DIR/sim_actual.raw.log" 2>&1
actual_rc="$?"
set -e
printf '%s\n' "$actual_rc" > "$RUN_DIR/sim_actual.rc"
[[ "$actual_rc" -eq 0 ]] || exit 31
grep -qx 'PASS M86-R3 actual-record differential phases=1728 descriptors=221184 outputs=221184 beats=835383 escape=1 stress_phases=128 backpressure_cycles=5215 fifo_full_cycles=4900 r1_cycle_mismatches=0' \
    "$RUN_DIR/sim_actual.raw.log"
grep -Eq 'cp_descriptor_128.* 1728 match' "$RUN_DIR/sim_actual.raw.log"
grep -Eq 'cp_return_to_load.* 1727 match' "$RUN_DIR/sim_actual.raw.log"
for log in "$RUN_DIR/sim_directed.raw.log" "$RUN_DIR/sim_actual.raw.log"; do
    if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' "$log"; then
        echo "M86-R3 functional/SVA failure signature: $log" >&2
        exit 32
    fi
done

{
    echo "status=PASS_M86_R3_PHASE_FSM_ACTUAL_RECORD_DIFFERENTIAL_VCS_SVA"
    echo "claim_scope=explicit_three_channel_phase_fsm_and_r1_cycle_equivalence_only"
    echo "directed_triple_contention_states=3"
    echo "bounded_next_loader_wait_cycles=8"
    echo "actual_phases=1728"
    echo "actual_descriptors=221184"
    echo "actual_outputs=221184"
    echo "actual_bank_read_issues=835383"
    echo "actual_r1_cycle_mismatches=0"
    echo "compiled_sram_macro=false"
    echo "real_escape_fallback=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M86-R3 phase FSM directed and actual-record VCS/SVA sealed at $RUN_DIR"
