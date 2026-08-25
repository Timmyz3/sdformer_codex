#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="$DC_HANDOFF_ROOT/runs/m99_phase_slack_vcs_r1_sealed_20260824"
M83_RECORDS="/tmp/m85_inputs/m83_cap11_phase_records.bin"
M83_OFFSETS="/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin"
VCS_ROOT="/opt/synopsys/vcs/V-2023.12-SP1"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M99 sealed VCS run: $RUN_DIR" >&2
    exit 2
fi
for external in "$M83_RECORDS" "$M83_OFFSETS"; do
    if [[ ! -f "$external" ]]; then
        echo "missing M99 external input: $external" >&2
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
RTL="rtl_m99/phase_slack_guarded_wordpacked_pwp_stream.sv"
SVA="verif_m99/phase_slack_guarded_wordpacked_pwp_stream_assertions.sv"
TB_DIRECTED="tb_m99/tb_m99_phase_slack_guarded_wordpacked_pwp_stream.sv"
TB_ACTUAL="tb_m99/tb_m99_phase_slack_actual_records.sv"
FILELIST="dc_handoff/filelists/date_m99_phase_slack_vcs.f"
METADATA="results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"
CONTRACT="contracts/m99_phase_slack_metadata_compiler_vcs_contract_r1_20260824.json"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$M85"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["$RTL"]="adb2dfd95ee3dd179cb373eb5ead937d9beb4db25648325634ebba755243b082"
    ["$SVA"]="461cafca5614231216652bc69a27c997e28f0d331b7a3dd958726d9297bb48de"
    ["$TB_DIRECTED"]="14eeb00be94d1338aefb37190e53f81ff03edd6eb2f98eef48515433f5843aff"
    ["$TB_ACTUAL"]="a3a2987164565659c8fe86aac8584651fba808eade00cbefcfa200a8fa1b3167"
    ["$FILELIST"]="12bcb401f2779407fed42577476c8c456eaff85f742daca31f259205a0ab1975"
    ["$METADATA"]="52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0"
    ["$CONTRACT"]="a89fde382fb19b639523a0b2d0b4500b498794a09ec960a529c25c390324c420"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$M85" "$RTL" "$SVA" "$TB_DIRECTED" "$TB_ACTUAL" \
        "$FILELIST" "$METADATA" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M99 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
records_sha="$(sha256sum "$M83_RECORDS" | awk '{print $1}')"
offsets_sha="$(sha256sum "$M83_OFFSETS" | awk '{print $1}')"
if [[ "$records_sha" != "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d" ]]; then
    echo "M99 M83 records SHA mismatch" >&2
    exit 11
fi
if [[ "$offsets_sha" != "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c" ]]; then
    echo "M99 M83 offsets SHA mismatch" >&2
    exit 12
fi
sha256sum "$M82" "$M85" "$RTL" "$SVA" "$TB_DIRECTED" "$TB_ACTUAL" \
    "$FILELIST" "$METADATA" "$CONTRACT" > "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' "$M83_RECORDS" "$records_sha" \
    "$(stat -c %s "$M83_RECORDS")" >> "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' "$M83_OFFSETS" "$offsets_sha" \
    "$(stat -c %s "$M83_OFFSETS")" >> "$RUN_DIR/input_sha256.txt"

export VCS_HOME="$VCS_ROOT" VCS_ARCH_OVERRIDE=linux
set +e
"$VCS_ROOT/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$RUN_DIR/csrc_directed" \
    -f "$FILELIST" -top tb_m99_phase_slack_guarded_wordpacked_pwp_stream \
    -o "$RUN_DIR/simv_directed" > "$RUN_DIR/compile_directed.raw.log" 2>&1
compile_directed_rc="$?"
set -e
printf '%s\n' "$compile_directed_rc" > "$RUN_DIR/compile_directed.rc"
if [[ "$compile_directed_rc" -ne 0 || ! -x "$RUN_DIR/simv_directed" ]]; then
    echo "M99 directed VCS compile failed rc=$compile_directed_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile_directed.raw.log"; then
    echo "M99 directed compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv_directed" -no_save \
    -assert report="$RUN_DIR/assert_directed.report" \
    -cm line+cond+tgl+fsm+assert > "$RUN_DIR/sim_directed.raw.log" 2>&1
sim_directed_rc="$?"
set -e
printf '%s\n' "$sim_directed_rc" > "$RUN_DIR/sim_directed.rc"
if [[ "$sim_directed_rc" -ne 0 ]]; then
    echo "M99 directed VCS simulation failed rc=$sim_directed_rc" >&2
    exit 30
fi
grep -qx 'PASS M99 M85-differential entries=128 beats=436 parser_cycles=640 stalls=10 poison_attacks=3 early_lookup_attacks=1 simultaneous_unloaded_attacks=1 simultaneous_loaded_priority_attacks=1' \
    "$RUN_DIR/sim_directed.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim_directed.raw.log"; then
    echo "M99 directed functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_phase_load cp_simultaneous_load_lookup \
        cp_loaded_lookup_priority cp_parser_first_entry cp_parser_middle_entry \
        cp_parser_final_entry cp_lookup_stall cp_escape cp_width9 cp_width10 \
        cp_width11 cp_metadata_error; do
    grep -Eq "$cover.*[1-9][0-9]* match" "$RUN_DIR/sim_directed.raw.log"
done

set +e
"$VCS_ROOT/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -cm line+cond+tgl+fsm+assert -Mdir="$RUN_DIR/csrc_actual" \
    -f "$FILELIST" -top tb_m99_phase_slack_actual_records \
    -o "$RUN_DIR/simv_actual" > "$RUN_DIR/compile_actual.raw.log" 2>&1
compile_actual_rc="$?"
set -e
printf '%s\n' "$compile_actual_rc" > "$RUN_DIR/compile_actual.rc"
if [[ "$compile_actual_rc" -ne 0 || ! -x "$RUN_DIR/simv_actual" ]]; then
    echo "M99 actual-record VCS compile failed rc=$compile_actual_rc" >&2
    exit 40
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile_actual.raw.log"; then
    echo "M99 actual compile log contains warning/error signature" >&2
    exit 41
fi

set +e
"$RUN_DIR/simv_actual" -no_save \
    -assert report="$RUN_DIR/assert_actual.report" \
    +RECORDS_BIN="$M83_RECORDS" +OFFSETS_BIN="$M83_OFFSETS" \
    +METADATA_BIN="$METADATA" -cm line+cond+tgl+fsm+assert \
    > "$RUN_DIR/sim_actual.raw.log" 2>&1
sim_actual_rc="$?"
set -e
printf '%s\n' "$sim_actual_rc" > "$RUN_DIR/sim_actual.rc"
if [[ "$sim_actual_rc" -ne 0 ]]; then
    echo "M99 actual-record VCS simulation failed rc=$sim_actual_rc" >&2
    exit 50
fi
grep -qx 'PASS M99 actual-record differential phases=1728 entries=221184 outputs=221184 escape=1 beats=835383 address_checks=835383 masked_nonzero_words=733459 ii_checks=219456 parser_cycles=221568 poison_attacks=3' \
    "$RUN_DIR/sim_actual.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim_actual.raw.log"; then
    echo "M99 actual functional/SVA failure signature found" >&2
    exit 51
fi
for cover in cp_phase_load cp_parser_first_entry cp_parser_middle_entry \
        cp_parser_final_entry cp_escape cp_width9 cp_width10 cp_width11 \
        cp_metadata_error; do
    grep -Eq "$cover.*[1-9][0-9]* match" "$RUN_DIR/sim_actual.raw.log"
done

{
    echo "status=PASS_M99_DIRECTED_AND_ACTUAL_RECORD_VCS_SVA"
    echo "exact_sha=true"
    echo "directed_entries=128"
    echo "actual_phases=1728"
    echo "actual_entries=221184"
    echo "actual_outputs=221184"
    echo "actual_beats=835383"
    echo "bank_address_checks=835383"
    echo "parser_edges_per_phase=128"
    echo "current_m86_zero_incremental_parser_cycles=false"
    echo "dc_admitted=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
sha256sum "$RUN_DIR"/*.raw.log "$RUN_DIR"/*.report \
    "$RUN_DIR"/RUN_COMPLETE.txt > "$RUN_DIR/output_sha256.txt"
run_complete=1
echo "PASS M99 directed plus actual-record VCS/SVA sealed at $RUN_DIR"
