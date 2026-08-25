#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m86_sync_banked_guarded_pwp_vcs_r1_sealed_20260823}"
M83_RECORDS="${M83_RECORDS:-/tmp/m85_inputs/m83_cap11_phase_records.bin}"
M83_OFFSETS="${M83_OFFSETS:-/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86 VCS run: $RUN_DIR" >&2
    exit 2
fi
for external in "$M83_RECORDS" "$M83_OFFSETS"; do
    if [[ ! -f "$external" ]]; then
        echo "missing M86 external input: $external" >&2
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
RTL="rtl_m86/sync_banked_guarded_pwp_frontend.sv"
SVA="verif_m86/sync_banked_guarded_pwp_frontend_assertions.sv"
TB="tb_m86/tb_sync_banked_guarded_pwp_frontend.sv"
FILELIST="dc_handoff/filelists/date_m86_sync_banked_guarded_pwp_frontend_vcs.f"
CONTRACT="contracts/m86_sync_banked_guarded_pwp_frontend_vcs_contract_r1_20260823.json"
METADATA="results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$M85"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["$RTL"]="edb06b7f4e891d4b00c8b49ace547efdf8daf84dc19716c710a6a343dc97f781"
    ["$SVA"]="8733048482c677e77be88044d55215bea98d75007f9cb4b7aba83d23e6ce0dd3"
    ["$TB"]="1ae9433a031a772d215d4ac032255042cffb757cd8c0a9e4f3934a76641b1386"
    ["$FILELIST"]="4038e7629d90957e23ed36387f8cbdbf3c2e161df1955b91047cdeab13f25230"
    ["$CONTRACT"]="d7bb4929abca9d3f9562c3a7d85bdaa769734a877e064c50eaa6173fc519578a"
    ["$METADATA"]="52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$M85" "$RTL" "$SVA" "$TB" "$FILELIST" \
        "$CONTRACT" "$METADATA"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M86 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
records_sha="$(sha256sum "$M83_RECORDS" | awk '{print $1}')"
offsets_sha="$(sha256sum "$M83_OFFSETS" | awk '{print $1}')"
if [[ "$records_sha" != "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d" ]]; then
    echo "M86 M83 records SHA mismatch" >&2
    exit 11
fi
if [[ "$offsets_sha" != "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c" ]]; then
    echo "M86 M83 offsets SHA mismatch" >&2
    exit 12
fi
sha256sum "$M82" "$M85" "$RTL" "$SVA" "$TB" "$FILELIST" \
    "$CONTRACT" "$METADATA" > "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' \
    "$M83_RECORDS" "$records_sha" "$(stat -c %s "$M83_RECORDS")" \
    >> "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' \
    "$M83_OFFSETS" "$offsets_sha" "$(stat -c %s "$M83_OFFSETS")" \
    >> "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_sync_banked_guarded_pwp_frontend -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M86 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M86 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save \
    +RECORDS_BIN="$M83_RECORDS" +OFFSETS_BIN="$M83_OFFSETS" \
    +METADATA_BIN="$METADATA" > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M86 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M86 sync-bank actual-record replay phases=1728 descriptors=221184 outputs=221184 beats=835383 always_ready_ii_checks=203200 stress_phases=128 backpressure_cycles=5261 fifo_full_cycles=4940 duplicate_row_attacks=1' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M86 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_phase_load cp_fifo_backpressure cp_fifo_full cp_width8 \
        cp_width9 cp_width10 cp_width11 cp_escape cp_protocol_attack; do
    grep -Eq "$cover.* [1-9][0-9]* match" "$RUN_DIR/sim.raw.log"
done
{
    echo "status=PASS_M86_SYNC_BANK_ACTUAL_RECORD_VCS_SVA"
    echo "claim_scope=one_cycle_sync_bank_fifo_m85_m82_functional_only"
    echo "phases=1728"
    echo "descriptors=221184"
    echo "outputs=221184"
    echo "bank_read_issues=835383"
    echo "always_ready_ii_checks=203200"
    echo "random_backpressure_phases=128"
    echo "backpressure_cycles=5261"
    echo "fifo_full_cycles=4940"
    echo "compiled_sram_macro=false"
    echo "real_escape_fallback=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M86 synchronous-bank actual-record VCS/SVA sealed at $RUN_DIR"
