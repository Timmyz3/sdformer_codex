#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m84_hierarchical_pwp_bank_mapper_vcs_r1_sealed_20260823}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M84 VCS run: $RUN_DIR" >&2
    exit 2
fi
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
RTL="rtl_m84/hierarchical_pwp_bank_mapper.sv"
SVA="verif_m84/hierarchical_pwp_bank_mapper_assertions.sv"
TB="tb_m84/tb_hierarchical_pwp_bank_mapper.sv"
FILELIST="dc_handoff/filelists/date_m84_hierarchical_pwp_bank_mapper_vcs.f"
EXPORTER="system_simulator/scripts/export_m84_hierarchical_pwp_geometry.py"
GEOMETRY="results/m84_hierarchical_pwp_geometry_r2_20260823/m84_phase_geometry.bin"
GEOMETRY_RECEIPT="results/m84_hierarchical_pwp_geometry_r2_20260823/m84_phase_geometry_receipt.json"
M82_REVIEW="reviews/m82_zero_bubble_pwp_stream_vcs_independent_hammer_r1_20260823/m82_independent_hammer.json"
M83_DECODE="reviews/m83_canonical_cap11_pwp_records_independent_hammer_r1_20260823/remote_independent_full_decode.json"
CONTRACT="contracts/m84_hierarchical_pwp_bank_mapper_vcs_contract_r1_20260823.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="8dafcf1e049dfee1a06999c93010a6e8c2458cc17c9a2de712b26d4fc40a2067"
    ["$SVA"]="c0454345f5e2640a10351f142faffa647e05699a140386d81aaa7faabf59870b"
    ["$TB"]="85fa11d271b7d10dee5b47cbabe890d34ea2320480a693f7971c676a6b785826"
    ["$FILELIST"]="00dc39c45b175558f457f2dab168f4e19e985c4f5c5d520b872d6d693631185e"
    ["$EXPORTER"]="b90afd4d5f7d91111ba54d08a9e91f5ae31d3b3c9809520b8d27b3c1d1313c76"
    ["$GEOMETRY"]="294ea28b95ca2ef5c4adcb77195aabd388fc5b0ebc16bc4a3affcb8800b18e5d"
    ["$GEOMETRY_RECEIPT"]="bcebdbe95e7f455add38cbf9781c804edaa923fe83a726d755ec20b45b6b8df1"
    ["$M82_REVIEW"]="0aee83d94557ae68c6957f2616482cb1f4674c8195cd9c2dcbf814bf28ffca55"
    ["$M83_DECODE"]="bf80065a8c1dbac10bbe94edfbfa48c04f2d4917d0e1fae7c0ee8f6ab1bf07c8"
    ["$CONTRACT"]="b5eea5505346eb208a7c8d5f3f96a2004ff26d6f26a53f6c00988537e2e5acd0"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$SVA" "$TB" "$FILELIST" "$EXPORTER" "$GEOMETRY" \
        "$GEOMETRY_RECEIPT" "$M82_REVIEW" "$M83_DECODE" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M84 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$SVA" "$TB" "$FILELIST" "$EXPORTER" "$GEOMETRY" \
    "$GEOMETRY_RECEIPT" "$M82_REVIEW" "$M83_DECODE" "$CONTRACT" \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_hierarchical_pwp_bank_mapper -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M84 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M84 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save +GEOMETRY_BIN="$GEOMETRY" \
    > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M84 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M84 exhaustive phases=1728 entries=221184 escape=1 beats=835382 cross_row=725103 invalid_attacks=1 metadata=74B_vs_256B' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M84 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_width8 cp_width9 cp_width10 cp_width11 cp_escape \
        cp_cross_row cp_invalid_code; do
    grep -q "$cover" "$RUN_DIR/sim.raw.log"
done
{
    echo "status=PASS_M84_FROZEN_CATALOG_EXHAUSTIVE_VCS_SVA"
    echo "claim_scope=hierarchical_address_and_interleaved_bank_mapper_only"
    echo "entries=221184"
    echo "regular_beats=835382"
    echo "cross_row_beats=725103"
    echo "metadata_bytes_per_phase=74"
    echo "physical_sram_response=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M84 exhaustive VCS/SVA sealed at $RUN_DIR"
