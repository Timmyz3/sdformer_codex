#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m85_guarded_wordpacked_pwp_stream_vcs_r1_sealed_20260823}"
M83_RECORDS="${M83_RECORDS:-/tmp/m85_inputs/m83_cap11_phase_records.bin}"
M83_OFFSETS="${M83_OFFSETS:-/tmp/m85_inputs/m83_cap11_phase_offsets_u32le.bin}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M85 VCS run: $RUN_DIR" >&2
    exit 2
fi
for external in "$M83_RECORDS" "$M83_OFFSETS"; do
    if [[ ! -f "$external" ]]; then
        echo "missing M85 external input: $external" >&2
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
RTL="rtl_m85/guarded_wordpacked_pwp_stream.sv"
SVA="verif_m85/guarded_wordpacked_pwp_stream_assertions.sv"
TB="tb_m85/tb_guarded_wordpacked_pwp_stream.sv"
FILELIST="dc_handoff/filelists/date_m85_guarded_wordpacked_pwp_stream_vcs.f"
REPACKER="system_simulator/scripts/repack_m85_canonical_74b_phase_metadata.py"
METADATA="results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin"
METADATA_RECEIPT="results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b_receipt.json"
M84_REVIEW="reviews/m84_hierarchical_pwp_bank_mapper_vcs_independent_hammer_r1_20260823/m84_hierarchical_pwp_bank_mapper_independent_hammer_review.json"
CONTRACT="contracts/m85_guarded_wordpacked_pwp_stream_vcs_contract_r1_20260823.json"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$RTL"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["$SVA"]="7403ad62988b5b082788b415713cee74982c47be15878648ca1294897c7fe2f7"
    ["$TB"]="6ee304eaacdf8d3881cb87a96c199b3fc89e01d6350eda4cb23bb07061ac4c21"
    ["$FILELIST"]="6cedb577fdced0fb76efc031a827aac3081f800a4768fd5d4b4a7025f4f5e5ca"
    ["$REPACKER"]="a9cde0022106d5934fb532ad25f2b98b5d0b1e418b8c1e0c38bb069c53549f6c"
    ["$METADATA"]="52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0"
    ["$METADATA_RECEIPT"]="f57bf511417b384279219dd3b1d86ead035a0eaac625510391a22881807307d7"
    ["$M84_REVIEW"]="d1e2b31f5b2c9ce9aca171c505a3797c4af42e2c4f0d6a3628e8edac967e531c"
    ["$CONTRACT"]="2f1225acb79ceaf16df35bc477dcd05c54bf0d299675cec388bce66cb1e576af"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$RTL" "$SVA" "$TB" "$FILELIST" "$REPACKER" \
        "$METADATA" "$METADATA_RECEIPT" "$M84_REVIEW" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M85 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
records_sha="$(sha256sum "$M83_RECORDS" | awk '{print $1}')"
offsets_sha="$(sha256sum "$M83_OFFSETS" | awk '{print $1}')"
if [[ "$records_sha" != "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d" ]]; then
    echo "M85 M83 records SHA mismatch" >&2
    exit 11
fi
if [[ "$offsets_sha" != "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c" ]]; then
    echo "M85 M83 offsets SHA mismatch" >&2
    exit 12
fi
sha256sum "$M82" "$RTL" "$SVA" "$TB" "$FILELIST" "$REPACKER" \
    "$METADATA" "$METADATA_RECEIPT" "$M84_REVIEW" "$CONTRACT" \
    > "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' "$M83_RECORDS" "$records_sha" \
    "$(stat -c %s "$M83_RECORDS")" >> "$RUN_DIR/input_sha256.txt"
printf 'external_path=%s sha256=%s bytes=%s\n' "$M83_OFFSETS" "$offsets_sha" \
    "$(stat -c %s "$M83_OFFSETS")" >> "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_guarded_wordpacked_pwp_stream -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M85 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M85 compile log contains warning/error signature" >&2
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
    echo "M85 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M85 actual-record integration phases=1728 entries=221184 outputs=221184 escape=1 beats=835383 masked_nonzero_words=733459 ii_checks=219456 metadata_poison_attacks=3' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M85 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_phase_load cp_escape cp_width9 cp_width10 cp_width11 \
        cp_metadata_error; do
    grep -q "$cover" "$RUN_DIR/sim.raw.log"
done
{
    echo "status=PASS_M85_ACTUAL_WORDPACKED_RECORD_TO_M82_VCS_SVA"
    echo "claim_scope=packed_metadata_mapper_final_mask_and_m82_only"
    echo "phases=1728"
    echo "entries=221184"
    echo "outputs=221184"
    echo "beats_including_escape=835383"
    echo "masked_nonzero_following_words=733459"
    echo "synchronous_sram=false"
    echo "real_escape_fallback=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M85 actual-record VCS/SVA sealed at $RUN_DIR"
