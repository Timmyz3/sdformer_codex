#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m79_precision_elastic_pwp_vcs_r1_sealed_20260823}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M79 directed VCS run: $RUN_DIR" >&2
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
            echo "run_dir=$RUN_DIR"
        } > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"
RTL="rtl_m79/precision_elastic_pwp_beat_assembler.sv"
SVA="verif_m79/precision_elastic_pwp_beat_assembler_assertions.sv"
TB="tb_m79/tb_precision_elastic_pwp_beat_assembler.sv"
FILELIST="dc_handoff/filelists/date_m79_precision_elastic_pwp_directed_vcs.f"
CONTRACT="contracts/m79_precision_elastic_pwp_assembler_vcs_contract_r1_20260823.json"
M78_RESULT="results/m78_precision_elastic_pwp_valid825_internal_dev_r1_20260823/m78_precision_elastic_pwp.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="00bf98d682759906a932c5518561393c5fc74104407e9df35ec3af42835fcad7"
    ["$SVA"]="dbea10891e951a8b502f02776e14915e2cf67c5d70ff024530c5a5599ecadad7"
    ["$TB"]="62ca1d6dd375c2a0307eb81d91cb5eb54466589fbd07789d479f374f45e92b87"
    ["$FILELIST"]="1403f615fed184aaa669df5689dace47ab9b9329e999a0eae4fc7288ef76d7c2"
    ["$CONTRACT"]="7ac5121c2c01885fbb227fd6c386f626b39d9440469d0f8548b41a8122a7ae7a"
    ["$M78_RESULT"]="00d2802eb8e4085fdf740f0183b23488ef2def5ca38f027c57ccba04f30064cc"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT" "$M78_RESULT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M79 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT" "$M78_RESULT" \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_precision_elastic_pwp_beat_assembler -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M79 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M79 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M79 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M79 directed transactions=136 beats=512 escapes=8 stalls=12 lanes=96 protocol_attacks=2 widths=8,9,10,11,12' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal' "$RUN_DIR/sim.raw.log"; then
    echo "M79 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_width8 cp_width9 cp_width10 cp_width11 cp_escape12 \
        cp_output_stall cp_protocol_fault; do
    grep -q "$cover" "$RUN_DIR/sim.raw.log"
done

{
    echo "status=PASS_M79_DIRECTED_VCS_SVA"
    echo "claim_scope=isolated_pwp_beat_assembler_only"
    echo "rtl_or_netlist_formality=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M79 directed VCS/SVA sealed at $RUN_DIR"
