#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m82_zero_bubble_pwp_stream_vcs_r1_sealed_20260823}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M82 VCS run: $RUN_DIR" >&2
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
RTL="rtl_m82/zero_bubble_elastic_pwp_stream.sv"
SVA="verif_m82/zero_bubble_elastic_pwp_stream_assertions.sv"
TB="tb_m82/tb_zero_bubble_elastic_pwp_stream.sv"
FILELIST="dc_handoff/filelists/date_m82_zero_bubble_pwp_stream_directed_vcs.f"
CONTRACT="contracts/m82_zero_bubble_elastic_pwp_stream_vcs_contract_r1_20260823.json"
M79_REVIEW="reviews/m79_precision_elastic_pwp_vcs_independent_hammer_r1_20260823/m79_independent_hammer.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$SVA"]="159bd670e82109fce9f5fa3f27570996a40e79067f63e7f79dbb25eab01013f9"
    ["$TB"]="9bd7f53880c57c0d9b2b86cdb1350020d27d90c14d0c3a81652c34a451a2e3a7"
    ["$FILELIST"]="c34fdc9bf75d76360c9e49a0b1fcf4fcc8a9876fa1294336177c1db3ef651ce2"
    ["$CONTRACT"]="b7003149cd7ef80871239b582a04ffa07817c04eecd03bcab2cf07dd406e9272"
    ["$M79_REVIEW"]="85d15ef419b73fd130986fbbbe0aab09488dde055bafea17eb493d286a89c958"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT" "$M79_REVIEW"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M82 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT" "$M79_REVIEW" \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_zero_bubble_elastic_pwp_stream -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M82 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M82 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M82 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M82 zero-bubble regular=129 escapes=8 starts=139 ii_checks=135 stalls=1 lanes=96 protocol_attacks=3 service=3,4,4,5' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M82 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_width8 cp_width9 cp_width10 cp_width11 cp_escape12 \
        cp_zero_bubble_boundary cp_output_stall cp_protocol_fault; do
    grep -q "$cover" "$RUN_DIR/sim.raw.log"
done
{
    echo "status=PASS_M82_ZERO_BUBBLE_DIRECTED_VCS_SVA"
    echo "claim_scope=isolated_elastic_pwp_stream_only"
    echo "bank_integration=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M82 zero-bubble VCS/SVA sealed at $RUN_DIR"
