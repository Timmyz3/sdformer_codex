#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m62_p48_directed_vcs_r1_sealed_20260823}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M62 directed VCS run: $RUN_DIR" >&2
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
RTL="rtl_m62/qfit_head_p48_signed_lane_fold.sv"
SVA="verif_m62/qfit_head_p48_signed_lane_fold_assertions.sv"
TB="tb_m62/tb_qfit_head_p48_signed_lane_fold.sv"
FILELIST="dc_handoff/filelists/date_m62_p48_directed_vcs.f"
CONTRACT="contracts/m62_p48_signed_lane_fold_directed_vcs_contract_r1_20260823.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f"
    ["$SVA"]="16a7907340711ab722ce1f2ec978da776004befef791d03d8bc34893d128cd05"
    ["$TB"]="f6b9a4ad2967af302a093b16f0cef37a99b389486e1cfaa86568ca548a6392e8"
    ["$FILELIST"]="65ee44ec7b0614c6619863dbdb60e56010d4d76f39f34f2ff02f3b1a5f006387"
    ["$CONTRACT"]="cc70780bcd539eec5badf420f4b8c2e58e6c4bd6c402d9b74041cce836233b24"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M62 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT" \
    > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

{
    echo "$VCS_HOME/bin/vcs -full64 -sverilog -assert svaext"
    echo "+define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps"
    echo "-Mdir=$RUN_DIR/csrc -f $FILELIST"
    echo "-top tb_qfit_head_p48_signed_lane_fold -o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_qfit_head_p48_signed_lane_fold -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M62 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M62 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M62 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M62 directed groups=241 events=787 zero=15 lanes=96 protocol_attacks=1 full8=1' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal' "$RUN_DIR/sim.raw.log"; then
    echo "M62 functional/SVA failure signature found" >&2
    exit 31
fi
grep -q 'cp_full_eight_source_event' "$RUN_DIR/sim.raw.log"
run_complete=1
echo "PASS M62 directed VCS/SVA sealed at $RUN_DIR"
