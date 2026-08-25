#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m62_p48_directed_negative_vcs_r2_sealed_20260823}"
RECEIPT="${RECEIPT:-$HW_ROOT/results/m62_p48_directed_negative_vcs_r2_20260823/m62_p48_directed_negative_vcs_receipt_r2.json}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M62-r2 sealed VCS run: $RUN_DIR" >&2
    exit 2
fi
if [[ -e "$RECEIPT" ]]; then
    echo "refusing to overwrite M62-r2 receipt: $RECEIPT" >&2
    exit 3
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
INPUTS="contracts/m62_p48_signed_lane_fold_directed_negative_inputs_r2_20260823.json"
SVA="verif_m62/qfit_head_p48_signed_lane_fold_negative_assertions_r2.sv"
TB="tb_m62/tb_qfit_head_p48_signed_lane_fold_negative_r2.sv"
FILELIST="dc_handoff/filelists/date_m62_p48_directed_negative_vcs_r2.f"
CONTRACT="contracts/m62_p48_signed_lane_fold_directed_negative_vcs_contract_r2_20260823.json"
VALIDATOR="verif_m62/validate_m62_p48_directed_negative_vcs_r2.py"

declare -A EXPECTED_SHA=(
    ["$RTL"]="4ba42f70e664d7fc30716a04678acc955612008a2be5a0dad693778bbd776f0f"
    ["$INPUTS"]="48c855cb3333f8c5392fe47a95969dfaa65d8ada6928b039ce9e3d27446c123a"
    ["$SVA"]="585d8828b311deb323a5a4dc15fbad08ee5490182c318f09ef7b07a1f2e0663d"
    ["$TB"]="5b5286deee43273aaa04df5e95d5c05db3c9c43df0e962dc6f8d2471f94d985b"
    ["$FILELIST"]="c3143b9a183bce0063f39a4b302539b2829a555665923281629d02fec58938bc"
    ["$CONTRACT"]="431fd824352684d85ce54e5f36c78c48fb011477d44bdf419bf859b4a2f40698"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$INPUTS" "$SVA" "$TB" "$FILELIST" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M62-r2 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$INPUTS" "$SVA" "$TB" "$FILELIST" "$CONTRACT" \
    > "$RUN_DIR/input_sha256.txt"

VCS_BIN="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
if [[ ! -x "$VCS_BIN" ]]; then
    echo "frozen Synopsys VCS binary missing: $VCS_BIN" >&2
    exit 11
fi
export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"

{
    echo "$VCS_BIN -full64 -sverilog -assert svaext -timescale=1ns/1ps"
    echo "-Mdir=$RUN_DIR/csrc -f $FILELIST"
    echo "-top tb_qfit_head_p48_signed_lane_fold_negative_r2"
    echo "-o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

set +e
"$VCS_BIN" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_qfit_head_p48_signed_lane_fold_negative_r2 \
    -o "$RUN_DIR/simv" > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M62-r2 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M62-r2 compile log contains warning/error signature" >&2
    exit 21
fi

echo "$RUN_DIR/simv -no_save" > "$RUN_DIR/sim.command.txt"
set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M62-r2 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'M62_R2_NEGATIVE_ASSERTION_MODULE_ACTIVE=1' "$RUN_DIR/sim.raw.log"
grep -qx 'PASS M62 R2 directed_negative legal_full8=6 lane_checks=576 attacks=5 attack_accepts=5 sticky_cycles=15 mismatches=0' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'Assertion failure|failed at|Offending|^Error|^Fatal' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M62-r2 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in \
    cp_legal_full8_0 cp_legal_full8_1 cp_legal_full8_2 \
    cp_legal_full8_3 cp_legal_full8_4 cp_legal_full8_5 \
    cp_near_positive_limit cp_near_negative_limit cp_five_cycle_stall_case \
    cp_attack_overlap cp_attack_invalid_slot cp_attack_reserved_negative_128 \
    cp_attack_no_signed_work cp_attack_accumulator_overflow; do
    grep -q "m62_r2_sva\.$cover," "$RUN_DIR/sim.raw.log"
done

python3 "$VALIDATOR" --run-dir "$RUN_DIR" --receipt "$RECEIPT" \
    > "$RUN_DIR/validator.log" 2>&1
run_complete=1
rm -f "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
echo "PASS M62-r2 directed-negative VCS/SVA sealed at $RUN_DIR"
echo "receipt=$RECEIPT"
