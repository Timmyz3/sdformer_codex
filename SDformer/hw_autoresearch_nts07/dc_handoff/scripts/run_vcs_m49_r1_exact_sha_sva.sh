#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m49_k2_ctx8_atomic_exact_sha_vcs_r1_20260823"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M49-r1 exact-SHA VCS run: $RUN_DIR" >&2
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
RTL="rtl_m49/qfit_k2_parent_delta_p8_l96_ctx8.sv"
NOTE="rtl_m49/M49_K2_CTX8_ATOMIC_UNION_R1.md"
SVA="verif_m49/qfit_k2_parent_delta_p8_l96_ctx8_assertions.sv"
MITER="verif_m49/replay_m49_vcs_ledger.py"
TB="tb_m49/tb_qfit_k2_parent_delta_p8_l96_ctx8.sv"
FILELIST="dc_handoff/filelists/date_m49_k2_ctx8_p8_l96_vcs.f"
CONTRACT="contracts/m49_k2_ctx8_atomic_union_exact_sha_vcs_contract_r1_20260823.json"
M45_CONTRACT="contracts/m45_dual_destination_bank_fused_integrated_schedule_contract_r2_20260823.json"
M45_RESULT="results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823/m45_r2_context8_primary_schedule.json"
M45_REVIEW="results/m45_dual_destination_bank_fused_integrated_schedule_r2_20260823/m45_r2_independent_hammer_review.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="7aa0956fd187133b5bfcc366d6181fff78e2b96b4850e79ff291fc33b5eb5027"
    ["$NOTE"]="34aa0786e38df1bdc2c512974c27fab06b935ae78aa059b7f4e2b5f040b18885"
    ["$SVA"]="3359b7cf278bc0fe1ca3b2b30e2ef18953407266a79f186cd2b29c1999d74770"
    ["$MITER"]="ca207a2b0ed6a5077f120808bd15509b86f999a3c477521506024f176a110c74"
    ["$TB"]="a9dbae6def1640a4eb8baf1d83e052addd1a8aba01f76b77faa137967614491b"
    ["$FILELIST"]="000e42e1414b9517779cfe19e16a60b82456aaf21140c4a1e3d0c33065f1b5d2"
    ["$CONTRACT"]="ca5340bcda23fb93fbea08f1a527eb056c7263fbc7a642723f28d6377458bdd5"
    ["$M45_CONTRACT"]="1c547c3ecd5d82c5dc8217297f19ca730748ac9526663f5449d8f13d867cd6b4"
    ["$M45_RESULT"]="0f16e75601fdb18f31f9bc36f6aae8a17a9e62a20f5c07e18226562e9ba0d37c"
    ["$M45_REVIEW"]="cc0110cd9a8e084adf2c6e58224a2a3f52144608c96be3f65bde132a4921d6a8"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$RTL" "$NOTE" "$SVA" "$MITER" "$TB" "$FILELIST" \
        "$CONTRACT" "$M45_CONTRACT" "$M45_RESULT" "$M45_REVIEW"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M49-r1 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$RTL" "$NOTE" "$SVA" "$MITER" "$TB" "$FILELIST" \
    "$CONTRACT" "$M45_CONTRACT" "$M45_RESULT" "$M45_REVIEW" \
    > "$RUN_DIR/input_sha256.txt"
sha256sum --strict -c "$RUN_DIR/input_sha256.txt" \
    > "$RUN_DIR/input_manifest_check.raw.log" 2>&1
printf '0\n' > "$RUN_DIR/input_manifest_check.rc"

VCS_REAL_ROOT="/opt/synopsys/vcs/V-2023.12-SP1"
VCS_SHIM="$RUN_DIR/vcs_home"
mkdir "$VCS_SHIM"
for vcs_entry in "$VCS_REAL_ROOT"/*; do
    ln -s "$vcs_entry" "$VCS_SHIM/$(basename "$vcs_entry")"
done
ln -s "$VCS_REAL_ROOT/linux64" "$VCS_SHIM/linux"
{
    echo "vcs_real_root=$VCS_REAL_ROOT"
    echo "vcs_launcher=$VCS_SHIM/bin/vcs"
    echo "linux_alias_target=$VCS_REAL_ROOT/linux64"
    echo "vcs_arch_override=linux"
} > "$RUN_DIR/vcs_tool_identity.txt"

{
    echo "VCS_ARCH_OVERRIDE=linux VCS_HOME=$VCS_SHIM $VCS_SHIM/bin/vcs"
    echo "-full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED"
    echo "-timescale=1ns/1ps -Mdir=$RUN_DIR/csrc -f $FILELIST"
    echo "-top tb_qfit_k2_parent_delta_p8_l96_ctx8 -o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

set +e
env VCS_ARCH_OVERRIDE=linux VCS_HOME="$VCS_SHIM" \
    "$VCS_SHIM/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_qfit_k2_parent_delta_p8_l96_ctx8 -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M49-r1 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq '^(Warning|Error)-|(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/compile.raw.log"; then
    echo "M49-r1 compile log contains warning/error/fatal signature" >&2
    exit 21
fi
echo "M49_R1_COMPILE=PASS rc=0 simv_executable=1" \
    > "$RUN_DIR/compile.success.marker"

echo "$RUN_DIR/simv +LEDGER=$RUN_DIR/m49_handshake_ledger.log" \
    > "$RUN_DIR/sim.command.txt"
set +e
"$RUN_DIR/simv" +LEDGER="$RUN_DIR/m49_handshake_ledger.log" \
    > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M49-r1 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'M49_ASSERTION_MODULE_ACTIVE=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M49_SVA_BOUND=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M49_ATTACKS reset_request_stall=1 reset_output_stall=1 unexpected_response=1 duplicate_launch_pair=1 duplicate_relaunch=1 response_context0_mismatch=1 response_context1_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 positive_overflow=1 negative_overflow=1' "$RUN_DIR/sim.raw.log"
grep -qx 'PASS M49 K2_CTX8_ATOMIC_DUAL_ENQUEUE legal_tags=28 outputs=28 requests=71 context8=1 meta16=1 complete16=1' "$RUN_DIR/sim.raw.log"
grep -q 'Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64' "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M49-r1 functional/SVA failure signature found" >&2
    exit 31
fi

for cover_name in cp_context8_full cp_metadata16_full cp_complete16_full \
        cp_metadata_full_pop_push cp_atomic_k2_push \
        cp_complete_credit_pop_push2 cp_k1_request cp_k2_shared_bank \
        cp_k2_partial_share cp_k2_no_share_cycle cp_request_stall \
        cp_response_stall cp_output_stall cp_zero_launch cp_fault; do
    cover_line="$(grep "${cover_name}," "$RUN_DIR/sim.raw.log")"
    cover_matches="$(printf '%s\n' "$cover_line" \
        | sed -nE 's/^.*,[[:space:]]+[0-9]+ attempts,[[:space:]]+([0-9]+) match$/\1/p')"
    if [[ -z "$cover_matches" || "$cover_matches" -eq 0 ]]; then
        echo "M49-r1 missing nonzero SVA cover: $cover_name" >&2
        exit 32
    fi
    printf '%s=%s\n' "$cover_name" "$cover_matches" \
        >> "$RUN_DIR/sva_cover_matches.txt"
done

/usr/bin/python3.6 "$MITER" \
    --ledger "$RUN_DIR/m49_handshake_ledger.log" \
    --output "$RUN_DIR/m49_ledger_replay.json" \
    > "$RUN_DIR/miter.raw.log" 2>&1
grep -qx 'PASS M49 LEDGER commands=28 groups=15 requests=71 outputs=28 reads=491 updates=789' \
    "$RUN_DIR/miter.raw.log"

{
    echo "status=PASS_M49_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER"
    echo "headline_admitted=false"
    echo "review_required=true"
    echo "compile_rc=0"
    echo "sim_rc=0"
    echo "functional_mismatch_count=0"
    echo "ledger_mismatch_count=0"
    echo "sva_assertion_failure_count=0"
    echo "legal_tags=28"
    echo "legal_outputs=28"
    echo "accepted_requests=71"
    echo "physical_unique_weight_row_issues=491"
    echo "logical_destination_updates=789"
    echo "M45_cycles_PPA_power_energy_system_admitted=false"
} > "$RUN_DIR/runner_status.txt"

find "$RUN_DIR" -type f ! -name output_sha256.txt -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/output_sha256.txt"
sha256sum --strict -c "$RUN_DIR/output_sha256.txt" \
    > "$RUN_DIR/output_manifest_check.raw.log" 2>&1
printf '0\n' > "$RUN_DIR/output_manifest_check.rc"
(
    cd "$RUN_DIR"
    sha256sum input_sha256.txt output_sha256.txt runner_status.txt \
        output_manifest_check.raw.log output_manifest_check.rc \
        > run_local_seal.sha256
    sha256sum --strict -c run_local_seal.sha256 \
        > run_local_seal_check.raw.log 2>&1
)
printf '0\n' > "$RUN_DIR/run_local_seal_check.rc"
{
    echo "M49_R1_RUN_COMPLETE=PASS"
    echo "all_functional_SVA_ledger_and_SHA_checks_complete=true"
    echo "claim_scope=STANDALONE_VCS_SVA_ONLY_PENDING_INDEPENDENT_HAMMER"
} > "$RUN_DIR/RUN_COMPLETE.txt"
(
    cd "$RUN_DIR"
    sha256sum run_local_seal.sha256 run_local_seal_check.raw.log \
        run_local_seal_check.rc RUN_COMPLETE.txt > completion_seal.sha256
    sha256sum --strict -c completion_seal.sha256 \
        > completion_seal_check.raw.log 2>&1
)
printf '0\n' > "$RUN_DIR/completion_seal_check.rc"

run_complete=1
find "$RUN_DIR" -type f -exec chmod 0444 {} +
find "$RUN_DIR" -type d -exec chmod 0555 {} +
echo "PASS M49-r1 exact-SHA VCS/SVA run sealed at $RUN_DIR"
