#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m54_k4_ctx16_atomic_exact_sha_vcs_r1_20260823"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M54-r1 exact-SHA VCS run: $RUN_DIR" >&2
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
            echo "first_failure_logs_are_never_relaunched_into_this_directory=true"
        } > "$RUN_DIR/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
}
trap on_exit EXIT

cd "$HW_ROOT"
RTL="rtl_m54/qfit_k4_parent_delta_p8_l96_ctx16.sv"
NOTE="rtl_m54/M54_K4_CTX16_ATOMIC_UNION_R1.md"
SVA="verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv"
MITER="verif_m54/replay_m54_vcs_ledger.py"
PREFLIGHT="verif_m54/validate_m54_k4_ctx16_preflight.py"
TB="tb_m54/tb_qfit_k4_parent_delta_p8_l96_ctx16.sv"
FILELIST="dc_handoff/filelists/date_m54_k4_ctx16_p8_l96_vcs.f"
CONTRACT="contracts/m54_k4_ctx16_atomic_union_exact_sha_vcs_contract_r1_20260823.json"
DIAGNOSTICS="results/m54_k4_ctx16_diagnostics_20260823/diagnostic_history.json"
M49_CONTRACT="contracts/m49_k2_ctx8_atomic_union_exact_sha_vcs_contract_r1_20260823.json"
M49_RECEIPT="contracts/m49_r1_exact_sha_vcs_receipt_r1_20260823.json"
M49_REVIEW="results/m49_r1_independent_hammer_20260823/m49_r1_independent_hammer_review.json"
M52_CONTRACT="contracts/m52_high_fanout_context16_dse_contract_r1_20260823.json"
M52_RESULT="results/m52_high_fanout_context16_dse_r1_20260823/m52_high_fanout_context16_dse.json"
M52_REVIEW="results/m52_high_fanout_context16_dse_r1_independent_hammer_20260823/m52_r1_independent_hammer_review.json"

declare -A EXPECTED_SHA=(
    ["$RTL"]="e06040f6aeac3f30b2d018d415b95ae2471f01632ce801d789b0c93421e4cf0a"
    ["$NOTE"]="e6dc1c9d7aeee207214899a07956fe80a42ea1d33bc2dc2d1cbe266555868220"
    ["$SVA"]="1338421c3ee3d12f70fb2b2299e76d6651c297500920b1ffb70989c90cc2a267"
    ["$MITER"]="40252627a43359fc78c4f95673888badf8fa9fe7fc93557e0d244cb3f962523e"
    ["$PREFLIGHT"]="6d3fa85461d38afedc3f61aee880783a7151a30e88635083a29f419a470ff2b8"
    ["$TB"]="2e451050daedeb78e837be44a8e108f60978a47127cfe16a5c8c24892627e17f"
    ["$FILELIST"]="dd3ab7ece0e37881fa31943b6a9a1c0d7e5f308628a91850979a7688d0ed679f"
    ["$CONTRACT"]="f1b224843cba23f9929cee4147d18e72acda05d10e4c04a0dda086dc7b05dc08"
    ["$DIAGNOSTICS"]="630b34f9593a86aaff81174050076d8980d264d286eeb887e11997397c50242f"
    ["$M49_CONTRACT"]="ca5340bcda23fb93fbea08f1a527eb056c7263fbc7a642723f28d6377458bdd5"
    ["$M49_RECEIPT"]="30bc288a16a2b317467481a625cb739805c72d16f0f643ca69ab09b73a65e0bc"
    ["$M49_REVIEW"]="25e4989b347f40c5512667dc2260d45a8601bcd56cc7ab1815a0e6ed0a92f3b4"
    ["$M52_CONTRACT"]="9aab440911d8a1dbe5b0465ca4af31427131b7a27bebb4f9cfc9451689e5a173"
    ["$M52_RESULT"]="d60567fecd891e9da0fc1b5bb0d88f4bb7e8e93faa92092037fc46d63dcde50b"
    ["$M52_REVIEW"]="2a40398bdf5acf4ff9d853e8eba954fe5b62f2e8209c3f8d36548994826797b4"
)
INPUTS=("$RTL" "$NOTE" "$SVA" "$MITER" "$PREFLIGHT" "$TB" "$FILELIST"
        "$CONTRACT" "$DIAGNOSTICS" "$M49_CONTRACT" "$M49_RECEIPT"
        "$M49_REVIEW" "$M52_CONTRACT" "$M52_RESULT" "$M52_REVIEW")

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "${INPUTS[@]}"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M54-r1 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "${INPUTS[@]}" > "$RUN_DIR/input_sha256.txt"
sha256sum --strict -c "$RUN_DIR/input_sha256.txt" \
    > "$RUN_DIR/input_manifest_check.raw.log" 2>&1
printf '0\n' > "$RUN_DIR/input_manifest_check.rc"

/usr/bin/python3.6 "$PREFLIGHT" --contract "$CONTRACT" \
    --output "$RUN_DIR/preflight_receipt.json" \
    > "$RUN_DIR/preflight.raw.log" 2>&1
grep -qx 'PASS M54 preflight exact-SHA sources=7 covers=32 attacks=10 no-DC' \
    "$RUN_DIR/preflight.raw.log"

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
    echo "dc_launched=false"
    echo "open_source_simulator_used=false"
} > "$RUN_DIR/vcs_tool_identity.txt"

{
    echo "VCS_ARCH_OVERRIDE=linux VCS_HOME=$VCS_SHIM $VCS_SHIM/bin/vcs"
    echo "-full64 -sverilog -assert svaext +define+SVA_RUNTIME_ENABLED"
    echo "-timescale=1ns/1ps -Mdir=$RUN_DIR/csrc -f $FILELIST"
    echo "-top tb_qfit_k4_parent_delta_p8_l96_ctx16 -o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

set +e
env VCS_ARCH_OVERRIDE=linux VCS_HOME="$VCS_SHIM" \
    "$VCS_SHIM/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_qfit_k4_parent_delta_p8_l96_ctx16 -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M54-r1 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq '^(Warning|Error)-|(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/compile.raw.log"; then
    echo "M54-r1 compile log contains warning/error/fatal signature" >&2
    exit 21
fi
echo "M54_R1_COMPILE=PASS rc=0 simv_executable=1" \
    > "$RUN_DIR/compile.success.marker"

echo "$RUN_DIR/simv -no_save +LEDGER=$RUN_DIR/m54_handshake_ledger.log" \
    > "$RUN_DIR/sim.command.txt"
set +e
"$RUN_DIR/simv" -no_save +LEDGER="$RUN_DIR/m54_handshake_ledger.log" \
    > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M54-r1 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'M54_ASSERTION_MODULE_ACTIVE=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M54_SVA_BOUND=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M54_ATTACKS reset_request_stall=1 reset_output_stall=1 unexpected_response=1 duplicate_context_launch=1 stale_response_tag=1 response_context_count_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 positive_overflow=1 negative_overflow=1' "$RUN_DIR/sim.raw.log"
grep -qx 'PASS M54 K4_CTX16_ATOMIC_UNION commands=67 outputs=67 groups=24 requests=53 context16=1 meta16=1 complete16=1 push4=1 pop13push4=1' "$RUN_DIR/sim.raw.log"
grep -q 'Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64' "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M54-r1 functional/SVA failure signature found" >&2
    exit 31
fi

COVERS=(
    cp_context16 cp_meta16 cp_complete16 cp_push4
    cp_complete13_pop_push4 cp_meta_tail_wrap cp_complete_tail_wrap
    cp_k1 cp_k2 cp_k2_full_share cp_k2_partial_share cp_k2_no_share
    cp_k3 cp_k3_full_share cp_k3_partial_share cp_k3_no_share
    cp_k4 cp_k4_full_share cp_k4_partial_share cp_k4_no_share
    cp_request_stall cp_response_stall cp_output_stall
    cp_zero_k1 cp_zero_k2 cp_zero_k3 cp_zero_k4
    cp_unexpected_response cp_duplicate_context_launch
    cp_response_mismatch cp_overflow cp_fault
)
for cover_name in "${COVERS[@]}"; do
    cover_line="$(grep "${cover_name}," "$RUN_DIR/sim.raw.log")"
    cover_matches="$(printf '%s\n' "$cover_line" \
        | sed -nE 's/^.*,[[:space:]]+[0-9]+ attempts,[[:space:]]+([0-9]+) match$/\1/p')"
    if [[ -z "$cover_matches" || "$cover_matches" -eq 0 ]]; then
        echo "M54-r1 missing nonzero SVA cover: $cover_name" >&2
        exit 32
    fi
    printf '%s=%s\n' "$cover_name" "$cover_matches" \
        >> "$RUN_DIR/sva_cover_matches.txt"
done

/usr/bin/python3.6 "$MITER" \
    --ledger "$RUN_DIR/m54_handshake_ledger.log" \
    --output "$RUN_DIR/m54_ledger_replay.json" \
    > "$RUN_DIR/miter.raw.log" 2>&1
grep -qx 'PASS M54 LEDGER commands=67 groups=24 requests=53 outputs=67 reads=381 updates=450' \
    "$RUN_DIR/miter.raw.log"

{
    echo "status=PASS_M54_R1_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER"
    echo "headline_admitted=false"
    echo "review_required=true"
    echo "compile_rc=0"
    echo "sim_rc=0"
    echo "functional_mismatch_count=0"
    echo "ledger_mismatch_count=0"
    echo "sva_assertion_failure_count=0"
    echo "commands=67"
    echo "outputs=67"
    echo "groups=24"
    echo "accepted_requests=53"
    echo "physical_unique_weight_row_issues=381"
    echo "logical_destination_updates=450"
    echo "context16_reached=true"
    echo "metadata16_reached=true"
    echo "complete16_reached=true"
    echo "atomic_push4_reached=true"
    echo "complete13_pop_push4_reached=true"
    echo "M52_cycles_as_RTL_cycles=false"
    echo "dc_launched=false"
    echo "PPA_power_energy_system_admitted=false"
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
    echo "M54_R1_RUN_COMPLETE=PASS"
    echo "all_functional_SVA_ledger_and_SHA_checks_complete=true"
    echo "claim_scope=STANDALONE_VCS_SVA_ONLY_PENDING_INDEPENDENT_HAMMER"
    echo "dc_launched=false"
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
echo "PASS M54-r1 exact-SHA VCS/SVA run sealed at $RUN_DIR"
