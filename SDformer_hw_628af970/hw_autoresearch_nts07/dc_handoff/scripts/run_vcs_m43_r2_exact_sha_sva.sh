#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/m43_parent_delta_p8_l96_vcs_r2_exact_sha_20260823"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M43-r2 exact-SHA VCS run: $RUN_DIR" >&2
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

RTL="rtl_m43/qfit_parent_delta_p8_l96_multicontext.sv"
ASSERTIONS="verif_m43/qfit_parent_delta_p8_l96_multicontext_assertions.sv"
TESTBENCH="tb_m43/tb_qfit_parent_delta_p8_l96_multicontext.sv"
FILELIST="dc_handoff/filelists/date_m43_parent_delta_p8_l96_vcs.f"
RUNNER="dc_handoff/scripts/run_vcs_m43_r2_exact_sha_sva.sh"
CONTRACT="contracts/m43_r2_multicontext_exact_sha_vcs_contract_r1_20260823.json"
SCHEDULE_RESULT="results/m43_tile_resident_parent_delta_schedule_r1_20260823/m43_spatial_parent_delta_schedule_final.json"
SCHEDULE_REVIEW="results/m43_tile_resident_parent_delta_schedule_r1_20260823/m43_r1_independent_hammer_review.json"
SCHEDULE_VALIDATOR="system_simulator/scripts/validate_m43_r1_independent_hammer_review.py"
M41_PIN="contracts/m41_r2_integer_oracle_release_pin_r1_20260823.json"
M42_REVIEW="results/m42_real_work_headroom_gate_r1_20260823/m42_r1_independent_hammer_review.json"

RTL_SHA="e70239b1ec9a7d4541b0ae8d0a8f55e252fa6c804b364ab126d8201e108e0deb"
ASSERTIONS_SHA="f531a7c2077b18c60483b933ab625baa3feda3de0abb91770be2056d9b436bc2"
TESTBENCH_SHA="3ce4e30cc8da53fc628356af55898ac6be686932faad32f203296101bc015ed4"
FILELIST_SHA="ec279e0f90813dc853cae22fd711df56da451f3fcf090f893a7d9ba342b24e01"
CONTRACT_SHA="ab66620671a5805ab4e0bd8b4a53e10a659712687aca4f723e92c73eb1150435"
SCHEDULE_RESULT_SHA="70c52dfc8ef1b223391a1c0699f6ada8ff999d2079370bcd9d3917c198a1c329"
SCHEDULE_REVIEW_SHA="ce82cdb6b93d6ffc3ab5235db63ed4559e8046d2a2f7adf2ad8c0be1bb1e5278"
SCHEDULE_VALIDATOR_SHA="ab983d733f2bb16394678b701fe8c56575323c662cd2a96c0410a236216168b2"
M41_PIN_SHA="0261520d1ca085c95bfd7cb994ce2d5d304bf3cf978ca2d34bddc80e281587f7"
M42_REVIEW_SHA="de7a6187b5a4a693023948045ae27480051713192564b74cf66055648cbc0d02"

check_sha() {
    local expected="$1"
    local path="$2"
    local observed
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "$expected" "$observed" >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "$expected" ]]; then
        echo "M43-r2 exact-SHA preflight mismatch: $path" >&2
        return 1
    fi
}

: > "$RUN_DIR/preflight_sha_checks.txt"
check_sha "$RTL_SHA" "$RTL"
check_sha "$ASSERTIONS_SHA" "$ASSERTIONS"
check_sha "$TESTBENCH_SHA" "$TESTBENCH"
check_sha "$FILELIST_SHA" "$FILELIST"
check_sha "$CONTRACT_SHA" "$CONTRACT"
check_sha "$SCHEDULE_RESULT_SHA" "$SCHEDULE_RESULT"
check_sha "$SCHEDULE_REVIEW_SHA" "$SCHEDULE_REVIEW"
check_sha "$SCHEDULE_VALIDATOR_SHA" "$SCHEDULE_VALIDATOR"
check_sha "$M41_PIN_SHA" "$M41_PIN"
check_sha "$M42_REVIEW_SHA" "$M42_REVIEW"

sha256sum \
    "$RTL" "$ASSERTIONS" "$TESTBENCH" "$FILELIST" "$RUNNER" \
    "$CONTRACT" "$SCHEDULE_RESULT" "$SCHEDULE_REVIEW" \
    "$SCHEDULE_VALIDATOR" "$M41_PIN" "$M42_REVIEW" \
    > "$RUN_DIR/input_sha256.txt"
set +e
sha256sum --strict -c "$RUN_DIR/input_sha256.txt" \
    > "$RUN_DIR/input_manifest_check.raw.log" 2>&1
input_manifest_rc="$?"
set -e
printf '%s\n' "$input_manifest_rc" > "$RUN_DIR/input_manifest_check.rc"
if [[ "$input_manifest_rc" -ne 0 ]]; then
    echo "M43-r2 input manifest verification failed" >&2
    exit 10
fi

# The host VCS launcher selects a legacy `linux` compiler directory although
# this installation ships `linux64`.  Keep the compatibility alias inside the
# non-overwriting run so the exact tool path is explicit and auditable.
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
    echo "-full64 -sverilog -assert svaext -debug_access+pp"
    echo "+define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps"
    echo "-Mdir=$RUN_DIR/csrc -f $FILELIST"
    echo "-top tb_qfit_parent_delta_p8_l96_multicontext -o $RUN_DIR/simv"
} > "$RUN_DIR/compile.command.txt"

set +e
env VCS_ARCH_OVERRIDE=linux VCS_HOME="$VCS_SHIM" \
    "$VCS_SHIM/bin/vcs" \
    -full64 -sverilog -assert svaext -debug_access+pp \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_qfit_parent_delta_p8_l96_multicontext \
    -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M43-r2 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq '^(Warning|Error)-|(^|[^[:alpha:]])(warning|error|fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/compile.raw.log"; then
    echo "M43-r2 compile log contains warning/error/fatal signature" >&2
    exit 21
fi
echo "M43_R2_COMPILE=PASS rc=0 simv_executable=1" \
    > "$RUN_DIR/compile.success.marker"

echo "$RUN_DIR/simv" > "$RUN_DIR/sim.command.txt"
set +e
"$RUN_DIR/simv" > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M43-r2 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi

pass_count="$(grep -c '^PASS M43 multicontext ' "$RUN_DIR/sim.raw.log")"
if [[ "$pass_count" -ne 1 ]]; then
    echo "M43-r2 requires exactly one internal PASS line, got $pass_count" >&2
    exit 31
fi
grep -qx 'PASS M43 multicontext packets=128 outputs=128 requests=1403 simultaneous=1218 max_contexts=4 request_stalls=191 output_stalls=26' \
    "$RUN_DIR/sim.raw.log"
grep -qx 'M43_ASSERTION_MODULE_ACTIVE=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M43_SVA_BOUND=1' "$RUN_DIR/sim.raw.log"
grep -qx 'M43_ATTACKS metadata_fifo_saturation=1 reset_saturated=1 reset_request_stall=1 unexpected_response=1 response_context_mismatch=1 response_bank_mismatch=1 overlapping_masks=1 accumulator_overflow=1 request_stability=1 output_stability=1' \
    "$RUN_DIR/sim.raw.log"
grep -q 'Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64' \
    "$RUN_DIR/sim.raw.log"

for cover_name in cp_four_contexts cp_metadata_full cp_request_stall \
        cp_output_stall cp_request_response_overlap cp_fault; do
    cover_line="$(grep "${cover_name}," "$RUN_DIR/sim.raw.log")"
    cover_matches="$(printf '%s\n' "$cover_line" \
        | sed -nE 's/^.*,[[:space:]]+[0-9]+ attempts,[[:space:]]+([0-9]+) match$/\1/p')"
    if [[ -z "$cover_matches" || "$cover_matches" -eq 0 ]]; then
        echo "M43-r2 missing nonzero SVA cover: $cover_name" >&2
        exit 32
    fi
    printf '%s=%s\n' "$cover_name" "$cover_matches" \
        >> "$RUN_DIR/sva_cover_matches.txt"
done

if grep -Eiq 'failed at|Offending|assertion[^[:cntrl:]]*(fail|error)|(^|[^[:alpha:]])(Error|Fatal)([^[:alpha:]]|$)' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M43-r2 functional/SVA failure signature found" >&2
    exit 33
fi

grep '^PASS M43 multicontext ' "$RUN_DIR/sim.raw.log" \
    > "$RUN_DIR/tb_internal_pass.marker"
{
    echo "status=PASS_M43_R2_EXACT_SHA_VCS_SVA_PENDING_INDEPENDENT_HAMMER"
    echo "headline_admitted=false"
    echo "review_required=true"
    echo "candidate_rtl_sha256=$RTL_SHA"
    echo "compile_rc=$compile_rc"
    echo "sim_rc=$sim_rc"
    echo "functional_mismatch_count=0"
    echo "sva_assertion_failure_count=0"
    echo "checkpoint_integer_miter_admitted=false"
    echo "integrated_cycles_3x_DC_STA_Formality_PPA_power_energy_system_admitted=false"
} > "$RUN_DIR/runner_status.txt"
echo "M43_R2_SIMULATION=PASS rc=0 functional_mismatch=0 sva_failure=0" \
    > "$RUN_DIR/simulation.success.marker"

# Hash every regular evidence/generated file present before the output manifest.
find "$RUN_DIR" -type f ! -name output_sha256.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > "$RUN_DIR/output_sha256.txt"
set +e
sha256sum --strict -c "$RUN_DIR/output_sha256.txt" \
    > "$RUN_DIR/output_manifest_check.raw.log" 2>&1
output_manifest_rc="$?"
set -e
printf '%s\n' "$output_manifest_rc" > "$RUN_DIR/output_manifest_check.rc"
if [[ "$output_manifest_rc" -ne 0 ]]; then
    echo "M43-r2 output manifest verification failed" >&2
    exit 40
fi

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
    echo "M43_R2_RUN_COMPLETE=PASS"
    echo "all_functional_and_SVA_checks_complete=true"
    echo "all_input_output_and_local_SHA_checks_complete=true"
    echo "claim_scope=VCS_SVA_ONLY_PENDING_INDEPENDENT_HAMMER"
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
echo "PASS M43-r2 exact-SHA VCS/SVA run sealed at $RUN_DIR"
