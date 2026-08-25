#!/usr/bin/env bash
set -euo pipefail

DC_HANDOFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_HANDOFF_ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$DC_HANDOFF_ROOT/runs/m86_r2_arbitrated_sync_bank_vcs_r1_sealed_20260823}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M86-R2 VCS run: $RUN_DIR" >&2
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
M82="rtl_m82/zero_bubble_elastic_pwp_stream.sv"
M85="rtl_m85/guarded_wordpacked_pwp_stream.sv"
M86="rtl_m86/sync_banked_guarded_pwp_frontend.sv"
RTL="rtl_m86_r2/arbitrated_sync_banked_guarded_pwp_frontend.sv"
SVA="verif_m86_r2/arbitrated_sync_banked_guarded_pwp_frontend_assertions.sv"
TB="tb_m86_r2/tb_arbitrated_sync_banked_guarded_pwp_frontend.sv"
FILELIST="dc_handoff/filelists/date_m86_r2_arbitrated_sync_banked_guarded_pwp_vcs.f"
CONTRACT="contracts/m86_r2_arbitrated_sync_bank_vcs_contract_r1_20260823.json"

declare -A EXPECTED_SHA=(
    ["$M82"]="2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f"
    ["$M85"]="ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0"
    ["$M86"]="edb06b7f4e891d4b00c8b49ace547efdf8daf84dc19716c710a6a343dc97f781"
    ["$RTL"]="099abd4d43d49d5ee2b1e6ec90430334ac317bfec2c3766e9ca780ac313fe22c"
    ["$SVA"]="6509c7796f9bccd90ce59252fc1e5e908afab940ed1fc41b69efacb643a5f5d6"
    ["$TB"]="1651dc8973fcd44c3229a83f393e9c1bc3da9689e51e0b152c995f98f28880a1"
    ["$FILELIST"]="8b290f146d922d4432cc69dc18c24d6247f978090699bc777ccb9ab12883a9c5"
    ["$CONTRACT"]="087312d3401af00c981dbda4f36c36a5cd6d3c983b5348fcbcc614926595fcda"
)

: > "$RUN_DIR/preflight_sha_checks.txt"
for path in "$M82" "$M85" "$M86" "$RTL" "$SVA" "$TB" "$FILELIST" "$CONTRACT"; do
    observed="$(sha256sum "$path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' \
        "$path" "${EXPECTED_SHA[$path]}" "$observed" \
        >> "$RUN_DIR/preflight_sha_checks.txt"
    if [[ "$observed" != "${EXPECTED_SHA[$path]}" ]]; then
        echo "M86-R2 exact-SHA preflight mismatch: $path" >&2
        exit 10
    fi
done
sha256sum "$M82" "$M85" "$M86" "$RTL" "$SVA" "$TB" "$FILELIST" \
    "$CONTRACT" > "$RUN_DIR/input_sha256.txt"

VCS_HOME="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
export VCS_HOME VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$VCS_HOME/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$RUN_DIR/csrc" -f "$FILELIST" \
    -top tb_arbitrated_sync_banked_guarded_pwp_frontend -o "$RUN_DIR/simv" \
    > "$RUN_DIR/compile.raw.log" 2>&1
compile_rc="$?"
set -e
printf '%s\n' "$compile_rc" > "$RUN_DIR/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$RUN_DIR/simv" ]]; then
    echo "M86-R2 VCS compile failed rc=$compile_rc" >&2
    exit 20
fi
if grep -Eiq 'Warning-\[|Error-\[|^Error' "$RUN_DIR/compile.raw.log"; then
    echo "M86-R2 compile log contains warning/error signature" >&2
    exit 21
fi

set +e
"$RUN_DIR/simv" -no_save > "$RUN_DIR/sim.raw.log" 2>&1
sim_rc="$?"
set -e
printf '%s\n' "$sim_rc" > "$RUN_DIR/sim.rc"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M86-R2 VCS simulation failed rc=$sim_rc" >&2
    exit 30
fi
grep -qx 'PASS M86-R2 arbitration loaded_descriptor_wins=1 unloaded_loader_wins=1 payload_accepts=462 descriptor_accepts=1 outputs=1 bank_issues=3 bank_responses=3 silent_deadlock=0' \
    "$RUN_DIR/sim.raw.log"
if grep -Eiq 'failed at|Offending|^Error|^Fatal|watchdog timeout' \
        "$RUN_DIR/sim.raw.log"; then
    echo "M86-R2 functional/SVA failure signature found" >&2
    exit 31
fi
for cover in cp_unloaded_contention cp_loaded_contention cp_legal_output; do
    grep -Eq "$cover.* [1-9][0-9]* match" "$RUN_DIR/sim.raw.log"
done
{
    echo "status=PASS_M86_R2_ARBITRATION_DIRECTED_VCS_SVA"
    echo "claim_scope=simultaneous_valid_silent_deadlock_repair_only"
    echo "loaded_descriptor_wins=1"
    echo "unloaded_loader_wins=1"
    echo "payload_accepts=462"
    echo "descriptor_accepts=1"
    echo "outputs=1"
    echo "bank_read_issues=3"
    echo "bank_responses=3"
    echo "silent_deadlock_cycles=0"
    echo "actual_record_replay=false"
    echo "compiled_sram_macro=false"
    echo "real_escape_fallback=false"
    echo "rtl_cycle_speedup=false"
    echo "paper_ppa_ready=false"
    echo "system_speedup=false"
    echo "headline=false"
} > "$RUN_DIR/RUN_COMPLETE.txt"
run_complete=1
echo "PASS M86-R2 arbitration VCS/SVA sealed at $RUN_DIR"
