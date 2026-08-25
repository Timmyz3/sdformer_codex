#!/usr/bin/env bash
set -euo pipefail

m67_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m67_hw="$(cd "$m67_dc_root/.." && pwd)"
m67_run="${M67_RUN_DIR:-$m67_dc_root/runs/m67_lookahead_pressure_exact_sha_vcs_r1_20260823}"
m67_contract="contracts/m67_lookahead_pressure_vcs_contract_r1_20260823.json"
m67_validator="dc_handoff/scripts/validate_m67_lookahead_pressure_vcs_r1.py"
m67_builder="dc_handoff/scripts/build_m67_lookahead_pressure_vcs_receipt_r1.py"
m67_filelist="dc_handoff/filelists/date_m67_lookahead_pressure_vcs.f"
m67_receipt="$m67_run/m67_lookahead_pressure_vcs_receipt_r1.json"
m67_real_vcs="/opt/synopsys/vcs/V-2023.12-SP1"
m67_complete=0

m67_fail_seal() {
    local m67_rc=$?
    if [[ -d "$m67_run" && "$m67_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m67_rc"
            echo "system_speedup_admitted=false"
            echo "headline_admitted=false"
            echo "ppa_admitted=false"
        } > "$m67_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    fi
    exit "$m67_rc"
}
trap m67_fail_seal EXIT

[[ ! -e "$m67_run" ]] || {
    echo "refusing to overwrite M67 exact-SHA run: $m67_run" >&2
    exit 2
}
mkdir -p "$m67_run/snapshot" "$m67_run/vcs_home"
cd "$m67_hw"

/usr/bin/python3.6 "$m67_validator" --mode preflight \
    > "$m67_run/preflight.raw.log" 2>&1
grep -qx 'PASS M67 exact-SHA preflight identities=9' \
    "$m67_run/preflight.raw.log"

m67_snapshot_inputs=(
    rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv
    verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv
    verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv
    tb_m67/tb_m67_qfit_k4_parent_delta_lookahead_pressure.sv
    dc_handoff/filelists/date_m67_lookahead_pressure_vcs.f
    dc_handoff/scripts/run_vcs_m67_lookahead_pressure_exact_sha_r1.sh
    dc_handoff/scripts/build_m67_lookahead_pressure_vcs_receipt_r1.py
    dc_handoff/scripts/validate_m67_lookahead_pressure_vcs_r1.py
    contracts/m67_lookahead_pressure_vcs_contract_r1_20260823.json
)
for m67_path in "${m67_snapshot_inputs[@]}"; do
    cp --parents "$m67_path" "$m67_run/snapshot"
done
(
    cd "$m67_run/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum \
        > "$m67_run/snapshot.sha256"
)

for m67_entry in "$m67_real_vcs"/*; do
    ln -s "$m67_entry" "$m67_run/vcs_home/$(basename "$m67_entry")"
done
ln -s "$m67_real_vcs/linux64" "$m67_run/vcs_home/linux"
{
    echo "VCS_ARCH_OVERRIDE=linux VCS_HOME=$m67_run/vcs_home"
    echo "$m67_run/vcs_home/bin/vcs -full64 -sverilog -assert svaext"
    echo "+define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps"
    echo "-Mdir=$m67_run/csrc -f $m67_filelist"
    echo "-top tb_m67_qfit_k4_parent_delta_lookahead_pressure -o $m67_run/simv"
} > "$m67_run/compile.command.txt"

set +e
env VCS_ARCH_OVERRIDE=linux VCS_HOME="$m67_run/vcs_home" \
    "$m67_run/vcs_home/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$m67_run/csrc" -f "$m67_filelist" \
    -top tb_m67_qfit_k4_parent_delta_lookahead_pressure \
    -o "$m67_run/simv" > "$m67_run/compile.raw.log" 2>&1
m67_rc=$?
set -e
printf '%s\n' "$m67_rc" > "$m67_run/compile.rc"
[[ "$m67_rc" -eq 0 && -x "$m67_run/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$m67_run/compile.raw.log"

echo "$m67_run/simv -no_save +LEDGER=$m67_run/m67_handshake_ledger.log" \
    > "$m67_run/sim.command.txt"
set +e
(
    cd "$m67_run"
    ./simv -no_save +LEDGER="$m67_run/m67_handshake_ledger.log"
) > "$m67_run/sim.raw.log" 2>&1
m67_rc=$?
set -e
printf '%s\n' "$m67_rc" > "$m67_run/sim.rc"
[[ "$m67_rc" -eq 0 ]]
grep -qx 'M66_LOOKAHEAD_ASSERTION_MODULE_ACTIVE=1' "$m67_run/sim.raw.log"
grep -qx 'M54_ASSERTION_MODULE_ACTIVE=1' "$m67_run/sim.raw.log"
grep -qx 'M67_SEAM_COVER seam=1 command=1 output=1 command_output=1 zero_wait=1' \
    "$m67_run/sim.raw.log"
grep -qx 'M67_SVA_BOUND=1' "$m67_run/sim.raw.log"
grep -qx 'PASS M67 K4_CTX16_ATOMIC_UNION commands=73 outputs=73 groups=30 requests=56 context16=1 meta16=1 complete16=1 push4=1 pop13push4=1' \
    "$m67_run/sim.raw.log"
! grep -Eiq 'Assertion failure|failed at|Offending|^Error|^Fatal' \
    "$m67_run/sim.raw.log"

/usr/bin/python3.6 "$m67_builder" --run "$m67_run" --output "$m67_receipt" \
    > "$m67_run/receipt_builder.raw.log" 2>&1
grep -q '^PASS M67 receipt builder ' "$m67_run/receipt_builder.raw.log"
/usr/bin/python3.6 "$m67_validator" --mode full --run "$m67_run" \
    --receipt "$m67_receipt" > "$m67_run/validator.raw.log" 2>&1
grep -q '^PASS M67 full validator ' "$m67_run/validator.raw.log"

m67_receipt_sha="$(sha256sum "$m67_receipt" | awk '{print $1}')"
{
    echo "status=PASS_EXACT_SHA_SYNOPSYS_VCS_PRESSURE_R1"
    echo "receipt_sha256=$m67_receipt_sha"
    echo "tamper_attacks_rejected=6"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "ppa_admitted=false"
} > "$m67_run/RUN_COMPLETE.txt"
(
    cd "$m67_run"
    find . -type f ! -name output_manifest.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
        > output_manifest.sha256
    sha256sum --strict -c output_manifest.sha256 \
        > output_check.raw.log 2>&1
)

find "$m67_run" -type f -exec chmod 0444 {} +
find "$m67_run" -type d -exec chmod 0555 {} +
m67_complete=1
trap - EXIT
echo "M67_PRESSURE_VCS=PASS run=$m67_run receipt_sha256=$m67_receipt_sha"
