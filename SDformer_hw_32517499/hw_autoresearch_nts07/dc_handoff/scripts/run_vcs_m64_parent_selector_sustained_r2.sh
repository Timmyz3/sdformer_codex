#!/usr/bin/env bash
set -euo pipefail

m64_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m64_hw="$(cd "$m64_dc_root/.." && pwd)"
m64_run="${M64_R2_RUN_DIR:-$m64_dc_root/runs/m64_parent_selector_sustained_vcs_r2_sealed_20260823}"
m64_receipt="$m64_run/m64_parent_selector_sustained_vcs_receipt_r2.json"
m64_vcs="/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"
m64_validator="dc_handoff/scripts/validate_m64_parent_selector_sustained_vcs_r2.py"
m64_builder="dc_handoff/scripts/build_m64_parent_selector_sustained_vcs_r2_receipt.py"
m64_filelist="dc_handoff/filelists/date_m64_parent_selector_sustained_vcs_r2.f"
m64_complete=0

m64_fail_seal() {
    local m64_rc=$?
    if [[ -d "$m64_run" && "$m64_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m64_rc"
            echo "system_speedup_admitted=false"
            echo "headline_admitted=false"
            echo "ppa_admitted=false"
        } > "$m64_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    fi
    exit "$m64_rc"
}
trap m64_fail_seal EXIT

[[ ! -e "$m64_run" ]] || {
    echo "refusing to overwrite additive M64-r2 run: $m64_run" >&2
    exit 2
}
[[ -x "$m64_vcs" ]] || {
    echo "frozen Synopsys VCS launcher missing: $m64_vcs" >&2
    exit 3
}
mkdir -p "$m64_run/snapshot"
cd "$m64_hw"

python3 "$m64_validator" --mode preflight > "$m64_run/preflight.raw.log" 2>&1
grep -q '^PASS M64-r2 exact-SHA preflight identities=11$' \
    "$m64_run/preflight.raw.log"

m64_snapshot_inputs=(
    rtl_m64/qfit_adaptive_parent_selector_p256.sv
    verif_m64/qfit_adaptive_parent_selector_p256_sustained_assertions_r2.sv
    tb_m64/tb_qfit_adaptive_parent_selector_p256_sustained_r2.sv
    dc_handoff/filelists/date_m64_parent_selector_sustained_vcs_r2.f
    dc_handoff/scripts/run_vcs_m64_parent_selector_sustained_r2.sh
    dc_handoff/scripts/build_m64_parent_selector_sustained_vcs_r2_receipt.py
    dc_handoff/scripts/validate_m64_parent_selector_sustained_vcs_r2.py
    contracts/m64_parent_selector_sustained_vcs_contract_r2_20260823.json
    contracts/m64_online_adaptive_parent_selector_directed_vcs_contract_r1_20260823.json
    dc_handoff/runs/m64_parent_selector_directed_vcs_r1b_20260823/m64_directed_vcs_receipt_r1.json
    tb_m64/tb_qfit_adaptive_parent_selector_p256.sv
)
for m64_path in "${m64_snapshot_inputs[@]}"; do
    cp --parents "$m64_path" "$m64_run/snapshot"
done
(
    cd "$m64_run/snapshot"
    find . -type f -print0 | sort -z | xargs -0 sha256sum \
        > "$m64_run/snapshot.sha256"
)

export VCS_HOME="/opt/synopsys/vcs/V-2023.12-SP1"
export VCS_ARCH_OVERRIDE="linux"
{
    echo "$m64_vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps"
    echo "-Mdir=$m64_run/csrc -f $m64_filelist"
    echo "-top tb_qfit_adaptive_parent_selector_p256_sustained_r2"
    echo "-o $m64_run/simv"
} > "$m64_run/compile.command.txt"

set +e
"$m64_vcs" -full64 -sverilog -assert svaext -timescale=1ns/1ps \
    -Mdir="$m64_run/csrc" -f "$m64_filelist" \
    -top tb_qfit_adaptive_parent_selector_p256_sustained_r2 \
    -o "$m64_run/simv" > "$m64_run/compile.raw.log" 2>&1
m64_rc=$?
set -e
printf '%s\n' "$m64_rc" > "$m64_run/compile.rc"
[[ "$m64_rc" -eq 0 && -x "$m64_run/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$m64_run/compile.raw.log"

echo "$m64_run/simv -no_save" > "$m64_run/sim.command.txt"
set +e
"$m64_run/simv" -no_save > "$m64_run/sim.raw.log" 2>&1
m64_rc=$?
set -e
printf '%s\n' "$m64_rc" > "$m64_run/sim.rc"
[[ "$m64_rc" -eq 0 ]]
grep -qx 'M64_R2_SUSTAINED_ASSERTION_MODULE_ACTIVE=1' "$m64_run/sim.raw.log"
grep -q '^PASS M64 R2 sustained ' "$m64_run/sim.raw.log"
! grep -Eiq 'Assertion failure|failed at|Offending|^Error|^Fatal' \
    "$m64_run/sim.raw.log"

python3 "$m64_builder" --run "$m64_run" --output "$m64_receipt" \
    > "$m64_run/receipt_builder.raw.log" 2>&1
grep -q '^PASS M64-r2 receipt builder ' "$m64_run/receipt_builder.raw.log"
python3 "$m64_validator" --mode full --run "$m64_run" \
    --receipt "$m64_receipt" > "$m64_run/validator.raw.log" 2>&1
grep -q '^PASS M64-r2 full validator ' "$m64_run/validator.raw.log"

(
    cd "$m64_run"
    sha256sum m64_parent_selector_sustained_vcs_receipt_r2.json \
        > receipt.sha256
)
m64_receipt_sha="$(sha256sum "$m64_receipt" | awk '{print $1}')"
{
    echo "status=PASS_EXACT_SHA_SYNOPSYS_VCS_SUSTAINED_R2"
    echo "receipt_sha256=$m64_receipt_sha"
    echo "tamper_attacks_rejected=6"
    echo "system_speedup_admitted=false"
    echo "headline_admitted=false"
    echo "ppa_admitted=false"
    echo "power_energy_admitted=false"
} > "$m64_run/RUN_COMPLETE.txt"

(
    cd "$m64_run"
    find . -type f ! -name output_manifest.sha256 \
        ! -name output_check.raw.log -print0 | sort -z | xargs -0 sha256sum \
        > output_manifest.sha256
    sha256sum --strict -c output_manifest.sha256 \
        > output_check.raw.log 2>&1
)

find "$m64_run" -type f -exec chmod 0444 {} +
find "$m64_run" -type d -exec chmod 0555 {} +
m64_complete=1
trap - EXIT
echo "M64_R2_SUSTAINED_VCS=PASS run=$m64_run receipt_sha256=$m64_receipt_sha"
