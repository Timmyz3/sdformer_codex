#!/usr/bin/env bash
set -euo pipefail

m64_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m64_hw="$(cd "$m64_dc_root/.." && pwd)"
m64_run=${M64_RUN_DIR:-$m64_dc_root/runs/m64_parent_selector_directed_vcs_r1_20260823}
m64_vcs=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
m64_complete=0

m64_fail_seal() {
    local m64_rc=$?
    if [[ -d "$m64_run" && "$m64_complete" -ne 1 ]]; then
        {
            echo "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE"
            echo "runner_exit_code=$m64_rc"
            echo "system_speedup_admitted=false"
        } > "$m64_run/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    fi
    exit "$m64_rc"
}
trap m64_fail_seal EXIT

[[ ! -e "$m64_run" ]] || {
    echo "refusing to overwrite M64 directed run: $m64_run" >&2
    exit 2
}
mkdir -p "$m64_run"
cd "$m64_hw"

declare -A m64_expected=(
    [rtl_m64/qfit_adaptive_parent_selector_p256.sv]=1178a0ae412a17059a2a2865025ff759b9fc351cbd7f20451f8621c92cce9fe8
    [verif_m64/qfit_adaptive_parent_selector_p256_assertions.sv]=b037f722667d8600c47b16f47293563cd6c70a22ed8a0da1d3af2e3a0c1c5b27
    [tb_m64/tb_qfit_adaptive_parent_selector_p256.sv]=82d317c8952771c8adc4fd61679b798aaee540d017bf1caf20bda5064544ffaf
    [dc_handoff/filelists/date_m64_parent_selector_directed_vcs.f]=561dab1f4d4e4d9d60633a79e437a683538ac01c1a8c375eeb3599f3bcc45591
    [contracts/m64_online_adaptive_parent_selector_directed_vcs_contract_r1_20260823.json]=c63d5265f56471a34cb5bc4c48b88260c4a61755fd14b5a89c669e9d5c81c5c0
)
: > "$m64_run/preflight_sha_checks.txt"
for m64_path in "${!m64_expected[@]}"; do
    m64_observed=$(sha256sum "$m64_path" | awk '{print $1}')
    printf 'path=%s expected=%s observed=%s\n' "$m64_path" \
        "${m64_expected[$m64_path]}" "$m64_observed" \
        >> "$m64_run/preflight_sha_checks.txt"
    [[ "$m64_observed" == "${m64_expected[$m64_path]}" ]]
done

export VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1
export VCS_ARCH_OVERRIDE=linux
echo "$m64_vcs -full64 -sverilog -assert svaext -f date_m64_parent_selector_directed_vcs.f" \
    > "$m64_run/compile.command.txt"
set +e
"$m64_vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps \
    -Mdir="$m64_run/csrc" \
    -f dc_handoff/filelists/date_m64_parent_selector_directed_vcs.f \
    -top tb_qfit_adaptive_parent_selector_p256 -o "$m64_run/simv" \
    > "$m64_run/compile.raw.log" 2>&1
m64_rc=$?
set -e
echo "$m64_rc" > "$m64_run/compile.rc"
[[ "$m64_rc" -eq 0 && -x "$m64_run/simv" ]]
! grep -Eiq 'Warning-\[|Error-\[|^Error' "$m64_run/compile.raw.log"

set +e
"$m64_run/simv" -no_save > "$m64_run/sim.raw.log" 2>&1
m64_rc=$?
set -e
echo "$m64_rc" > "$m64_run/sim.rc"
[[ "$m64_rc" -eq 0 ]]
grep -qx 'PASS M64 selector tests=4096 outputs=4096 parent_hits=1271,974,988,863 stalls=1074' \
    "$m64_run/sim.raw.log"
! grep -Eiq 'failed at|Offending|^Error|^Fatal' "$m64_run/sim.raw.log"

/usr/bin/python3.6 dc_handoff/scripts/build_m64_parent_selector_directed_vcs_receipt.py \
    --run "$m64_run" --output "$m64_run/m64_directed_vcs_receipt_r1.json"
/usr/bin/python3.6 dc_handoff/scripts/validate_m64_parent_selector_directed_vcs.py \
    --run "$m64_run" > "$m64_run/validation.raw.log" 2>&1
grep -q '^PASS M64 directed validator ' "$m64_run/validation.raw.log"
{
    echo "status=PASS_EXACT_SHA_DIRECTED_VCS_SVA"
    echo "system_speedup_admitted=false"
    echo "dc_sta_formality_admitted=false"
    echo "paper_ppa_ready=false"
} > "$m64_run/RUN_COMPLETE.txt"
(
    cd "$m64_run"
    find . -type f ! -name output.sha256 ! -name output_check.raw.log \
        -print0 | sort -z | xargs -0 sha256sum > output.sha256
    sha256sum --strict -c output.sha256 > output_check.raw.log 2>&1
)
m64_complete=1
trap - EXIT
echo "M64_DIRECTED_VCS=PASS run=$m64_run"
