#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M405R3_RUN_DIR:-${task_hw_root}/results/m405r3_selected_slice_integration_vcs_r1_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"/{elastic,prefix,integration,m384}
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv"]="819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744"
    ["rtl_m405/m405_q32_serial16_zero_stop_controller.sv"]="f412ab817eb29ab303da9ec011379a853efc567fa5a6a29a73fced52cf503b42"
    ["rtl_m405/m405_q32_elastic_selected_slice.sv"]="91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd"
    ["verif_m405/m405_exact_elastic_pwp_issue_adapter_assertions.sv"]="372aaac64c02cc91ed97029338830cd5712202f7db9c5b550215428f633d1f4e"
    ["verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv"]="1e82a8e5ffbe80d0ae7a19a9d78899a2646d73f1994dc73796048dcfe2a47f11"
    ["verif_m405/m405_q32_elastic_selected_slice_assertions.sv"]="71a190e373ec0016cc09314276d03f3b40d7e7731c108b3734bc29c384abfa4b"
    ["tb_m405/tb_m405_exact_elastic_pwp_issue_adapter.sv"]="e26353cbed7c33ec39186218d1372f944820b493feb6dc8f38e37653756d721f"
    ["tb_m405/tb_m405_q32_serial16_zero_stop_controller.sv"]="5fef2c349b7d201af70969c5173e97b060bd427f750fcd8977eb7ae7aa93632a"
    ["tb_m405/tb_m405r3_q32_elastic_integration_repair.sv"]="72066f9b3562b9a15a7d531d8a21b125c9e53d609e00de29280b6c851b12a582"
    ["rtl_m384/m384_active_descriptor_streaming_controller.sv"]="7a93b60b327d7a92fb19028d754e3d5ed444c91c5a8d8a7ddd50ce03bb679512"
    ["verif_m384/m384_active_descriptor_streaming_controller_assertions.sv"]="214f41d45674c539fae3fec67d3988e894f04a0f0df2d87d132bc44c6d672d27"
    ["tb_m384/tb_m384_active_descriptor_streaming_controller.sv"]="ba76fc274b5a04b4d39f8187daee1980af69284662ab6b7cf072ce719efcde99"
    ["dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f"]="039288d349194b4e638732cf4dee67772b3d79c4d8df691aca1c5608950a88d5"
    ["dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f"]="0b2e19735cfb8ba8060bd5bbf161b808548e1361b9f8b38cd44600cdb689bbf1"
    ["dc_handoff/filelists/date_m405r3_q32_elastic_integration_vcs.f"]="3d6f4753c8a8a790d6970c64ded407b93ec86e7eff149b85058505463044f5b0"
    ["dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_vcs.f"]="8414f0b0a854ab83a394a322a661d89ad14b73cb563597f875e002085bb2dd82"
    ["contracts/m405r3_selected_slice_integration_vcs_contract_r1_20260826.json"]="70c1433fb4cb1ff91ffafbe8797e3222b2082ac73997fe8719cfd2cd847566a6"
    ["results/m406_m405r2_selected_leafs_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256"]="0bb224d6c48136d3bc921b1203646e83dfce61435f50d75b1955f53fcd5f25ee"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: >"${task_run}/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "${task_path}" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "${task_path}" \
        "${task_expected[${task_path}]}" "${task_observed}" \
        >>"${task_run}/preflight_sha_checks.txt"
    [[ "${task_observed}" == "${task_expected[${task_path}]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" >"${task_run}/input_sha256.txt"
cp contracts/m405r3_selected_slice_integration_vcs_contract_r1_20260826.json \
    "${task_run}/contract.json"
export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

run_one() {
    local task_name="$1" task_filelist="$2" task_top="$3" task_seed="$4"
    set +e
    "${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="${task_run}/${task_name}/csrc" -f "${task_filelist}" \
        -top "${task_top}" -o "${task_run}/${task_name}/simv" \
        >"${task_run}/${task_name}/compile.log" 2>&1
    local task_rc=$?
    set -e
    printf '%s\n' "${task_rc}" >"${task_run}/${task_name}/compile.rc"
    [[ ${task_rc} -eq 0 && -x "${task_run}/${task_name}/simv" ]] || return 20
    grep -Eiq 'Warning-\[|Error-\[|^Error' \
        "${task_run}/${task_name}/compile.log" && return 21 || true
    set +e
    "${task_run}/${task_name}/simv" "+ntb_random_seed=${task_seed}" \
        -no_save -cm assert \
        -assert "report=${task_run}/${task_name}/assert.report" \
        >"${task_run}/${task_name}/sim.log" 2>&1
    task_rc=$?
    set -e
    printf '%s\n' "${task_rc}" >"${task_run}/${task_name}/sim.rc"
    [[ ${task_rc} -eq 0 ]] || return 22
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout|mismatches=[1-9]' \
        "${task_run}/${task_name}/sim.log" \
        "${task_run}/${task_name}/assert.report" && return 23 || true
}

run_one elastic dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f \
    tb_m405_exact_elastic_pwp_issue_adapter 4055120260826 || exit $?
grep -Eq 'PASS M405A exact elastic PWP blocks=386 narrow=100 wide=286 contributions=672 .* protocol_attacks=4 atomic_leaks=0' \
    "${task_run}/elastic/sim.log" || exit 30

run_one prefix dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f \
    tb_m405_q32_serial16_zero_stop_controller 4055220260826 || exit $?
grep -Eq 'PASS M405B q32 serial16 rows=64 pass0=64 pass1=[1-9][0-9]* early=[1-9][0-9]* outputs=64 .* protocol_attacks=3 tie_lowest_id=true' \
    "${task_run}/prefix/sim.log" || exit 31

run_one integration dc_handoff/filelists/date_m405r3_q32_elastic_integration_vcs.f \
    tb_m405r3_q32_elastic_integration_repair 4055320260826 || exit $?
grep -Eq 'PASS M405R3 integration config_live_through_replay=1 legal_replay_after_last=1 legal_phase_release=1 global_fault_attacks=2 post_fault_accepts=0 accept_equations=true sticky_global_quiescence=true' \
    "${task_run}/integration/sim.log" || exit 32

run_one m384 dc_handoff/filelists/date_m384_active_descriptor_streaming_controller_vcs.f \
    tb_m384_active_descriptor_streaming_controller 4055420260826 || exit $?
grep -Eq 'PASS M384 active descriptor streaming controller phases=4 replays=8 bundles=10804 pwp_runs=[1-9][0-9]* .* protocol_attacks=10 .* mismatches=0 exact_compaction=true direct_address_runs=true' \
    "${task_run}/m384/sim.log" || exit 33

python3 - "${task_run}" \
    "${task_run}/m405r3_selected_slice_integration_vcs_receipt_r1.json" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def text(name): return (root/name/"sim.log").read_text(errors="replace")
a = re.search(r"blocks=(\d+) narrow=(\d+) wide=(\d+) contributions=(\d+).*protocol_attacks=(\d+) atomic_leaks=(\d+)", text("elastic"))
b = re.search(r"rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) outputs=(\d+) stalls=(\d+) protocol_attacks=(\d+)", text("prefix"))
m = re.search(r"phases=(\d+) replays=(\d+) bundles=(\d+) pwp_runs=(\d+).*protocol_attacks=(\d+).*mismatches=(\d+)", text("m384"))
if not a or not b or not m: raise SystemExit("missing M405R3 PASS ledger")
av,bv,mv = ([int(x) for x in z.groups()] for z in (a,b,m))
out={"schema":"m405r3_selected_slice_integration_vcs_receipt_v1",
 "status":"PASS_M405R3_DIRECTED_INTEGRATION_REPAIR_AND_Q32_M384",
 "tool":"Synopsys VCS V-2023.12-SP1",
 "elastic":{"blocks":av[0],"narrow":av[1],"wide":av[2],"contributions":av[3],"protocol_attacks":av[4],"atomic_leaks":av[5]},
 "prefix":{"rows":bv[0],"pass0":bv[1],"pass1":bv[2],"early":bv[3],"outputs":bv[4],"stalls":bv[5],"protocol_attacks":bv[6],"legal_phase_release":1},
 "integration":{"config_live_through_replay":True,"legal_replay_after_last":1,"legal_phase_release":1,"global_fault_attacks":2,"post_fault_accepts":0,"accept_equations":True,"sticky_global_quiescence":True},
 "m384":{"phases":mv[0],"replays":mv[1],"bundles":mv[2],"pwp_runs":mv[3],"protocol_attacks":mv[4],"mismatches":mv[5],"pwp_stride_bytes":640,"tile0_pwp_base":6240,"tile1_pwp_base":38912},
 "claim_boundary":{"directed_leaf_rtl":True,"directed_integration_rtl":True,"m384_q32_directed_regression":True,"full_static_codec_vcs":False,"full_real_q32_miter_vcs":False,"dc":False,"rtl_measured_speedup":False,"system_speedup":False,"headline":False}}
Path(sys.argv[2]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' PASS_M405R3_SELECTED_SLICE_INTEGRATION_SYNOPSYS_VCS \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M405R3 VCS sealed at ${task_run}"
