#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M405R2_RUN_DIR:-${task_hw_root}/results/m405r2_selected_leafs_directed_vcs_r1_20260826}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}/elastic" "${task_run}/prefix"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv"]="819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744"
    ["rtl_m405/m405_q32_serial16_zero_stop_controller.sv"]="bf509984690e8b6fec83477df7c6c7e223c4c08a984fa82ca3d3348f602af613"
    ["rtl_m405/m405_q32_elastic_selected_slice.sv"]="5bf3c10f892751487688ceecdd9b68a4ad784e3a9a52cb60fe8cb96d66669184"
    ["verif_m405/m405_exact_elastic_pwp_issue_adapter_assertions.sv"]="372aaac64c02cc91ed97029338830cd5712202f7db9c5b550215428f633d1f4e"
    ["verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv"]="4cbb1557dbda9b525893dc4d2a1627e38a5e2d73feda827228e9e5b5d3ad023d"
    ["tb_m405/tb_m405_exact_elastic_pwp_issue_adapter.sv"]="e26353cbed7c33ec39186218d1372f944820b493feb6dc8f38e37653756d721f"
    ["tb_m405/tb_m405_q32_serial16_zero_stop_controller.sv"]="a1f32275bdec9ee860d97982c24946ddaa074ece8476cb5f8358a414ba42ff1e"
    ["dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f"]="039288d349194b4e638732cf4dee67772b3d79c4d8df691aca1c5608950a88d5"
    ["dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f"]="0b2e19735cfb8ba8060bd5bbf161b808548e1361b9f8b38cd44600cdb689bbf1"
    ["dc_handoff/filelists/date_m405_selected_slice_rtl.f"]="b3795bbbf04c36d2baad79c2e9dfd93393e8f04315a85864c3b7ca129d72952e"
    ["contracts/m405_selected_leafs_directed_vcs_contract_r1_20260826.json"]="fc1f6eb425a78498c8d2071b6e90ac9f8ad4458892e1f2eb291136400892ab64"
    ["contracts/m405_selected_leafs_directed_vcs_contract_r2_20260826.json"]="3352f728e1bf4b836714fabf70fb7de728ad724e25079c1d7b13abed567fd90d"
    ["results/m403_m401_elastic_pwp_selected_rtl_prereview_r1_20260826/SHA256SUMS.seal.sha256"]="d97b6132a70e32e676eaada660e4315ef1d62e6577d5e8f7d664e409f0b565ea"
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
cp contracts/m405_selected_leafs_directed_vcs_contract_r2_20260826.json \
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
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog timeout' \
        "${task_run}/${task_name}/sim.log" \
        "${task_run}/${task_name}/assert.report" && return 23 || true
    # VCS omits a failure-summary line when only cover properties are
    # reported. The fail/Offending scan above is therefore the fail-closed
    # zero-assertion-failure gate; requiring a nonexistent summary is wrong.
}

run_one elastic \
    dc_handoff/filelists/date_m405_exact_elastic_pwp_adapter_directed_vcs.f \
    tb_m405_exact_elastic_pwp_issue_adapter 405120260826 || exit $?
grep -Eq 'PASS M405A exact elastic PWP blocks=386 narrow=100 wide=286 contributions=672 .* no_gap_wide=66 protocol_attacks=4 atomic_leaks=0' \
    "${task_run}/elastic/sim.log" || exit 30

run_one prefix \
    dc_handoff/filelists/date_m405_q32_serial16_zero_stop_directed_vcs.f \
    tb_m405_q32_serial16_zero_stop_controller 405220260826 || exit $?
grep -Eq 'PASS M405B q32 serial16 rows=64 pass0=64 pass1=[1-9][0-9]* early=[1-9][0-9]* outputs=64 .* protocol_attacks=3 tie_lowest_id=true source_scratch_reads=0 descriptor_scratch=0 task_adjacency_observed=true' \
    "${task_run}/prefix/sim.log" || exit 31

python3 - "${task_run}/elastic/sim.log" "${task_run}/prefix/sim.log" \
    "${task_run}/m405r2_selected_leafs_directed_vcs_receipt_r1.json" <<'PY'
import json, re, sys
from pathlib import Path
a = re.search(r"blocks=(\d+) narrow=(\d+) wide=(\d+) contributions=(\d+) stalls=(\d+) no_gap_wide=(\d+) protocol_attacks=(\d+) atomic_leaks=(\d+)", Path(sys.argv[1]).read_text())
b = re.search(r"rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) outputs=(\d+) stalls=(\d+) protocol_attacks=(\d+)", Path(sys.argv[2]).read_text())
if not a or not b: raise SystemExit("missing PASS ledger")
av, bv = [int(x) for x in a.groups()], [int(x) for x in b.groups()]
out = {"schema":"m405r2_selected_leafs_directed_vcs_receipt_v1",
       "status":"PASS_M405R2_DIRECTED_LEAF_BRINGUP_ONLY",
       "tool":"Synopsys VCS V-2023.12-SP1",
       "elastic":{"blocks":av[0],"narrow_blocks":av[1],"wide_blocks":av[2],"contributions":av[3],"stall_cycles":av[4],"no_gap_wide_blocks":av[5],"protocol_attacks":av[6],"atomic_leaks":av[7]},
       "prefix":{"rows":bv[0],"pass0_tasks":bv[1],"pass1_tasks":bv[2],"early_stops":bv[3],"outputs":bv[4],"stall_cycles":bv[5],"protocol_attacks":bv[6]},
       "r1_failure_was_testbench_sampling_only":True,
       "claim_boundary":{"directed_leaf_rtl":True,"full_static_codec_vcs":False,"full_real_q32_miter_vcs":False,"m384_q32_regression":False,"integrated_vcs":False,"dc":False,"rtl_measured_speedup":False,"system_speedup":False,"headline":False}}
Path(sys.argv[3]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
PY

sha256sum "${task_runner}" >"${task_run}/runner_sha256.txt"
printf '%s\n' PASS_M405R2_SELECTED_EXACT_LEAFS_DIRECTED_SYNOPSYS_VCS \
    >"${task_run}/RUN_COMPLETE.txt"
find "${task_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${task_run}/RUN_MANIFEST.sha256"
sha256sum "${task_run}/RUN_MANIFEST.sha256" \
    >"${task_run}/RUN_MANIFEST.seal.sha256"
task_complete=1
echo "PASS M405r2 directed VCS sealed at ${task_run}"
