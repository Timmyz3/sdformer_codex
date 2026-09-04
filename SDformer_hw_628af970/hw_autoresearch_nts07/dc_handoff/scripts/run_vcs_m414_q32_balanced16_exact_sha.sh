#!/usr/bin/env bash
set -euo pipefail

m414_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
m414_hw="$(cd "${m414_dc_root}/.." && pwd)"
m414_runner="$(realpath "${BASH_SOURCE[0]}")"
m414_run="${M414_VCS_RUN_DIR:-${m414_hw}/results/m414_q32_balanced16_vcs_r1_20260826}"
m414_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
m414_config="${m414_hw}/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_phase_config_768.memh"
m414_rows="${m414_hw}/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/m410r2_h67_q32_runtime_rows_32.memh"
m414_contract="contracts/m414_q32_balanced16_zero_stop_vcs_contract_r1_20260826.json"

m414_sha() { sha256sum "$1" | awk '{print $1}'; }
m414_expect() {
    local m414_path=$1
    local m414_expected=$2
    [[ -f "${m414_path}" ]] || exit 3
    [[ "$(m414_sha "${m414_path}")" == "${m414_expected}" ]] || exit 3
}

[[ ! -e "${m414_run}" ]] || exit 5
mkdir -p "${m414_run}"/{directed,integration,full_runtime}
m414_complete=0
trap 'm414_rc=$?; if [[ ${m414_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${m414_rc}" >"${m414_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${m414_hw}"

m414_expect "${m414_vcs}/bin/vcs" 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287
m414_expect "${m414_contract}" 239bc95d4b460672cdc57bc2c3694a5ea3962b1ece2e3c2f64c6464a6fa24dd5
m414_expect rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
m414_expect rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv 819bee3d13d80519778a6f23218b15afec97d2d6677693f1014a2ba38e2c8744
m414_expect rtl_m405/m405_q32_elastic_selected_slice.sv 91a47ee17a85b35224fa59047971292346e8ef806b0acaadd9b42d88dcb476fd
m414_expect verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv 1e82a8e5ffbe80d0ae7a19a9d78899a2646d73f1994dc73796048dcfe2a47f11
m414_expect verif_m405/m405_q32_elastic_selected_slice_assertions.sv 71a190e373ec0016cc09314276d03f3b40d7e7731c108b3734bc29c384abfa4b
m414_expect tb_m405/tb_m405_q32_serial16_zero_stop_controller.sv 5fef2c349b7d201af70969c5173e97b060bd427f750fcd8977eb7ae7aa93632a
m414_expect tb_m405/tb_m410_h67_q32_full_runtime_vcs.sv 3f0c13b3bd8c03797b805cb01db216f48c81840e1413c8b037b277e47334d751
m414_expect tb_m405/tb_m405r3_q32_elastic_integration_repair.sv 72066f9b3562b9a15a7d531d8a21b125c9e53d609e00de29280b6c851b12a582
m414_expect dc_handoff/filelists/date_m414_q32_balanced16_directed_vcs.f ec5e5b17e66388a057aafe70f3ef0fc5325edbddc757625ebcf53ee996cd73e0
m414_expect dc_handoff/filelists/date_m414_q32_balanced16_full_runtime_vcs.f 39d702d1702b768cc63863d2aa07a7c0d74d699beb6e07608d73c4f619cf9e79
m414_expect dc_handoff/filelists/date_m414_balanced_selected_slice_integration_vcs.f fec7067b3686cc992edc0bdc35836dc2ef8f960acbf9bbf9c5dc9708b609756e
m414_expect "${m414_config}" b35178f06529940403fea28b3d04dbf56eb8686a90f11e20d5678e6cfb348c04
m414_expect "${m414_rows}" 6e03352b89eff1955825334b4dedd991db8c975a9ef6662fe0317e73ccfa8334
m414_expect results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256 31abafb9e39e2a9fa39b348b0ab9954805ec94e58f1006a6f2d57e5d24946efc
m414_expect contracts/m413_m412_dual_dc_independent_hammer_contract_r1_20260826.json 50e145edf32ee2ed00ae1c028b79960f55602fda867604f26c70313b3e95941d
m414_expect results/m413_m412_dual_dc_independent_hammer_r1_20260826/m413_m412_dual_dc_independent_hammer_review_r1.json de9bf0d1d0da77f13185e91f7f23681255b2966782a7a33a8db191633973728d
m414_expect results/m413_m412_dual_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 e783d69a4cd266130eb0c5a49eedbb7fe2adb212926fa34e6bcdbe74ca51c571
m414_expect docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

(
    cd results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m414_run}/stimulus_seal_check.log" 2>&1
(
    cd results/m413_m412_dual_dc_independent_hammer_r1_20260826
    sha256sum -c SHA256SUMS
    sha256sum -c SHA256SUMS.seal.sha256
) >"${m414_run}/m413_seal_check.log" 2>&1
sha256sum \
    rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
    rtl_m405/m405_exact_elastic_pwp_issue_adapter.sv \
    rtl_m405/m405_q32_elastic_selected_slice.sv \
    verif_m405/m405_q32_serial16_zero_stop_controller_assertions.sv \
    verif_m405/m405_q32_elastic_selected_slice_assertions.sv \
    tb_m405/tb_m405_q32_serial16_zero_stop_controller.sv \
    tb_m405/tb_m410_h67_q32_full_runtime_vcs.sv \
    tb_m405/tb_m405r3_q32_elastic_integration_repair.sv \
    dc_handoff/filelists/date_m414_q32_balanced16_directed_vcs.f \
    dc_handoff/filelists/date_m414_q32_balanced16_full_runtime_vcs.f \
    dc_handoff/filelists/date_m414_balanced_selected_slice_integration_vcs.f \
    "${m414_config}" "${m414_rows}" "${m414_contract}" \
    results/m413_m412_dual_dc_independent_hammer_r1_20260826/SHA256SUMS.seal.sha256 \
    docs/359_DATE终局冻结_20260813.md >"${m414_run}/input_sha256.txt"
cp "${m414_contract}" "${m414_run}/contract.json"
export VCS_HOME="${m414_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"

m414_compile() {
    local m414_name=$1
    local m414_filelist=$2
    local m414_top=$3
    local m414_out="${m414_run}/${m414_name}"
    set +e
    "${m414_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="${m414_out}/csrc" -f "${m414_filelist}" \
        -top "${m414_top}" -o "${m414_out}/simv" \
        >"${m414_out}/compile.log" 2>&1
    local m414_rc=$?
    set -e
    printf '%s\n' "${m414_rc}" >"${m414_out}/compile.rc"
    [[ ${m414_rc} -eq 0 && -x "${m414_out}/simv" ]] || return 20
    grep -Eiq 'Warning-\[|Error-\[|^Error' "${m414_out}/compile.log" && return 21 || true
}

m414_sim_directed() {
    local m414_name=$1
    local m414_seed=$2
    local m414_out="${m414_run}/${m414_name}"
    set +e
    "${m414_out}/simv" "+ntb_random_seed=${m414_seed}" -no_save \
        -cm assert -assert "report=${m414_out}/assert.report" \
        >"${m414_out}/sim.log" 2>&1
    local m414_rc=$?
    set -e
    printf '%s\n' "${m414_rc}" >"${m414_out}/sim.rc"
    [[ ${m414_rc} -eq 0 ]] || return 22
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
        "${m414_out}/sim.log" "${m414_out}/assert.report" && return 23 || true
}

m414_compile directed dc_handoff/filelists/date_m414_q32_balanced16_directed_vcs.f \
    tb_m405_q32_serial16_zero_stop_controller
m414_sim_directed directed 4140120260826
grep -Eq 'PASS M405B q32 serial16 rows=64 pass0=64 pass1=61 early=1 outputs=64 .* protocol_attacks=3 tie_lowest_id=true .* task_adjacency_observed=true' \
    "${m414_run}/directed/sim.log" || exit 30

m414_compile integration dc_handoff/filelists/date_m414_balanced_selected_slice_integration_vcs.f \
    tb_m405r3_q32_elastic_integration_repair
m414_sim_directed integration 4140220260826
grep -Eq 'PASS M405R3 integration config_live_through_replay=1 legal_replay_after_last=1 legal_phase_release=1 global_fault_attacks=2 post_fault_accepts=0 accept_equations=true sticky_global_quiescence=true' \
    "${m414_run}/integration/sim.log" || exit 31

m414_compile full_runtime dc_handoff/filelists/date_m414_q32_balanced16_full_runtime_vcs.f \
    tb_m410_h67_q32_full_runtime_vcs
m414_start="$(date +%s)"
set +e
"${m414_run}/full_runtime/simv" +ntb_random_seed=4140320260826 \
    "+M410_CONFIG=${m414_config}" "+M410_ROWS=${m414_rows}" \
    -no_save -cm assert \
    -assert "report=${m414_run}/full_runtime/assert.report" \
    >"${m414_run}/full_runtime/sim.log" 2>&1
m414_rc=$?
set -e
m414_end="$(date +%s)"
printf '%s\n' "${m414_rc}" >"${m414_run}/full_runtime/sim.rc"
printf '%s\n' "$((m414_end-m414_start))" >"${m414_run}/full_runtime/sim_wall_seconds.txt"
[[ ${m414_rc} -eq 0 ]] || exit 32
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|mismatches=[1-9]' \
    "${m414_run}/full_runtime/sim.log" \
    "${m414_run}/full_runtime/assert.report" && exit 33 || true
grep -Eq 'PASS M410 full ordered q32 runtime phases=17280 configs=17280 rows=51840000 pass0=51840000 pass1=16037540 early=3751608 pwp=16971357 results=51840000 metadata_mismatches=0 arithmetic_mismatches=0 config_mismatches=0 task_flag_mismatches=0 tie_lowest_id=true exact_runtime_order=true system_speedup=false headline=false cycles=[0-9]+' \
    "${m414_run}/full_runtime/sim.log" || exit 34

python3 - "${m414_run}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
directed = (root / "directed/sim.log").read_text(errors="replace")
integration = (root / "integration/sim.log").read_text(errors="replace")
runtime = (root / "full_runtime/sim.log").read_text(errors="replace")
d = re.search(r"rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) outputs=(\d+) stalls=(\d+) protocol_attacks=(\d+)", directed)
r = re.search(r"phases=(\d+) configs=(\d+) rows=(\d+) pass0=(\d+) pass1=(\d+) early=(\d+) pwp=(\d+) results=(\d+) metadata_mismatches=(\d+) arithmetic_mismatches=(\d+) config_mismatches=(\d+) task_flag_mismatches=(\d+).*cycles=(\d+)", runtime)
if not d or not r or "PASS M405R3 integration" not in integration:
    raise SystemExit("missing M414 PASS payload")
dv = [int(x) for x in d.groups()]
rv = [int(x) for x in r.groups()]
expected_d = [64, 64, 61, 1, 64, 11, 3]
expected_r = [17280, 17280, 51840000, 51840000, 16037540,
              3751608, 16971357, 51840000, 0, 0, 0, 0]
if dv != expected_d or rv[:12] != expected_r:
    raise SystemExit("M414 ledger drift")
receipt = {
    "schema": "m414_q32_balanced16_zero_stop_vcs_receipt_v1",
    "status": "PASS_M414_EXACT_BALANCED16_DIRECTED_INTEGRATION_FULL_RUNTIME_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "directed": {"rows": dv[0], "pass0_tasks": dv[1],
                 "pass1_tasks": dv[2], "early_stops": dv[3],
                 "outputs": dv[4], "stall_cycles": dv[5],
                 "protocol_attacks": dv[6]},
    "integration": {"config_live_through_replay": 1,
                    "legal_replay_after_last": 1,
                    "legal_phase_release": 1,
                    "global_fault_attacks": 2,
                    "post_fault_accepts": 0},
    "full_runtime": {"phases": rv[0], "configs": rv[1],
                     "source_rows": rv[2], "pass0_tasks": rv[3],
                     "pass1_tasks": rv[4], "early_stops": rv[5],
                     "pwp_rows": rv[6], "results": rv[7],
                     "mismatches": sum(rv[8:12]),
                     "matcher_task_cycles": rv[3] + rv[4],
                     "m401_matcher_cycles": rv[3] + rv[4] + 2*rv[0],
                     "testbench_raw_cycles_not_speed": rv[12]},
    "full_runtime_wall_seconds": int((root / "full_runtime/sim_wall_seconds.txt").read_text()),
    "equivalence": {"lowest_id_total_order": True,
                    "added_pipeline_stages": 0,
                    "task_ledger_change": 0,
                    "result_or_flag_mismatches": 0},
    "claim_boundary": {"exact_balanced_tournament_rtl": True,
                       "directed_vcs": True,
                       "integration_vcs": True,
                       "full_runtime_vcs": True,
                       "cycle_speedup_changed": False,
                       "accuracy_changed": False,
                       "dc": False, "formality": False,
                       "primetime": False, "energy": False,
                       "system_speedup": False, "date_headline": False},
}
(root / "m414_q32_balanced16_zero_stop_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

sha256sum "${m414_runner}" >"${m414_run}/runner_sha256.txt"
printf '%s\n' PASS_M414_EXACT_BALANCED16_DIRECTED_INTEGRATION_FULL_RUNTIME_VCS \
    >"${m414_run}/RUN_COMPLETE.txt"
find "${m414_run}" -type f ! -name simv ! -path '*/csrc/*' \
    ! -path '*/simv.daidir/*' ! -path '*/simv.vdb/*' \
    ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
    -print0 | sort -z | xargs -0 sha256sum \
    >"${m414_run}/RUN_MANIFEST.sha256"
sha256sum "${m414_run}/RUN_MANIFEST.sha256" \
    >"${m414_run}/RUN_MANIFEST.seal.sha256"
m414_complete=1
echo "PASS_M414_EXACT_BALANCED16_VCS run=${m414_run}"
