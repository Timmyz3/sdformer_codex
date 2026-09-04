#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${task_hw_root}/results/m518_matched_fixed_t10_atlif_vcs_r9_exact_20260827"
task_admission="${task_hw_root}/contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r9_20260827.json"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_expected_runner="${M518_EXPECTED_RUNNER_SHA256:-}"
task_expected_admission="${M518_EXPECTED_STATIC_ADMISSION_SHA256:-}"
task_observed_runner="$(sha256sum "${task_runner}" | awk '{print $1}')"

[[ ! -v M518_RUN_DIR ]] || exit 5
[[ "${task_expected_runner}" =~ ^[0-9a-f]{64}$ ]] || exit 3
[[ "${task_observed_runner}" == "${task_expected_runner}" ]] || exit 4
[[ "${task_expected_admission}" =~ ^[0-9a-f]{64}$ ]] || exit 6
[[ -f "${task_admission}" ]] || exit 7
[[ "$(sha256sum "${task_admission}" | awk '{print $1}')" == \
    "${task_expected_admission}" ]] || exit 8
[[ -f "${task_admission}.sha256" && \
   -f "${task_admission}.sha256.seal.sha256" ]] || exit 9
(cd "$(dirname "${task_admission}")" && \
    sha256sum -c "$(basename "${task_admission}").sha256" >/dev/null && \
    sha256sum -c "$(basename "${task_admission}").sha256.seal.sha256" >/dev/null) || exit 9
python3 - "${task_admission}" "${task_observed_runner}" <<'PY'
import json
import math
import sys

def reject(value):
    raise ValueError("non-finite JSON constant: " + value)

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    admission = json.load(handle, parse_constant=reject)

def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, child in value.items():
            finite(key)
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)

finite(admission)
if admission.get("authorized_runner_sha256") != sys.argv[2]:
    raise SystemExit("M518 r9 static admission runner identity mismatch")
if admission.get("authorized_invocations") != 1:
    raise SystemExit("M518 r9 static admission invocation count mismatch")
if admission.get("vcs_authorized") is not True:
    raise SystemExit("M518 r9 static admission does not authorize VCS")
if admission.get("dc_authorized") is not False:
    raise SystemExit("M518 r9 static admission unexpectedly authorizes DC")
if admission.get("required_result_path") != \
        "results/m518_matched_fixed_t10_atlif_vcs_r9_exact_20260827":
    raise SystemExit("M518 r9 static admission result identity drift")
PY
[[ ! -e "${task_run}" ]] || exit 2
mkdir "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m518/m518_matched_fixed_t10_atlif.sv"]="8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6"
    ["verif_m518/m518_matched_fixed_t10_atlif_assertions.sv"]="89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5"
    ["tb_m518/tb_m518_matched_fixed_t10_atlif.sv"]="8877512040c0677de58bc88c1cacd8056bb6f20026c24e3794f633682d962e56"
    ["dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f"]="09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea"
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r9_20260827.json"]="f99767c17e33000012de31873169544e68f1e9b8eaf3724257595d666004b11b"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r8_20260827.json"]="68055a1385918909eaee5f881b1e226ca8d8f03a1609917c51e0faf55d42fe9b"
    ["dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r8_exact.sh"]="fe457d7bbf93e72e913c55427696fb782dcc00dee80c74b1f4dba9c3edd01a52"
    ["reviews/m518_candidate_static_hammer_r8_independent_20260827/m518_candidate_static_hammer_r8_independent_20260827.json"]="07ea5f13a92e41cd50f0af68269e0846bdb71aa2dd38af5d1609c4b82a0582d9"
    ["reviews/m518_candidate_static_hammer_r8_independent_20260827/SHA256SUMS"]="1db05274448121be017abb63e031603d97fe591a0abc57cb371bc5bfe7f73923"
    ["reviews/m518_candidate_static_hammer_r8_independent_20260827/SHA256SUMS.seal.sha256"]="0650035952f888aac2cfd4345a995ffdc7442b0c761b2b0632b6356a58c7d26a"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r8_20260827.json"]="e28022f96b6f0026905c796d977568e5ca69bd9c6d9ec9882be7bd3dc768f5ff"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r8_20260827.json.sha256"]="ac5f7a8a24bc572a8a43d0640f2df1f0417f46d7a6ac9a242255e0e753366ce3"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_launch_admission_r8_20260827.json.sha256.seal.sha256"]="e71f3501f29b44a0fed0b9a5e15d54986c342dff8b5a6ccb46de038f18dfc0a1"
    ["reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827.json"]="ac7769c902322fcff2b52130d04356da42942fedc0eae81a27f52d0c4c32e2a3"
    ["reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827.md"]="d8991cd9384ef853cb501bf72d40ee6bf3d4acbe133ecc43ca017c68593483d2"
    ["reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/RUN_COMPLETE"]="3bbba2689dd1187571867ba5cfed3577c3b9e3eb4e84b70af42c3091fa2890a7"
    ["reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/SHA256SUMS"]="c6ba447e15b640c6c5d3d3ae4c788fffb1b167318a7f6d879da4537a907f6c1b"
    ["reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/SHA256SUMS.seal.sha256"]="ad35cd9fdc51efcfd084916d3f119267af837bdb99279bebecb9a7c9004bb2cb"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/RUN_FAILED_OR_INCOMPLETE.txt"]="0b5e0cf33d68ff4c29b8cc7f237a2328c09123b5e5edd7c000e60582bc95d466"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/compile.rc"]="9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/sim.rc"]="9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/compile.log"]="dcdf66aeddc9087a413bf2f557a1b361a40c41005dbf8b2a59bdad2ddf0249d0"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/sim.log"]="5d7a2e49e5b54c53b8525d1bff8b6f570ec9d62b0ba15bb219f4f36cee0171e2"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/assert.report"]="79713cbaf2826081c48e306537c3379716be836c3e2d61b62dc7e1c2496a1d1c"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/input_sha256.txt"]="a7546ee6be31f102649531b95b983cbedf2dcf4c89369249de01318fce280e27"
    ["results/m518_matched_fixed_t10_atlif_vcs_r8_exact_20260827/vcs_id.txt"]="f62d91ed6be84f085e21a12dcdaad502bfc130dbbe9d0726714b6cfc1ea26439"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

task_verify_inputs() {
    local destination="$1" negative="$2"
    local path expected observed mismatch=0
    mkdir -p "${destination}"
    : >"${destination}/preflight_sha_checks.txt"
    while IFS= read -r path; do
        expected="${task_expected[${path}]}"
        if [[ "${negative}" == 1 && "${path}" == \
              "tb_m518/tb_m518_matched_fixed_t10_atlif.sv" ]]; then
            expected="0000000000000000000000000000000000000000000000000000000000000000"
        fi
        if [[ -f "${path}" ]]; then
            observed="$(sha256sum "${path}" | awk '{print $1}')"
        else
            observed="MISSING"
        fi
        printf 'path=%s expected=%s observed=%s\n' \
            "${path}" "${expected}" "${observed}" \
            >>"${destination}/preflight_sha_checks.txt"
        [[ "${observed}" == "${expected}" ]] || mismatch=1
    done < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort)
    [[ ${mismatch} -eq 0 ]] || return 10
}

task_negative_dir="${task_run}/negative_preflight_control"
set +e
task_verify_inputs "${task_negative_dir}" 1
task_negative_rc=$?
set -e
printf '%s\n' "${task_negative_rc}" >"${task_negative_dir}/negative_preflight.rc"
[[ ${task_negative_rc} -eq 10 ]] || exit 11
[[ ! -e "${task_negative_dir}/compile.log" && \
   ! -e "${task_negative_dir}/simv" && \
   ! -e "${task_negative_dir}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r9.json" && \
   ! -e "${task_negative_dir}/RUN_COMPLETE.txt" ]] || exit 12
printf '%s\n' \
    "EXPECTED_FAIL_M518_R9_WRONG_TB_SHA_EXIT10_NO_TOOL_NO_POSITIVE_RECEIPT" \
    >"${task_negative_dir}/NEGATIVE_CONTROL_COMPLETE.txt"
(
    cd "${task_negative_dir}"
    find . -type f ! -name NEGATIVE_MANIFEST.sha256 \
        ! -name NEGATIVE_MANIFEST.seal.sha256 -print0 | LC_ALL=C sort -z \
        | xargs -0 sha256sum >NEGATIVE_MANIFEST.sha256
    sha256sum NEGATIVE_MANIFEST.sha256 >NEGATIVE_MANIFEST.seal.sha256
)

task_verify_inputs "${task_run}" 0 || exit 10
printf 'runner_expected=%s\nrunner_observed=%s\nstatic_admission=%s\n' \
    "${task_expected_runner}" "${task_observed_runner}" \
    "${task_expected_admission}" >"${task_run}/runner_identity.txt"
for task_sealed_dir in \
    reviews/m518_candidate_static_hammer_r8_independent_20260827 \
    reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827; do
    task_label="$(basename "${task_sealed_dir}")"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS) \
        >"${task_run}/${task_label}_manifest_check.txt"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS.seal.sha256) \
        >"${task_run}/${task_label}_outer_seal_check.txt"
done
while IFS= read -r path; do sha256sum "${path}"; done \
    < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort) \
    >"${task_run}/input_sha256.txt"
cp contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r9_20260827.json \
    "${task_run}/contract_draft.json"
cp "${task_admission}" "${task_run}/static_launch_admission.json"

python3 - <<'PY'
import hashlib
import json
import math
import re

def load_json(path):
    def reject(value):
        raise ValueError("non-finite JSON constant: " + value)
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject)
    def finite(member):
        if isinstance(member, float) and not math.isfinite(member):
            raise ValueError("non-finite JSON number")
        if isinstance(member, dict):
            for key, child in member.items():
                finite(key)
                finite(child)
        elif isinstance(member, list):
            for child in member:
                finite(child)
    finite(value)
    return value

def read(path):
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()

tb_path = "tb_m518/tb_m518_matched_fixed_t10_atlif.sv"
rtl_path = "rtl_m518/m518_matched_fixed_t10_atlif.sv"
sva_path = "verif_m518/m518_matched_fixed_t10_atlif_assertions.sv"
tb = read(tb_path)
rtl = read(rtl_path)
sva = read(sva_path)
contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r9_20260827.json")
failure = load_json(
    "reviews/m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827/"
    "m518_vcs_r8_release_partial_raw_failure_hammer_r6_20260827.json")
if contract.get("execution_state", {}).get("r9_static_readmission") is not False:
    raise SystemExit("M518 r9 contract self-authorization drift")
if failure.get("status") != \
        "DIAGNOSTIC_CONFIRMED__R8_V16_REDUNDANT_NEGEDGE_BUBBLE__R9_TB_ONLY_READMISSION_REQUIRED":
    raise SystemExit("M518 r8 failure review status drift")
if failure.get("failure_localization", {}).get("classification") != \
        "TESTBENCH_STIMULUS_CADENCE_P1__NOT_RTL_OR_ORACLE_P0":
    raise SystemExit("M518 r8 failure classification drift")
if failure.get("decision", {}).get("r9_vcs_authorized") is not False:
    raise SystemExit("M518 r8 failure review unexpectedly authorizes r9 VCS")
old = "#0.2;release_valid=1'b1;raw_valid=1'b1;"
new = "@(negedge clk_core);release_valid=1'b1;raw_valid=1'b1;"
if tb.count(old) != 1:
    raise SystemExit("M518 r9 release cadence fragment cardinality drift")
r8_tb = tb.replace(old, new, 1)
if hashlib.sha256(r8_tb.encode("utf-8")).hexdigest() != \
        "d03fd23a19046d7b96819f2f8b7753a03cb2cf3454564579b03647026a480de2":
    raise SystemExit("M518 r9 one-fragment reverse does not recover frozen r8 TB")
if tb.count("@(negedge clk_core);result_ready=1'b0;#0.2;") != 1:
    raise SystemExit("M518 r9 frozen V08 line-765 #0.2 settle drift")
if "$deposit" in tb or re.search(r"\bforce\s+u_dut\.", tb) or \
        re.search(r"\brelease\s+u_dut\.", tb):
    raise SystemExit("M518 r9 forbidden TB state-write instrumentation")
if re.findall(r"u_dut\.[A-Za-z_][A-Za-z0-9_]*(?:\[[^\n;]*\])?\s*(?:<=|=(?!=))", tb):
    raise SystemExit("M518 r9 hierarchical DUT-state LHS")
if rtl.count("always_ff") != 1 or re.search(r"always\s*@\s*\(posedge", rtl):
    raise SystemExit("M518 r9 DUT sequential ownership drift")
if "bind " in tb or "bind " in sva:
    raise SystemExit("M518 r9 writing-bind risk")
assertions = set(re.findall(r"\b(ap_[A-Za-z0-9_]+)\s*:", sva))
covers = set(re.findall(r"\b(cp_[A-Za-z0-9_]+)\s*:", sva))
if len(assertions) != 51 or len(covers) != 25:
    raise SystemExit("M518 r9 assertion/cover cardinality drift")
required = (
    "v06_hold_dense_issue=1'b1;",
    "v06_first_empty_fill_bank1=1'b1;",
    "send_frame_tile(legal_config,payload_bank1,tag_bank1);",
    "send_frame_tile(legal_config,payload_bank0,tag_bank0);",
    "finish_context(2,first_accept_time,1'b0,measured_cycles);",
    "#0.2;release_valid=1'b1;raw_valid=1'b1;",
    "@(negedge clk_core);result_ready=1'b0;#0.2;",
    "V08 phase12 close stall was not atomic",
    "V08 phase16 close stall was not atomic",
    "v06_legal_fill_harness=1 phase12_stall=1 phase16_stall=1",
)
if any(fragment not in tb for fragment in required):
    raise SystemExit("M518 r9 V06/V08/oracle preservation drift")
pass_line = (
    "PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 "
    "clean_N4=80 random_contexts=4 rail_boundary_points=6 "
    "zero_tile_held_edges=8 zero_tile_fault_transitions=1 "
    "release_state_attacks=5 reset_attacks=9 reset_partial_config=1 "
    "reset_partial_raw=1 reset_dense_c0=1 reset_dense_c11=1 "
    "reset_dense_c12=1 reset_dense_c15=1 reset_dense_c16=1 "
    "reset_fifo_full_close=1 reset_quarantine=1 clean_after_reset_N1=9 "
    "sequential_oldest=1 v06_legal_fill_harness=1 phase12_stall=1 "
    "phase16_stall=1 padding_attacks=216 raw_attacks=7 config_attacks=5 "
    "fault_edge_pop_push=1 slot_tuples_per_tile=1600 multiplier_slots=96 "
    "issue_cycles=17 vcs_only=true dc=false formality=false ptpx=false "
    "speedup=false ppa=false headline=false")
if tb.count(pass_line) != 1:
    raise SystemExit("M518 r9 exact PASS signature drift")
PY

printf '%s\n' \
    "PASS_M518_R9_TB_ONLY_RELEASE_CADENCE_REVERSE_SHA_PREFLIGHT_NO_COMPILE_YET" \
    >"${task_run}/PREFLIGHT_COMPLETE.txt"
export VCS_HOME="${task_vcs}" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
[[ -x "${task_vcs}/bin/vcs" ]] || exit 18
set +e
"${task_vcs}/bin/vcs" -full64 -ID >"${task_run}/vcs_id.txt" 2>&1
task_rc=$?
set -e
[[ ${task_rc} -eq 0 ]] || exit 19
grep -Fq 'V-2023.12-SP1' "${task_run}/vcs_id.txt" || exit 19

set +e
"${task_vcs}/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED +define+M518_VCS_V06_HARNESS \
    -timescale=1ns/1ps -cm assert \
    -Mdir="${task_run}/csrc" \
    -f dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f \
    -top tb_m518_matched_fixed_t10_atlif -o "${task_run}/simv" \
    >"${task_run}/compile.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/compile.rc"
[[ ${task_rc} -eq 0 && -x "${task_run}/simv" ]] || exit 20
grep -Eiq 'Error-\[|^Error|Fatal:' "${task_run}/compile.log" && exit 21 || true

set +e
"${task_run}/simv" +ntb_random_seed=51820260827 -no_save -cm assert \
    -assert report="${task_run}/assert.report" \
    >"${task_run}/sim.log" 2>&1
task_rc=$?
set -e
printf '%s\n' "${task_rc}" >"${task_run}/sim.rc"
[[ ${task_rc} -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog|timeout' \
    "${task_run}/sim.log" "${task_run}/assert.report" && exit 23 || true

task_pass='PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 clean_N4=80 random_contexts=4 rail_boundary_points=6 zero_tile_held_edges=8 zero_tile_fault_transitions=1 release_state_attacks=5 reset_attacks=9 reset_partial_config=1 reset_partial_raw=1 reset_dense_c0=1 reset_dense_c11=1 reset_dense_c12=1 reset_dense_c15=1 reset_dense_c16=1 reset_fifo_full_close=1 reset_quarantine=1 clean_after_reset_N1=9 sequential_oldest=1 v06_legal_fill_harness=1 phase12_stall=1 phase16_stall=1 padding_attacks=216 raw_attacks=7 config_attacks=5 fault_edge_pop_push=1 slot_tuples_per_tile=1600 multiplier_slots=96 issue_cycles=17 vcs_only=true dc=false formality=false ptpx=false speedup=false ppa=false headline=false'
grep -Fq "${task_pass}" "${task_run}/sim.log" || exit 30
task_required_covers=(
    cp_first_issue cp_first_close cp_tail_close cp_close_stall
    cp_phase12_stall cp_phase16_stall cp_result_stall cp_fifo_full
    cp_full_pop_push cp_raw_backpressure cp_release_wait cp_release
    cp_context_retire cp_fault cp_zero_tile_fault cp_config_frame_fault
    cp_raw_frame_fault cp_fault_with_pop_push cp_dual_ready_oldest_bank1
    cp_beat0 cp_beat1 cp_beat2 cp_beat3 cp_beat4 cp_reset_recovery
)
for task_cover in "${task_required_covers[@]}"; do
    grep -Eq "${task_cover}, .* [1-9][0-9]* match" \
        "${task_run}/assert.report" || exit 31
done

python3 - "${task_run}/assert.report" \
    "${task_negative_dir}/NEGATIVE_MANIFEST.sha256" \
    "${task_negative_dir}/NEGATIVE_MANIFEST.seal.sha256" \
    "${task_run}/vcs_id.txt" "${task_observed_runner}" \
    "${task_expected_admission}" \
    "${task_run}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r9.json" <<'PY'
import hashlib
import json
import math
import re
import sys

def digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()

with open(sys.argv[1], "r", encoding="utf-8", errors="replace") as handle:
    report = handle.read()
covers = {name: int(count) for name, count in re.findall(
    r"u_sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match", report)}
required = {
    "cp_first_issue", "cp_first_close", "cp_tail_close", "cp_close_stall",
    "cp_phase12_stall", "cp_phase16_stall", "cp_result_stall",
    "cp_fifo_full", "cp_full_pop_push", "cp_raw_backpressure",
    "cp_release_wait", "cp_release", "cp_context_retire", "cp_fault",
    "cp_zero_tile_fault", "cp_config_frame_fault", "cp_raw_frame_fault",
    "cp_fault_with_pop_push", "cp_dual_ready_oldest_bank1",
    "cp_beat0", "cp_beat1", "cp_beat2", "cp_beat3", "cp_beat4",
    "cp_reset_recovery",
}
if set(covers) != required or any(covers[name] < 1 for name in required):
    raise SystemExit("M518 r9 assertion cover drift")
receipt = {
    "schema": "m518_matched_fixed_t10_atlif_author_vcs_receipt_v9",
    "status": "PASS_M518_R9_TB_ONLY_RELEASE_CADENCE_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN",
    "role": "author_campaign_not_independent_review",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "actual_vcs_id_sha256": digest(sys.argv[4]),
    "exact_runner_sha256": sys.argv[5],
    "static_launch_admission_sha256": sys.argv[6],
    "tb_only_repair": {
        "post_send_config_skew_ns": 0.2,
        "r8_tb_reverse_sha_exact": True,
        "v08_line765_settle_preserved": True,
        "rtl_sha_unchanged": True,
        "sva_sha_unchanged": True,
        "phase_flow_unchanged": True,
        "v06_unchanged": True,
        "expected_cycles_unchanged": True,
        "numeric_oracle_unchanged": True,
        "cover_requirements_unchanged": True,
    },
    "negative_control": {
        "status": "EXPECTED_FAIL_WRONG_TB_SHA_EXIT10_BEFORE_TOOL",
        "exit_code": 10,
        "manifest_sha256": digest(sys.argv[2]),
        "outer_seal_file_sha256": digest(sys.argv[3]),
    },
    "campaign": {
        "tests": "SEALED_V01_V20_WITH_ORIGINAL_NUMBERING",
        "clean_cycles_N1": 29,
        "clean_cycles_N4": 80,
        "sequential_oldest_attacks": 1,
        "v06_legal_fill_harness": 1,
        "numeric_mismatches": 0,
        "assertion_failures": 0,
        "assertion_cover_matches": covers,
    },
    "claim_boundary": {
        "author_vcs": True,
        "independent_review": False,
        "simulation_only_harness_is_paper_hardware": False,
        "dc": False,
        "formality": False,
        "sta": False,
        "ptpx": False,
        "power": False,
        "energy": False,
        "speedup": False,
        "ppa": False,
        "system_speedup": False,
        "headline": False,
    },
}
with open(sys.argv[7], "w", encoding="utf-8") as handle:
    json.dump(receipt, handle, allow_nan=False, indent=2, sort_keys=True)
    handle.write("\n")
with open(sys.argv[7], "r", encoding="utf-8") as handle:
    round_trip = json.load(
        handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
if round_trip != receipt:
    raise SystemExit("strict finite M518 r9 receipt round-trip drift")
for value in covers.values():
    if not math.isfinite(float(value)):
        raise SystemExit("non-finite cover count")
PY

printf '%s  %s\n' "${task_observed_runner}" "${task_runner}" \
    >"${task_run}/runner_sha256.txt"
printf '%s\n' \
    "PASS_M518_R9_TB_ONLY_RELEASE_CADENCE_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN" \
    >"${task_run}/RUN_COMPLETE.txt"
(
    cd "${task_run}"
    find . -type f ! -name simv ! -path './csrc/*' \
        ! -path './simv.daidir/*' ! -path './simv.vdb/*' \
        ! -name RUN_MANIFEST.sha256 ! -name RUN_MANIFEST.seal.sha256 \
        ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum >RUN_MANIFEST.sha256
    sha256sum RUN_MANIFEST.sha256 >RUN_MANIFEST.seal.sha256
    cp RUN_MANIFEST.sha256 SHA256SUMS
    sha256sum SHA256SUMS >SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M518 r9 exact-SHA VCS author campaign at ${task_run}"
