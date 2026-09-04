#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M518_RUN_DIR:-${task_hw_root}/results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_expected_runner="${M518_EXPECTED_RUNNER_SHA256:-}"
task_observed_runner="$(sha256sum "${task_runner}" | awk '{print $1}')"

# Exact runner authorization is required out of band before any result, negative
# control, tool query, compile, or publication side effect.
[[ "${task_expected_runner}" =~ ^[0-9a-f]{64}$ ]] || exit 3
[[ "${task_observed_runner}" == "${task_expected_runner}" ]] || exit 4
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m518/m518_matched_fixed_t10_atlif.sv"]="90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a"
    ["verif_m518/m518_matched_fixed_t10_atlif_assertions.sv"]="977f95652bb788047549d58ff94e416f00542c9d3e63fa6f83e09fe582c910f4"
    ["tb_m518/tb_m518_matched_fixed_t10_atlif.sv"]="e7973a91d04b9f20542b04c58a213e2c8929259768701664dda76721720d2888"
    ["dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f"]="09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea"
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r5_20260827.json"]="51b81bbad3eaa05269e209a68019cb9aa9a29e822d03c283df2f5c68fb2ff996"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r4_20260827.json"]="5a424e5c58c83a1047fde09108a2a812a704175396a974db7d79041a4a5d66cc"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/m518_atlif_fixed_baseline_spec_r1_20260827.md"]="f50376b28f4f69ab8d257d06ea40553ece43d59cde24e90c4e78e5437afba083"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/m518_atlif_fixed_baseline_spec_r1_20260827.json"]="a4b57569d86dca3f0f906565d9b5f7be97335946ac91e38a536d73dca3f2bee1"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/RUN_COMPLETE"]="09dab11ee1ceeafa810c6f91889db8c2929aab41302290ac36f6510d88d2200a"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/SHA256SUMS"]="177851f6d773c78366382b1cd1e3a64d6e47e06edab0c0fd7c732ba2fdf63d74"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/SHA256SUMS.seal.sha256"]="1a06765ec9bf602cbd2e4b5bda938360713e91a9befa65e1b68aff7e29974bb0"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/m518_vcs_compile_failure_hammer_r2_20260827.json"]="3a1b31f5c5d7c99ed4541b1c70838ef2fe87c23162ce62f64cae97dc04af3938"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/m518_vcs_compile_failure_hammer_r2_20260827.md"]="731b274e125fcef2bb824035797c66a6c07715a2339b6fd4d7695298bc1c00c6"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/RUN_COMPLETE"]="6d891a6914bafb9314b4513be413c182f6263fc2e85f7a766ba1d73e17616a43"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/SHA256SUMS"]="e865499448d81012ee9112fa6dcc48093edb9a6f263d6f548288d6c77d6c4c1c"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/SHA256SUMS.seal.sha256"]="860162d25c9af9118960145582132ea975b32cd959957dfa2793ad7f9927cd0b"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/RUN_FAILED_OR_INCOMPLETE.txt"]="0b71bda8d02ca4d69532170e5592679f03858fb0b3a4464fc4a68ca839e2ee23"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/compile.log"]="a44557c1eb9e997197f88dad19c122116410c33fd5372ae30395d864ab1de2da"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/compile.rc"]="ce8bafb38615aeb5d44ebbabe78ec14ac35a5de87bdc5ad5ea82a72656024ce4"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/vcs_id.txt"]="f62d91ed6be84f085e21a12dcdaad502bfc130dbbe9d0726714b6cfc1ea26439"
    ["dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r4_exact.sh"]="d656d11dc32e11e018c7035112567a5b0b2de52dc5e2ad6073778295883ef55b"
    ["reviews/m518_candidate_static_hammer_r4_20260827/m518_candidate_static_hammer_r4_20260827.json"]="d989582c2a5ea15ed2a2e8cee223c242440c09ac0f795b3295a6b19344eba3ac"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

task_verify_inputs() {
    local task_destination="$1"
    local task_negative="$2"
    local task_path task_expected_sha task_observed_sha task_mismatch=0
    mkdir -p "${task_destination}"
    : >"${task_destination}/preflight_sha_checks.txt"
    while IFS= read -r task_path; do
        task_expected_sha="${task_expected[${task_path}]}"
        if [[ "${task_negative}" == "1" && "${task_path}" == \
                "rtl_m518/m518_matched_fixed_t10_atlif.sv" ]]; then
            task_expected_sha="0000000000000000000000000000000000000000000000000000000000000000"
        fi
        if [[ -f "${task_path}" ]]; then
            task_observed_sha="$(sha256sum "${task_path}" | awk '{print $1}')"
        else
            task_observed_sha="MISSING"
        fi
        printf 'path=%s expected=%s observed=%s\n' \
            "${task_path}" "${task_expected_sha}" "${task_observed_sha}" \
            >>"${task_destination}/preflight_sha_checks.txt"
        if [[ "${task_observed_sha}" != "${task_expected_sha}" ]]; then
            task_mismatch=1
        fi
    done < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort)
    [[ ${task_mismatch} -eq 0 ]] || return 10
}

# Automatic negative control remains structurally disjoint from positive VCS.
task_negative_dir="${task_run}/negative_preflight_control"
set +e
task_verify_inputs "${task_negative_dir}" 1
task_negative_rc=$?
set -e
printf '%s\n' "${task_negative_rc}" >"${task_negative_dir}/negative_preflight.rc"
[[ ${task_negative_rc} -eq 10 ]] || exit 11
[[ ! -e "${task_negative_dir}/compile.log" \
    && ! -e "${task_negative_dir}/simv" \
    && ! -e "${task_negative_dir}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r5.json" \
    && ! -e "${task_negative_dir}/RUN_COMPLETE.txt" ]] || exit 12
printf '%s\n' \
    "EXPECTED_FAIL_M518_R5_WRONG_RTL_SHA_EXIT10_NO_TOOL_NO_POSITIVE_RECEIPT" \
    >"${task_negative_dir}/NEGATIVE_CONTROL_COMPLETE.txt"
(
    cd "${task_negative_dir}"
    find . -type f ! -name NEGATIVE_MANIFEST.sha256 \
        ! -name NEGATIVE_MANIFEST.seal.sha256 -print0 | LC_ALL=C sort -z \
        | xargs -0 sha256sum >NEGATIVE_MANIFEST.sha256
    sha256sum NEGATIVE_MANIFEST.sha256 >NEGATIVE_MANIFEST.seal.sha256
)

task_verify_inputs "${task_run}" 0 || exit 10
printf 'runner_expected=%s\nrunner_observed=%s\n' \
    "${task_expected_runner}" "${task_observed_runner}" \
    >"${task_run}/runner_identity.txt"

task_spec_dir="reviews/m518_atlif_fixed_baseline_spec_r1_20260827"
(
    cd "${task_spec_dir}"
    sha256sum -c SHA256SUMS
) >"${task_run}/sealed_spec_manifest_check.txt"
(
    cd "${task_spec_dir}"
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_run}/sealed_spec_outer_seal_check.txt"

task_failure_review_dir="reviews/m518_vcs_compile_failure_hammer_r2_20260827"
(
    cd "${task_failure_review_dir}"
    sha256sum -c SHA256SUMS
) >"${task_run}/r4_failure_review_manifest_check.txt"
(
    cd "${task_failure_review_dir}"
    sha256sum -c SHA256SUMS.seal.sha256
) >"${task_run}/r4_failure_review_outer_seal_check.txt"

while IFS= read -r task_path; do sha256sum "${task_path}"; done \
    < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort) \
    >"${task_run}/input_sha256.txt"
cp contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r5_20260827.json \
    "${task_run}/contract_draft.json"

python3 - <<'PY'
import hashlib
import json
import math
import re
from pathlib import Path

def reject_nonfinite(value):
    raise ValueError("non-finite JSON constant: " + value)

def assert_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            assert_finite(key)
            assert_finite(member)
    elif isinstance(value, list):
        for member in value:
            assert_finite(member)

def load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=reject_nonfinite)
    assert_finite(value)
    return value

rtl_path = Path("rtl_m518/m518_matched_fixed_t10_atlif.sv")
sva_path = Path("verif_m518/m518_matched_fixed_t10_atlif_assertions.sv")
tb_path = Path("tb_m518/tb_m518_matched_fixed_t10_atlif.sv")
filelist_path = Path(
    "dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f")
contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r5_20260827.json")
r4_contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r4_20260827.json")
failure_review = load_json(
    "reviews/m518_vcs_compile_failure_hammer_r2_20260827/"
    "m518_vcs_compile_failure_hammer_r2_20260827.json")
rtl = rtl_path.read_text(encoding="utf-8")
sva = sva_path.read_text(encoding="utf-8")
tb = tb_path.read_text(encoding="utf-8")

if contract.get("status") != \
        "DRAFT_R5_EXACT_SIX_TOKEN_RENAME_NOT_STATICALLY_READMITTED_NOT_EXECUTED_DO_NOT_CITE":
    raise SystemExit("M518 r5 contract status drift")
if contract.get("execution_state", {}).get("r5_static_readmission") is not False:
    raise SystemExit("M518 r5 self-authorization drift")
if failure_review.get("status") != \
        "DIAGNOSTIC_CONFIRMED__R4_SYSTEMVERILOG_RESERVED_TOKEN_FAILURE__R5_STATIC_READMISSION_REQUIRED":
    raise SystemExit("M518 sealed r4 failure classification drift")
if failure_review.get("decision", {}).get("r5_runner_execution_authorized") \
        is not False:
    raise SystemExit("M518 failure audit unexpectedly authorizes r5")
if failure_review.get("minimum_r5_repair", {}).get(
        "all_six_word_tokens_must_change") is not True:
    raise SystemExit("M518 six-token repair requirement drift")

# Exact mutation proof: reversing precisely six word tokens reconstructs the
# frozen r4 RTL byte for byte. This excludes every other RTL edit.
if re.search(r"\bwithin\b", rtl):
    raise SystemExit("reserved identifier within remains in M518 r5 RTL")
if len(re.findall(r"\btap_within\b", rtl)) != 6:
    raise SystemExit("M518 r5 tap_within token count is not exactly six")
restored, replacements = re.subn(r"\btap_within\b", "within", rtl)
if replacements != 6 or hashlib.sha256(restored.encode("utf-8")).hexdigest() \
        != "09b1d976595f13885da917dd33b9ce87c403750eb8bb6e42a9aff1379a93412a":
    raise SystemExit("M518 r5 reverse mutation does not reconstruct r4 RTL")

expected_filelist = [
    "rtl_m518/m518_matched_fixed_t10_atlif.sv",
    "verif_m518/m518_matched_fixed_t10_atlif_assertions.sv",
    "tb_m518/tb_m518_matched_fixed_t10_atlif.sv",
]
if [line.strip() for line in filelist_path.read_text().splitlines()
        if line.strip()] != expected_filelist:
    raise SystemExit("M518 r5 filelist topology drift")

def public_ports(path, module):
    source = Path(path).read_text(encoding="utf-8")
    start = source.index("module " + module)
    header = source[start:source.index(");", start) + 2]
    ports = []
    for line in header.splitlines():
        match = re.match(
            r"\s*(input|output)\s+logic(?:\s+\[([^]]+)\])?\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*,?$", line)
        if match:
            ports.append(match.groups())
    return ports

m273_ports = public_ports(
    "rtl_m273/m273_integrated_rank3_atlif.sv",
    "m273_integrated_rank3_atlif")
m518_ports = public_ports(rtl_path, "m518_matched_fixed_t10_atlif")
if len(m273_ports) != 50 or m518_ports != m273_ports:
    raise SystemExit("M518 r5 public-port signature is not exact M273r2")

tuples = []
active_counts = []
for cycle in range(17):
    cycle_tuples = []
    for slot in range(96):
        active = True
        if cycle <= 11:
            beat, sub = divmod(cycle, 3)
            scalar, tap = divmod(slot, 3)
            target = (2 * beat + scalar // 16, scalar % 16, 3 * sub + tap)
        elif cycle <= 15 and slot < 32:
            beat = cycle - 12
            target = (2 * beat + slot // 16, slot % 16, 9)
        elif cycle <= 15:
            scalar, tap = divmod(slot - 32, 2)
            target = (8 + scalar // 16, scalar % 16,
                      2 * (cycle - 12) + tap)
        elif slot < 32:
            target = (8 + slot // 16, slot % 16, 8)
        elif slot < 64:
            scalar = slot - 32
            target = (8 + scalar // 16, scalar % 16, 9)
        else:
            active = False
        if active:
            cycle_tuples.append(target)
            tuples.append(target)
    active_counts.append(len(cycle_tuples))
expected_tuples = {(row, lane, time_index)
                   for row in range(10) for lane in range(16)
                   for time_index in range(10)}
if active_counts != [96] * 16 + [64] or len(tuples) != 1600 \
        or len(set(tuples)) != 1600 or set(tuples) != expected_tuples:
    raise SystemExit("M518 r5 frozen schedule bijection drift")

expected_ids = ["V%02d" % number for number in range(1, 21)]
if list(r4_contract.get("sealed_v01_v20", {}).keys()) != expected_ids:
    raise SystemExit("r4 source-of-truth does not preserve ordered V01-V20")
if contract.get("preserved_campaign", {}).get("tests") != 20:
    raise SystemExit("r5 campaign count drift")

for fragment in (
        "localparam int MULTIPLIERS = 96",
        "localparam int PAYLOAD_BITS = 1064",
        "logic [(OUTPUTS*ACC_W)-1:0] acc_state_observe",
        "config_candidate[CONFIG_BITS-1:PAYLOAD_BITS]=='0",
        "fifo_push=dense_issue&&(dense_selected_cycle>=12);",
        "if(config_accept&&!config_frame_error)",
        "if(raw_accept&&!raw_frame_error)"):
    if fragment not in rtl:
        raise SystemExit("missing M518 r5 RTL mechanism: " + fragment)

required_tb = (
    "build_random_case", "build_rail_case", "enqueue_expected_frame",
    "oldest_selection_sequential_attack", "release_partial_raw_attack",
    "release_dense_phase_attack", "release_fifo_drain_attack",
    "targeted_phase12_phase16_stalls", "reset_state_attacks",
    "total_random_contexts!=4", "total_rail_boundary_points!=6",
    "build_rail_case(-8388608", "held_release_edges<8",
    "total_zero_tile_held_release_edges!=8",
    "total_zero_tile_fault_transitions!=1",
    "total_release_state_attacks!=5", "total_reset_attacks!=9",
    "reset_partial_config_attacks!=1", "reset_partial_raw_attacks!=1",
    "reset_dense_c0_attacks!=1", "reset_dense_c11_attacks!=1",
    "reset_dense_c12_attacks!=1", "reset_dense_c15_attacks!=1",
    "reset_dense_c16_attacks!=1", "reset_fifo_full_close_attacks!=1",
    "reset_quarantine_attacks!=1", "total_clean_after_reset_probes!=9",
)
for fragment in required_tb:
    if fragment not in tb:
        raise SystemExit("missing sealed M518 TB mechanism: " + fragment)

required_sva = (
    "ap_config_accept_known", "ap_raw_accept_known",
    "ap_raw_occupancy_exact", "ap_oldest_bank1",
    "ap_dense_start_ownership", "ap_fifo_conservation_exact",
    "ap_close_stall_holds", "acc_state_internal",
    "ap_full_pop_push_atomic", "ap_tile_done_exact", "ap_busy_exact",
    "ap_context_cycle_progress", "ap_context_retire_count",
    "cp_phase12_stall", "cp_phase16_stall",
    "cp_dual_ready_oldest_bank1", "cp_zero_tile_fault",
    "cp_beat0", "cp_beat1", "cp_beat2", "cp_beat3", "cp_beat4",
    "cp_reset_recovery",
)
for fragment in required_sva:
    if fragment not in sva:
        raise SystemExit("missing sealed M518 SVA mechanism: " + fragment)

pass_line = (
    "PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 "
    "clean_N4=80 random_contexts=4 rail_boundary_points=6 "
    "zero_tile_held_edges=8 zero_tile_fault_transitions=1 "
    "release_state_attacks=5 reset_attacks=9 reset_partial_config=1 "
    "reset_partial_raw=1 reset_dense_c0=1 reset_dense_c11=1 "
    "reset_dense_c12=1 reset_dense_c15=1 reset_dense_c16=1 "
    "reset_fifo_full_close=1 reset_quarantine=1 clean_after_reset_N1=9 "
    "sequential_oldest=1 phase12_stall=1 phase16_stall=1 "
    "padding_attacks=216 raw_attacks=7 config_attacks=5 "
    "fault_edge_pop_push=1 slot_tuples_per_tile=1600 "
    "multiplier_slots=96 issue_cycles=17 vcs_only=true dc=false "
    "formality=false ptpx=false speedup=false ppa=false headline=false")
if pass_line not in tb:
    raise SystemExit("M518 r5 PASS signature drift")
PY

printf '%s\n' \
    "PASS_M518_R5_EXACT_SIX_TOKEN_REVERSE_SHA_AND_STATIC_PREFLIGHT_NO_COMPILE_YET" \
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
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
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

task_pass='PASS M518 matched Fixed T10 ATLIF sealed_V01_V20 clean_N1=29 clean_N4=80 random_contexts=4 rail_boundary_points=6 zero_tile_held_edges=8 zero_tile_fault_transitions=1 release_state_attacks=5 reset_attacks=9 reset_partial_config=1 reset_partial_raw=1 reset_dense_c0=1 reset_dense_c11=1 reset_dense_c12=1 reset_dense_c15=1 reset_dense_c16=1 reset_fifo_full_close=1 reset_quarantine=1 clean_after_reset_N1=9 sequential_oldest=1 phase12_stall=1 phase16_stall=1 padding_attacks=216 raw_attacks=7 config_attacks=5 fault_edge_pop_push=1 slot_tuples_per_tile=1600 multiplier_slots=96 issue_cycles=17 vcs_only=true dc=false formality=false ptpx=false speedup=false ppa=false headline=false'
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
    "${task_run}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r5.json" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

report = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
covers = {}
for name, count in re.findall(
        r"u_sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match",
        report):
    covers[name] = int(count)
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
    raise SystemExit("assertion cover drift: %r" % covers)
vcs_id = Path(sys.argv[4]).read_text(encoding="utf-8", errors="replace")
if "V-2023.12-SP1" not in vcs_id:
    raise SystemExit("actual VCS identity drift")

receipt = {
    "schema": "m518_matched_fixed_t10_atlif_author_vcs_receipt_v5",
    "status": "PASS_M518_R5_EXACT_SIX_TOKEN_REPAIR_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN",
    "role": "author_campaign_not_independent_review",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "actual_vcs_id_sha256": digest(sys.argv[4]),
    "exact_runner_sha256": sys.argv[5],
    "source_repair": {
        "r4_rtl_sha256": "09b1d976595f13885da917dd33b9ce87c403750eb8bb6e42a9aff1379a93412a",
        "r5_rtl_sha256": "90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a",
        "old_word_tokens": 0,
        "new_word_tokens": 6,
        "reverse_sha_matches_r4": True,
        "semantic_change_intended": False,
    },
    "r4_failure_review": {
        "status": "SEALED_R4_SYSTEMVERILOG_RESERVED_TOKEN_FAILURE_BOUND",
        "review_json_sha256": digest(
            "reviews/m518_vcs_compile_failure_hammer_r2_20260827/"
            "m518_vcs_compile_failure_hammer_r2_20260827.json"),
        "member_manifest_sha256": digest(
            "reviews/m518_vcs_compile_failure_hammer_r2_20260827/SHA256SUMS"),
        "outer_seal_file_sha256": digest(
            "reviews/m518_vcs_compile_failure_hammer_r2_20260827/"
            "SHA256SUMS.seal.sha256"),
        "r4_result_admitted": False,
    },
    "negative_control": {
        "status": "EXPECTED_FAIL_WRONG_RTL_SHA_EXIT10_BEFORE_TOOL",
        "exit_code": 10,
        "compile_or_simv_created": False,
        "positive_receipt_created": False,
        "manifest_sha256": digest(sys.argv[2]),
        "outer_seal_file_sha256": digest(sys.argv[3]),
    },
    "campaign": {
        "tests": "SEALED_V01_V20_WITH_ORIGINAL_NUMBERING",
        "clean_cycles_N1": 29,
        "clean_cycles_N4": 80,
        "random_contexts": 4,
        "q24_boundary_points": 6,
        "lower_rail_observable_threshold": -8388608,
        "zero_tile_held_release_edges": 8,
        "zero_tile_fault_transitions": 1,
        "release_state_attacks": 5,
        "reset_attacks": 9,
        "clean_after_reset_exact_N1_29": 9,
        "sequential_oldest_attacks": 1,
        "padding_attacks": 216,
        "config_frame_attacks": 5,
        "raw_frame_attacks": 7,
        "numeric_mismatches": 0,
        "assertion_failures": 0,
        "assertion_cover_matches": covers,
    },
    "claim_boundary": {
        "author_vcs": True,
        "independent_review": False,
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
receipt_path = Path(sys.argv[6])
receipt_path.write_text(
    json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8")
round_trip = json.loads(
    receipt_path.read_text(encoding="utf-8"),
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("non-finite JSON constant: " + value)))
def assert_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            assert_finite(key)
            assert_finite(member)
    elif isinstance(value, list):
        for member in value:
            assert_finite(member)
assert_finite(round_trip)
if round_trip != receipt:
    raise SystemExit("strict finite M518 r5 receipt round-trip drift")
PY

printf '%s  %s\n' "${task_observed_runner}" "${task_runner}" \
    >"${task_run}/runner_sha256.txt"
printf '%s\n' \
    "PASS_M518_R5_EXACT_SIX_TOKEN_REPAIR_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN" \
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
echo "PASS M518 r5 exact-SHA VCS author campaign at ${task_run}"
