#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M518_RUN_DIR:-${task_hw_root}/results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_expected_runner="${M518_EXPECTED_RUNNER_SHA256:-}"
task_observed_runner="$(sha256sum "${task_runner}" | awk '{print $1}')"

[[ "${task_expected_runner}" =~ ^[0-9a-f]{64}$ ]] || exit 3
[[ "${task_observed_runner}" == "${task_expected_runner}" ]] || exit 4
[[ ! -e "${task_run}" ]] || exit 2
mkdir -p "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m518/m518_matched_fixed_t10_atlif.sv"]="90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a"
    ["verif_m518/m518_matched_fixed_t10_atlif_assertions.sv"]="89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5"
    ["tb_m518/tb_m518_matched_fixed_t10_atlif.sv"]="e7973a91d04b9f20542b04c58a213e2c8929259768701664dda76721720d2888"
    ["dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f"]="09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea"
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r6_20260827.json"]="153f733bb231e746980a255fc200b6e8738cf48e051db1c0e52ba4f329ef4341"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r5_20260827.json"]="51b81bbad3eaa05269e209a68019cb9aa9a29e822d03c283df2f5c68fb2ff996"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r4_20260827.json"]="5a424e5c58c83a1047fde09108a2a812a704175396a974db7d79041a4a5d66cc"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/m518_atlif_fixed_baseline_spec_r1_20260827.md"]="f50376b28f4f69ab8d257d06ea40553ece43d59cde24e90c4e78e5437afba083"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/m518_atlif_fixed_baseline_spec_r1_20260827.json"]="a4b57569d86dca3f0f906565d9b5f7be97335946ac91e38a536d73dca3f2bee1"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/RUN_COMPLETE"]="09dab11ee1ceeafa810c6f91889db8c2929aab41302290ac36f6510d88d2200a"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/SHA256SUMS"]="177851f6d773c78366382b1cd1e3a64d6e47e06edab0c0fd7c732ba2fdf63d74"
    ["reviews/m518_atlif_fixed_baseline_spec_r1_20260827/SHA256SUMS.seal.sha256"]="1a06765ec9bf602cbd2e4b5bda938360713e91a9befa65e1b68aff7e29974bb0"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/m518_vcs_sva_compile_failure_hammer_r3_20260827.json"]="060df0209dc7043aafd8bbaad3d185aa5ae94493eff1536800f9d6a67d4966c2"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/m518_vcs_sva_compile_failure_hammer_r3_20260827.md"]="9a708bf7915f1354c9b80da360303cb2d3f111dce2fcf512b54cfd8a142dd020"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/RUN_COMPLETE"]="2f508794e684720a27ac5d3a4dea745a6ddce1b12f1ca3473e5174e075e682a5"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/SHA256SUMS"]="c7ccc08b34e808663b0d12597646ccfa2e06357d00f9b1084b89dc21c171eb73"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/SHA256SUMS.seal.sha256"]="ce7d72c4c704349093452e59f9fe6739d8318598f12b574abc2bd5b994637585"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/m518_vcs_compile_failure_hammer_r2_20260827.json"]="3a1b31f5c5d7c99ed4541b1c70838ef2fe87c23162ce62f64cae97dc04af3938"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/m518_vcs_compile_failure_hammer_r2_20260827.md"]="731b274e125fcef2bb824035797c66a6c07715a2339b6fd4d7695298bc1c00c6"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/RUN_COMPLETE"]="6d891a6914bafb9314b4513be413c182f6263fc2e85f7a766ba1d73e17616a43"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/SHA256SUMS"]="e865499448d81012ee9112fa6dcc48093edb9a6f263d6f548288d6c77d6c4c1c"
    ["reviews/m518_vcs_compile_failure_hammer_r2_20260827/SHA256SUMS.seal.sha256"]="860162d25c9af9118960145582132ea975b32cd959957dfa2793ad7f9927cd0b"
    ["reviews/m518_candidate_static_hammer_r5_20260827/m518_candidate_static_hammer_r5_20260827.json"]="0394ce4271e22e79e859b0bf4a0741240a67819b3845259b24e397f2a2e7ea39"
    ["reviews/m518_candidate_static_hammer_r5_20260827/RUN_COMPLETE"]="e2bb4cc31ccd6371cb063d173ef638a0999f0273f13c9b6e41a59cb2216f93ba"
    ["reviews/m518_candidate_static_hammer_r5_20260827/SHA256SUMS"]="9294f15bb1319ce4da3f89bd2f30d3b8294a63dd7a3485d29a70c3539e52d048"
    ["reviews/m518_candidate_static_hammer_r5_20260827/SHA256SUMS.seal.sha256"]="69b5cbc0d005405afef9a3c621b0e5ec9903425c389167994451dcb899d2c411"
    ["results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827/RUN_FAILED_OR_INCOMPLETE.txt"]="0b71bda8d02ca4d69532170e5592679f03858fb0b3a4464fc4a68ca839e2ee23"
    ["results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827/compile.log"]="b08daef096915851d4a162b62825b2550ecac4f97fbd6b5cee3d328fb274620a"
    ["results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827/compile.rc"]="ce8bafb38615aeb5d44ebbabe78ec14ac35a5de87bdc5ad5ea82a72656024ce4"
    ["results/m518_matched_fixed_t10_atlif_vcs_r5_exact_20260827/vcs_id.txt"]="f62d91ed6be84f085e21a12dcdaad502bfc130dbbe9d0726714b6cfc1ea26439"
    ["dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r5_exact.sh"]="854f152ad23bcc3e353953dee93d0b88f24eab2b4f34261bd88c3c3560a7312a"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/RUN_FAILED_OR_INCOMPLETE.txt"]="0b71bda8d02ca4d69532170e5592679f03858fb0b3a4464fc4a68ca839e2ee23"
    ["results/m518_matched_fixed_t10_atlif_vcs_r4_exact_20260827/compile.log"]="a44557c1eb9e997197f88dad19c122116410c33fd5372ae30395d864ab1de2da"
    ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)

task_verify_inputs() {
    local destination="$1" negative="$2"
    local path expected observed mismatch=0
    mkdir -p "${destination}"
    : >"${destination}/preflight_sha_checks.txt"
    while IFS= read -r path; do
        expected="${task_expected[${path}]}"
        if [[ "${negative}" == "1" && "${path}" == \
                "verif_m518/m518_matched_fixed_t10_atlif_assertions.sv" ]]; then
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
[[ ! -e "${task_negative_dir}/compile.log" \
    && ! -e "${task_negative_dir}/simv" \
    && ! -e "${task_negative_dir}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r6.json" \
    && ! -e "${task_negative_dir}/RUN_COMPLETE.txt" ]] || exit 12
printf '%s\n' \
    "EXPECTED_FAIL_M518_R6_WRONG_SVA_SHA_EXIT10_NO_TOOL_NO_POSITIVE_RECEIPT" \
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

for task_sealed_dir in \
    reviews/m518_atlif_fixed_baseline_spec_r1_20260827 \
    reviews/m518_vcs_compile_failure_hammer_r2_20260827 \
    reviews/m518_candidate_static_hammer_r5_20260827 \
    reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827; do
    task_label="$(basename "${task_sealed_dir}")"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS) \
        >"${task_run}/${task_label}_manifest_check.txt"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS.seal.sha256) \
        >"${task_run}/${task_label}_outer_seal_check.txt"
done

while IFS= read -r path; do sha256sum "${path}"; done \
    < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort) \
    >"${task_run}/input_sha256.txt"
cp contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r6_20260827.json \
    "${task_run}/contract_draft.json"

python3 - <<'PY'
import hashlib
import json
import math
import re
from pathlib import Path

def load_json(path):
    def reject(value):
        raise ValueError("non-finite JSON constant: " + value)
    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       parse_constant=reject)
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

rtl_path = Path("rtl_m518/m518_matched_fixed_t10_atlif.sv")
sva_path = Path("verif_m518/m518_matched_fixed_t10_atlif_assertions.sv")
tb_path = Path("tb_m518/tb_m518_matched_fixed_t10_atlif.sv")
filelist_path = Path(
    "dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f")
contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r6_20260827.json")
r4_contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r4_20260827.json")
r5_failure = load_json(
    "reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/"
    "m518_vcs_sva_compile_failure_hammer_r3_20260827.json")
sva = sva_path.read_text(encoding="utf-8")
tb = tb_path.read_text(encoding="utf-8")

if contract.get("execution_state", {}).get("r6_static_readmission") is not False:
    raise SystemExit("M518 r6 contract self-authorization drift")
if r5_failure.get("status") != \
        "DIAGNOSTIC_CONFIRMED__R5_SVA_UNMATCHED_PAREN_FAILURE__R6_STATIC_READMISSION_REQUIRED":
    raise SystemExit("M518 r5 failure classification drift")
if r5_failure.get("decision", {}).get("r6_runner_execution_authorized") \
        is not False:
    raise SystemExit("M518 r5 audit unexpectedly authorizes r6")

old_suffix = "(raw_owned_internal[0]&&!raw_ready_internal[0]));"
new_suffix = "(raw_owned_internal[0]&&!raw_ready_internal[0])));"
if sva.count(new_suffix) != 1 or sva.count(old_suffix) != 0:
    raise SystemExit("M518 r6 exact SVA suffix cardinality drift")
restored = sva.replace(new_suffix, old_suffix, 1)
if len(sva.encode("utf-8")) != len(restored.encode("utf-8")) + 1:
    raise SystemExit("M518 r6 SVA is not exactly one byte longer")
if hashlib.sha256(restored.encode("utf-8")).hexdigest() != \
        "977f95652bb788047549d58ff94e416f00542c9d3e63fa6f83e09fe582c910f4":
    raise SystemExit("M518 r6 reverse mutation does not reconstruct r5 SVA")
statement = "\n".join(sva.splitlines()[136:143])
if statement.count("(") != 8 or statement.count(")") != 8:
    raise SystemExit("M518 r6 target assertion parentheses are not 8/8")

expected_filelist = [
    "rtl_m518/m518_matched_fixed_t10_atlif.sv",
    "verif_m518/m518_matched_fixed_t10_atlif_assertions.sv",
    "tb_m518/tb_m518_matched_fixed_t10_atlif.sv",
]
if [line.strip() for line in filelist_path.read_text().splitlines()
        if line.strip()] != expected_filelist:
    raise SystemExit("M518 r6 filelist topology drift")

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

if public_ports(rtl_path, "m518_matched_fixed_t10_atlif") != public_ports(
        "rtl_m273/m273_integrated_rank3_atlif.sv",
        "m273_integrated_rank3_atlif"):
    raise SystemExit("M518 r6 public-port signature drift")

expected_ids = ["V%02d" % number for number in range(1, 21)]
if list(r4_contract.get("sealed_v01_v20", {}).keys()) != expected_ids:
    raise SystemExit("M518 V01-V20 source-of-truth drift")
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
if any(fragment not in sva for fragment in required_sva):
    raise SystemExit("M518 r6 required SVA fragment drift")
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
    raise SystemExit("M518 r6 PASS signature drift")
PY

printf '%s\n' \
    "PASS_M518_R6_EXACT_ONE_CHARACTER_SVA_REVERSE_SHA_PREFLIGHT_NO_COMPILE_YET" \
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
    "${task_run}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r6.json" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

report = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
covers = {name: int(count) for name, count in re.findall(
    r"u_sva\.(cp_[A-Za-z0-9_]+),\s+\d+ attempts,\s+(\d+) match",
    report)}
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
    raise SystemExit("M518 r6 assertion cover drift")
receipt = {
    "schema": "m518_matched_fixed_t10_atlif_author_vcs_receipt_v6",
    "status": "PASS_M518_R6_EXACT_ONE_CHARACTER_SVA_REPAIR_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN",
    "role": "author_campaign_not_independent_review",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "actual_vcs_id_sha256": digest(sys.argv[4]),
    "exact_runner_sha256": sys.argv[5],
    "source_repair": {
        "r5_sva_sha256": "977f95652bb788047549d58ff94e416f00542c9d3e63fa6f83e09fe582c910f4",
        "r6_sva_sha256": "89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5",
        "bytes_inserted": 1,
        "reverse_sha_matches_r5": True,
        "semantic_change_intended": False,
    },
    "r5_failure_review": {
        "review_json_sha256": digest(
            "reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/"
            "m518_vcs_sva_compile_failure_hammer_r3_20260827.json"),
        "member_manifest_sha256": digest(
            "reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/SHA256SUMS"),
        "outer_seal_file_sha256": digest(
            "reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/"
            "SHA256SUMS.seal.sha256"),
        "r5_result_admitted": False,
    },
    "negative_control": {
        "status": "EXPECTED_FAIL_WRONG_SVA_SHA_EXIT10_BEFORE_TOOL",
        "exit_code": 10,
        "manifest_sha256": digest(sys.argv[2]),
        "outer_seal_file_sha256": digest(sys.argv[3]),
    },
    "campaign": {
        "tests": "SEALED_V01_V20_WITH_ORIGINAL_NUMBERING",
        "clean_cycles_N1": 29,
        "clean_cycles_N4": 80,
        "random_contexts": 4,
        "q24_boundary_points": 6,
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
    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for key, member in value.items():
            finite(key)
            finite(member)
    elif isinstance(value, list):
        for member in value:
            finite(member)
finite(round_trip)
if round_trip != receipt:
    raise SystemExit("strict finite M518 r6 receipt round-trip drift")
PY

printf '%s  %s\n' "${task_observed_runner}" "${task_runner}" \
    >"${task_run}/runner_sha256.txt"
printf '%s\n' \
    "PASS_M518_R6_EXACT_ONE_CHARACTER_SVA_REPAIR_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN" \
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
echo "PASS M518 r6 exact-SHA VCS author campaign at ${task_run}"
