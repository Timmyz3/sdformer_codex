#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "${task_dc_root}/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
[[ ! -v M518_RUN_DIR ]] || exit 5
task_run="${task_hw_root}/results/m518_matched_fixed_t10_atlif_vcs_r7_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
task_expected_runner="${M518_EXPECTED_RUNNER_SHA256:-}"
task_observed_runner="$(sha256sum "${task_runner}" | awk '{print $1}')"

[[ "${task_expected_runner}" =~ ^[0-9a-f]{64}$ ]] || exit 3
[[ "${task_observed_runner}" == "${task_expected_runner}" ]] || exit 4
[[ ! -e "${task_run}" ]] || exit 2
mkdir "${task_run}"
task_complete=0
trap 'task_rc=$?; if [[ ${task_complete} -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "${task_rc}" >"${task_run}/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "${task_hw_root}"

declare -A task_expected=(
    ["rtl_m518/m518_matched_fixed_t10_atlif.sv"]="8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6"
    ["verif_m518/m518_matched_fixed_t10_atlif_assertions.sv"]="89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5"
    ["tb_m518/tb_m518_matched_fixed_t10_atlif.sv"]="a2de78ac5a3c537e03113f06552a09808426170d188d39e462b500b0c865eb12"
    ["dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f"]="09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea"
    ["rtl_m273/m273_integrated_rank3_atlif.sv"]="11d5c6c4f5f0c44ea0a8c2b815683a2e1ab2dbb007bd3afdca0d8ae9e901067d"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r7_20260827.json"]="3f046e9390427374edd3f40cec9297138f619726c72fd92be5d4ff4cdeb55037"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r6_20260827.json"]="153f733bb231e746980a255fc200b6e8738cf48e051db1c0e52ba4f329ef4341"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r5_20260827.json"]="51b81bbad3eaa05269e209a68019cb9aa9a29e822d03c283df2f5c68fb2ff996"
    ["contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r4_20260827.json"]="5a424e5c58c83a1047fde09108a2a812a704175396a974db7d79041a4a5d66cc"
    ["dc_handoff/scripts/run_vcs_m518_matched_fixed_t10_atlif_r6_exact.sh"]="050db5ce70013ba0b61093ce2abbb544b645542af55e48061a1d9bc3e60c2a4d"
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
    ["reviews/m518_candidate_static_hammer_r5_20260827/m518_candidate_static_hammer_r5_20260827.json"]="0394ce4271e22e79e859b0bf4a0741240a67819b3845259b24e397f2a2e7ea39"
    ["reviews/m518_candidate_static_hammer_r5_20260827/m518_candidate_static_hammer_r5_20260827.md"]="48eeaa9db995e6861ff7e2edeab3f42955ff7cc4f985f6f50ab977c5ee24be4c"
    ["reviews/m518_candidate_static_hammer_r5_20260827/RUN_COMPLETE"]="e2bb4cc31ccd6371cb063d173ef638a0999f0273f13c9b6e41a59cb2216f93ba"
    ["reviews/m518_candidate_static_hammer_r5_20260827/SHA256SUMS"]="9294f15bb1319ce4da3f89bd2f30d3b8294a63dd7a3485d29a70c3539e52d048"
    ["reviews/m518_candidate_static_hammer_r5_20260827/SHA256SUMS.seal.sha256"]="69b5cbc0d005405afef9a3c621b0e5ec9903425c389167994451dcb899d2c411"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/m518_vcs_sva_compile_failure_hammer_r3_20260827.json"]="060df0209dc7043aafd8bbaad3d185aa5ae94493eff1536800f9d6a67d4966c2"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/m518_vcs_sva_compile_failure_hammer_r3_20260827.md"]="9a708bf7915f1354c9b80da360303cb2d3f111dce2fcf512b54cfd8a142dd020"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/RUN_COMPLETE"]="2f508794e684720a27ac5d3a4dea745a6ddce1b12f1ca3473e5174e075e682a5"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/SHA256SUMS"]="c7ccc08b34e808663b0d12597646ccfa2e06357d00f9b1084b89dc21c171eb73"
    ["reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827/SHA256SUMS.seal.sha256"]="ce7d72c4c704349093452e59f9fe6739d8318598f12b574abc2bd5b994637585"
    ["reviews/m518_candidate_static_hammer_r6_independent_20260827/m518_candidate_static_hammer_r6_independent_20260827.json"]="efbafd249f090ed51a897391b473caaa993b9818489ed32d1c37423bc0be7688"
    ["reviews/m518_candidate_static_hammer_r6_independent_20260827/m518_candidate_static_hammer_r6_independent_20260827.md"]="105c0867318a8f0d91050406ceb7e6c32b2e7d8e0bd9ce3eca8f7547cbff42f5"
    ["reviews/m518_candidate_static_hammer_r6_independent_20260827/RUN_COMPLETE"]="eabf51b717da1978bcd12f5172abbec3074beca07f4fcdb6f48dc247b211ca4a"
    ["reviews/m518_candidate_static_hammer_r6_independent_20260827/SHA256SUMS"]="69002b63aef7045ec3fe9fd7a5b5809824d06ab8c20fadaeaa57aa367af361a0"
    ["reviews/m518_candidate_static_hammer_r6_independent_20260827/SHA256SUMS.seal.sha256"]="8c405537149027abf2dae7e52ce9fb2252048cce5d0348e08136ad58c3e0f476"
    ["reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/m518_vcs_tb_icpd_failure_hammer_r4_20260827.json"]="d0afdae91116a12f051b8241603ea634b72b125682962ac53d7b4064d7b199d3"
    ["reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/m518_vcs_tb_icpd_failure_hammer_r4_20260827.md"]="08e0de88e638d4eb54eafd627e900435f3886c4353280de516fa80ba26a48be2"
    ["reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/RUN_COMPLETE"]="b64dab527daac3ce5fb0523141f035c1f1d8bd672bc7f1a53958310d8ed6db6e"
    ["reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/SHA256SUMS"]="cd68a909c5335ddb68a4cd4d6bf929cfe555b9a056654e9b671b787b7e6fd605"
    ["reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/SHA256SUMS.seal.sha256"]="a9ae5da67f8f393382d61369d47c58b58c94f7ce266720511a7d1db8564c36e9"
    ["results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827/RUN_FAILED_OR_INCOMPLETE.txt"]="0b71bda8d02ca4d69532170e5592679f03858fb0b3a4464fc4a68ca839e2ee23"
    ["results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827/compile.log"]="08aeb6f7a81b36e70a52bcd083a16cb3f19d3e9c6291dd9c0b35a7bcdf4bfd33"
    ["results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827/compile.rc"]="ce8bafb38615aeb5d44ebbabe78ec14ac35a5de87bdc5ad5ea82a72656024ce4"
    ["results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827/vcs_id.txt"]="f62d91ed6be84f085e21a12dcdaad502bfc130dbbe9d0726714b6cfc1ea26439"
    ["results/m518_matched_fixed_t10_atlif_vcs_r6_exact_20260827/input_sha256.txt"]="266e639a1d771c16b6f7d9f7d27a0ff0e1458e43d64de56a6a96a96846f55dc6"
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
[[ ! -e "${task_negative_dir}/compile.log" \
    && ! -e "${task_negative_dir}/simv" \
    && ! -e "${task_negative_dir}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r7.json" \
    && ! -e "${task_negative_dir}/RUN_COMPLETE.txt" ]] || exit 12
printf '%s\n' \
    "EXPECTED_FAIL_M518_R7_WRONG_TB_SHA_EXIT10_NO_TOOL_NO_POSITIVE_RECEIPT" \
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
    reviews/m518_vcs_sva_compile_failure_hammer_r3_20260827 \
    reviews/m518_candidate_static_hammer_r6_independent_20260827 \
    reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827; do
    task_label="$(basename "${task_sealed_dir}")"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS) \
        >"${task_run}/${task_label}_manifest_check.txt"
    (cd "${task_sealed_dir}" && sha256sum -c SHA256SUMS.seal.sha256) \
        >"${task_run}/${task_label}_outer_seal_check.txt"
done

while IFS= read -r path; do sha256sum "${path}"; done \
    < <(printf '%s\n' "${!task_expected[@]}" | LC_ALL=C sort) \
    >"${task_run}/input_sha256.txt"
cp contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r7_20260827.json \
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

def replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise SystemExit("M518 r7 reverse-proof cardinality drift: " + label)
    return source.replace(old, new, 1)

def preprocess_harness(source, defined):
    output = []
    stack = []
    active = True
    for line in source.splitlines(True):
        stripped = line.strip()
        if stripped == "`ifdef M518_VCS_V06_HARNESS":
            stack.append((active, defined))
            active = active and defined
        elif stripped == "`else" and stack:
            parent, condition = stack[-1]
            active = parent and not condition
        elif stripped == "`endif" and stack:
            parent, _ = stack.pop()
            active = parent
        elif active:
            output.append(line)
    if stack:
        raise SystemExit("M518 r7 unterminated harness preprocessor block")
    return "".join(output)

def public_ports(source, module):
    start = source.index("module " + module)
    header = source[start:source.index(");", start) + 2]
    ports = []
    for line in header.splitlines():
        match = re.match(
            r"\s*,?\s*(input|output)\s+logic(?:\s+\[([^]]+)\])?\s+"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*,?$", line)
        if match:
            ports.append(match.groups())
    return ports

rtl_path = Path("rtl_m518/m518_matched_fixed_t10_atlif.sv")
sva_path = Path("verif_m518/m518_matched_fixed_t10_atlif_assertions.sv")
tb_path = Path("tb_m518/tb_m518_matched_fixed_t10_atlif.sv")
filelist_path = Path(
    "dc_handoff/filelists/date_m518_matched_fixed_t10_atlif_directed_vcs.f")
rtl = rtl_path.read_text(encoding="utf-8")
sva = sva_path.read_text(encoding="utf-8")
tb = tb_path.read_text(encoding="utf-8")
contract = load_json(
    "contracts/m518_matched_fixed_t10_atlif_vcs_contract_draft_r7_20260827.json")
r6_failure = load_json(
    "reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/"
    "m518_vcs_tb_icpd_failure_hammer_r4_20260827.json")

if contract.get("execution_state", {}).get("r7_static_readmission") is not False:
    raise SystemExit("M518 r7 contract self-authorization drift")
if r6_failure.get("status") != \
        "DIAGNOSTIC_CONFIRMED__R6_TB_HIERARCHICAL_DEPOSIT_VS_ALWAYS_FF_ICPD__R7_STATIC_READMISSION_REQUIRED":
    raise SystemExit("M518 r6 failure review status drift")
if r6_failure.get("root_cause_classification", {}).get("classification") != \
        "TESTBENCH_INSTRUMENTATION_P1__NOT_RTL_P0":
    raise SystemExit("M518 r6 failure classification drift")
if r6_failure.get("decision", {}).get("r7_vcs_execution_authorized") is not False:
    raise SystemExit("M518 r6 failure review unexpectedly authorizes r7 VCS")

if "$deposit" in tb:
    raise SystemExit("M518 r7 TB retains $deposit")
if re.search(r"\bforce\s+u_dut\.", tb) or re.search(r"\brelease\s+u_dut\.", tb):
    raise SystemExit("M518 r7 TB retains hierarchical force/release")
hierarchical_lhs = re.findall(
    r"u_dut\.[A-Za-z_][A-Za-z0-9_]*(?:\[[^\n;]*\])?\s*(?:<=|=(?!=))", tb)
if hierarchical_lhs:
    raise SystemExit("M518 r7 TB retains hierarchical DUT-state LHS")
if rtl.count("always_ff") != 1 or re.search(r"always\s*@\s*\(posedge", rtl):
    raise SystemExit("M518 r7 DUT single always_ff ownership drift")
if "bind " in tb or "bind " in sva:
    raise SystemExit("M518 r7 writing-bind risk")

production_ports = public_ports(
    preprocess_harness(rtl, False), "m518_matched_fixed_t10_atlif")
harness_ports = public_ports(
    preprocess_harness(rtl, True), "m518_matched_fixed_t10_atlif")
m273_ports = public_ports(
    Path("rtl_m273/m273_integrated_rank3_atlif.sv").read_text(encoding="utf-8"),
    "m273_integrated_rank3_atlif")
if len(production_ports) != 50 or production_ports != m273_ports:
    raise SystemExit("M518 r7 macro-absent 50-port production identity drift")
if len(harness_ports) != 52 or harness_ports[:50] != production_ports:
    raise SystemExit("M518 r7 macro-present 52-port prefix drift")
if [item[2] for item in harness_ports[50:]] != [
        "v06_hold_dense_issue", "v06_first_empty_fill_bank1"]:
    raise SystemExit("M518 r7 simulation-only harness port drift")

r6_rtl = rtl
r6_rtl = replace_once(r6_rtl, '''`ifdef M518_VCS_V06_HARNESS
    ,input logic                     v06_hold_dense_issue
    ,input logic                     v06_first_empty_fill_bank1
`endif
''', "", "rtl ports")
r6_rtl = replace_once(r6_rtl, '''`ifdef M518_VCS_V06_HARNESS
        if(!fill_active_q&&raw_owned_q==0&&v06_first_empty_fill_bank1)
            raw_target_bank=1'b1;
`endif
''', "", "rtl first-bank selector")
r6_rtl = replace_once(r6_rtl, '''        dense_issue=dense_source_valid
            &&(dense_selected_cycle<12||fifo_credit)&&!protocol_error_q
`ifdef M518_VCS_V06_HARNESS
            &&!v06_hold_dense_issue
`endif
            ;
''', '''        dense_issue=dense_source_valid
            &&(dense_selected_cycle<12||fifo_credit)&&!protocol_error_q;
''', "rtl dense hold")
if hashlib.sha256(r6_rtl.encode("utf-8")).hexdigest() != \
        "90e0304bd8fa5bae5f4cf523d8ab7c62b42878a0ce17b75bd62f8b9288600a6a":
    raise SystemExit("M518 r7 macro-only RTL reverse proof does not recover r6")

r6_tb = tb
r6_tb = replace_once(r6_tb, '''`ifdef M518_VCS_V06_HARNESS
    logic v06_hold_dense_issue,v06_first_empty_fill_bank1;
`endif
''', "", "tb harness declaration")
r6_tb = replace_once(r6_tb,
    "    integer total_sequential_oldest_attacks,total_v06_harness_activations;\n",
    "    integer total_sequential_oldest_attacks;\n", "tb counter declaration")
r6_tb = replace_once(r6_tb, '''`ifdef M518_VCS_V06_HARNESS
            v06_hold_dense_issue=1'b0;v06_first_empty_fill_bank1=1'b0;
`endif
''', "", "tb reset controls")
new_v06 = '''`ifdef M518_VCS_V06_HARNESS
            // Hold issue, steer only the first empty fill to bank1, and use the
            // production five-beat raw path for both completed banks. No TB
            // process writes a DUT state variable.
            v06_hold_dense_issue=1'b1;
            v06_first_empty_fill_bank1=1'b1;
            send_frame_tile(legal_config,payload_bank1,tag_bank1);
            v06_first_empty_fill_bank1=1'b0;
            send_frame_tile(legal_config,payload_bank0,tag_bank0);
            if(u_dut.raw_ready_q!==2'b11||u_dut.raw_owned_q!==2'b11
                    ||u_dut.raw_order1_q>=u_dut.raw_order0_q
                    ||u_dut.raw_tag1_q!==tag_bank1
                    ||u_dut.raw_tag0_q!==tag_bank0
                    ||debug_raw_beats!=10||debug_tiles_loaded!=2
                    ||stage1_issue)
                $fatal(1,"V06 legal-fill harness failed to construct bank1-oldest dual-ready state");
            total_v06_harness_activations=total_v06_harness_activations+1;
            v06_hold_dense_issue=1'b0;
`else
            $fatal(1,"V06 requires M518_VCS_V06_HARNESS");
`endif
'''
old_v06 = '''            enqueue_expected_tile(1,601,tag_bank1);
            enqueue_expected_tile(1,600,tag_bank0);
            @(negedge clk_core);
            // The eager one-fill/one-dense transport cannot naturally retain two
            // ready banks. Deposit a legal completed-bank snapshot, then cross a
            // real issue edge and check all sequential ownership/tag effects.
            $deposit(u_dut.raw_bank0_q,payload_bank0);
            $deposit(u_dut.raw_bank1_q,payload_bank1);
            $deposit(u_dut.raw_tag0_q,tag_bank0);
            $deposit(u_dut.raw_tag1_q,tag_bank1);
            $deposit(u_dut.raw_order0_q,32'd9);
            $deposit(u_dut.raw_order1_q,32'd3);
            $deposit(u_dut.raw_owned_q,2'b11);
            $deposit(u_dut.raw_ready_q,2'b11);
            $deposit(u_dut.tiles_loaded_q,32'd2);
            $deposit(u_dut.raw_beats_q,32'd10);
'''
r6_tb = replace_once(r6_tb, new_v06, old_v06, "tb V06 legal fill")
r6_tb = replace_once(r6_tb, '''`ifdef M518_VCS_V06_HARNESS
        v06_hold_dense_issue=1'b0;v06_first_empty_fill_bank1=1'b0;
`endif
''', "", "tb initial controls")
r6_tb = replace_once(r6_tb,
    "        total_sequential_oldest_attacks=0;total_v06_harness_activations=0;\n",
    "        total_sequential_oldest_attacks=0;\n", "tb counter init")
r6_tb = replace_once(r6_tb,
    "                ||total_v06_harness_activations!=1\n", "", "tb closure")
r6_tb = replace_once(r6_tb, "oldest=%0d harness=%0d c12=",
                     "oldest=%0d c12=", "tb fatal format")
r6_tb = replace_once(r6_tb,
    "                total_v06_harness_activations,\n", "", "tb fatal argument")
r6_tb = replace_once(r6_tb,
    "sequential_oldest=1 v06_legal_fill_harness=1 phase12_stall=1",
    "sequential_oldest=1 phase12_stall=1", "tb PASS field")
if hashlib.sha256(r6_tb.encode("utf-8")).hexdigest() != \
        "e7973a91d04b9f20542b04c58a213e2c8929259768701664dda76721720d2888":
    raise SystemExit("M518 r7 targeted TB reverse proof does not recover r6")

required_v06 = (
    "send_frame_tile(legal_config,payload_bank1,tag_bank1);",
    "send_frame_tile(legal_config,payload_bank0,tag_bank0);",
    "u_dut.raw_ready_q!==2'b11", "u_dut.raw_owned_q!==2'b11",
    "u_dut.raw_order1_q>=u_dut.raw_order0_q",
    "u_dut.raw_tag1_q!==tag_bank1", "u_dut.raw_tag0_q!==tag_bank0",
    "if(!stage1_issue||u_dut.dense_selected_raw_bank!==1'b1",
    "u_dut.dense_raw_bank_q!==1'b1", "u_dut.raw_ready_q!==2'b01",
    "u_dut.raw_owned_q!==2'b11", "u_dut.dense_tag!==tag_bank1",
    "finish_context(2,first_accept_time,1'b0,measured_cycles);",
)
if any(fragment not in tb for fragment in required_v06):
    raise SystemExit("M518 r7 V06 oracle drift")
if tb.count("total_v06_harness_activations=total_v06_harness_activations+1") != 1:
    raise SystemExit("M518 r7 harness activation cardinality drift")

expected_filelist = [
    "rtl_m518/m518_matched_fixed_t10_atlif.sv",
    "verif_m518/m518_matched_fixed_t10_atlif_assertions.sv",
    "tb_m518/tb_m518_matched_fixed_t10_atlif.sv",
]
if [line.strip() for line in filelist_path.read_text().splitlines()
        if line.strip()] != expected_filelist:
    raise SystemExit("M518 r7 filelist topology drift")
assertions = set(re.findall(r"\b(ap_[A-Za-z0-9_]+)\s*:", sva))
covers = set(re.findall(r"\b(cp_[A-Za-z0-9_]+)\s*:", sva))
if len(assertions) != 51 or len(covers) != 25:
    raise SystemExit("M518 r7 assertion/cover cardinality drift")

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
    raise SystemExit("M518 r7 exact PASS signature drift")

for script in Path("dc_handoff/scripts").glob("*"):
    name = script.name.lower()
    if not script.is_file() or not any(token in name for token in (
            "run_dc", "formality", "run_fm", "run_pt", "ptpx")):
        continue
    if "M518_VCS_V06_HARNESS" in script.read_text(
            encoding="utf-8", errors="ignore"):
        raise SystemExit("M518 r7 simulation define leaked into physical flow: " + str(script))
PY

printf '%s\n' \
    "PASS_M518_R7_LEGAL_FILL_HARNESS_REVERSE_SHA_AND_NO_HIERARCHICAL_WRITE_PREFLIGHT_NO_COMPILE_YET" \
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
    "${task_run}/m518_matched_fixed_t10_atlif_author_vcs_receipt_r7.json" <<'PY'
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
    raise SystemExit("M518 r7 assertion cover drift")
receipt = {
    "schema": "m518_matched_fixed_t10_atlif_author_vcs_receipt_v7",
    "status": "PASS_M518_R7_SIMULATION_ONLY_LEGAL_FILL_HARNESS_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN",
    "role": "author_campaign_not_independent_review",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "actual_vcs_id_sha256": digest(sys.argv[4]),
    "exact_runner_sha256": sys.argv[5],
    "r6_failure_review": {
        "review_json_sha256": digest(
            "reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/"
            "m518_vcs_tb_icpd_failure_hammer_r4_20260827.json"),
        "member_manifest_sha256": digest(
            "reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/SHA256SUMS"),
        "outer_seal_file_sha256": digest(
            "reviews/m518_vcs_tb_icpd_failure_hammer_r4_20260827/"
            "SHA256SUMS.seal.sha256"),
        "r6_result_admitted": False,
    },
    "legal_fill_harness": {
        "compile_define": "M518_VCS_V06_HARNESS",
        "hierarchical_state_writes": 0,
        "production_ports_without_define": 50,
        "simulation_ports_with_define": 52,
        "first_fill_bank": 1,
        "second_fill_bank": 0,
        "five_beat_raw_frames": 2,
        "harness_activations": 1,
        "r6_rtl_reverse_sha_exact": True,
        "r6_tb_reverse_sha_exact": True,
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
        "random_contexts": 4,
        "q24_boundary_points": 6,
        "zero_tile_held_release_edges": 8,
        "zero_tile_fault_transitions": 1,
        "release_state_attacks": 5,
        "reset_attacks": 9,
        "clean_after_reset_exact_N1_29": 9,
        "sequential_oldest_attacks": 1,
        "v06_legal_fill_harness": 1,
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
    raise SystemExit("strict finite M518 r7 receipt round-trip drift")
PY

printf '%s  %s\n' "${task_observed_runner}" "${task_runner}" \
    >"${task_run}/runner_sha256.txt"
printf '%s\n' \
    "PASS_M518_R7_SIMULATION_ONLY_LEGAL_FILL_HARNESS_SEALED_V01_V20_SYNOPSYS_VCS_AUTHOR_CAMPAIGN" \
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
echo "PASS M518 r7 exact-SHA VCS author campaign at ${task_run}"
