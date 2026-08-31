#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m489_fc2_canonical_8bank_equal_bandwidth_vcs_r1c_exact_20260827"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M349 sealed VCS run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
 ["rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"]="8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0"
 ["rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"]="529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267"
 ["rtl_m218/m218_fc2_tagged_slice_service_island.sv"]="f6537081977e9dc09e968fad800b333604b4573ee2e9361960483349fe1e8ad1"
 ["rtl_m219/m219_fc2_k1_cropped_tagged_slice_service_island.sv"]="75c4690ec04653084fb59fd75c5ba7ac329807975d76c9ffc43b6304bd4e1d47"
 ["rtl_m342/m342_fc2_standalone_raw4_acc24.sv"]="309759bfa6eeb303143e707bd3df269eddcd31e34e79ed662d507c363ba4d904"
 ["rtl_m349/m349_fc2_k1x8_raw4_acc24.sv"]="ddcf6c051a43813f84fe94a789f209160d522e8a8be79a3fc7b572133393b2c9"
 ["rtl_m488/m488_fc2_bundle_to_8bank_adapter.sv"]="b9024112bb3e3f27ebed60c92437aa136a23fd954568c89413e05724931d4c1b"
 ["rtl_m489/m489_fc2_k8_canonical_8bank_raw4_acc24.sv"]="98ab10e6036c0db0a8faf623317ccf4a1326b950bb11ea9b51ff05c45a6ae550"
 ["verif_m216/m216_fc2_raw4_to_source_cap_frontend_assertions.sv"]="1c8afec4c8035f60237156b93e9af05c4565eaa9eaa4c2527c35356e841689f0"
 ["verif_m218/m218_fc2_tagged_slice_service_assertions.sv"]="030f3cde04488a3d08e42bb074289ea96d022cbc4fc6c0446fc2fac711a16f45"
 ["verif_m219/m219_fc2_k1_cropped_tagged_slice_service_assertions.sv"]="378a81dcd9fc258dd568d8ee283be842b80d632c56315a9126cac074948bd93c"
 ["verif_m342/m342_fc2_standalone_raw4_acc24_assertions.sv"]="530e8883a7cd019dac727d366fba9589adda8b1c8ff6b1f60f23171fefb7d333"
 ["verif_m349/m349_fc2_equal_bandwidth_assertions.sv"]="b11db468b6ef34a463932f64ace8ef2f91bc75f4a9da9fa645d97392695a8226"
 ["verif_m488/m488_fc2_bundle_to_8bank_adapter_assertions.sv"]="fd49748432a286a44ddc99ec58e4ca0d8bcbbf33a17e3ce1d642308d257946ba"
 ["tb_m349/m349_fc2_scalar_bank_memory_model.sv"]="4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa"
 ["tb_m489/tb_m489_fc2_canonical_8bank_equal_bandwidth_raw4_acc24.sv"]="0719905b20a5fa9cd8136f0054beac7785d658916f423280b7065c50139b8d70"
 ["dc_handoff/filelists/date_m489_fc2_canonical_8bank_equal_bandwidth_raw4_acc24_vcs.f"]="90c9370b7d7e8cdba06aa4d523cc9cc8886a65f5242d007eeb85d9395dfc3a9e"
 ["contracts/m489_fc2_canonical_8bank_equal_bandwidth_vcs_contract_r1_20260827.json"]="9730f35fd6ac3cecc4a8b0da02f797d945c0613608092cbe376df99f5fb2a3b2"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
    task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
    printf 'path=%s expected=%s observed=%s\n' "$task_path" \
        "${task_expected[$task_path]}" "$task_observed" \
        >> "$task_run/preflight_sha_checks.txt"
    [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"
sha256sum "$task_runner" > "$task_run/runner_sha256.txt"

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m489_fc2_canonical_8bank_equal_bandwidth_raw4_acc24_vcs.f \
    -top tb_m489_fc2_canonical_8bank_equal_bandwidth_raw4_acc24 \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" \
    && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=349025 -no_save \
    -assert report="$task_run/assert.report" -cm assert \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true

grep -Fq 'PASS M489 canonical-8bank equal-bandwidth FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 service_sva_bound=true adapter_sva_bound=true racefree_cycle_monitor=true request_stalls=705 result_stalls=45 raw_stalls=1162 full8_requests=882 k1x8_full_issue=885 candidate_younger_before_older=994 baseline_younger_before_older=7024' \
    "$task_run/sim.log" || exit 30
grep -Fq 'M489 canonical equalbw B=1 events=20 k8_cycles=59 k1x8_cycles=51 speedup=0.864407 tuple_mismatches=0 weight_mismatches=0' "$task_run/sim.log" || exit 31
grep -Fq 'M489 canonical equalbw B=2 events=41 k8_cycles=143 k1x8_cycles=131 speedup=0.916084 tuple_mismatches=0 weight_mismatches=0' "$task_run/sim.log" || exit 32
grep -Fq 'M489 canonical equalbw B=4 events=90 k8_cycles=505 k1x8_cycles=486 speedup=0.962376 tuple_mismatches=0 weight_mismatches=0' "$task_run/sim.log" || exit 33
grep -Fq 'M489 canonical equalbw B=8 events=110 k8_cycles=1246 k1x8_cycles=1231 speedup=0.987961 tuple_mismatches=0 weight_mismatches=0' "$task_run/sim.log" || exit 34
grep -Fq 'M489 canonical equalbw B=1 events=0 k8_cycles=14 k1x8_cycles=14 speedup=1.000000 tuple_mismatches=0 weight_mismatches=0' "$task_run/sim.log" || exit 35

for task_cover in cp_b1 cp_b2 cp_b4 cp_b8 cp_all_eight_lane_group \
        cp_eight_requests_same_cycle cp_request_backpressure \
        cp_result_stall cp_done cp_protocol_fault; do
    grep -Eq "baseline\.m349_top_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 40
done
for task_cover in cp_k8_request cp_same_cycle_replace cp_result_stall cp_done; do
    grep -Eq "candidate\.core\.g_k8\.service\.m349_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 41
done
for task_bank in 0 1 2 3 4 5 6 7; do
    for task_cover in cp_k1_request cp_same_cycle_replace cp_result_stall cp_done; do
        grep -Eq "baseline\.g_lane\[${task_bank}\]\.service\.m349_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
            "$task_run/assert.report" || exit 42
    done
done
for task_cover in cp_full_eight_bank_request cp_eight_responses_same_cycle \
        cp_out_of_order_bundle_response cp_retire_then_slot_reuse \
        cp_protocol_attack; do
    grep -Eq "candidate\.memory_adapter\.m488_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 43
done

python3 - "$task_run" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
text = (root / "sim.log").read_text()
pattern = re.compile(
    r"M489 canonical equalbw B=(\d+) events=(\d+) k8_cycles=(\d+) "
    r"k1x8_cycles=(\d+) speedup=([0-9.]+) tuple_mismatches=(\d+) "
    r"weight_mismatches=(\d+)"
)
observed = [tuple(match.groups()) for match in pattern.finditer(text)]
expected = [
    ("1", "20", "59", "51", "0.864407", "0", "0"),
    ("2", "41", "143", "131", "0.916084", "0", "0"),
    ("4", "90", "505", "486", "0.962376", "0", "0"),
    ("8", "110", "1246", "1231", "0.987961", "0", "0"),
    ("1", "0", "14", "14", "1.000000", "0", "0"),
]
if observed != expected:
    raise SystemExit(f"unexpected M489 rows: {observed!r}")

rows = []
product = 1.0
for output_blocks, events, k8, k1x8, ratio, _, _ in observed[:4]:
    row = {
        "output_blocks": int(output_blocks),
        "events": int(events),
        "k8_cycles": int(k8),
        "k1x8_cycles": int(k1x8),
        "equal_bandwidth_cycle_ratio": int(k1x8) / int(k8),
    }
    product *= row["equal_bandwidth_cycle_ratio"]
    rows.append(row)
receipt = {
    "schema": "m489_fc2_canonical_8bank_equal_bandwidth_vcs_receipt_v1",
    "status": "PASS_M489_FC2_CANONICAL_8BANK_EQUAL_BANDWIDTH_EXACT_VCS",
    "exact_sha": True,
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 349025,
    "candidate": "frozen M216 SOURCE_CAP=8 plus frozen M218 O8/FIFO4 plus loop-free M488 canonical 8-bank adapter",
    "baseline": "frozen M216 SOURCE_CAP=8 dispatcher plus eight frozen M219 K1 services; aggregate O64/FIFO32",
    "fairness": {
        "logical_banks_each": 8,
        "word_bits_per_bank": 128,
        "peak_bank_words_per_cycle_each": 8,
        "peak_weight_bits_per_cycle_each": 1024,
        "same_scalar_bank_memory_model": True,
        "same_raw_weight_request_response_result_and_done_trajectories": True,
        "response_visibility": "12 of every 17 edge ordinals",
    },
    "clean_cases": 10,
    "zero_event_cases": 2,
    "por_midflight_cases": 2,
    "protocol_attacks": 4,
    "numeric_mismatches": 0,
    "transaction_multiset_mismatches": 0,
    "weight_mismatches": 0,
    "service_sva_bound_and_active": True,
    "race_free_inclusive_cycle_definition": "header_accept through token_done_accept, inclusive",
    "request_stalls": 705,
    "result_stalls": 45,
    "raw_stalls": 1162,
    "candidate_full_eight_bank_requests": 882,
    "baseline_eight_scalar_same_cycle_issues": 885,
    "candidate_younger_before_older_responses": 994,
    "baseline_younger_before_older_responses": 7024,
    "cycle_rows": rows,
    "geomean_equal_bandwidth_cycle_ratio": product ** 0.25,
    "aggregate_equal_bandwidth_cycle_ratio": 1899 / 1953,
    "positive_equal_bandwidth_cycle_speedup": False,
    "interpretation": "After including the canonical independent-bank adapter, shared K8 is 1.2 to 13.6 percent slower than K1x8 in directed nonzero cases. The remaining admissible hypothesis is throughput-per-area and energy efficiency, not cycle speedup.",
    "claim_boundary": {
        "canonical_common_sram_interface": True,
        "equal_bandwidth_standalone_cycle_ratio_measured": True,
        "positive_coalescing_control_cycle_speedup": False,
        "matched_dc": False,
        "complete_fc2": False,
        "complete_ffn": False,
        "physical_speedup": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(root / "m489_fc2_canonical_8bank_equal_bandwidth_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
(root / "m489_fc2_canonical_8bank_equal_bandwidth_vcs_receipt_r1.txt").write_text(
    "\n".join([
        "status=PASS_M489_FC2_CANONICAL_8BANK_EQUAL_BANDWIDTH_EXACT_VCS",
        "exact_sha=true",
        "tool=Synopsys_VCS_V-2023.12-SP1",
        "equal_bandwidth=true",
        "peak_bank_words_per_cycle_each=8",
        "peak_weight_bits_per_cycle_each=1024",
        "k8_cycles_b1_b2_b4_b8=59,143,505,1246",
        "k1x8_cycles_b1_b2_b4_b8=51,131,486,1231",
        "equal_bandwidth_cycle_ratio_b1_b2_b4_b8=0.8644067797,0.9160839161,0.9623762376,0.9879614767",
        f"geomean_equal_bandwidth_cycle_ratio={product ** 0.25}",
        "aggregate_equal_bandwidth_cycle_ratio=0.9723502304",
        "positive_coalescing_control_cycle_speedup=false",
        "numeric_mismatches=0",
        "transaction_multiset_mismatches=0",
        "weight_mismatches=0",
        "candidate_younger_before_older_responses=994",
        "baseline_younger_before_older_responses=7024",
        "complete_fc2=false",
        "complete_ffn=false",
        "physical_speedup=false",
        "system_speedup=false",
        "paper_ppa_ready=false",
        "headline=false",
    ]) + "\n"
)
(root / "README.md").write_text(
    "# M489 canonical eight-bank FC2 VCS\n\n"
    "Exact-SHA Synopsys VCS passes B1/B2/B4/B8 and zero-event cases for "
    "the M218+M488 K8 candidate and eight-M219 K1x8 baseline behind the "
    "same scalar bank model. K8 is 1.2--13.6% slower on the nonzero "
    "directed cases, so no cycle speedup is admitted. M218, M488 and all "
    "eight M219 "
    "service SVA instances are bound and active; request-ID scoreboards "
    "observe younger-before-older retirement on both sides; expanded "
    "request/response/weight/result comparison has zero mismatch.\n\n"
    "Boundary: raw FC2 activity to signed Acc24 only. This is not complete "
    "FC2/FFN, PPA, physical, system, or headline evidence.\n"
)
PY

printf 'PASS_M489_FC2_CANONICAL_8BANK_EQUAL_BANDWIDTH_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
(
    cd "$task_run"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M489 exact VCS sealed at $task_run"
