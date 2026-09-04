#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="${M497_RUN_DIR:-$task_hw_root/results/m497_fc2_canonical_k1_vs_k1x8_vcs_r1_exact_20260827}"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M497 sealed VCS run" >&2
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
 ["rtl_m342/m342_fc2_standalone_raw4_acc24.sv"]="aa017fd3cf18557214d2542f0047d2f0fc7ac6f16a7bcdfc7cd2336568f6e27d"
 ["rtl_m349/m349_fc2_k1x8_raw4_acc24.sv"]="ddcf6c051a43813f84fe94a789f209160d522e8a8be79a3fc7b572133393b2c9"
 ["rtl_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter.sv"]="e5f3022e23736216f61482e1e33638d84c9a39dfb807c1c2fc53a14c90696456"
 ["rtl_m499/m499_fc2_k1_no_reuse_8bank_raw4_acc24.sv"]="fdbdc0751b491e6559038df03a481b9fdd927f725e9380b14ec630f85d887205"
 ["verif_m216/m216_fc2_raw4_to_source_cap_frontend_assertions.sv"]="1c8afec4c8035f60237156b93e9af05c4565eaa9eaa4c2527c35356e841689f0"
 ["verif_m218/m218_fc2_tagged_slice_service_assertions.sv"]="030f3cde04488a3d08e42bb074289ea96d022cbc4fc6c0446fc2fac711a16f45"
 ["verif_m219/m219_fc2_k1_cropped_tagged_slice_service_assertions.sv"]="378a81dcd9fc258dd568d8ee283be842b80d632c56315a9126cac074948bd93c"
 ["verif_m342/m342_fc2_standalone_raw4_acc24_assertions.sv"]="530e8883a7cd019dac727d366fba9589adda8b1c8ff6b1f60f23171fefb7d333"
 ["verif_m349/m349_fc2_equal_bandwidth_assertions.sv"]="b11db468b6ef34a463932f64ace8ef2f91bc75f4a9da9fa645d97392695a8226"
 ["verif_m499/m499_fc2_bundle_to_8bank_no_reuse_adapter_assertions.sv"]="28b137431102c6a45a98eadba7b06a1bd94105f9e406df87fd02819f133cc8a0"
 ["tb_m349/m349_fc2_scalar_bank_memory_model.sv"]="4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa"
 ["tb_m497/tb_m497_fc2_canonical_k1_vs_k1x8_raw4_acc24.sv"]="cd1b88f4d0e8259ec0e8766c99e9c48aaf3eaf3e4df701609314d66830fcb831"
 ["dc_handoff/filelists/date_m497_fc2_canonical_k1_vs_k1x8_raw4_acc24_vcs.f"]="6dc1fe6aac5f7d5b1ead433577c7d69e3647dbb4f8ee727f208e189de28e945a"
 ["contracts/m506_fc2_synth_portable_onehot_vcs_contract_r1_20260827.json"]="0899801fa9bf29bd5619ff942353ced907c76b1a5ec3c06214801508826485cd"
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
    -f dc_handoff/filelists/date_m497_fc2_canonical_k1_vs_k1x8_raw4_acc24_vcs.f \
    -top tb_m497_fc2_canonical_k1_vs_k1x8_raw4_acc24 \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
! grep -Eiq 'Error-\[|^Error' "$task_run/compile.log" || exit 21
[[ "$(grep -Ec 'Warning-\[' "$task_run/compile.log" || true)" -eq 1 ]] || exit 21
grep -Fq 'Warning-[BTNL] Bind target in parent library is not loaded' \
    "$task_run/compile.log" || exit 21
grep -Fq "target of bind statement 'm218_fc2_tagged_slice_service_island'" \
    "$task_run/compile.log" || exit 21

set +e
"$task_run/simv" +ntb_random_seed=349025 -no_save \
    -assert report="$task_run/assert.report" -cm assert \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
! grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "$task_run/sim.log" "$task_run/assert.report" || exit 23

grep -Fq 'PASS M497 canonical-K1 versus K1x8 FC2 VCS clean_cases=10 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 service_sva_bound=true adapter_sva_bound=true racefree_cycle_monitor=true request_stalls=3499 result_stalls=46 raw_stalls=4540 single1_requests=8052 k1x8_full_issue=885 candidate_younger_before_older=6434 baseline_younger_before_older=7024' "$task_run/sim.log" || exit 30
for task_row in \
 'B=1 events=20 k1_cycles=253 k1x8_cycles=51 k1x8_speedup_vs_k1=4.960784' \
 'B=2 events=41 k1_cycles=773 k1x8_cycles=131 k1x8_speedup_vs_k1=5.900763' \
 'B=4 events=90 k1_cycles=3154 k1x8_cycles=486 k1x8_speedup_vs_k1=6.489712' \
 'B=8 events=110 k1_cycles=7659 k1x8_cycles=1231 k1x8_speedup_vs_k1=6.221771' \
 'B=1 events=0 k1_cycles=14 k1x8_cycles=14 k1x8_speedup_vs_k1=1.000000'; do
    grep -Fq "M497 canonical K1 versus K1x8 $task_row tuple_mismatches=0 weight_mismatches=0" \
        "$task_run/sim.log" || exit 31
done

for task_cover in cp_b1 cp_b2 cp_b4 cp_b8 cp_all_eight_lane_group \
        cp_eight_requests_same_cycle cp_request_backpressure \
        cp_result_stall cp_done cp_protocol_fault; do
    grep -Eq "baseline\.m349_top_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 40
done
for task_cover in cp_k1_request cp_same_cycle_replace cp_result_stall cp_done; do
    grep -Eq "candidate\.core\.g_k1\.service\.m349_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 41
done
for task_bank in 0 1 2 3 4 5 6 7; do
    for task_cover in cp_k1_request cp_same_cycle_replace cp_result_stall cp_done; do
        grep -Eq "baseline\.g_lane\[${task_bank}\]\.service\.m349_bound_service_sva\.${task_cover}, .* [1-9][0-9]* match" \
            "$task_run/assert.report" || exit 42
    done
done
for task_cover in cp_pending_request_stall cp_out_of_order_bundle_response \
        cp_retire_then_slot_reuse cp_cutthrough_bundle_response \
        cp_protocol_attack; do
    grep -Eq "candidate\.memory_adapter\.m499_sva\.${task_cover}, .* [1-9][0-9]* match" \
        "$task_run/assert.report" || exit 43
done

python3 - "$task_run" <<'PY'
import json
import pathlib
import re
import sys
from functools import reduce
from operator import mul

root = pathlib.Path(sys.argv[1])
text = (root / "sim.log").read_text()
pattern = re.compile(
    r"M497 canonical K1 versus K1x8 B=(\d+) events=(\d+) "
    r"k1_cycles=(\d+) k1x8_cycles=(\d+) k1x8_speedup_vs_k1=([0-9.]+) "
    r"tuple_mismatches=(\d+) weight_mismatches=(\d+)"
)
observed = [tuple(match.groups()) for match in pattern.finditer(text)]
expected = [
    ("1", "20", "253", "51", "4.960784", "0", "0"),
    ("2", "41", "773", "131", "5.900763", "0", "0"),
    ("4", "90", "3154", "486", "6.489712", "0", "0"),
    ("8", "110", "7659", "1231", "6.221771", "0", "0"),
    ("1", "0", "14", "14", "1.000000", "0", "0"),
]
if observed != expected:
    raise SystemExit("unexpected M497 rows: {!r}".format(observed))
rows = []
ratios = []
for output_blocks, events, k1, k1x8, ratio, _, _ in observed[:4]:
    exact_ratio = int(k1) / int(k1x8)
    ratios.append(exact_ratio)
    rows.append({
        "output_blocks": int(output_blocks),
        "events": int(events),
        "canonical_k1_cycles": int(k1),
        "replicated_k1x8_cycles": int(k1x8),
        "k1x8_speedup_vs_k1": exact_ratio,
    })
geomean = reduce(mul, ratios, 1.0) ** 0.25
aggregate = sum(row["canonical_k1_cycles"] for row in rows) / sum(
    row["replicated_k1x8_cycles"] for row in rows)
receipt = {
    "schema": "m497_fc2_canonical_k1_vs_k1x8_vcs_receipt_v1",
    "status": "PASS_M497_FC2_CANONICAL_K1_VS_K1X8_EXACT_VCS",
    "exact_sha": True,
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 349025,
    "candidate": "frozen M216/M219 K1 plus M499 no-reuse 8-bank adapter",
    "baseline": "frozen M216 dispatcher plus eight frozen M219 K1 services",
    "fairness": {
        "logical_banks_each": 8,
        "word_bits_per_bank": 128,
        "peak_bank_words_per_cycle_k1": 1,
        "peak_bank_words_per_cycle_k1x8": 8,
        "same_scalar_bank_memory_model": True,
        "same_raw_payload_and_reference_arithmetic": True,
        "same_external_visibility_schedule": True,
    },
    "clean_cases": 10,
    "por_midflight_cases": 2,
    "protocol_attacks": 4,
    "numeric_mismatches": 0,
    "transaction_multiset_mismatches": 0,
    "weight_mismatches": 0,
    "cycle_rows": rows,
    "geomean_k1x8_speedup_vs_k1": geomean,
    "aggregate_k1x8_speedup_vs_k1": aggregate,
    "interpretation": "K1x8 is 8x replicated bank/service bandwidth and is 5.86x faster geometrically than one K1 service. This is the low-bandwidth endpoint of a resource-performance Pareto, not a same-resource or system speedup.",
    "claim_boundary": {
        "directed_fc2_slice_vcs": True,
        "bandwidth_scaling_measured": True,
        "same_resource_speedup": False,
        "frozen_h67_replay": False,
        "complete_fc2": False,
        "complete_ffn": False,
        "physical": False,
        "power": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(root / "m497_fc2_canonical_k1_vs_k1x8_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
(root / "m497_fc2_canonical_k1_vs_k1x8_vcs_receipt_r1.txt").write_text(
    "\n".join([
        "status=PASS_M497_FC2_CANONICAL_K1_VS_K1X8_EXACT_VCS",
        "exact_sha=true",
        "tool=Synopsys_VCS_V-2023.12-SP1",
        "canonical_k1_cycles_b1_b2_b4_b8=253,773,3154,7659",
        "replicated_k1x8_cycles_b1_b2_b4_b8=51,131,486,1231",
        "k1x8_speedup_vs_k1_b1_b2_b4_b8=" + ",".join(str(x) for x in ratios),
        "geomean_k1x8_speedup_vs_k1={}".format(geomean),
        "aggregate_k1x8_speedup_vs_k1={}".format(aggregate),
        "same_resource_speedup=false",
        "system_speedup=false",
        "paper_ppa_ready=false",
        "headline=false",
    ]) + "\n")
(root / "README.md").write_text(
    "# M497 canonical K1 versus replicated K1x8\n\n"
    "Exact-SHA Synopsys VCS validates identical FC2 slice arithmetic and "
    "transaction multisets behind the same eight scalar SRAM models. K1x8 "
    "is 8x replicated service/bank bandwidth and reaches 5.86x geometric "
    "cycle speedup over canonical K1. This is a bandwidth-scaling Pareto "
    "endpoint, not same-resource or system speedup. M499 removes a K1-only "
    "three-layer same-edge slot-reuse feedback path without changing M490/M492.\n")
PY

printf 'PASS_M497_FC2_CANONICAL_K1_VS_K1X8_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
(
    cd "$task_run"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M497 exact VCS sealed at $task_run"
