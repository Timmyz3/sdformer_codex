#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_hw_root/results/m342_fc2_standalone_raw4_acc24_directed_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M342 sealed VCS run" >&2
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
 ["verif_m216/m216_fc2_raw4_to_source_cap_frontend_assertions.sv"]="1c8afec4c8035f60237156b93e9af05c4565eaa9eaa4c2527c35356e841689f0"
 ["verif_m218/m218_fc2_tagged_slice_service_assertions.sv"]="030f3cde04488a3d08e42bb074289ea96d022cbc4fc6c0446fc2fac711a16f45"
 ["verif_m219/m219_fc2_k1_cropped_tagged_slice_service_assertions.sv"]="378a81dcd9fc258dd568d8ee283be842b80d632c56315a9126cac074948bd93c"
 ["verif_m342/m342_fc2_standalone_raw4_acc24_assertions.sv"]="530e8883a7cd019dac727d366fba9589adda8b1c8ff6b1f60f23171fefb7d333"
 ["tb_m342/m342_fc2_eight_bank_memory_model.sv"]="38c64af719ab8728a051892532b3fabfabb1bcd79b560d5ea28108fac4023517"
 ["tb_m342/tb_m342_fc2_standalone_raw4_acc24.sv"]="557b0b3a7c838da46fb5d78ee1cf2f112d418e7bb06bd85edfd0b488ea79bae8"
 ["dc_handoff/filelists/date_m342_fc2_standalone_raw4_acc24_directed_vcs.f"]="d2599fcfd05b87e8a8f0ad83eccd7b39987f8960db77b3740dd32bcbfe240b28"
 ["contracts/m342_fc2_standalone_raw4_acc24_directed_vcs_contract_r1_20260825.json"]="ab3d14c1e60b0c8a4e927e570e8714ec4879db16126f0f622e17569b93a4e26d"
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

export VCS_HOME="$task_vcs" VCS_ARCH_OVERRIDE="${VCS_ARCH_OVERRIDE:-linux}"
set +e
"$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
    +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
    -Mdir="$task_run/csrc" \
    -f dc_handoff/filelists/date_m342_fc2_standalone_raw4_acc24_directed_vcs.f \
    -top tb_m342_fc2_standalone_raw4_acc24 \
    -o "$task_run/simv" > "$task_run/compile.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/compile.rc"
[[ "$task_rc" -eq 0 && -x "$task_run/simv" ]] || exit 20
grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_run/compile.log" \
    && exit 21 || true

set +e
"$task_run/simv" +ntb_random_seed=342025 -no_save \
    -assert report="$task_run/assert.report" -cm assert \
    > "$task_run/sim.log" 2>&1
task_rc=$?
set -e
echo "$task_rc" > "$task_run/sim.rc"
[[ "$task_rc" -eq 0 ]] || exit 22
grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
    "$task_run/sim.log" "$task_run/assert.report" && exit 23 || true

grep -Fq 'PASS M342 standalone raw4-to-Acc24 VCS clean_cases=10 B1_B2_B4_B8=true zero_tokens=2 reset_cases=2 protocol_attacks=4 numeric_mismatches=0 K8_K1_reference_exact=true request_stalls=1541 result_stalls=44 raw_stalls=3698 full8_requests=882 ooo_responses=9262' \
    "$task_run/sim.log" || exit 30
grep -Fq 'M342 clean cap=8 B=1 events=20 cycles=42 results=6 mismatches=0' "$task_run/sim.log" || exit 31
grep -Fq 'M342 clean cap=1 B=1 events=20 cycles=160 results=6 mismatches=0' "$task_run/sim.log" || exit 32
grep -Fq 'M342 clean cap=8 B=2 events=41 cycles=112 results=12 mismatches=0' "$task_run/sim.log" || exit 33
grep -Fq 'M342 clean cap=1 B=2 events=41 cycles=602 results=12 mismatches=0' "$task_run/sim.log" || exit 34
grep -Fq 'M342 clean cap=8 B=4 events=90 cycles=410 results=24 mismatches=0' "$task_run/sim.log" || exit 35
grep -Fq 'M342 clean cap=1 B=4 events=90 cycles=2566 results=24 mismatches=0' "$task_run/sim.log" || exit 36
grep -Fq 'M342 clean cap=8 B=8 events=110 cycles=1027 results=48 mismatches=0' "$task_run/sim.log" || exit 37
grep -Fq 'M342 clean cap=1 B=8 events=110 cycles=6235 results=48 mismatches=0' "$task_run/sim.log" || exit 38
for task_instance in candidate baseline; do
    for task_cover in cp_b1 cp_b2 cp_b4 cp_b8 cp_group_stall \
            cp_memory_request_stall cp_result_stall cp_final_done \
            cp_protocol_attack; do
        grep -Eq "${task_instance}\.sva\.${task_cover}, .* [1-9][0-9]* match" \
            "$task_run/assert.report" || exit 40
    done
done
grep -Eq 'candidate\.sva\.cp_full_eight_source_request, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 41
grep -Eq 'baseline\.sva\.cp_single_source_request, .* [1-9][0-9]* match' \
    "$task_run/assert.report" || exit 42

python3 - "$task_run" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = [
    {"output_blocks": 1, "events": 20, "k8_cycles": 42,
     "k1_cycles": 160},
    {"output_blocks": 2, "events": 41, "k8_cycles": 112,
     "k1_cycles": 602},
    {"output_blocks": 4, "events": 90, "k8_cycles": 410,
     "k1_cycles": 2566},
    {"output_blocks": 8, "events": 110, "k8_cycles": 1027,
     "k1_cycles": 6235},
]
product = 1.0
for row in rows:
    row["standalone_cycle_speedup"] = row["k1_cycles"] / row["k8_cycles"]
    product *= row["standalone_cycle_speedup"]
receipt = {
    "schema": "m342_fc2_standalone_raw4_acc24_vcs_receipt_v1",
    "status": "PASS_M342_STANDALONE_RAW4_TO_ACC24_EXACT_VCS",
    "exact_sha": True,
    "tool": "Synopsys VCS V-2023.12-SP1",
    "seed": 342025,
    "candidate": "M216 SOURCE_CAP=8 plus M218",
    "baseline": "M216 SOURCE_CAP=1 plus M219",
    "memory": {"logical_banks": 8, "word_bits_per_bank": 128,
               "latency_cycles": 4, "out_of_order": True,
               "same_active_bank_reads": True},
    "clean_cases": 10,
    "zero_event_tokens": 2,
    "common_por_midflight_cases": 2,
    "protocol_attacks": 4,
    "numeric_mismatches": 0,
    "k8_k1_software_reference_exact": True,
    "request_stalls": 1541,
    "result_stalls": 44,
    "raw_stalls": 3698,
    "full_eight_source_requests": 882,
    "observed_out_of_order_responses": 9262,
    "cycle_rows": rows,
    "geomean_standalone_cycle_speedup": product ** 0.25,
    "mapping": {"source": "96*beat+8*row+bank",
                "destination": "96*block+16*slice+lane"},
    "claim_boundary": {
        "standalone_raw4_to_acc24_fc2": True,
        "standalone_cycle_speedup": True,
        "complete_model_layer": False,
        "complete_fc2": False,
        "complete_ffn": False,
        "physical_speedup": False,
        "system_speedup": False,
        "paper_ppa_ready": False,
        "headline": False,
    },
}
(root / "m342_fc2_standalone_raw4_acc24_vcs_receipt_r1.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

{
    echo status=PASS_M342_STANDALONE_RAW4_TO_ACC24_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo clean_cases=10
    echo b1_b2_b4_b8_both_caps=true
    echo zero_event_tokens=2
    echo common_por_midflight_cases=2
    echo protocol_attacks=4
    echo numeric_mismatches=0
    echo k8_k1_software_reference_exact=true
    echo k8_cycles_b1_b2_b4_b8=42,112,410,1027
    echo k1_cycles_b1_b2_b4_b8=160,602,2566,6235
    echo standalone_speedup_b1_b2_b4_b8=3.8095238095,5.375,6.2585365854,6.0710808179
    echo geomean_standalone_cycle_speedup=5.2813748451
    echo complete_model_layer=false
    echo complete_fc2=false
    echo complete_ffn=false
    echo physical_speedup=false
    echo system_speedup=false
    echo paper_ppa_ready=false
    echo headline=false
} > "$task_run/m342_fc2_standalone_raw4_acc24_vcs_receipt_r1.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
(
    cd "$task_run"
    find . -type f ! -name SHA256SUMS -print0 | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)
printf 'PASS_M342_STANDALONE_RAW4_TO_ACC24_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M342 exact VCS sealed at $task_run"
