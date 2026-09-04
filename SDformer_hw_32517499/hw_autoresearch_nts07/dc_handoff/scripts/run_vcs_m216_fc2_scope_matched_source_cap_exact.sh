#!/usr/bin/env bash
set -euo pipefail

task_dc_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_dc_root/.." && pwd)"
task_run="$task_hw_root/results/m216_fc2_scope_matched_source_cap_vcs_r1_exact_20260825"
task_vcs="${VCS_HOME:-/opt/synopsys/vcs/V-2023.12-SP1}"
[[ ! -e "$task_run" ]] || {
    echo "refusing to overwrite M216 sealed VCS run" >&2
    exit 2
}
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT
cd "$task_hw_root"

declare -A task_expected=(
 ["rtl_m214/m214_fc2_raw4_to_descriptor4_terminal_hint_compactor.sv"]="e278da8b0deaa0dda07b0477930453daa40b0331399a3941b743d604d0b102a5"
 ["rtl_m214/m214_fc2_descriptor4_same_done_load_frontend.sv"]="e9384a4825d6d0fde11679e74ec5e3973d17da325e6c8df40d7491ce203c0317"
 ["rtl_m214/m214_fc2_raw4_to_same_done_load_frontend.sv"]="d5caa7f3431761bacde2190412215ef84346a64b3b0559e7cff3116c63f97862"
 ["rtl_m216/m216_fc2_descriptor4_source_cap_frontend.sv"]="8295393bf91a9bfc64a2253aaff60db97df5df587ab9b77d56996afee82cb2a0"
 ["rtl_m216/m216_fc2_raw4_to_source_cap_frontend.sv"]="529e463802fec72716ac6592d31e7668104a5463ff92499a98ec7314c8e88267"
 ["verif_m216/m216_fc2_raw4_to_source_cap_frontend_assertions.sv"]="1c8afec4c8035f60237156b93e9af05c4565eaa9eaa4c2527c35356e841689f0"
 ["tb_m216/tb_m216_fc2_raw4_to_source_cap_frontend.sv"]="e840a1c302ab6f84ce914e5d65f4553496d47e199dbe9487e2ea13591b94461c"
 ["tb_m216/tb_m216_fc2_source_cap8_shadow_miter.sv"]="0a9f832c276194a737e2ea229ad77b4deea0b94e899c88021de70eeb8b654324"
 ["tb_m216/tb_m216_fc2_source_cap_tail_sweep.sv"]="6856bda3d0b46586e0037c0d26685c21e2a231edfa9530d9b6b0b4f75179a6ec"
 ["tb_m216/tb_m216_fc2_k1_dense_bank96.sv"]="8e420d441e083152f59bf6d82bcad95a3a943674dd0c04f098e4d15ab6b3f06f"
 ["dc_handoff/filelists/date_m216_fc2_source_cap_vcs.f"]="d4e1dd2847988ce731d8e46dca663518e6ae532b23e5bba37faa920d30e5e5bd"
 ["dc_handoff/filelists/date_m216_fc2_source_cap8_shadow_miter_vcs.f"]="4723648de923b55325eb50beb8c8d70b8ac8f40532b9f36b93976bf964167f53"
 ["dc_handoff/filelists/date_m216_fc2_source_cap_tail_sweep_vcs.f"]="967d6604f4d1c34c2b026b5476ccca0895988a03d2a103c0a4b192324614c180"
 ["dc_handoff/filelists/date_m216_fc2_k1_dense_bank96_vcs.f"]="636a0726fc2562356d8c669af012785195ee42dbc8e49cff5e2e0f13c056c800"
 ["system_simulator/scripts/explore_m214_fc2_same_cycle_done_load_recurrence.py"]="01a870f85ae62208d9d9c145021a476a59b26cb2fc0fad343d2ed51006517b5e"
 ["system_simulator/scripts/explore_m216_fc2_source_cap_recurrence.py"]="1bed35da65287b48bcaee0e5181bcfae01c3dbc41ea1927d9eccbf94dfaf380b"
 ["contracts/m216_fc2_scope_matched_source_cap_vcs_contract_r1_20260825.json"]="5c7bca14548488c478d12c246936460a3ce09a6d65655203e2302f08d6afca68"
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
run_vcs() {
    local task_name="$1" task_filelist="$2" task_top="$3"
    local task_dir="$task_run/$task_name"
    mkdir "$task_dir"
    set +e
    "$task_vcs/bin/vcs" -full64 -sverilog -assert svaext \
        +define+SVA_RUNTIME_ENABLED -timescale=1ns/1ps -cm assert \
        -Mdir="$task_dir/csrc" -f "$task_filelist" -top "$task_top" \
        -o "$task_dir/simv" > "$task_dir/compile.log" 2>&1
    local task_rc=$?
    set -e
    echo "$task_rc" > "$task_dir/compile.rc"
    [[ "$task_rc" -eq 0 && -x "$task_dir/simv" ]] || exit 20
    grep -Eiq 'Warning-\[|Error-\[|^Error' "$task_dir/compile.log" \
        && exit 21 || true
    set +e
    "$task_dir/simv" +ntb_random_seed=216025 -no_save \
        -assert report="$task_dir/assert.report" -cm assert \
        > "$task_dir/sim.log" 2>&1
    task_rc=$?
    set -e
    echo "$task_rc" > "$task_dir/sim.rc"
    [[ "$task_rc" -eq 0 ]] || exit 22
    grep -Eiq 'failed at|Offending|^Error|^Fatal|Fatal:|watchdog' \
        "$task_dir/sim.log" "$task_dir/assert.report" && exit 23 || true
}

run_vcs k1_directed \
    dc_handoff/filelists/date_m216_fc2_source_cap_vcs.f \
    tb_m216_fc2_raw4_to_source_cap_frontend
run_vcs k8_shadow_miter \
    dc_handoff/filelists/date_m216_fc2_source_cap8_shadow_miter_vcs.f \
    tb_m216_fc2_source_cap8_shadow_miter
run_vcs k1_tail \
    dc_handoff/filelists/date_m216_fc2_source_cap_tail_sweep_vcs.f \
    tb_m216_fc2_source_cap_tail_sweep
run_vcs k1_dense \
    dc_handoff/filelists/date_m216_fc2_k1_dense_bank96_vcs.f \
    tb_m216_fc2_k1_dense_bank96

grep -Fq 'PASS M216 source-cap cycle co-sim VCS source_cap=1' \
    "$task_run/k1_directed/sim.log" || exit 30
grep -Fq 'PASS M216 K8 shadow is cycle-identical to M214' \
    "$task_run/k8_shadow_miter/sim.log" || exit 31
grep -Fxq 'PASS M216 source-cap tail sweep source_cap=1 cases=256' \
    "$task_run/k1_tail/sim.log" || exit 32
grep -Fq 'PASS M216 K1 dense bank96 VCS events=3072 groups=24576 done=1 dense_packet_accepts=8' \
    "$task_run/k1_dense/sim.log" || exit 33
grep -Fq 'header_to_done_cycles=24585 source_conservation=true' \
    "$task_run/k1_dense/sim.log" || exit 34
[[ "$(grep -c '^M216TAIL ' "$task_run/k1_tail/sim.log")" -eq 256 ]] \
    || exit 35
grep -Eq 'cp_group_accept, .* 24576 match' \
    "$task_run/k1_dense/assert.report" || exit 36
grep -Eq 'cp_descriptor_bank_sum_48, .* 8 match' \
    "$task_run/k1_dense/assert.report" || exit 37

python3 system_simulator/scripts/explore_m216_fc2_source_cap_recurrence.py \
    --sweep-log "$task_run/k1_tail/sim.log" \
    --output "$task_run/model/m216_k1_rtl_control_recurrence.json" \
    > "$task_run/model_stdout.log"
python3 - "$task_run" <<'PY'
import importlib.util
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
hw = pathlib.Path.cwd()

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m214 = load(hw / "system_simulator/scripts/explore_m214_fc2_same_cycle_done_load_recurrence.py", "m214_exact")
m216 = load(hw / "system_simulator/scripts/explore_m216_fc2_source_cap_recurrence.py", "m216_exact")
records = []
for blocks in (1, 2, 4, 8):
    depth = m214.GEOMETRY[blocks][1]
    for mode in range(4):
        for seed in range(16):
            payload = m214.sweep_payload(blocks, mode, seed)
            old = m214.simulate_m214(payload, depth, blocks)["cycles"]
            shadow = m216.simulate_m216(
                payload, depth, blocks, source_cap=8)["cycles"]
            records.append({"blocks": blocks, "mode": mode, "seed": seed,
                            "m214_cycles": old, "m216_k8_cycles": shadow})
assert len(records) == 256
mismatches = [item for item in records
              if item["m214_cycles"] != item["m216_k8_cycles"]]
result = {
    "schema": "m216_k8_model_vs_m214_identity_v1",
    "status": "PASS_EXACT_256_CASE_MODEL_IDENTITY" if not mismatches
              else "FAIL_MODEL_IDENTITY",
    "cases": len(records), "mismatches": len(mismatches),
    "all_records": records, "mismatch_records": mismatches,
    "claim_boundary": {"model_identity_only": True,
                       "physical_speedup": False,
                       "system_speedup": False, "headline": False},
}
path = root / "model/m216_k8_vs_m214_model_identity.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
assert not mismatches

k1 = json.loads((root / "model/m216_k1_rtl_control_recurrence.json").read_text())
assert k1["status"] == "PASS_EXACT_256_CASE_VCS"
assert k1["cases"] == 256 and k1["mismatches"] == 0
assert k1["source_caps"] == [1]
PY

{
    echo status=PASS_M216_FC2_SCOPE_MATCHED_SOURCE_CAP_EXACT_VCS
    echo exact_sha=true
    echo tool=Synopsys_VCS_V-2023.12-SP1
    echo k1_directed_reference_mismatches=0
    echo k8_m214_cycle_miter_mismatches=0
    echo k1_recurrence_cases=256
    echo k1_recurrence_mismatches=0
    echo k8_m214_model_identity_cases=256
    echo k8_m214_model_identity_mismatches=0
    echo dense_stage3_events=3072
    echo dense_stage3_groups=24576
    echo dense_stage3_header_to_done_cycles=24585
    echo scope_matched_k1_k8=true
    echo complete_fc2=false
    echo physical_speedup=false
    echo system_speedup=false
    echo headline=false
} > "$task_run/m216_fc2_scope_matched_source_cap_vcs_receipt_r1.txt"
sha256sum "$0" > "$task_run/runner_sha256.txt"
(
    cd "$task_run"
    find . -type f ! -name SHA256SUMS -print0 | sort -z \
        | xargs -0 sha256sum > SHA256SUMS
)
printf 'PASS_M216_FC2_SCOPE_MATCHED_SOURCE_CAP_EXACT_VCS\n' \
    > "$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M216 exact VCS sealed at $task_run"
