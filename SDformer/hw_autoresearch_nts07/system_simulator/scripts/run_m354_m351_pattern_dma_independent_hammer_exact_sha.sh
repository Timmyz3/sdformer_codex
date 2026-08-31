#!/usr/bin/env bash
set -euo pipefail

task_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_hw_root="$(cd "$task_script_dir/../.." && pwd)"
task_sdformer_root="$(cd "$task_hw_root/.." && pwd)"
task_contract="$task_hw_root/contracts/m354_m351_pattern_dma_independent_hammer_contract_r1_20260825.json"
task_result="$task_hw_root/results/m354_m351_pattern_dma_independent_hammer_r1_20260825"
task_tmp="$(mktemp -d /tmp/m354_m351_exact.XXXXXX)"
task_complete=0
cleanup() {
    task_rc=$?
    if [[ $task_complete -ne 1 ]]; then
        if [[ ! -e "$task_result" ]]; then mkdir -p "$task_result"; fi
        printf 'status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n' \
            "$task_rc" > "$task_result/RUN_FAILED_OR_INCOMPLETE.txt"
    fi
    rm -rf "$task_tmp"
}
trap cleanup EXIT

[[ ! -e "$task_result" ]] || {
    echo "refusing to overwrite M354 sealed review" >&2
    exit 2
}
cd "$task_hw_root"

[[ "$(sha256sum "$task_contract" | awk '{print $1}')" == \
    "85f78df45022943b5b4463af7182500e6a87ca1f1517d6c3cadc60e9e43ea885" ]] \
    || exit 10
[[ "$(sha256sum system_simulator/scripts/analyze_m354_m351_pattern_dma_independent_hammer.py | awk '{print $1}')" == \
    "80e4291e0015976c47b77db468a3e433d7660f01bdc38330de6131744e5aa043" ]] \
    || exit 11

python3 - "$task_contract" "$task_hw_root" > "$task_tmp/preflight_sha_checks.txt" <<'PY'
import hashlib
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
contract = json.loads(contract_path.read_text())
for name, identity in sorted(contract["inputs"].items()):
    path = root / identity["path"]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"name={name} path={identity['path']} expected={identity['sha256']} observed={observed}")
    if observed != identity["sha256"]:
        raise SystemExit(12)
PY

{
    cd results/m339_q128_selective_pwp_kfirst_cycle_r1_20260825
    sha256sum -c SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS
    cd "$task_hw_root/results/m344_output_block_tiled_q128_kfirst_r1_20260825"
    sha256sum -c SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS
    cd "$task_hw_root/results/m347_m344_output_block_tiled_q128_independent_hammer_r1_20260825"
    sha256sum -c SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS
    cd "$task_sdformer_root"
    sha256sum -c hw_autoresearch_nts07/results/m351_m344_pattern_dma_correction_overlay_r1_20260825/SHA256SUMS.seal.sha256
    sha256sum -c hw_autoresearch_nts07/results/m351_m344_pattern_dma_correction_overlay_r1_20260825/SHA256SUMS
} > "$task_tmp/parent_seal_verification.log"
cd "$task_hw_root"

python3 system_simulator/scripts/analyze_m351_m344_pattern_dma_correction_overlay.py \
    --contract contracts/m351_m344_pattern_dma_correction_overlay_contract_r1_20260825.json \
    --output-dir "$task_tmp/m351_replay" \
    > "$task_tmp/m351_exact_sha_replay.log" 2>&1
cmp "$task_tmp/m351_replay/m351_m344_pattern_dma_correction_overlay_r1.json" \
    results/m351_m344_pattern_dma_correction_overlay_r1_20260825/m351_m344_pattern_dma_correction_overlay_r1.json

python3 system_simulator/scripts/analyze_m339_q128_selective_pwp_kfirst_cycle.py \
    --contract contracts/m339_q128_selective_pwp_kfirst_cycle_contract_r1_20260825.json \
    --output-dir "$task_tmp/m339_replay" \
    > "$task_tmp/m339_exact_sha_replay.log" 2>&1
cmp "$task_tmp/m339_replay/m339_q128_selective_pwp_kfirst_cycle_r1.json" \
    results/m339_q128_selective_pwp_kfirst_cycle_r1_20260825/m339_q128_selective_pwp_kfirst_cycle_r1.json

python3 system_simulator/scripts/analyze_m354_m351_pattern_dma_independent_hammer.py \
    --contract "$task_contract" \
    --m351-replay "$task_tmp/m351_replay/m351_m344_pattern_dma_correction_overlay_r1.json" \
    --m339-replay "$task_tmp/m339_replay/m339_q128_selective_pwp_kfirst_cycle_r1.json" \
    --output-dir "$task_result" > "$task_tmp/m354_analysis.log" 2>&1

cp "$task_tmp/preflight_sha_checks.txt" "$task_result/"
cp "$task_tmp/parent_seal_verification.log" "$task_result/"
cp "$task_tmp/m351_exact_sha_replay.log" "$task_result/"
cp "$task_tmp/m339_exact_sha_replay.log" "$task_result/"
cp "$task_tmp/m354_analysis.log" "$task_result/"
sha256sum \
    "$task_tmp/m351_replay/m351_m344_pattern_dma_correction_overlay_r1.json" \
    "$task_tmp/m339_replay/m339_q128_selective_pwp_kfirst_cycle_r1.json" \
    > "$task_result/exact_replay_sha256.txt"
sha256sum "$0" > "$task_result/runner_sha256.txt"

python3 - "$task_result/m354_m351_pattern_dma_independent_hammer_r1.json" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["status"] == "PASS_M354_INDEPENDENT_M351_HAMMER"
assert payload["score_0_to_100"] == 91
assert payload["verdict"]["p0_count"] == 0
assert payload["verdict"]["p1_count"] == 0
assert payload["verdict"]["p2_count"] == 3
assert len(payload["all_q_output_tile_port_matcher_rows"]) == 16
assert payload["capacity_recompute"]["fixed_physical_cache_plus_descriptor_bytes"] == 101536
assert payload["exact_sha_replay"]["m351_byte_identical"] is True
assert payload["exact_sha_replay"]["m339_byte_identical"] is True
assert payload["recurrence_boundary_audit"]["top_level_cycle_bound_false"] is True
PY

grep -Fq 'M351_PASS q128_o1_shared96_serial16_analytical=1.396902x physical=101536B cycle_admitted=false' \
    "$task_result/m351_exact_sha_replay.log" || exit 30
grep -Fq 'M339_PASS q16=1.540642x/ws16 q32=1.692877x/ws32 q64=1.857852x/ws64 q128=2.043940x/ws128' \
    "$task_result/m339_exact_sha_replay.log" || exit 31
grep -Fq 'M354_PASS score=91 p0=0 p1=0 p2=3 q128_shared_serial=1.396902x physical=101536B' \
    "$task_result/m354_analysis.log" || exit 32

printf 'PASS_M354_INDEPENDENT_M351_HAMMER\n' > "$task_result/RUN_COMPLETE.txt"
(
    cd "$task_result"
    find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
        -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M354 independent M351 hammer sealed at $task_result"
