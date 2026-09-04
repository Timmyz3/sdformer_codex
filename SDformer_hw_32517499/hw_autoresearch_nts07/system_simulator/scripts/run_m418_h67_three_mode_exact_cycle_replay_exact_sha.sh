#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
runner_path="${script_dir}/$(basename "${BASH_SOURCE[0]}")"
cd "${hw_root}"

contract="contracts/m418_h67_three_mode_exact_cycle_replay_contract_r1_20260826.json"
analyzer="system_simulator/scripts/analyze_m418_h67_three_mode_exact_cycle_replay.py"
output_dir="${1:-results/m418_h67_three_mode_exact_cycle_replay_r1_20260826}"
expected_contract="7b54b715d5c0b3af6c27a48ff8c1ade234adb3f822c8ca2169ff3131fb6c1e34"
expected_analyzer="4496b6477b2270698144ed3770d9df60b2b7879ac10bfef21c8a8da1d704d4a2"

expect_sha() {
    local path="$1"
    local expected="$2"
    local observed
    test -f "${path}" || { echo "M418 missing input: ${path}" >&2; exit 20; }
    observed="$(sha256sum "${path}" | awk '{print $1}')"
    test "${observed}" = "${expected}" || {
        echo "M418 exact-SHA mismatch: ${path} expected=${expected} observed=${observed}" >&2
        exit 21
    }
}

expect_sha "${contract}" "${expected_contract}"
expect_sha "${analyzer}" "${expected_analyzer}"
test ! -e "${output_dir}" || {
    echo "M418 refusing overwrite: ${output_dir}" >&2
    exit 22
}

prereq_log="$(mktemp "${TMPDIR:-/tmp}/m418_prereq.XXXXXX")"
run_log="$(mktemp "${TMPDIR:-/tmp}/m418_run.XXXXXX")"
cleanup() {
    rm -f "${prereq_log}" "${run_log}"
}
trap cleanup EXIT

{
    (cd "${hw_root}/.." &&
        sha256sum -c hw_autoresearch_nts07/results/m397_h67_fixed_product_qo_finite_dse_r1_20260826/SHA256SUMS &&
        sha256sum -c hw_autoresearch_nts07/results/m397_h67_fixed_product_qo_finite_dse_r1_20260826/SHA256SUMS.seal.sha256)
    (cd "${hw_root}/.." &&
        sha256sum -c hw_autoresearch_nts07/results/m401_h67_q32_elastic_pwp_full_replay_r1_20260826/SHA256SUMS &&
        sha256sum -c hw_autoresearch_nts07/results/m401_h67_q32_elastic_pwp_full_replay_r1_20260826/SHA256SUMS.seal.sha256)
    (cd "${hw_root}/.." &&
        sha256sum -c hw_autoresearch_nts07/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS &&
        sha256sum -c hw_autoresearch_nts07/results/m410r2_h67_q32_full_runtime_vcs_stimulus_r2_20260826/SHA256SUMS.seal.sha256)
} >"${prereq_log}" 2>&1

if ! python3 "${analyzer}" --contract "${contract}" \
        --output-dir "${output_dir}" >"${run_log}" 2>&1; then
    cat "${run_log}" >&2
    exit 23
fi
mv "${prereq_log}" "${output_dir}/prerequisite_double_seal_check.log"
mv "${run_log}" "${output_dir}/run.log"
cp "${contract}" "${output_dir}/contract.json"

{
    sha256sum "${contract}"
    sha256sum "${analyzer}"
    sha256sum "${runner_path}"
    python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
contract = json.loads((root / "contracts/m418_h67_three_mode_exact_cycle_replay_contract_r1_20260826.json").read_text())
for name, identity in sorted(contract["inputs"].items()):
    path = root / identity["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != identity["sha256"]:
        raise SystemExit("post-run M418 SHA drift: " + name)
    print(digest + "  " + identity["path"])
PY
} >"${output_dir}/input_sha256.txt"

python3 - "${output_dir}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
result = json.loads((out / "m418_h67_three_mode_exact_cycle_replay_r1.json").read_text())
if result["status"] != "PASS_M418_FORMAL_THREE_MODE_EXECUTABLE_REPLAY":
    raise SystemExit("M418 status gate failed")
expected = {
    "dense16_same_resource": 6636544610,
    "zero_elided_bit_sparse_exact_reproduction": 742148386,
    "m401_combined_exact_reproduction": 641790704,
}
for variant, cycles in expected.items():
    if result["variants"][variant]["cycles"] != cycles:
        raise SystemExit("M418 cycle gate failed: " + variant)
if result["execution_gates"]["phase_records"] != 51840:
    raise SystemExit("M418 phase-record gate failed")
for filename in (
    "dense16_per_phase_timestamps_components.csv",
    "zero_elided_per_phase_timestamps_components.csv",
    "m401_combined_per_phase_timestamps_components.csv",
):
    with (out / filename).open(newline="") as handle:
        if sum(1 for _ in csv.DictReader(handle)) != 17280:
            raise SystemExit("M418 per-phase CSV extent gate failed: " + filename)
PY

(
    cd "${output_dir}"
    find . -maxdepth 1 -type f \
        ! -name SHA256SUMS \
        ! -name SHA256SUMS.seal.sha256 \
        -printf '%P\n' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
    sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
    sha256sum -c SHA256SUMS.seal.sha256
)

sha256sum docs/359_DATE终局冻结_20260813.md | \
    grep -q '^dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4 '
echo "M418_RUN_COMPLETE output=${output_dir}"
