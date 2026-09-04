#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
hw_root=$(cd -- "${script_dir}/../.." && pwd -P)

contract="${hw_root}/contracts/m710_decoder_temporal_delta_legal_tap_product_work_contract_r1_20260828.json"
analyzer="${hw_root}/system_simulator/analyze_m710_decoder_temporal_delta_legal_tap_work.py"
tests="${hw_root}/system_simulator/tests/test_m710_decoder_temporal_delta_legal_tap_work.py"
output="${hw_root}/results/m710_h67_decoder_temporal_delta_legal_tap_product_work_r1_20260828"

expected_contract_sha="9234a517c4fab185a4ae2d0a2b5bc76f41181125510ca35da03fbe0dda4e5132"
expected_analyzer_sha="526a36c367af915fdba4daaa8754cb33922fbe7dde327ee307d32464ddcb8296"
expected_tests_sha="897182961ea18486e79258746e03f13c5f10d1bbca514366bec7fc36f6ac8171"
expected_docs359_sha="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

[[ "$(sha256sum "${contract}" | awk '{print $1}')" == "${expected_contract_sha}" ]]
[[ "$(sha256sum "${analyzer}" | awk '{print $1}')" == "${expected_analyzer_sha}" ]]
[[ "$(sha256sum "${tests}" | awk '{print $1}')" == "${expected_tests_sha}" ]]
[[ "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" == "${expected_docs359_sha}" ]]

[[ ! -e "${output}" ]]
if compgen -G "${output}.staging.*" >/dev/null; then
  echo "M710_FAIL stale staging exists" >&2
  exit 20
fi

python3 -m py_compile "${analyzer}" "${tests}"
python3 -m unittest "${tests}"
python3 "${analyzer}" --contract "${contract}" --output-dir "${output}"

(
  cd -- "${output}"
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.seal.sha256
)

[[ "$(find "${output}" -maxdepth 1 -type f | wc -l)" -eq 8 ]]
[[ "$(find "${output}" -mindepth 1 -maxdepth 1 -type l | wc -l)" -eq 0 ]]
jq -e '
  .status == "PASS_CPU_PRODUCT_WORK_AUDIT__FRESH_REVIEW_REQUIRED" and
  .verdict == "KILL_N2_NO_RTL" and
  .fast_kill_gate.pass == false and
  .fast_kill_gate.all_four_modules_regress_above_one == true and
  .claim_boundary.product_work_regression == true and
  ([.claim_boundary.cycles, .claim_boundary.speedup,
    .claim_boundary.system_speedup, .claim_boundary.accuracy,
    .claim_boundary.numeric_bridge, .claim_boundary.rtl,
    .claim_boundary.vcs, .claim_boundary.eda, .claim_boundary.dc,
    .claim_boundary.formality, .claim_boundary.ptpx,
    .claim_boundary.energy, .claim_boundary.ppa,
    .claim_boundary.date_headline] | all(. == false))
' "${output}/summary.json" >/dev/null

[[ "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" == "${expected_docs359_sha}" ]]
echo "M710_RUNNER_PASS output=${output}"
