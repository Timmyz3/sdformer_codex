#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
contract="${hw_root}/contracts/m423b_h67_cycle_aware_q32_heldout_once_contract_r1_20260826.json"
analyzer="${hw_root}/system_simulator/scripts/analyze_m423b_h67_cycle_aware_q32_heldout_once.py"
run_dir="${hw_root}/results/m423b_h67_cycle_aware_q32_heldout_once_r1_20260826"
consumed="${hw_root}/results/M423B_M40_ONE_SHOT_CONSUMED_20260826.marker"

test "$(sha256sum "${contract}" | awk '{print $1}')" = "d43151f7cf4a42ea14c2db3aa49762b4b18ec97b432db256fc25d3b90806d57b"
test "$(sha256sum "${analyzer}" | awk '{print $1}')" = "993111ed28a6fd1e5a80378d92e5eb519fe9561ba626dbbd81e20a1f19e814cf"
test "$(sha256sum "${hw_root}/results/m423a_trainonly_cycle_aware_q32_catalog_r1_20260826/SHA256SUMS.seal.sha256" | awk '{print $1}')" = "718d6707841175ae50c2e90f762b43aaaa215c35dce8b18230a34c2aaef2c6c3"
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
test ! -e "${run_dir}"
test ! -e "${consumed}"

printf '%s\n' \
  'M423B M40 one-shot held-out execution consumed.' \
  'The sealed M423a catalog must not be tuned or replayed against M40 again.' \
  'system_speedup=false; scope=four_H67_bottleneck_Conv_only.' > "${consumed}"

PYTHONDONTWRITEBYTECODE=1 python3 "${analyzer}" \
  --contract "${contract}" \
  --output-dir "${run_dir}"

cp "${contract}" "${run_dir}/contract.json"
cp "${consumed}" "${run_dir}/M423B_M40_ONE_SHOT_CONSUMED.marker"
(
  cd "${run_dir}"
  find . -maxdepth 1 -type f \
    ! -name 'SHA256SUMS' ! -name 'SHA256SUMS.seal.sha256' \
    -printf '%f\n' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
echo "M423B_ONE_SHOT_SEALED ${run_dir}"
