#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
contract="${hw_root}/contracts/m423a_trainonly_cycle_aware_q32_catalog_contract_r1_20260826.json"
builder="${hw_root}/system_simulator/scripts/build_m423_trainonly_cycle_aware_q32_catalog.py"
run_dir="${hw_root}/results/m423a_trainonly_cycle_aware_q32_catalog_r1_20260826"

test "$(sha256sum "${contract}" | awk '{print $1}')" = "734c39aa54a80e5324b463d16a93484bc964b3165925cc954cfd0547e9761729"
test "$(sha256sum "${builder}" | awk '{print $1}')" = "8f71e5972765365635ab6562df0da1115e38bb2bb01dadca6b7e88b6f02e96eb"
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
test ! -e "${run_dir}"

PYTHONDONTWRITEBYTECODE=1 python3 "${builder}" \
  --contract "${contract}" \
  --output-dir "${run_dir}"

cp "${contract}" "${run_dir}/contract.json"
(
  cd "${run_dir}"
  find . -maxdepth 1 -type f \
    ! -name 'SHA256SUMS' ! -name 'SHA256SUMS.seal.sha256' \
    -printf '%f\n' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
echo "M423A_SEALED ${run_dir}"
