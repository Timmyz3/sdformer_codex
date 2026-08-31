#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
contract="${hw_root}/contracts/m423b_recovery_h67_cycle_aware_q32_heldout_contract_r2_20260826.json"
analyzer="${hw_root}/system_simulator/scripts/analyze_m423b_h67_cycle_aware_q32_heldout_once.py"
run_dir="${hw_root}/results/m423b_recovery_h67_cycle_aware_q32_heldout_r2_20260826"
recovery_marker="${hw_root}/results/M423B_RECOVERY_COMPLETED_EVALUATION_CONSUMED_20260826.marker"

test "$(sha256sum "${contract}" | awk '{print $1}')" = "e4dc6a591a909496b608a7437dabd5678e8667f44ea0464fbf98cfb4c2314d8e"
test "$(sha256sum "${analyzer}" | awk '{print $1}')" = "d776819f74b2cb8d21eed706dd8cd5f5adb01434bfec46c1e3cdddf3253844ee"
test "$(sha256sum "${hw_root}/results/m423a_trainonly_cycle_aware_q32_catalog_r1_20260826/m423_trainonly_cycle_aware_q32_catalog_r1.json" | awk '{print $1}')" = "edfa5ee560a52380c5d22e841af1ed8e6e8241a7e7f021ffdb74735067ab65b4"
test "$(sha256sum "${hw_root}/results/m423b_h67_cycle_aware_q32_heldout_once_r1_20260826/SHA256SUMS.seal.sha256" | awk '{print $1}')" = "f441abf6e4b5ab96319b25366d3b8df2a1e184b564844508bcd018fae4966e4e"
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
test ! -e "${run_dir}"
test ! -e "${recovery_marker}"

printf '%s\n' \
  'M423B recovery completed-evaluation allowance consumed.' \
  'r1 read all payloads but completed zero phase rows; recovery may complete one evaluation.' \
  'No post-recovery M40 replay or catalog tuning is allowed.' > "${recovery_marker}"

PYTHONDONTWRITEBYTECODE=1 python3 "${analyzer}" \
  --contract "${contract}" \
  --output-dir "${run_dir}"

cp "${contract}" "${run_dir}/contract.json"
cp "${recovery_marker}" "${run_dir}/M423B_RECOVERY_COMPLETED_EVALUATION_CONSUMED.marker"
(
  cd "${run_dir}"
  find . -maxdepth 1 -type f \
    ! -name 'SHA256SUMS' ! -name 'SHA256SUMS.seal.sha256' \
    -printf '%f\n' | LC_ALL=C sort | xargs sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
)
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
echo "M423B_RECOVERY_SEALED ${run_dir}"
