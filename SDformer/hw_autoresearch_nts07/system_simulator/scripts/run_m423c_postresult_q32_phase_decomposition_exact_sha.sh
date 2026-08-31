#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
hw_root="$(cd "${script_dir}/../.." && pwd)"
contract="${hw_root}/contracts/m423c_postresult_q32_phase_decomposition_contract_r1_20260826.json"
deriver="${hw_root}/system_simulator/scripts/derive_m423c_postresult_q32_phase_decomposition.py"
run_dir="${hw_root}/results/m423c_postresult_q32_phase_decomposition_r1_20260826"

test "$(sha256sum "${contract}" | awk '{print $1}')" = "9456b08cfb66c430cbf886d300a8fafe5fda2929fd1bb7a86c1b6a8c6df51204"
test "$(sha256sum "${deriver}" | awk '{print $1}')" = "02b513c6f02a455d3384d62d0042a2bbfc675c65cf5d13c643faa8234f0582f9"
test "$(sha256sum "${hw_root}/results/m423b_recovery_h67_cycle_aware_q32_heldout_r2_20260826/SHA256SUMS.seal.sha256" | awk '{print $1}')" = "1a7685f4561f83cbc82b3aad2b717d9b860946689a2411ed4ef1d3e3fd914cff"
test "$(sha256sum "${hw_root}/docs/359_DATE终局冻结_20260813.md" | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
test ! -e "${run_dir}"

PYTHONDONTWRITEBYTECODE=1 python3 "${deriver}" \
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
echo "M423C_SEALED ${run_dir}"
