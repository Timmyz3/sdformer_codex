#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

analyzer="system_simulator/scripts/analyze_m453b_h67_hierarchical_q32x3_secondary_replay.py"
contract="contracts/m453b_h67_hierarchical_q32x3_secondary_replay_contract_final_r1_20260826.json"
micro_dir="results/m453b_final_freeze_micro_r1_20260826"
output_dir="results/m453b_h67_hierarchical_q32x3_secondary_replay_final_r1_20260826"

printf '%s  %s\n' \
  '84c3a2c79ad5926ba72a4727aac64b6ca29b7530a3120ada350d68d8ce12f6ca' \
  "${analyzer}" | sha256sum -c -
printf '%s  %s\n' \
  '3292e54ef0bf64b96b421de0d5b374a1552573bf29868e5f218fdaeac1bd2c4f' \
  "${contract}" | sha256sum -c -
printf '%s  %s\n' \
  '3e81ee1ae67f7c86c483a2673d8e184818115b203104fd2c22d9df768444d8c2' \
  "${micro_dir}/SHA256SUMS.seal.sha256" | sha256sum -c -
printf '%s  %s\n' \
  'dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4' \
  'docs/359_DATE终局冻结_20260813.md' | sha256sum -c -
(cd "${micro_dir}" && sha256sum -c SHA256SUMS.seal.sha256 && \
  sha256sum -c SHA256SUMS)
test "$(jq -r .status "${micro_dir}/m453b_final_freeze_micro_receipt_r1.json")" = \
  'PASS_M453B_FINAL_FREEZE_MICRO_M40_NOT_READ'
test ! -e "${output_dir}"

python3 "${analyzer}" --contract "${contract}" --output-dir "${output_dir}"

(cd "${output_dir}" && sha256sum -c SHA256SUMS.seal.sha256 && \
  sha256sum -c SHA256SUMS)
