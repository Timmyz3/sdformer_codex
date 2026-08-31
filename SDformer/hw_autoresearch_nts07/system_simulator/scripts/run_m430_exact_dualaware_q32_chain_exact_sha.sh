#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${HW_ROOT}"

EXPECTED_DOCS359="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_TRAIN_SCRIPT="6b9c94f1da3a7588561b5b20ac6693481d80fb06ee6a0c39bd1eaf024b24c881"
EXPECTED_HELDOUT_SCRIPT="af9da7d8cb3610bc547347d425452be68c6738aa86a8aa76a9efc554447f4377"
EXPECTED_TRAIN_CONTRACT="46325aec003fd68f2988fe3bfe22c988714fe57c7f71055c4214ee730f897c07"
EXPECTED_HELDOUT_CONTRACT="261cb8fc3fec3d08570f55423da71188b3b8c17b5537f695309075d16f72c912"

test "$(sha256sum docs/359_DATE终局冻结_20260813.md | awk '{print $1}')" = "${EXPECTED_DOCS359}"
test "$(sha256sum system_simulator/scripts/build_m430_trainonly_dualaware_q32_catalog.py | awk '{print $1}')" = "${EXPECTED_TRAIN_SCRIPT}"
test "$(sha256sum system_simulator/scripts/analyze_m430b_h67_dualaware_q32_heldout_once.py | awk '{print $1}')" = "${EXPECTED_HELDOUT_SCRIPT}"
test "$(sha256sum contracts/m430a_trainonly_dualaware_q32_catalog_contract_r1_20260826.json | awk '{print $1}')" = "${EXPECTED_TRAIN_CONTRACT}"
test "$(sha256sum contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json | awk '{print $1}')" = "${EXPECTED_HELDOUT_CONTRACT}"

python3 -m py_compile \
  system_simulator/scripts/build_m430_trainonly_dualaware_q32_catalog.py \
  system_simulator/scripts/analyze_m430b_h67_dualaware_q32_heldout_once.py

python3 system_simulator/scripts/build_m430_trainonly_dualaware_q32_catalog.py \
  --contract contracts/m430a_trainonly_dualaware_q32_catalog_contract_r1_20260826.json \
  --output-dir results/m430a_trainonly_dualaware_q32_catalog_r1_20260826

(
  cd results/m430a_trainonly_dualaware_q32_catalog_r1_20260826
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.seal.sha256
)

python3 system_simulator/scripts/analyze_m430b_h67_dualaware_q32_heldout_once.py \
  --contract contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json \
  --output-dir results/m430b_h67_dualaware_q32_heldout_once_r1_20260826

(
  cd results/m430b_h67_dualaware_q32_heldout_once_r1_20260826
  sha256sum -c SHA256SUMS
  sha256sum -c SHA256SUMS.seal.sha256
)

