#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

test "$(sha256sum neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1233_motion_final_checkpoint_unified_hardware_selection_interface_r2.py | awk '{print $1}')" = "1227b0746776aff1103937ba5557f325e97e5c8fa751a2593136ece9674f8462"
test "$(sha256sum hw_autoresearch_nts07/tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py | awk '{print $1}')" = "b63bbc1191f9ac972c589d3d75f3532a0de67f5d561fce179a7fa4996653063d"
test "$(sha256sum hw_autoresearch_nts07/contracts/m1233_motion_final_checkpoint_unified_capture_selection_interface_successor_source_contract_r1_20260830.json | awk '{print $1}')" = "835c1f04b57a743de4d90ba233e14b3453cd18da6b5c439f141f781092d7631c"
test "$(sha256sum neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1227_motion_final_checkpoint_unified_hardware_r1.py | awk '{print $1}')" = "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b"
test "$(sha256sum hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md | awk '{print $1}')" = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

/opt/anaconda3/envs/python310/bin/python3.10 -m json.tool \
  hw_autoresearch_nts07/contracts/m1233_motion_final_checkpoint_unified_capture_selection_interface_successor_source_contract_r1_20260830.json >/dev/null
/opt/anaconda3/envs/python310/bin/python3.10 -m py_compile \
  neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1233_motion_final_checkpoint_unified_hardware_selection_interface_r2.py \
  hw_autoresearch_nts07/tests/test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source.py
/opt/anaconda3/envs/python310/bin/python3.10 -m unittest -q \
  hw_autoresearch_nts07.tests.test_m1233_motion_final_checkpoint_unified_capture_selection_interface_source

test ! -e hw_autoresearch_nts07/results/m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830
test ! -e hw_autoresearch_nts07/results/.m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.attempt_consumed
test ! -e hw_autoresearch_nts07/results/.m1233_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.production.log

echo PASS_M1233_MECHANICAL_CHECKS__SOURCE_ONLY__16_TESTS__NAMESPACES_FRESH
