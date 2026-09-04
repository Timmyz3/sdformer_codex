#!/usr/bin/env bash
set -euo pipefail

m361r2_hw="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
m361r2_output="${M361R2_OUTPUT_DIR:-${m361r2_hw}/results/m361r2_wide_partition_exact_pattern_dse_replay_r1_20260825}"
m361r2_contract="contracts/m361r2_wide_partition_exact_pattern_dse_contract_r1_20260825.json"
m361r2_script="system_simulator/scripts/analyze_m361r2_wide_partition_exact_pattern_dse.py"

m361r2_expect() {
    local path=$1
    local expected=$2
    [[ -f "${path}" ]] || exit 3
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]] || exit 3
}

[[ ! -e "${m361r2_output}" ]] || exit 5
cd "${m361r2_hw}"
m361r2_expect "${m361r2_script}" c8a264bb5ce94bed0c71b98ca9bd9e070c79621dbf17e694ac52f108c7b0ff7a
m361r2_expect "${m361r2_contract}" fdffb634c3026c8b724a93c408f2b125d48f0605088835bf29d95ddf854dd881
m361r2_expect "contracts/m361_wide_partition_exact_pattern_dse_contract_r1_20260825.json" fbcc336a7437f972107a7099190aaea644f73d9d665f1c1e77daded1b04a72ae
m361r2_expect "system_simulator/scripts/analyze_m43_tile_resident_parent_delta_schedule.py" a4ddebf4687b32c65735c591a6526f43b7274777ace4e3ca90d19a2d04adb1c3
m361r2_expect "system_handoff/incoming/m73_h67_ep35_train_calibration_sources_s32_r1_20260823/m73_train_calibration_source_manifest.json" 3fb3468066fe1f7d61f5e39398cb2f8655643080f03e5b1deb58ef2911db17e2
m361r2_expect "system_handoff/incoming/m248_paft_ep4_running_bn_bottleneck_sources_s10_r1_20260825/m248_paft_ep4_running_bn_bottleneck_source_manifest.json" 6ba74414093edc1bf7d165b8904d8ac68bfdcdb3a49151203932e5c3aea92b0b
m361r2_expect "results/m338_trainonly_nested_q128_catalog_r1_20260825/m338_trainonly_nested_q128_catalog_r1.json" b7c9e19166d3abfbe696df74dcfb99ef65607d209b18ca51b8456f15bcb6c2b1
m361r2_expect "docs/359_DATE终局冻结_20260813.md" dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

python3 "${m361r2_script}" --contract "${m361r2_contract}" \
    --output-dir "${m361r2_output}" | tee "${m361r2_output}.log"
python3 - "${m361r2_output}/m361r2_wide_partition_exact_pattern_dse_r1.json" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
assert payload["status"] == "PASS_M361R2_TRAIN_ONLY_K32_K64_CATALOG_DISJOINT_S10_EXACT_WORK_NO_CYCLES"
assert payload["admission"]["cycle_speedup"] is False
assert payload["admission"]["system_speedup"] is False
assert payload["admission"]["date_headline"] is False
PY
sha256sum "${m361r2_output}/m361r2_wide_partition_exact_pattern_dse_r1.json" \
    >"${m361r2_output}/SHA256SUMS"
sha256sum "${m361r2_output}/SHA256SUMS" \
    >"${m361r2_output}/SHA256SUMS.seal.sha256"
echo "PASS_M361R2_EXACT_SHA_REPLAY output=${m361r2_output}"
