#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_exact_metadata_cascade_profile_20260809}"
PROFILE="$ROOT/results/h67_fullres_ep30_t450_profile100_20260805/nts11_hardware_p0_profile.md"
SAMPLES="$ROOT/results/h67_fullres_ep30_t450_profile100_20260805/sample_workload.csv"
VECTORS="$ROOT/tb_h67/vectors/h67_ep30_fullres_t450_all12_20260805/h67_checkpoint_rows.txt"
RQTB="$ROOT/results/h67_rqtb_strong_baseline_2s_t450_20260809/report.json"

mkdir -p "$OUT"
rm -f "$OUT/status.tsv" "$OUT/source_hashes.sha256" "$OUT/source_hash_check.log"

cd "$ROOT"
PYTHONPATH=. python3 -m unittest \
  tests.test_profile_h67_exact_metadata_cascade \
  >"$OUT/unittest.log" 2>&1

python3 scripts/profile_h67_exact_metadata_cascade.py \
  --profile-md "$PROFILE" \
  --sample-csv "$SAMPLES" \
  --vectors "$VECTORS" \
  --rqtb-report "$RQTB" \
  --output-dir "$OUT" \
  --selected-bundle 8 \
  >"$OUT/profile.log" 2>&1

python3 - "$OUT/report.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("报告状态不是 PASS")
if not report.get("admission", {}).get("pass"):
    raise SystemExit("架构准入未通过")
if report["sample0_exact"]["zero_k_score_classes"] != [0, 1, 2]:
    raise SystemExit("zero-K score class 未闭合为 [0,1,2]")
if report["cycle_model"]["zk_decoupled_cycle_reduction"] < 0.10:
    raise SystemExit("解耦周期模型未达到 10% 门槛")
PY

sha256sum \
  "$ROOT/scripts/profile_h67_exact_metadata_cascade.py" \
  "$ROOT/tests/test_profile_h67_exact_metadata_cascade.py" \
  "$ROOT/sim_h67/run_h67_exact_metadata_cascade_profile.sh" \
  >"$OUT/source_hashes.sha256"
sha256sum -c "$OUT/source_hashes.sha256" >"$OUT/source_hash_check.log"

printf '%s\t%s\n' \
  "Python 单元测试" "PASS" \
  "profile100/sample0/RQTB 架构准入" "PASS" \
  "源码 SHA-256 自校验" "PASS" \
  >"$OUT/status.tsv"

echo "PASS H67 exact metadata cascade architecture admission"
