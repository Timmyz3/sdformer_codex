#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${RESULT_DIR:-$ROOT/results/h67_mssb5_score_pair_screen_20260810}"
LIB="$ROOT/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib"
BUILD="$OUT/build"
LOGS="$OUT/logs"
mkdir -p "$BUILD" "$LOGS"
cd "$ROOT"

RTL=(
  rtl_h67/h67_motionxor_score_q7.sv
  rtl_h67/h67_direct_score_pair.sv
  rtl_h67/h67_balanced_popcount32.sv
  rtl_h67/h67_cse7_score_pair.sv
  rtl_h67/h67_ssr5_score_pair.sv
  rtl_h67/h67_mssb5_score_pair.sv
)

iverilog -g2012 -Wall -s tb_h67_mssb5_score_pair \
  -o "$BUILD/mssb5.vvp" "${RTL[@]}" tb_h67/tb_h67_mssb5_score_pair.sv \
  >"$LOGS/iverilog_build.log" 2>&1
vvp "$BUILD/mssb5.vvp" >"$LOGS/iverilog.log" 2>&1
grep -q '^PASS tb_h67_mssb5_score_pair vectors=20516 errors=0$' \
  "$LOGS/iverilog.log"

rm -rf "$BUILD/verilator"
verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-WIDTHTRUNC \
  -Wno-WIDTHEXPAND -Wno-DECLFILENAME -Wno-BLKSEQ \
  --top-module tb_h67_mssb5_score_pair --Mdir "$BUILD/verilator" \
  "${RTL[@]}" tb_h67/tb_h67_mssb5_score_pair.sv \
  >"$LOGS/verilator_build.log" 2>&1
"$BUILD/verilator/Vtb_h67_mssb5_score_pair" >"$LOGS/verilator.log" 2>&1
grep -q '^PASS tb_h67_mssb5_score_pair vectors=20516 errors=0$' \
  "$LOGS/verilator.log"

for top in h67_direct_score_pair h67_cse7_score_pair h67_ssr5_score_pair h67_mssb5_score_pair; do
  verilator --lint-only -Wall -Wno-fatal -Wno-UNUSEDSIGNAL \
    -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-DECLFILENAME \
    --top-module "$top" "${RTL[@]}" >"$LOGS/verilator_lint_${top}.log" 2>&1
  yosys -l "$LOGS/nangate45_${top}.log" -p "
    read_liberty -lib $LIB;
    read_verilog -sv ${RTL[*]};
    hierarchy -check -top $top;
    proc; flatten; opt; techmap; opt;
    abc -fast -liberty $LIB;
    clean; check -assert; stat -liberty $LIB;
    write_verilog -noattr $BUILD/${top}_mapped.v
  " >/dev/null
  {
    echo "read_liberty $LIB"
    echo "read_verilog $BUILD/${top}_mapped.v"
    echo "link_design $top"
    echo 'set_max_delay 5 -from [all_inputs] -to [all_outputs]'
    echo 'report_checks -from [all_inputs] -to [all_outputs] -path_delay max -format full -digits 6'
    echo 'exit'
  } | sta >"$LOGS/sta_${top}.log" 2>&1
  grep -q 'data arrival time' "$LOGS/sta_${top}.log"
done

python3 -m unittest tests.test_report_h67_mssb5_score_pair \
  >"$LOGS/python_unit_tests.log" 2>&1
python3 scripts/report_h67_mssb5_score_pair.py --output-dir "$OUT" \
  >"$LOGS/report_stdout.log" 2>&1
grep -q '"decision": "ADMIT_ROW_TOP_INTEGRATION"' "$LOGS/report_stdout.log"

sha256sum "${RTL[@]}" tb_h67/tb_h67_mssb5_score_pair.sv \
  scripts/report_h67_mssb5_score_pair.py \
  tests/test_report_h67_mssb5_score_pair.py \
  sim_h67/run_h67_mssb5_score_pair_checks.sh "$LIB" \
  >"$OUT/source_input_sha256.txt"

tar --sort=name --mtime='UTC 2026-08-10' --owner=0 --group=0 --numeric-owner \
  -cf - "${RTL[@]}" tb_h67/tb_h67_mssb5_score_pair.sv \
  scripts/report_h67_mssb5_score_pair.py \
  tests/test_report_h67_mssb5_score_pair.py \
  sim_h67/run_h67_mssb5_score_pair_checks.sh \
  | gzip -n >"$OUT/source_bundle.tar.gz"

sha256sum "$OUT"/logs/*.log "$OUT"/build/*_mapped.v \
  "$OUT"/report.json "$OUT"/report.md "$OUT"/source_input_sha256.txt \
  "$OUT"/source_bundle.tar.gz >"$OUT/result_sha256.txt"

python3 - "$OUT/complete.json" "$OUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[2]).resolve()
hashes = output / "result_sha256.txt"
value = {
    "schema": "h67_mssb5_score_pair_screen_complete_v1",
    "status": "ADMIT_ROW_TOP_INTEGRATION",
    "evidence": "[rtl]+[开放逻辑映射代理]+[开放网表STA代理]",
    "output_directory": str(output),
    "result_sha256_file_sha256": hashlib.sha256(hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
PY
sha256sum "$OUT/result_sha256.txt" "$OUT/complete.json" \
  >"$OUT/receipt_sha256.txt"

echo 'PASS Motion MSSB5 score-pair leaf screening'
