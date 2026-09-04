#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BUILD_DIR:-${ROOT}/build_new_arch/local5_qsilent_order}"
OUT="${RESULT_DIR:-${ROOT}/results/local5_qsilent_order_20260813}"
mkdir -p "${BUILD}" "${OUT}"

RTL=(
  "${ROOT}/rtl_local5/local5_shiftmax5_q17.sv"
  "${ROOT}/rtl_qfit/qfit_tagged_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_xorbank_compactor4.sv"
  "${ROOT}/rtl_qfit/qfit_local5_score_leaf.sv"
  "${ROOT}/rtl_qfit/qfit_local5_qsilent_score_leaf.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_score_leaf_assertions.sv"
  "${ROOT}/verif_qfit/qfit_local5_qsilent_score_leaf_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_local5_qsilent_random_backpressure.sv"
OBJ="${BUILD}/obj"

verilator --binary --timing --assert -Wall -Wno-fatal \
  -Wno-PINMISSING -Wno-PINCONNECTEMPTY \
  --top-module tb_qfit_local5_qsilent_random_backpressure \
  --Mdir "${OBJ}" "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
  >"${OUT}/build.log" 2>&1

for seed in 1 7 29 113 509 2027 6553 16381; do
  "${OBJ}/Vtb_qfit_local5_qsilent_random_backpressure" "+SEED=${seed}" \
    | tee "${OUT}/seed_${seed}.log"
done

python3 - "${OUT}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pattern = re.compile(
    r"QSILENT_RANDOM_BP seed=(\d+) issued=(\d+) retired=(\d+) "
    r"stalls=(\d+) overlap=(\d+) cycles=(\d+)"
)
rows = []
for path in sorted(root.glob("seed_*.log")):
    text = path.read_text(encoding="utf-8")
    matches = pattern.findall(text)
    if len(matches) != 1 or text.count(
        "PASS tb_qfit_local5_qsilent_random_backpressure"
    ) != 1:
        raise SystemExit(f"fail-closed parse error: {path}")
    seed, issued, retired, stalls, overlap, cycles = map(int, matches[0])
    if issued != 512 or retired != 512 or stalls == 0 or overlap == 0:
        raise SystemExit(f"coverage contract failed: {path}")
    rows.append({
        "seed": seed,
        "issued": issued,
        "retired": retired,
        "stalled_outputs": stalls,
        "overlap_accepts": overlap,
        "cycles": cycles,
    })
if len(rows) != 8:
    raise SystemExit(f"expected 8 seeds, found {len(rows)}")
report = {
    "schema": "local5_qsilent_order_random_backpressure_v1",
    "status": "PASS",
    "evidence": "[rtl]+[sva]",
    "transactions": sum(row["issued"] for row in rows),
    "retire_mismatches": 0,
    "seeds": rows,
    "claim_boundary": "Leaf protocol verification; not a performance result.",
}
(root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
PY

sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" "${BASH_SOURCE[0]}" \
  >"${OUT}/source_sha256.txt"

echo "PASS Local5 Q-silent issue-order random-backpressure checks"
