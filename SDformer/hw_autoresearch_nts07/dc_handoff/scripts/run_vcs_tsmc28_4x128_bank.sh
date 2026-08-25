#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/dc_handoff/scripts/run_vcs_tsmc28_4x128_bank.sh"
REPO_MANIFEST="$ROOT/dc_handoff/macro_manifests/tsmc28_128x128_1rw_20260822.json"
ASSET_DIR="${TSMC28_SRAM_ASSET_DIR:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/macro_assets/tsmc28_128x128_1rw_20260821}"
MODEL="${TSMC28_SRAM_MODEL:-$ASSET_DIR/ts1n28hpcphvtb128x128m4s_180a_ssg0p9v125c.v}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs/tsmc28_4x128_bank_vcs_20260822}"

command -v vcs >/dev/null
test -s "$MODEL"
test -s "$ASSET_DIR/SHA256SUMS"
test -s "$REPO_MANIFEST"
mkdir -p "$OUTPUT_DIR"
vcs -full64 -sverilog -timescale=1ns/1ps +define+UNIT_DELAY \
  -top tb_qfit_sync_1rw_acc_bank_tsmc28_4x128 \
  -o "$OUTPUT_DIR/simv" \
  "$MODEL" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$ROOT/tb_qfit/tb_qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  -l "$OUTPUT_DIR/compile.log"
"$OUTPUT_DIR/simv" -l "$OUTPUT_DIR/sim.log"
rg -q 'PASS_TSMC28_4X128_LOGICAL_128X512_BANK' "$OUTPUT_DIR/sim.log"
git_head="$(git -C "$ROOT" rev-parse HEAD)"
python3 - "$OUTPUT_DIR/run_manifest.json" "$git_head" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema": "qfit_tsmc28_4x128_bank_vcs_run_v1",
    "status": "PASS_VENDOR_FUNCTIONAL_MODEL",
    "git_head_at_run": sys.argv[2],
    "logical_bank": {"depth": 128, "width_bits": 512},
    "native_macro_instances": 4,
    "test_scope": {
        "all_address_deterministic_write_read": True,
        "mixed_transactions": 512,
        "disabled_output_hold": True,
        "write_cycle_output_hold": True,
        "same_address_read_write_read": True,
        "illegal_control_fail_noisy": True,
    },
    "claim_boundary": (
        "Single-bank VCS vendor functional-model evidence with UNIT_DELAY. "
        "Not full-core integration, timing-aware simulation, STA, PPA, or FPS."
    ),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
sha256sum \
  "$MODEL" \
  "$ASSET_DIR/SHA256SUMS" \
  "$RUNNER" "$REPO_MANIFEST" \
  "$ROOT/rtl_qfit/qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$ROOT/tb_qfit/tb_qfit_sync_1rw_acc_bank_tsmc28_4x128.sv" \
  "$OUTPUT_DIR/run_manifest.json" \
  "$OUTPUT_DIR/compile.log" "$OUTPUT_DIR/sim.log" \
  > "$OUTPUT_DIR/evidence.sha256"
sha256sum -c "$OUTPUT_DIR/evidence.sha256"
