#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m467r2_conv3x3_execution_island_vcs_r1b_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M467R2 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m467r2_conv3x3_execution_island_vcs_contract_r1_20260826.json aaef7e2581c50497122de0e4e1fd5f576912b44b53b1e5a0153cd94753038453
check_sha rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
check_sha rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
check_sha rtl_m467/m467_conv3x3_execution_island.sv 4f32a49fc06a26ca9aed0a9af5b37818340b0ce87cd8b3da34a5ba96f268b56f
check_sha verif_m467/m467_conv3x3_execution_island_assertions.sv d02939289e5823b890785e4e84d692b50b0f67fdfb0a166aa1627a4d7e15c8db
check_sha tb_m467/tb_m467_conv3x3_execution_island.sv 9edfe1ccbe447070d8d30dc68a09a85fb2dbaeee086d650963864c0d79a76f4c
check_sha dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f 5a7c1e63f5e08fafb57850ac60dda15c8f1f56ad450e6a103f621e0bd48265e2
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

test ! -e "$RUN_DIR" || { echo "M467R2 result directory exists: $RUN_DIR" >&2; exit 3; }
mkdir -p "$RUN_DIR"
cp contracts/m467r2_conv3x3_execution_island_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m467r2_conv3x3_execution_island_vcs_contract_r1_20260826.json \
  rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
  rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
  rtl_m467/m467_conv3x3_execution_island.sv \
  verif_m467/m467_conv3x3_execution_island_assertions.sv \
  tb_m467/tb_m467_conv3x3_execution_island.sv \
  dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f \
  docs/359_DATE终局冻结_20260813.md > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m467_conv3x3_execution_island \
  -f dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"
grep -q '^PASS M467R2 directed ' "$RUN_DIR/sim.log"
! grep -q 'Error-\|Assertion failed\|Fatal:' "$RUN_DIR/sim.log"

echo PASS_M467R2_ROW_ADDRESSED_OPERATOR_PERSISTENT_CONV3X3_ISLAND_SYNOPSYS_VCS > "$RUN_DIR/RUN_COMPLETE.txt"
python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
receipt = {
    "milestone": "M467R2_ROW_ADDRESSED_OPERATOR_PERSISTENT_CONV3X3_ISLAND",
    "status": "PASS_SYNOPSYS_VCS_DIRECTED_R2",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "production_geometry": "8 block banks x depth 3000 x 96 x signed19",
    "directed_geometry": "ROWS_PER_PHASE=2, three accumulated phases plus reset-recovery phase",
    "counts": {"phases": 4, "rows": 8, "active_rows": 5,
               "descriptor_writes": 5, "descriptor_reads": 10,
               "pwp_requests": 24, "weight_requests": 80,
               "plus_requests": 24, "minus_requests": 24,
               "accumulator_reads_including_commit_scan": 48,
               "accumulator_writes": 40, "same_address_forward_hits": 8,
               "operator_commit_vectors": 16, "protocol_attacks": 1},
    "coverage": {"tiles": 2, "blocks": 4, "commit_exact": True,
                 "same_phase_distinct_rows": True,
                 "cross_phase_sram_old_psum": True,
                 "nonlast_phase_commit_zero": True, "reset_recovery": True,
                 "stalls": True, "illegal_protocol_fail_closed": True},
    "memory_boundary": "behavioral descriptor/payload/accumulator memories; external block-bank SP-style read/write port cut; no macro PPA",
    "claim_boundary": {"m430_absolute_timestamps": False,
                       "rtl_measured_517m": False, "system_speedup": False,
                       "paper_ppa": False, "macro_ppa": False,
                       "energy": False, "headline": False},
}
(r / "m467r2_vcs_receipt_r2.json").write_text(json.dumps(receipt, indent=2)+"\n")
files = [p for p in sorted(r.iterdir()) if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
(r / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
(r / "SHA256SUMS.seal.sha256").write_text(f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY
