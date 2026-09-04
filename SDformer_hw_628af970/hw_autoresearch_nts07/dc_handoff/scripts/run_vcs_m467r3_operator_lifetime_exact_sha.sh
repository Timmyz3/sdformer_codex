#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m467r3_operator_lifetime_zero_semantics_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M467R3 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m467r3_operator_lifetime_zero_semantics_vcs_contract_r1_20260826.json 4f526bb81a635b5ec6536e6b4d7885577c6dfb64ff2776011445a6068fab980f
check_sha contracts/m467r2_revoked_by_m467r3_operator_lifetime_p0_20260826.json fa0bdbe562327a7b6358b862628c5b92cd7437331c4302607ddb3ddc829a70cf
check_sha rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
check_sha rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
check_sha rtl_m467/m467_conv3x3_execution_island.sv 44516ef411fac080a8a637e1b6bc67129bc209f77d5fc4f0645e1064aaceee87
check_sha verif_m467/m467_conv3x3_execution_island_assertions.sv 54b44e5d7c677f3cb1811359176844e213d225554365da05317ecba8367f8e0c
check_sha tb_m467/tb_m467_conv3x3_execution_island.sv 2cbc0ffc0a6336355afaccf0f3733e61d9917ede2cc163ec5f7370cab50ba33a
check_sha dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f 5a7c1e63f5e08fafb57850ac60dda15c8f1f56ad450e6a103f621e0bd48265e2
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

test ! -e "$RUN_DIR" || { echo "M467R3 result directory exists: $RUN_DIR" >&2; exit 3; }
mkdir -p "$RUN_DIR"
cp contracts/m467r3_operator_lifetime_zero_semantics_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
cp contracts/m467r2_revoked_by_m467r3_operator_lifetime_p0_20260826.json "$RUN_DIR/r2_revocation.json"
sha256sum \
  contracts/m467r3_operator_lifetime_zero_semantics_vcs_contract_r1_20260826.json \
  contracts/m467r2_revoked_by_m467r3_operator_lifetime_p0_20260826.json \
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
grep -q '^PASS M467R3 directed ' "$RUN_DIR/sim.log"
! grep -q 'Error-\|Assertion failed\|Fatal:' "$RUN_DIR/sim.log"

echo PASS_M467R3_OPERATOR_LIFETIME_ZERO_SEMANTICS_SYNOPSYS_VCS > "$RUN_DIR/RUN_COMPLETE.txt"
python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
receipt = {
    "milestone": "M467R3_OPERATOR_LIFETIME_ZERO_SEMANTICS",
    "status": "PASS_EXACT_SHA_SYNOPSYS_VCS",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "production_geometry": "8 block banks x depth 3000 x 96 x signed19 plus 1 live bit/vector",
    "directed_geometry": "ROWS_PER_PHASE=2, two back-to-back nonzero operators without reset, then fault/reset recovery",
    "counts": {"operators": 2, "phases_before_fault_reset": 4, "accepted_rows_before_fault_reset": 8,
               "active_rows": 6, "descriptor_writes": 6, "descriptor_reads": 12,
               "pwp_requests": 32, "weight_requests": 96,
               "plus_requests": 32, "minus_requests": 32,
               "accumulator_reads": 40, "accumulator_writes": 48,
               "same_address_forward_hits": 8, "zero_initializations": 24,
               "zero_commits": 8, "operator_commit_vectors": 32,
               "protocol_attacks": 1},
    "coverage": {"same_phase_distinct_rows": True, "cross_phase_sram_old_psum": True,
                 "same_operator_forwarding": True, "no_reset_between_operators": True,
                 "stale_sram_read_suppressed": True, "stale_forward_suppressed": True,
                 "untouched_second_operator_row_commits_zero": True,
                 "nonlast_phase_commit_zero": True, "commit_exact": True,
                 "pwp_fallback_plus_minus": True, "stalls": True,
                 "reset_recovery": True, "illegal_protocol_fail_closed": True},
    "memory_boundary": "external 1824-bit accumulator data is deliberately poisoned and never physically cleared; one-bit live sidecar supplies exact zero semantics",
    "claim_boundary": {"performance": False, "ppa": False, "macro_ppa": False,
                       "system_or_full_network": False, "energy": False, "headline": False}
}
(r / "m467r3_vcs_receipt_r1.json").write_text(json.dumps(receipt, indent=2)+"\n")
files = [p for p in sorted(r.iterdir()) if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
(r / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
(r / "SHA256SUMS.seal.sha256").write_text(f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY
