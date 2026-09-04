#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m467r4_row_shared_live_scoreboard_vcs_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M467R4 exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m467r4_row_shared_live_invariant_vcs_contract_r1_20260826.json ea6cc8169a84692190b19a9f32bd69c7897dbe96e5620734ac2221732957b3b6
check_sha rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
check_sha rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
check_sha rtl_m467/m467_conv3x3_execution_island.sv 703a94516b49456f407c1ad4d849493f0db11819d3167a563534cfe1eb4baca0
check_sha verif_m467/m467_conv3x3_execution_island_assertions.sv 1ebc2364128998fe8871ce9cdb9d4713a08ce80cab16f76cf5c956019cdc35c9
check_sha tb_m467/tb_m467_conv3x3_execution_island.sv abdb676c9b6d1ccbbc744186a8c58016118a9257f2d4aad6496e3f801d8896e8
check_sha dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f 5a7c1e63f5e08fafb57850ac60dda15c8f1f56ad450e6a103f621e0bd48265e2
check_sha results/m467r4_row_live_premature_mutation_attack_r1_20260826/m467r4_mutation_attack.json 385d6b1ee5eaaea9df4d55ea0becd7823eea94969eeee77f513a3f899c13ea37
check_sha results/m467r4_row_live_premature_mutation_attack_r1_20260826/SHA256SUMS 969c0a6a1936b25f14c11a07a9075ee1f0e67301dd6f303d4b996cb38b52e609
check_sha results/m467r4_row_live_premature_mutation_attack_r1_20260826/SHA256SUMS.seal.sha256 2936fa1d797c205267946453aa4d37746f452206b29f8361fcfc808e8dc18c49
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

test ! -e "$RUN_DIR" || { echo "M467R4 result directory exists: $RUN_DIR" >&2; exit 3; }
mkdir -p "$RUN_DIR"
cp contracts/m467r4_row_shared_live_invariant_vcs_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m467r4_row_shared_live_invariant_vcs_contract_r1_20260826.json \
  rtl_m414/m414_q32_balanced16_zero_stop_controller.sv \
  rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv \
  rtl_m467/m467_conv3x3_execution_island.sv \
  verif_m467/m467_conv3x3_execution_island_assertions.sv \
  tb_m467/tb_m467_conv3x3_execution_island.sv \
  dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f \
  results/m467r4_row_live_premature_mutation_attack_r1_20260826/m467r4_mutation_attack.json \
  results/m467r4_row_live_premature_mutation_attack_r1_20260826/SHA256SUMS \
  results/m467r4_row_live_premature_mutation_attack_r1_20260826/SHA256SUMS.seal.sha256 \
  docs/359_DATE终局冻结_20260813.md > "$RUN_DIR/input_sha256.txt"

vcs -full64 -sverilog -assert svaext -timescale=1ns/1ps \
  -top tb_m467_conv3x3_execution_island \
  -f dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f \
  -o "$RUN_DIR/simv" -Mdir="$RUN_DIR/csrc" \
  2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"
grep -q '^PASS M467R4 directed ' "$RUN_DIR/sim.log"
! grep -q 'Error-\|Assertion failed\|Fatal:' "$RUN_DIR/sim.log"

echo PASS_M467R4_ROW_SHARED_LIVE_SCOREBOARD_SYNOPSYS_VCS > "$RUN_DIR/RUN_COMPLETE.txt"
python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
receipt = {
    "milestone": "M467R4_ROW_SHARED_LIVE_SCOREBOARD",
    "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_FUNCTIONAL_ONLY",
    "tool": "Synopsys VCS V-2023.12-SP1",
    "storage": {"m467r3_slot_live_bits": 24000, "m467r4_row_live_bits": 3000,
                "logical_reduction": "8x", "synchronous_macro_latency_charged": False,
                "boot_scrub_or_reset_physicalized": False, "ppa_admitted": False},
    "counts": {"operators": 2, "phases_including_reset_recovery": 5,
               "accepted_rows": 10, "active_rows": 6,
               "descriptor_writes": 6, "descriptor_reads": 12,
               "pwp_requests": 32, "narrow_pwp_requests": 8,
               "signed_negative_pwp_requests": 8, "weight_requests": 96,
               "plus_requests": 32, "minus_requests": 32,
               "accumulator_reads": 40, "accumulator_writes": 48,
               "same_address_forward_hits": 8, "zero_initializations": 24,
               "zero_init_slot_mask": 255, "zero_commits": 8,
               "row_live_sets": 3, "row_live_clears": 3,
               "operator_commit_vectors": 32, "protocol_attacks": 1},
    "coverage": {"same_phase_distinct_rows": True, "cross_phase_sram_old_psum": True,
                 "no_reset_between_operators": True, "stale_sram_read_suppressed": True,
                 "stale_forward_suppressed": True, "untouched_commit_zero": True,
                 "row_live_set_only_slot7": True, "row_live_clear_only_slot7": True,
                 "all_slots_zero_initialized_before_first_row_live": True,
                 "accumulator_read_stall_stable": True,
                 "accumulator_write_stall_stable": True,
                 "narrow_signed_negative_pwp_exact": True,
                 "commit_exact": True, "reset_recovery": True,
                 "illegal_protocol_fail_closed": True},
    "mutation_attack": {"status": "COUNTEREXAMPLE_REPRODUCED",
                        "premature_set_first_bad_slot": 1,
                        "premature_clear_first_bad_slot": 1},
    "claim_boundary": {"functional_only": True, "cycle_measurement": False,
                       "performance": False, "ppa": False, "macro_ppa": False,
                       "system_or_full_network": False, "energy": False,
                       "headline": False}
}
(r / "m467r4_vcs_receipt_r1.json").write_text(json.dumps(receipt, indent=2)+"\n")
files = [p for p in sorted(r.iterdir()) if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
(r / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
(r / "SHA256SUMS.seal.sha256").write_text(f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY
