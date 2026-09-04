#!/usr/bin/env bash
set -euo pipefail

HW_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HW_ROOT"
RUN_DIR="${1:-results/m467r3_back_to_back_operator_contamination_attack_r1_20260826}"

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  test "$actual" = "$expected" || {
    echo "M467R3 attack exact-SHA mismatch path=$path expected=$expected actual=$actual" >&2
    exit 2
  }
}

check_sha contracts/m467r3_back_to_back_operator_contamination_attack_contract_r1_20260826.json eaaef5f1a83592fae23687ece1fcd01ef9fd1b744b21683df91e1c296244b26b
check_sha rtl_m414/m414_q32_balanced16_zero_stop_controller.sv a290feff90b9aa6c282fedf99a284e4afe2cff96dc5f7bc79b04e76b97144f1f
check_sha rtl_m433/m433_exact_dualbank_coread_pwp_adapter.sv 75ad462a584ea46bd1043bb6a21d82b5687e7ab392995b28d707c248a5f96046
check_sha rtl_m467/m467_conv3x3_execution_island.sv 4f32a49fc06a26ca9aed0a9af5b37818340b0ce87cd8b3da34a5ba96f268b56f
check_sha verif_m467/m467_conv3x3_execution_island_assertions.sv d02939289e5823b890785e4e84d692b50b0f67fdfb0a166aa1627a4d7e15c8db
check_sha tb_m467/tb_m467_conv3x3_execution_island.sv 087e3d072bc7789db6e3fbbe0621fe3338ede8df5d76be86f2d02c36c0d752f5
check_sha dc_handoff/filelists/date_m467_conv3x3_execution_island_vcs.f 5a7c1e63f5e08fafb57850ac60dda15c8f1f56ad450e6a103f621e0bd48265e2
check_sha docs/359_DATE终局冻结_20260813.md dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4

test ! -e "$RUN_DIR" || { echo "M467R3 attack result directory exists: $RUN_DIR" >&2; exit 3; }
mkdir -p "$RUN_DIR"
cp contracts/m467r3_back_to_back_operator_contamination_attack_contract_r1_20260826.json "$RUN_DIR/contract.json"
sha256sum \
  contracts/m467r3_back_to_back_operator_contamination_attack_contract_r1_20260826.json \
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
set +e
"$RUN_DIR/simv" -no_save 2>&1 | tee "$RUN_DIR/sim.log"
SIM_RC="${PIPESTATUS[0]}"
set -e
grep -q 'commit mismatch slot=0 row=0' "$RUN_DIR/sim.log"
! grep -q '^PASS M467R2 directed ' "$RUN_DIR/sim.log"
printf '%s\n' "$SIM_RC" > "$RUN_DIR/sim_exit_code.txt"
echo REPRODUCED_M467R2_BACK_TO_BACK_OPERATOR_CONTAMINATION > "$RUN_DIR/ATTACK_REPRODUCED.txt"

python3 - "$RUN_DIR" <<'PY'
import hashlib, json, pathlib, sys
r = pathlib.Path(sys.argv[1])
receipt = {
    "milestone": "M467R3_BACK_TO_BACK_OPERATOR_CONTAMINATION_ATTACK",
    "status": "REPRODUCED_EXPECTED_FAIL_SYNOPSYS_VCS",
    "failure": "second operator row0 commit consumed stale operator-1 accumulator data despite zero mathematical initialization",
    "old_rtl_sha256": "4f32a49fc06a26ca9aed0a9af5b37818340b0ce87cd8b3da34a5ba96f268b56f",
    "reset_between_operators": False,
    "claims": {"performance": False, "ppa": False, "system_or_full_network": False, "headline": False}
}
(r / "m467r3_attack_receipt_r1.json").write_text(json.dumps(receipt, indent=2)+"\n")
files = [p for p in sorted(r.iterdir()) if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}]
(r / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in files))
(r / "SHA256SUMS.seal.sha256").write_text(f"{hashlib.sha256((r/'SHA256SUMS').read_bytes()).hexdigest()}  SHA256SUMS\n")
PY
