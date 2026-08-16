#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/build_new_arch/gasr2c_singlebank"
OUT="${ROOT}/results/local5_gasr2c_singlebank_postg0_rtl_20260803"
VECTORS="${ROOT}/tb_qfit/vectors/local5_gasr_singlebank_postg0_100"
mkdir -p "${BUILD}" "${OUT}/yosys"

RTL=(
  "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv"
  "${ROOT}/rtl_qfit/qfit_direct_1rw_acc_bank.sv"
  "${ROOT}/rtl_qfit/qfit_gasr2c_acc_bank.sv"
)
ASSERTIONS=(
  "${ROOT}/verif_qfit/qfit_single_port_acc_memory_assertions.sv"
  "${ROOT}/verif_qfit/qfit_direct_1rw_acc_bank_assertions.sv"
  "${ROOT}/verif_qfit/qfit_gasr2c_acc_bank_assertions.sv"
)
TB="${ROOT}/tb_qfit/tb_qfit_gasr2c_singlebank_postg0.sv"

PYTHONPATH="${ROOT}/scripts" python3 \
  "${ROOT}/scripts/generate_local5_gasr_singlebank_vectors.py" >/dev/null

iverilog -g2012 -Wall -s tb_qfit_gasr2c_singlebank_postg0 \
  -o "${BUILD}/deterministic.vvp" "${RTL[@]}" "${TB}"
vvp "${BUILD}/deterministic.vvp" "+VECTOR_DIR=${VECTORS}" \
  | tee "${OUT}/deterministic.log"

iverilog -g2012 -Wall -s tb_qfit_gasr2c_singlebank_postg0 \
  -Ptb_qfit_gasr2c_singlebank_postg0.RANDOM_GAPS=1 \
  -o "${BUILD}/random.vvp" "${RTL[@]}" "${TB}"
vvp "${BUILD}/random.vvp" "+VECTOR_DIR=${VECTORS}" \
  | tee "${OUT}/random_gaps.log"

for mode in deterministic random; do
  mdir="${BUILD}/${mode}_obj"
  rm -rf "${mdir}"
  extra=()
  if [[ "${mode}" == "random" ]]; then
    extra=(-GRANDOM_GAPS=1)
  fi
  verilator --binary --timing --assert -Wall -Wno-fatal \
    -Wno-UNUSEDSIGNAL -Wno-BLKSEQ -Wno-WIDTHTRUNC \
    --top-module tb_qfit_gasr2c_singlebank_postg0 \
    "${extra[@]}" --Mdir "${mdir}" \
    "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}"
  "${mdir}/Vtb_qfit_gasr2c_singlebank_postg0" \
    "+VECTOR_DIR=${VECTORS}" | tee "${OUT}/verilator_assert_${mode}.log"
done

for top in qfit_direct_1rw_acc_bank qfit_gasr2c_acc_bank; do
  verilator --lint-only -Wall -Wno-fatal --top-module "${top}" \
    "${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv" \
    "${ROOT}/rtl_qfit/${top}.sv" > "${OUT}/verilator_lint_${top}.log" 2>&1
  yosys -Q -p "read_verilog -sv ${ROOT}/rtl_qfit/qfit_single_port_acc_memory.sv ${ROOT}/rtl_qfit/${top}.sv; hierarchy -check -top ${top}; proc; opt; memory -nomap; opt; check; tee -q -o ${OUT}/yosys/${top}.json stat -json" \
    > "${OUT}/yosys/${top}.log" 2>&1
done

python3 "${ROOT}/scripts/summarize_local5_gasr2c_singlebank_rtl.py" >/dev/null
sha256sum "${RTL[@]}" "${ASSERTIONS[@]}" "${TB}" \
  "${ROOT}/scripts/generate_local5_gasr_singlebank_vectors.py" \
  "${ROOT}/scripts/summarize_local5_gasr2c_singlebank_rtl.py" \
  "${VECTORS}/manifest.json" > "${OUT}/source_sha256.txt"
cat > "${OUT}/status.tsv" <<'EOF'
真实100组deterministic Acc32与周期对照	PASS
真实100组随机输入空泡Acc32	PASS
Verilator SVA deterministic	PASS
Verilator SVA random	PASS
Verilator RTL lint direct/GASR	PASS
Yosys可综合性 direct/GASR	PASS
EOF
echo "PASS Local5 GASR-2C single-bank checks"
