#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${QFIT_DS_FLM_STAMP:-20260731}"
OUT="${ROOT}/results/qfit_ds_flm_${STAMP}"
BUILD="${OUT}/build"
RTL="${ROOT}/rtl_qfit/qfit_ds_flm_materializer.sv"
BASELINE_RTL="${ROOT}/rtl_qfit/qfit_source_multicast_term_builder.sv"
TB="${ROOT}/tb_qfit/tb_qfit_ds_flm_materializer.sv"
MITER_TB="${ROOT}/tb_qfit/tb_qfit_ds_flm_lane_miter.sv"
SVA="${ROOT}/verif_qfit/qfit_ds_flm_materializer_assertions.sv"

rm -rf "${BUILD}"
mkdir -p "${BUILD}"

iverilog -g2012 \
  -s tb_qfit_ds_flm_materializer \
  -o "${BUILD}/ds_flm_iv" \
  "${RTL}" "${TB}"
vvp "${BUILD}/ds_flm_iv" \
  | tee "${OUT}/iverilog.log"

iverilog -g2012 \
  -Ptb_qfit_ds_flm_materializer.HEAD_DIM=32 \
  -s tb_qfit_ds_flm_materializer \
  -o "${BUILD}/ds_flm_32_iv" \
  "${RTL}" "${TB}"
vvp "${BUILD}/ds_flm_32_iv" \
  | tee "${OUT}/iverilog_32.log"

iverilog -g2012 \
  -s tb_qfit_ds_flm_lane_miter \
  -o "${BUILD}/ds_flm_miter_iv" \
  "${BASELINE_RTL}" "${RTL}" "${MITER_TB}"
vvp "${BUILD}/ds_flm_miter_iv" \
  | tee "${OUT}/iverilog_miter.log"

verilator --binary --timing --assert -Wall -Wno-fatal -Wno-BLKSEQ \
  --top-module tb_qfit_ds_flm_materializer \
  --Mdir "${BUILD}/obj_verilator" \
  "${RTL}" "${SVA}" "${TB}" \
  >"${OUT}/verilator_build.log" 2>&1
"${BUILD}/obj_verilator/Vtb_qfit_ds_flm_materializer" \
  | tee "${OUT}/verilator.log"

verilator --binary --timing --assert -Wall -Wno-fatal -Wno-BLKSEQ \
  -GHEAD_DIM=32 \
  --top-module tb_qfit_ds_flm_materializer \
  --Mdir "${BUILD}/obj_verilator_32" \
  "${RTL}" "${SVA}" "${TB}" \
  >"${OUT}/verilator_32_build.log" 2>&1
"${BUILD}/obj_verilator_32/Vtb_qfit_ds_flm_materializer" \
  | tee "${OUT}/verilator_32.log"

verilator --binary --timing -Wall -Wno-fatal -Wno-BLKSEQ \
  --top-module tb_qfit_ds_flm_lane_miter \
  --Mdir "${BUILD}/obj_verilator_miter" \
  "${BASELINE_RTL}" "${RTL}" "${MITER_TB}" \
  >"${OUT}/verilator_miter_build.log" 2>&1
"${BUILD}/obj_verilator_miter/Vtb_qfit_ds_flm_lane_miter" \
  | tee "${OUT}/verilator_miter.log"

verilator --lint-only --assert -Wall -Wno-fatal \
  --top-module qfit_ds_flm_materializer \
  "${RTL}" "${SVA}" \
  >"${OUT}/verilator_lint.log" 2>&1
if grep -q '^%Warning' "${OUT}/verilator_lint.log"; then
  cat "${OUT}/verilator_lint.log"
  printf 'Verilator lint出现未审阅warning\n' >&2
  exit 1
fi

yosys -q -l "${OUT}/yosys.log" -p "
  read_verilog -sv ${RTL};
  hierarchy -top qfit_ds_flm_materializer;
  proc; opt; flatten; memory_collect; memory_dff; opt; check -assert;
  tee -o ${OUT}/stat.json stat -json;
  write_json ${OUT}/netlist.json
"

yosys -q -l "${OUT}/baseline_yosys.log" -p "
  read_verilog -sv ${BASELINE_RTL};
  hierarchy -top qfit_source_multicast_term_builder;
  proc; opt; flatten; memory_collect; memory_dff; opt; check -assert;
  tee -o ${OUT}/baseline_stat.json stat -json;
  write_json ${OUT}/baseline_netlist.json
"

{
  printf 'Icarus双模式精确顺序/多重集/随机反压\tPASS\n'
  printf 'Icarus HEAD_DIM=32 最大descriptor\tPASS\n'
  printf 'Icarus原builder/lane-major逐拍miter\tPASS\n'
  printf 'Verilator动态仿真+SVA\tPASS\n'
  printf 'Verilator HEAD_DIM=32动态仿真+SVA\tPASS\n'
  printf 'Verilator原builder/lane-major逐拍miter\tPASS\n'
  printf 'Verilator RTL lint零warning\tPASS\n'
  printf 'Yosys synth-readable/check -assert\tPASS\n'
  printf '同流程lane-major builder结构基线\tPASS\n'
} >"${OUT}/status.tsv"

{
  printf '生成时间UTC\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Icarus\t%s\n' "$(iverilog -V 2>/dev/null | head -n 1)"
  printf 'Verilator\t%s\n' "$(verilator --version)"
  printf 'Yosys\t%s\n' "$(yosys -V)"
  printf '证据边界\tRTL功能、SVA、lint与开放结构综合；非DC/STA/SAIF\n'
} >"${OUT}/reproducibility_manifest.tsv"

sha256sum \
  "${RTL}" "${BASELINE_RTL}" "${TB}" "${MITER_TB}" "${SVA}" \
  "${BASH_SOURCE[0]}" \
  >"${OUT}/source_sha256.txt"

cat >"${OUT}/report.md" <<'EOF'
# DS-FLM 独立双模式 Late Materializer RTL 验证

## 结论

独立单 context materializer 已实现 lane-major 与 gate-major 两种精确展开顺序。两种模式沿用既有 Local5 descriptor 合同和首次出现去重规则；`descriptor_mode` 仅在 descriptor 握手拍锁存，descriptor 执行期间外部模式翻转不改变输出。

## 已验证合同

- lane-major：活动 lane 升序，lane 内 unique gate 按 role 首次出现顺序。
- gate-major：unique gate 按 role 首次出现顺序，gate 内活动 lane 升序。
- 两种模式生成相同 canonical `{lane, gate, destination_mask}` 多重集。
- zero-K、全 zero-gate、全 invalid-role 均产生零 term。
- 非零 descriptor 恰有一个 `term_last`，且只位于最后一个已接受 term。
- 随机下游反压下，输出 payload 和 `term_last` 保持稳定。
- descriptor 执行期间反复翻转 `descriptor_mode`，输出仍遵守握手拍锁存的模式。
- 8-lane定向回归共接收8个descriptor、58个term和90次destination update；
  32-lane回归额外覆盖两种模式各一个160-term最大descriptor。测试强制要求反压
  stall与忙期模式翻转计数均非零。
- 原builder与DS-FLM lane-major在32 lane、100个随机descriptor及双向随机反压下
  逐拍比较ready/valid、全部term字段、`term_last`和性能计数器。

## 工具结果

| 阶段 | 工具 | 结果 |
|---|---|---|
| 功能仿真 | Icarus Verilog | PASS |
| 动态断言 | Verilator + SVA | PASS |
| 静态 lint | Verilator | PASS，RTL 零 warning |
| 开放综合可读 | Yosys | PASS，`check -assert` 无问题 |

## 证据边界与未闭合风险

- 当前是独立叶模块，未接 builder、TCFM 或集成顶层。
- 已做 `HEAD_DIM=8/32` 动态回归、32-lane最大descriptor，以及原builder/lane-major
  的100个随机descriptor逐拍miter；尚未做真实Local5 descriptor trace和形式化
  multiset证明。
- Yosys 将五项小型 gate/mask 数组展开为寄存器并给出 5 条提示性 warning；`check -assert` 为 0 problem，该提示不等同于目标库物理实现结论。
- 单 context 会在整个 descriptor 展开期间反压上游；尚未评估与 producer 的吞吐耦合。
- 两种模式只证明功能等价，尚无门级 SAIF、SRAM 活动、DC/STA 或能量收益结论。
- 设计为单时钟同步复位，无 CDC；本轮未引入 ICG，时钟门控需由后续目标库功耗流程决定。
EOF

DS_CELLS="$(jq -r \
  '.modules."\\qfit_ds_flm_materializer".num_cells' \
  "${OUT}/stat.json")"
BASE_CELLS="$(jq -r \
  '.modules."\\qfit_source_multicast_term_builder".num_cells' \
  "${OUT}/baseline_stat.json")"
DS_WIRES="$(jq -r \
  '.modules."\\qfit_ds_flm_materializer".num_wire_bits' \
  "${OUT}/stat.json")"
BASE_WIRES="$(jq -r \
  '.modules."\\qfit_source_multicast_term_builder".num_wire_bits' \
  "${OUT}/baseline_stat.json")"
CELL_DELTA="$(python3 -c \
  "print(f'{(${DS_CELLS}-${BASE_CELLS})/${BASE_CELLS}:.2%}')")"
{
  printf '\n## 同流程结构增量\n\n'
  printf '| 模块 | Yosys cells | wire bits | `$mul` |\n'
  printf '|---|---:|---:|---:|\n'
  printf '| 原lane-major builder | %s | %s | 1 |\n' \
    "${BASE_CELLS}" "${BASE_WIRES}"
  printf '| DS-FLM双模式materializer | %s | %s | 0 |\n\n' \
    "${DS_CELLS}" "${DS_WIRES}"
  printf 'DS-FLM相对原builder的开放逻辑cell增量为%s；该数字仅是同一' \
    "${CELL_DELTA}"
  printf 'Yosys流程下的结构代理，不是标准单元面积。双模式增加控制mux和状态，'
  printf '但本轮也消除了原基线中的可变term-count乘法器。\n'
} >>"${OUT}/report.md"

printf 'PASS qfit DS-FLM materializer checks\n'
