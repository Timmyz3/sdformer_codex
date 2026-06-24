# UniBin-H60 RTL Skill 流程详细审阅

**日期**：2026-06-23  
**审阅对象**：

```text
rtl_dc/unibin_h60_core_dc.sv
tb_dc/tb_unibin_h60_core_dc.sv
rtl_allbinary/*.v
tb_allbinary/tb_unibin_h60_modules.v
sim_dc/*
sim_allbinary/*
```

**使用 skill**：

1. `rtl-design`：按 module planning、RTL coding、lint、CDC/RDC、synth readiness、signoff readiness 审阅。
2. `erie-verilog-generator`：按 ASIC Verilog quality、RTL-MD constraints、independent static lint 审阅。
3. `functional-verification`：按 test plan、directed tests、coverage 缺口审阅。

说明：当前环境未暴露 `digital-chip-design-agents:rtl-design-orchestrator` 和 `verification-orchestrator`，因此本轮按 skill 文档手动执行 staged review，并记录每阶段 evidence。

---

## 1. 总结结论

当前 RTL 状态可以这样定义：

```text
SystemVerilog 模块级 H60 core：可仿真、可 Verilator lint、可 Yosys synth/check。
Erie strict Verilog-2001 handoff：未通过。
软件 bit-accurate 等价：未完成，因为缺 PyTorch golden vector 和完整 head_dim vector datapath。
完整 accelerator：未完成，因为缺 SRAM wrapper、descriptor controller、TTB shell、window dataflow。
```

本轮 fresh verification 结果：

| 项目 | 结果 |
|---|---|
| `sim_dc/run_iverilog_dc.sh` | PASS |
| `sim_dc/run_verilator_lint.sh` | PASS，无 warning 输出 |
| `sim_dc/run_yosys_synth.sh` | PASS，0 problems，cells=24313，memories=0 |
| `sim_allbinary/run_all_checks.sh` | PASS |
| Erie lint `rtl_dc` | FAIL，6 error / 2 warning |
| Erie lint `rtl_allbinary` | 部分通过；`binary_popcount_consensus`、`shiftmax_int8_unit`、`ttb_skip_unit` 仍有 strict style error |

最重要的审阅判断：

1. **可以继续作为 SystemVerilog H60 core 主线推进**。
2. **不能直接交 Erie strict 或 Verilog-2001 style signoff**。
3. **不能声称软件网络等价**，直到 golden vector checker 通过。
4. **不能报最终面积**，因为 Yosys `memories=0` 表明 row buffer 仍是寄存器/逻辑展开，不是 SRAM macro。

---

## 2. rtl-design Skill 审阅

### 2.1 Stage: module_planning

`rtl_dc/unibin_h60_core_dc.sv` 当前职责：

```text
load one H60 token row
→ compute all-binary TX/SC raw score
→ row mean centering
→ row max
→ exp2 approximation
→ power-of-two Shiftmax gate
→ gated scalar K output
```

模块接口：

| 接口组 | 状态 |
|---|---|
| 配置接口 `cfg_*` | 有 |
| 输入 ready-valid | 有 |
| 输出 ready-valid | 有 |
| perf counter | 有 |
| SRAM/memory interface | 无，当前内部数组 |
| descriptor interface | 无 |
| head_dim lane/vector interface | 无 |

审阅结论：

| 检查项 | 结果 | 说明 |
|---|---|---|
| 单一职责 | 部分通过 | 当前 core 把 score/Shiftmax/buffer/FSM 放在一个模块，适合原型，不适合最终 timing closure |
| 端口定义 | 通过 | 方向、类型、位宽明确 |
| 单时钟域 | 通过 | 仅 `clk_core` |
| reset domain | 通过 | 单 active-low reset |
| top-level wiring only | 不适用 | 当前不是顶层 |
| datapath/control 分离 | 未通过 | FSM 和 datapath 混在一个模块，后续建议拆 leaf modules |

建议模块层级：

```text
unibin_h60_core
├── rv_row_loader
├── binary_consensus_score_q7
├── row_center_max_unit
├── shiftmax_pow2_q8
├── gated_k_scalar_or_lane
└── perf_counter
```

### 2.2 Stage: rtl_coding

正向项：

1. `rtl_dc` 使用 `default_nettype none`。
2. 端口显式声明。
3. `always_comb` 有默认赋值，未见 latch 风险。
4. `always_ff` 使用 nonblocking assignment。
5. 同步 reset 风格适合 ASIC。
6. 没有 raw gated clock。
7. 没有 `initial` 或 `#delay` 出现在 DUT RTL。

问题：

| 严重度 | 问题 | 文件/位置 | 说明 |
|---|---|---|---|
| P1 | function-heavy datapath | `unibin_h60_core_dc.sv:76,88,128,167,181` | Verilator/Yosys 可接受，但 Erie strict 不接受，timing review 也不够模块化 |
| P1 | 组合除法仍存在 | `consensus_score`、`score_mean_w` | score/head_dim 和 row mean 用 `/`，对 ASIC timing/PPA 不友好 |
| P1 | 内部数组未 SRAM 化 | `score_mem_q/k_value_mem_q/k_event_mem_q/exp_mem_q` | `memories=0`，面积不真实 |
| P1 | scalar K output | `in_k_value/out_gated_k` | 不是完整 `head_dim` vector datapath |
| P2 | `cfg_n_tokens=0` 被解释为 `MAX_TOKENS` | `cfg_start` 初始化逻辑 | 可作为协议定义，但应写入接口 spec 或改成非法配置 |
| P2 | `cfg_alpha0/center_scores` 不可运行时配置 | parameter 固化 | 当前可接受，CSR 版本后续再加 |

### 2.3 Stage: lint_check

Fresh command：

```bash
./sim_dc/run_verilator_lint.sh
```

结果：

```text
PASS，无 warning 输出
```

`sim_allbinary/run_all_checks.sh` 中逐 top Verilator lint：

```text
PASS: Verilator lint completed for all UniBin-H60 tops
```

审阅结论：

```text
SystemVerilog/Verilator lint 通过；
Erie strict lint 未通过，见第 3 节。
```

### 2.4 Stage: CDC/RDC

CDC：

```text
仅一个 clock domain: clk_core
无跨时钟输入同步器
```

RDC：

```text
单 rst_n_core
同步 reset 使用方式：always_ff @(posedge clk_core) + if (!rst_n_core)
```

审阅结论：

| 项目 | 结果 |
|---|---|
| CDC | 当前无 CDC crossing |
| RDC | 当前单 reset domain |
| 风险 | 外部 `cfg_start/in_valid/out_ready` 必须与 `clk_core` 同步；若来自异步域，外层 wrapper 必须加 synchronizer/async FIFO |

### 2.5 Stage: synth_check

Fresh command：

```bash
./sim_dc/run_yosys_synth.sh
```

结果：

```text
Found and reported 0 problems.
Number of memories: 0
Number of cells: 24313
```

审阅结论：

| 项目 | 结果 | 说明 |
|---|---|---|
| Yosys check | 通过 | 0 problems |
| generic cells | 24313 | 只能作为趋势数 |
| memories | 0 | 不能作为最终 SRAM 后面积 |
| unmapped tech cells | 未评估 | 尚未接 DC library |
| timing/WNS | 未评估 | 尚无 SDC/library |

---

## 3. erie-verilog-generator Skill 审阅

### 3.1 Erie ASIC Quality 对照

通过项：

| 规则方向 | 状态 |
|---|---|
| no raw gated clocks | 通过 |
| testbench 与 RTL 分离 | 通过 |
| no delay in RTL | 通过 |
| no simulation system task in RTL | 通过 |
| explicit reset polarity | 通过 |
| no internal tri-state | 通过 |

未通过或需改进：

| 规则方向 | 状态 | 说明 |
|---|---|---|
| no function/task in generated RTL | 未通过 | `rtl_dc` 使用 5 个 function |
| parameterized loop strict bound | 未通过 | `popcount32` 和部分 allbinary loop |
| arithmetic width review | 部分通过 | Verilator 已无 warning，但 Erie literal/base warning 仍存在 |
| timing-reviewable datapath | 部分通过 | 大组合 function 不利于后续 timing closure |

### 3.2 Erie 静态 lint 结果

Fresh command：

```bash
python /root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py --mode rtl <file>
```

`rtl_dc/unibin_h60_core_dc.sv`：

```text
6 error(s), 2 warning(s)
```

主要项：

```text
NO_TASK_FUNCTION x5
FOR_CONST_BOUNDS x1
LITERAL_BASE_WIDTH x2
```

`rtl_allbinary`：

| 文件 | Erie 结果 |
|---|---|
| `binary_atlif_state_unit.v` | 0 error / 0 warning |
| `binary_atlif_unit.v` | 0 error / 0 warning |
| `gated_k_unit.v` | 0 error / 0 warning |
| `unibin_h60_token_core.v` | 0 error / 0 warning |
| `binary_popcount_consensus.v` | 1 error / 4 warning |
| `ttb_skip_unit.v` | 2 error / 2 warning |
| `shiftmax_int8_unit.v` | 6 error / 8 warning |

审阅结论：

```text
当前 RTL 不是 Erie strict handoff 形态。
如果目标是 Verilog-2001 strict，需要拆 function、固定 loop bound 或生成专用 HEAD_DIM=32 版本。
```

### 3.3 Erie 修复建议

推荐新建一个 strict 目录，而不是直接污染当前 SystemVerilog 主线：

```text
rtl_release/
├── popcount32_fixed.v
├── binary_consensus_score_q7.v
├── exp2_lut_q8.v
├── ceil_log2_u32.v
├── shiftmax_pow2_q8.v
└── unibin_h60_core_release.v
```

这样做的好处：

1. `rtl_dc` 保持快速研究迭代；
2. `rtl_release` 满足 Erie/Verilog-2001 handoff；
3. 两者用同一套 golden vector 做 equivalence check。

---

## 4. functional-verification Skill 审阅

### 4.1 当前 test plan 覆盖

`tb_dc` 当前覆盖：

| 场景 | 状态 |
|---|---|
| reset | 覆盖 |
| normal 4-token row | 覆盖 |
| empty token counter | 覆盖 |
| early `in_last` | 覆盖 |
| output backpressure | 覆盖 |
| 162-token synthetic row | 覆盖 |
| negative `k_value` | 覆盖 |
| `out_last` token index | 覆盖 |

`tb_allbinary` 当前覆盖：

| 算子 | 状态 |
|---|---|
| binary ATLIF comparator | 覆盖 |
| stateful ATLIF leak/reset | 覆盖 |
| popcount consensus raw score | 覆盖 |
| TTB empty detector | 覆盖 |
| shiftmax_int8 smoke | 覆盖 |
| gated-K multiply | 覆盖 |

### 4.2 缺失 verification 项

| 优先级 | 缺口 | 说明 |
|---|---|---|
| P0 | PyTorch golden vector checker | 软件等价声明前必须完成 |
| P0 | score/gate/output 数值逐 token 对比 | 现在 testbench 只查计数和非零 gate |
| P1 | SVA bind checker | ready-valid 稳定性、emit bound、done 时序 |
| P1 | random/backpressure stress | 当前 backpressure 只有一处 directed |
| P1 | reset-during-transaction | 当前没有运行中 reset |
| P1 | boundary config | `cfg_n_tokens=1/0/162/>MAX` |
| P2 | coverage metrics | 当前没有 functional coverage |

### 4.3 建议 V-plan

| feature_id | 描述 | directed test | assertion/monitor |
|---|---|---|---|
| F001 | cfg/start 协议 | start_when_idle | `cfg_start` only accepted in `ST_IDLE` |
| F002 | input ready-valid | input_hold_when_not_ready | input stable when `valid && !ready` |
| F003 | early last | early_last_short_row | issued count equals actual loaded |
| F004 | max tokens | max_162_row | `out_last` at token 161 |
| F005 | output backpressure | output_stall | output stable when `valid && !ready` |
| F006 | score datapath | golden_score | compare raw/centered score |
| F007 | shiftmax datapath | golden_gate | compare exp/row_sum/denom/gate |
| F008 | gated K | golden_gated_k | compare output |
| F009 | reset | reset_mid_row | all counters/FSM reset cleanly |

---

## 5. 软件一致性审阅

当前 `rtl_dc` 已比最初版本更接近 all-binary deployment config：

```text
TX = (overlap + alpha0 * same_zero) / head_dim
SC = overlap / head_dim
score = TX + mu * SC
row mean centering
Shiftmax next-power-of-two denominator
```

但仍不能声明 bit-accurate：

1. 软件 `alpha0=0.02`，RTL 使用 `ALPHA0_Q8=5`，即 `0.01953125`。
2. 软件 `torch.pow(2.0, shifted)`，RTL 是 Q7 + 16-entry fractional LUT 近似。
3. 软件 `hardware_gate_step=1/128`，RTL gate 输出是 8-bit integer，目前未建立同一量纲的 JSON checker。
4. 软件 `attn = k_orig * gate` 是 head_dim vector，RTL 仍是 scalar `in_k_value/out_gated_k`。

因此软件等价 signoff 必须依赖：

```text
export PyTorch golden row
→ RTL sim dump or checker
→ compare score/gate/gated output
→ record tolerance and quantization convention
```

---

## 6. 关键风险清单

### P0

1. 缺 PyTorch golden vector checker。
2. 缺完整 head_dim vector/lane datapath。

### P1

1. Erie strict 未通过。
2. 缺 SVA bind checker。
3. row buffer 未 SRAM wrapper 化。
4. `score_mean_w` 和 score/head_dim 仍含组合除法，后续 PPA 可能不理想。
5. `shiftmax_int8_unit` 是 legacy 原型，`numerator` 被 Yosys 展开成寄存器，不应扩到 162-token 主线。

### P2

1. `cfg_n_tokens=0` 协议需要明确。
2. `cfg_alpha0/cfg_center_scores` 是否需要 CSR 化需要架构决策。
3. 缺 descriptor、TTB issue、window partition/reverse、skip/residual shell。

---

## 7. 建议下一步

最短路径：

1. 新增 `export_unibin_h60_golden.py`，导出 1-2 个真实 window 的 Q/K/score/gate/output。
2. 新增 `tb_dc/tb_unibin_h60_core_golden.sv` 或 Python cocotb checker。
3. 加 SVA bind file：

```text
out_valid && !out_ready -> out_token_idx/out_gate/out_gated_k stable
state==ST_EMIT -> emit_idx_q < n_tokens_q
done -> issued_tokens_q == n_tokens_q
cfg_start accepted only in IDLE
```

4. 若需要 Erie strict 交付，再拆 `rtl_release` leaf modules。
5. golden 通过后，再设计 head_dim lane streaming 和 SRAM wrapper。

---

## 8. Signoff 状态

| signoff 项 | 当前状态 |
|---|---|
| module smoke simulation | 通过 |
| Verilator lint | 通过 |
| Yosys generic synth/check | 通过 |
| Erie strict lint | 未通过 |
| CDC/RDC | 单域假设下通过；外部异步输入未覆盖 |
| functional coverage | 未建立 |
| software golden equivalence | 未建立 |
| DC/library/SDC timing | 未建立 |
| SRAM macro PPA | 未建立 |

当前可对外表述：

```text
UniBin-H60 已有可综合 SystemVerilog 模块级原型，并通过 directed simulation、Verilator lint 和 Yosys synth/check。
它还不是软件 bit-accurate 或 Erie strict/Verilog-2001 handoff RTL。
```

