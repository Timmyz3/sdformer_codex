# UniBin-H60 RTL 双 Skill 审阅与软件一致性验证

**日期**：2026-06-23  
**审阅对象**：

```text
rtl_dc/unibin_h60_core_dc.sv
rtl_allbinary/*.v
tb_dc/tb_unibin_h60_core_dc.sv
tb_allbinary/tb_unibin_h60_modules.v
sim_dc/*
sim_allbinary/*
```

**使用口径**：

1. 按 `rtl-design` skill 做 SystemVerilog/ASIC RTL 设计审阅、lint、CDC/RDC、综合准备检查。
2. 按 `erie-verilog-generator` skill 做 Verilog-2001 风格、静态 lint、ASIC 质量兼容性检查。
3. 由于当前环境没有暴露 `digital-chip-design-agents:rtl-design-orchestrator` 和 `verification-orchestrator` 工具，本轮按 skill 规则手动执行，并把这个限制作为审阅边界。

---

## 1. 总结结论

当前 RTL **可以作为 H60 硬件方案的模块级原型和 DC/Yosys 入口**，但还不能声明为：

```text
软件 H60 bit-accurate 等价 RTL
完整 accelerator RTL
最终 DATE PPA signoff RTL
```

主要原因不是语法跑不过，而是语义层还有几处必须对齐：

1. 当前 RTL 的 H60 score 是 all-binary 简化公式，和软件 `h60` 配置里的 TX/SC 公式不完全一致。
2. 当前 RTL 没有实现软件配置中的 `center_scores: true`。
3. 当前 RTL 没有实现软件配置中的 `consensus_score_norm: head_dim`。
4. `rtl_dc` 的 Shiftmax gate 使用精确除法风格，不是软件 `shiftmax` 的 next-power-of-two denominator。
5. `rtl_dc` 只处理每个 token 一个 `k_value`，不是完整 head_dim 向量输出。
6. `in_last` 如果早于 `cfg_n_tokens` 到达，RTL 仍按 `cfg_n_tokens` 输出，存在读未写 token buffer 的风险。

因此，本轮审阅建议：

```text
先不要继续扩大顶层控制；
先把 H60 score/gate/gated-K 与 PyTorch golden vector 对齐；
再做 SRAM wrapper 和顶层 accelerator shell。
```

---

## 2. 已执行验证

### 2.1 rtl_dc directed simulation

命令：

```bash
cd /root/private_data/work/SDformer/hw_autoresearch_nts07
./sim_dc/run_iverilog_dc.sh
```

结果：

```text
PASS: unibin_h60_core_dc directed test passed
```

说明：

1. reset、`cfg_start`、4-token 输入、一个 empty token、输出计数器能跑通。
2. Icarus 有 testbench `$display/$finish` 在 `always_ff` 内的 warning，属于 testbench 写法 warning，不是 DUT 综合问题。
3. Icarus 提示 `unique case` 被忽略，这是仿真器能力限制，不是 RTL 语法错误。

### 2.2 rtl_dc Verilator lint

命令：

```bash
./sim_dc/run_verilator_lint.sh
```

结果：通过，无输出。

### 2.3 rtl_dc Yosys synth/check

命令：

```bash
./sim_dc/run_yosys_synth.sh
```

关键结果：

```text
Found and reported 0 problems.
Number of memories: 0
Number of cells: 31912
```

解释：

`Number of memories: 0` 不代表不需要 SRAM，而是当前 row buffer 被综合成寄存器/逻辑。后续如果要报真实面积，需要把 `score_mem_q/k_value_mem_q/k_event_mem_q/exp_mem_q` 替换成 SRAM wrapper 或明确保持小 RF。

### 2.4 rtl_allbinary 全量脚本

命令：

```bash
./sim_allbinary/run_all_checks.sh
```

结果：

```text
PASS: UniBin-H60 module smoke tests passed
PASS: Verilator lint completed for all UniBin-H60 tops
PASS: Yosys synthesis/check completed for all UniBin-H60 tops
PASS: all UniBin-H60 RTL checks completed
```

Yosys 统计摘要：

| top | memories | cells | 备注 |
|---|---:|---:|---|
| `binary_atlif_unit` | 0 | 79 | 组合 comparator |
| `binary_atlif_state_unit` | 0 | 481 | 带 membrane state |
| `binary_popcount_consensus` | 0 | 1158 | popcount + TX/SC |
| `ttb_skip_unit` | 0 | 438 | empty/active count |
| `shiftmax_int8_unit` | 0 | 16363 | 组合 Shiftmax 原型，面积最大 |
| `gated_k_unit` | 0 | 374 | 小乘法/门控 |
| `unibin_h60_token_core` | 0 | 1652 | token 级组合原型 |

`shiftmax_int8_unit` 有：

```text
Warning: Replacing memory \numerator with list of registers.
```

这说明 `numerator` 数组没有被综合成 memory，而是展开成寄存器/逻辑。对小 `MAX_TOKENS=8` 可接受；对 162-token row 不应照此复制。

### 2.5 Erie 静态 lint

逐文件运行：

```bash
python /root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py --mode rtl <file>
```

结果摘要：

| 文件 | Erie 结果 | 主要问题 |
|---|---|---|
| `binary_atlif_state_unit.v` | 0 error | 通过 |
| `binary_atlif_unit.v` | 0 error | 通过 |
| `gated_k_unit.v` | 0 error | 通过 |
| `unibin_h60_token_core.v` | 0 error | 通过 |
| `binary_popcount_consensus.v` | 1 error | `for` bound 使用参数，Erie strict 期望常量 loop bound |
| `ttb_skip_unit.v` | 2 errors | wire declaration with init；参数化 loop bound |
| `shiftmax_int8_unit.v` | 6 errors | Verilog function；参数化 loop bound |
| `unibin_h60_core_dc.sv` | 5 errors | SystemVerilog function；参数化 loop bound |

解释：

Erie skill 偏 Verilog-2001 生成风格，尤其不喜欢 `function` 和参数化 loop。当前 `rtl_dc` 是 SystemVerilog 风格，能过 Verilator/Yosys，但不满足 Erie strict style。若后续要交一个更保守的 Verilog-2001 RTL 包，应把 helper function 拆成 leaf module 或展开成显式组合逻辑。

---

## 3. 关键软件一致性审阅

### 3.1 软件 h60 的真实公式

当前 all-binary deployment config 仍然是：

```yaml
bsa_attention:
  mode: h60
  center_scores: true
  preserve_mean: true
  alpha0: 0.02
  mismatch_penalty: 0.0
  score_scale: 1.0
  consensus_score_norm: head_dim
  value_mode: threshold
  single_active_penalty: 0.0
  bipolar_mu: 0.05
  hardware_quant_enabled: true
  hardware_mu_pow2_shift: 4
  hardware_score_step: 0.0078125
  hardware_score_min: -2.0
  hardware_score_max: 2.0
  hardware_gate_step: 0.0078125
  hardware_gate_min: 0.0
  hardware_gate_max: 2.0
```

软件 `h60` 分支的顺序是：

```text
tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
scores = tx_scores + mu * sc_scores
if center_scores:
    scores = scores - scores.mean(dim=2, keepdim=True)
scores = hardware_score_quant(scores)
gate = shiftmax(scores)
if preserve_mean:
    gate = gate * n_tokens
gate = hardware_gate_quant(gate)
attn = k_orig * gate
```

对应源码位置：

```text
bsa_attention.py:1866-1894
```

软件 `shiftmax` 是：

```text
shifted = scores - row_max
numerator = 2^shifted
denominator = 2^ceil(log2(sum(numerator)))
gate = numerator / denominator
```

对应源码位置：

```text
bsa_attention.py:137-147
```

### 3.2 当前 rtl_dc score 公式

`rtl_dc/unibin_h60_core_dc.sv` 当前公式是：

```text
q_active = popcount(Q)
k_active = popcount(K)
overlap  = popcount(Q & K)
mismatch = q_active + k_active - 2 * overlap
TX       = overlap
SC       = overlap - mismatch
score    = TX + (mu_q8 * SC >> 8)
```

对应 RTL：

```text
unibin_h60_core_dc.sv:83-112
```

这个公式是一个硬件友好的 all-binary proxy，但和当前软件配置不完全一致：

| 项目 | 软件 h60 all-binary 输入下 | 当前 RTL |
|---|---|---|
| TX | `same_nonzero + alpha0 * same_zero`，再按 `head_dim` 归一化 | `overlap` |
| SC | binary 0/1 情况下近似 `overlap`，默认无 single-active penalty，再按 `head_dim` 归一化 | `overlap - mismatch` |
| `alpha0=0.02` | 有 silent/silent 小奖励 | 无 |
| `center_scores=true` | row 内减均值 | 无 |
| `consensus_score_norm=head_dim` | 除以 head_dim | 无 |
| `hardware_score_step=1/128` | score 量化到 1/128 | RTL 内部整数 score |

结论：

```text
当前 RTL 不是现有 PyTorch h60 的 bit-accurate 实现。
它是一个候选硬件 proxy，需要软件侧明确导出同公式 golden，或者 RTL 补齐软件公式。
```

### 3.3 当前 rtl_dc Shiftmax 与软件不同

`rtl_dc` 当前 gate 计算：

```text
scaled = exp_value * 255
if preserve_mean:
    scaled = scaled * n_tokens
gate = scaled / row_sum
```

对应 RTL：

```text
unibin_h60_core_dc.sv:128-152
```

软件 Shiftmax 的 denominator 是 next power of two：

```text
denominator = 2^ceil(log2(sum(numerator)))
```

所以软件是 power-of-two denominator，当前 `rtl_dc` 是 exact division by `row_sum`。这会导致 gate 数值系统性不同。

`rtl_allbinary/shiftmax_int8_unit.v` 反而更接近 power-of-two denominator，因为它有 `ceil_log2_u32(row_sum)` 和 shift 逻辑；但它是组合原型，`MAX_TOKENS=8`，不适合直接扩到 162。

建议：

1. `rtl_dc` 中把 `gate_from_exp` 改成 next-power-of-two denominator。
2. 或者软件 deployment config 改成 exact L1/softmax-ish gate，并重新验证精度。
3. DATE 论文如果强调 Shiftmax 硬件友好，建议 RTL 必须实现 power-of-two denominator，不能用综合除法作为主方案。

### 3.4 当前 rtl_dc 没有实现完整 head_dim 输出

软件 `h60` 输出是：

```text
attn = k_orig.mul(gate)
```

其中 `k_orig` 是 token × head_dim 的向量。

当前 `rtl_dc` 输入只有：

```text
in_k_value: signed [7:0]
in_k_bits : [HEAD_DIM-1:0]
```

输出只有：

```text
out_gated_k: signed [15:0]
```

也就是每个 token 一个 scalar `k_value`，不是完整 head_dim vector。它适合作为 H60 row gate/gated scalar smoke core，但不是完整 attention output datapath。

建议下一版接口改成以下二选一：

1. **vector lane streaming**：每个 token 的 K value 按 channel lane 分多拍输入/输出。
2. **event-only gated-K**：如果 K value 也二值化，则输出变成 gate add/skip，另用 accumulator 还原 channel。

### 3.5 `in_last` 与 `cfg_n_tokens` 协议有潜在 bug

当前 `ST_LOAD` 在以下条件进入下一状态：

```text
in_last || load_idx_q == n_tokens_q - 1
```

但 `n_tokens_q` 在 `cfg_start` 时固定为 `cfg_n_tokens`，如果 `in_last` 提前到达，RTL 没有把 `n_tokens_q` 改成实际加载数量。后续 `ST_SUM_EXP/ST_EMIT` 仍会读出 `cfg_n_tokens` 个 token，其中一部分 buffer 未写。

对应 RTL：

```text
unibin_h60_core_dc.sv:173-176
unibin_h60_core_dc.sv:223-250
```

建议二选一：

1. 如果 `cfg_n_tokens` 是唯一权威长度，则移除或忽略 `in_last`。
2. 如果 `in_last` 是合法提前结束，则在接收最后 token 时锁存：

```text
actual_n_tokens_q <= load_idx_q + 1
```

并用 `actual_n_tokens_q` 驱动后续 scan/emit。

这是功能正确性问题，建议 P0 修复。

---

## 4. RTL 设计质量审阅

### 4.1 rtl-design 口径

优点：

1. `rtl_dc` 使用 `default_nettype none`。
2. 单时钟域 `clk_core`，没有 CDC。
3. `rtl_dc` 使用同步复位风格，适合 ASIC。
4. ready-valid 基本接口清楚。
5. Verilator/Yosys 可通过。

问题：

| 严重级别 | 问题 | 说明 |
|---|---|---|
| P0 | 软件公式未 bit-accurate 对齐 | score、center、head_dim norm、Shiftmax denominator 都有差异 |
| P0 | `in_last` 早停协议风险 | 可能读未写 token buffer |
| P1 | `gate_from_exp` 使用组合除法 | 面积/时序差，不符合 Shiftmax power-of-two 故事 |
| P1 | row buffers 未 SRAM wrapper 化 | 当前 Yosys memory=0，面积不真实 |
| P1 | `out_valid` 时 `out_gate/out_gated_k` 为组合输出 | 下游 backpressure 时虽然 index 不变，但大除法/乘法在输出端组合路径上 |
| P1 | 缺 SVA/协议 checker | 没有 valid-ready 稳定性、start/busy、cfg range assertion |
| P2 | `ST_FIND_MAX` 现在只是 pipeline boundary | 可以保留，但文档要说明，或用于后续 mean-centering |
| P2 | 缺 clock gating plan | skill 要求高 gating 机会域显式 ICG；当前还没有功耗 RTL 策略 |

### 4.2 Erie 口径

Erie strict 更偏保守 Verilog-2001：

1. 不希望 RTL 里使用 `function/task`；
2. 不希望 wire declaration 同时初始化；
3. 不喜欢参数化 loop bound；
4. 更偏显式、可波形调试的 leaf module。

因此当前两套 RTL 分层建议是：

| 目录 | 定位 | 是否适合继续 |
|---|---|---|
| `rtl_allbinary` | 算子原型/模块烟测 | 适合保留作为参考和单元 golden |
| `rtl_dc` | DC-ready 最小 H60 core | 适合作为主线，但需修 P0 软件一致性 |
| 未来 `rtl_release` | Verilog-2001/ASIC handoff 包 | 建议从 `rtl_dc` 重构而来 |

---

## 5. 待优化清单

### P0：必须先处理

1. **确定硬件公式到底对齐哪个软件公式**  
   当前软件 config 不是 RTL 公式。要么改软件部署模式导出 `Binary Consensus Score Engine` golden，要么改 RTL 实现软件公式。

2. **补 PyTorch golden vector**  
   至少导出：

```text
Q event
K event
tx_score
sc_score
centered fused score
quantized score
exp numerator
row_sum
shiftmax gate
quantized gate
k_orig
attn output
```

3. **修 `in_last`/`cfg_n_tokens` 协议**  
   要么 `cfg_n_tokens` 权威，要么 early `in_last` 更新 actual length。

4. **把 rtl_dc Shiftmax 改成 next-power-of-two denominator**  
   否则和软件 `shiftmax` 与论文故事不一致。

### P1：进入 DC/PPA 前处理

1. row buffer SRAM wrapper 化。
2. `gate_from_exp` 去除组合除法。
3. 输出端增加寄存器切 pipeline，避免除法/乘法直接挂在 output path。
4. 增加 SVA：

```text
cfg_start only accepted in IDLE
in_valid && !in_ready -> input stable
out_valid && !out_ready -> output stable
emit_count == actual_n_tokens
done only after last output handshake
cfg_n_tokens <= MAX_TOKENS
```

5. 增加 directed tests：

```text
n_tokens=1
n_tokens=MAX_TOKENS
in_last early
out_ready backpressure
all empty tokens
all active tokens
negative k_value
score saturation
row_sum boundary
```

### P2：结构优化

1. 把 popcount、score fuse、Shiftmax、gated-K 拆成 leaf modules，便于 Erie strict 和 DC timing。
2. 增加 descriptor-driven wrapper，避免 top-level 控制直接绑死 H60 row。
3. `rtl_allbinary/shiftmax_int8_unit.v` 的 power-of-two 归一化思路可迁移到 `rtl_dc`，但需要流水化和支持 162 token。
4. 增加 clock enable 或 ICG wrapper 计划，利用 TTB empty skip 降低 activity。

---

## 6. 建议的下一步

最小下一步不是写完整 accelerator，而是做一个 **H60 bit-accurate closure loop**：

```text
PyTorch export golden row
→ RTL testbench read vector
→ compare score/gate/output
→ 修 score/Shiftmax/quant
→ 再进入 SRAM wrapper 和 accelerator shell
```

建议新建：

```text
entrypoints/export_unibin_h60_golden.py
tb_dc/golden/
tb_dc/tb_unibin_h60_core_golden.sv
sim_dc/run_golden_check.sh
```

只有这个闭环通过后，当前 RTL 才能从“硬件方案原型”升级为“软件网络设计一致的 RTL”。

