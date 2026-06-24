# UniBin-H60 RTL P0 修复与复验结果

**日期**：2026-06-23  
**对象**：`rtl_dc/unibin_h60_core_dc.sv`、`tb_dc/tb_unibin_h60_core_dc.sv`  
**背景**：基于 `28_UniBinH60_RTL双Skill审阅与软件一致性验证.md` 的 P0/P1 问题，本轮只修最小关键路径，不扩大完整 accelerator 顶层。

---

## 1. 本轮修复结论

本轮已经修复三个关键问题：

1. `in_last` 早于 `cfg_n_tokens` 时，后续 scan/emit 现在按实际加载 token 数执行。
2. `rtl_dc` 的 H60 score 从原来的硬件 proxy 改为更贴近当前 all-binary deployment config 的公式：

```text
TX = (overlap + alpha0 * same_zero) / head_dim
SC = overlap / head_dim
score = TX + mu * SC
score_q = Q7 fixed-point
centered_score = score_q - row_mean(score_q)
```

3. `rtl_dc` 的 Shiftmax gate 从 exact division by `row_sum` 改为 next-power-of-two denominator：

```text
denom_shift = ceil(log2(row_sum))
gate = exp2(score - row_max) * 255 * n_tokens >> denom_shift
```

这使 RTL 方向更接近软件 `shiftmax()`：

```text
2^score / 2^ceil(log2(sum(2^score)))
```

---

## 2. 具体 RTL 改动

### 2.1 score 公式修复

原 RTL：

```text
q_active = popcount(Q)
k_active = popcount(K)
overlap  = popcount(Q & K)
mismatch = q_active + k_active - 2 * overlap
TX       = overlap
SC       = overlap - mismatch
score    = TX + mu * SC
```

修复后：

```text
same_zero = head_dim - q_active - k_active + overlap
TX        = overlap + alpha0 * same_zero
SC        = overlap
score     = (TX + mu * SC) / head_dim
```

当前 RTL 使用：

```text
SCORE_FRAC = 7
ALPHA0_Q8  = 5
cfg_mu_q8  = 16
```

对应：

```text
score step = 1/128
alpha0 ≈ 5/256 = 0.01953125
mu = 16/256 = 1/16
```

这与 deployment config 的主要硬件量化口径一致：

```yaml
hardware_score_step: 0.0078125
hardware_mu_pow2_shift: 4
hardware_gate_step: 0.0078125
alpha0: 0.02
```

注意：`alpha0=5/256` 是 `0.02` 的硬件近似，不是完全相等。

### 2.2 row mean centering

软件配置：

```yaml
center_scores: true
```

修复后 RTL 在 `ST_FIND_MAX` 阶段执行：

```text
centered_score_i = raw_score_i - mean(raw_score_row)
score_mem[i]     = centered_score_i
row_max          = max(centered_score_i)
```

这样后续 `ST_SUM_EXP` 使用 centered score 做 Shiftmax。

### 2.3 early in_last 修复

原问题：

```text
cfg_n_tokens = 6
实际只输入 3 个 token 且第 3 个 in_last=1
后续仍按 6 个 token emit
```

修复后：

```text
if in_last || load_idx == n_tokens_q - 1:
    n_tokens_q <= load_idx + 1
```

因此后续 `ST_FIND_MAX/ST_SUM_EXP/ST_EMIT` 均按实际 token 数执行。

### 2.4 Shiftmax denominator 修复

原 RTL：

```text
gate = scaled / row_sum
```

修复后：

```text
denom_shift = ceil_log2(row_sum)
gate = scaled >> denom_shift
```

这消除了组合除法，也让 RTL 和 DATE 论文中的 Shiftmax 硬件友好叙事一致。

### 2.5 fractional exp2 近似

因为 score 现在是 Q7 定点，`exp2_approx_q8()` 不再只按整数 delta shift，而是加入 16 档 fractional LUT：

```text
frac_idx = abs(delta_q7)[6:3]
frac_value ≈ 256 * 2^(-frac_idx/16)
exp = frac_value >> integer_shift
```

这是硬件近似，不是完整浮点 `pow(2.0, x)`。

---

## 3. testbench 扩展

`tb_dc/tb_unibin_h60_core_dc.sv` 新增第二帧测试：

```text
cfg_n_tokens = 6
实际输入 token = 3
第 3 个 token 设置 in_last = 1
out_ready 先拉低两拍
检查 out_valid/out_token_idx/out_gate 在 backpressure 下稳定
检查最终 perf_tokens_loaded/perf_issued_tokens = 3
检查 out_last 出现在 token_idx = 2
```

覆盖了两个之前缺失的风险：

1. early `in_last`；
2. output ready-valid backpressure。

---

## 4. 复验结果

### 4.1 rtl_dc iverilog directed simulation

命令：

```bash
cd /root/private_data/work/SDformer/hw_autoresearch_nts07
./sim_dc/run_iverilog_dc.sh
```

结果：

```text
PASS: unibin_h60_core_dc directed test passed
```

### 4.2 rtl_dc Verilator lint

命令：

```bash
./sim_dc/run_verilator_lint.sh
```

结果：

```text
PASS，无 warning 输出
```

### 4.3 rtl_dc Yosys synth/check

命令：

```bash
./sim_dc/run_yosys_synth.sh
```

结果：

```text
Found and reported 0 problems.
Number of memories: 0
Number of cells: 24289
```

对比修复前：

```text
修复前 cells ≈ 31912
修复后 cells ≈ 24289
```

下降的主要原因是去掉 exact division，改成 next-power-of-two shift denominator。

### 4.4 rtl_allbinary 复验

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

### 4.5 Erie strict lint

命令：

```bash
python /root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py --mode rtl rtl_dc/unibin_h60_core_dc.sv
```

结果：

```text
6 error(s), 2 warning(s)
```

主要原因仍是：

1. SystemVerilog `function`；
2. 参数化 `for` loop bound；
3. Erie strict 偏 Verilog-2001 生成风格。

解释：

这不影响当前 Verilator/Yosys 的 SystemVerilog RTL 验证，但如果要做 Erie strict 0-error handoff，需要把 `popcount32/consensus_score/exp2_approx_q8/ceil_log2_u32/gate_from_exp` 拆成 leaf module 或显式组合逻辑。

---

## 5. 仍未解决的边界

本轮没有解决以下问题：

### 5.1 还不是完整 head_dim vector datapath

当前接口仍是：

```text
in_k_value: scalar signed 8-bit
out_gated_k: scalar signed 16-bit
```

软件是：

```text
attn = k_orig * gate
```

其中 `k_orig` 是 token × head_dim 向量。因此当前 RTL 仍是 H60 row/gate/gated-scalar core，不是完整 attention output vector datapath。

### 5.2 还没有 PyTorch golden vector bit-accurate 闭环

虽然公式已经更贴近 deployment config，但仍需要导出 golden：

```text
Q event / K event
raw TX / SC
centered score
quantized score
exp numerator
row_sum
gate
gated output
```

然后用 RTL testbench 对逐 token 数值。

### 5.3 Erie strict 还未清零

当前 RTL 适合 SystemVerilog ASIC flow；若要 Erie strict 风格，需要下一步做结构化拆分。

### 5.4 SRAM wrapper 还未完成

Yosys 仍显示：

```text
Number of memories: 0
```

所以当前面积仍是寄存器展开后的逻辑面积，不是 SRAM macro 后面积。

---

## 6. 下一步建议

优先级建议：

1. 导出 PyTorch golden row，并做 RTL bit-accurate checker。
2. 把 scalar `in_k_value/out_gated_k` 扩展成 head_dim lane streaming。
3. 把 helper function 拆成 leaf modules，做 Erie strict 0-error 版本。
4. 把 row buffer 替换成 SRAM wrapper。
5. 再继续做 descriptor/controller/TTB issue shell。

当前状态可以描述为：

```text
P0 协议和主要 Shiftmax/score 口径已修；
模块级 SystemVerilog RTL 验证通过；
仍需 golden vector 和 vector datapath 才能声明软件网络等价。
```

