# Grok RTL 问题清单评估与修复结果

**日期**：2026-06-23  
**输入文档**：`docs/30_交给另一Agent的RTL问题清单.md`  
**本轮范围**：只处理低风险、可直接验证的问题；不在本轮强行展开完整 head_dim datapath、PyTorch golden vector、SRAM wrapper 或 descriptor controller。

---

## 1. 总体判断

Grok 清单大部分判断是有道理的，尤其是：

1. `rtl_allbinary` 与 `rtl_dc` score 语义分叉；
2. 缺 PyTorch golden vector；
3. `tb_dc` 没有覆盖 162-token row；
4. `unused_abs_delta & 1'b0`、`unused_score_reduce & 1'b0` 这类 lint hack 应清理；
5. Erie strict 仍未通过；
6. `shiftmax_int8_unit` 扩到 162 token 可能面积爆炸。

但其中有几项不适合本轮直接修完：

| 问题 | 本轮判断 |
---|---|
| PyTorch golden vector | 必须做，但需要软件导出脚本和新 checker，单独开任务 |
| head_dim vector datapath | 属于接口/微架构升级，不应混入小修 |
| Erie strict 0-error | 需要把 function 拆 leaf module，属于结构重构 |
| SRAM wrapper/controller/window 语义 | 系统级后续，不属于 H60 core P0 小修 |

---

## 2. 本轮已修

### 2.1 清理 `rtl_dc` lint hack

原代码：

```text
unused_abs_delta = ^{abs_delta[SCORE_W-1], abs_delta[2:0]};
exp2_approx_q8 = (unused_abs_delta & 1'b0) ? 16'd0 : ...
```

问题：

这是为了压 Verilator unused-bit warning 的恒 false 分支，不是干净 RTL。

修复后：

```text
int_shift  = abs_delta[SCORE_W-1:SCORE_FRAC]
frac_round = frac_idx + OR(low_fraction_bits)
frac_idx   = rounded fractional index
```

也就是把原先没用到的高位和低位都纳入 exp2 近似：

1. 高位参与 integer shift saturation；
2. 低位参与 fractional LUT index rounding；
3. 删除恒 false 分支。

### 2.2 扩展 `tb_dc` 到 162-token row

原 testbench：

```text
MAX_TOKENS = 8
```

修复后：

```text
MAX_TOKENS = 162
```

新增第三帧 synthetic directed test：

```text
cfg_n_tokens = 162
输入 162 个 token
最后一个 token in_last = 1
检查 perf_tokens_loaded = 162
检查 perf_issued_tokens = 162
检查 cumulative out_count = 169
检查 out_last 在 token_idx = 161
```

这覆盖了 Grok 清单里的 “TB 未覆盖 162-token window”。

### 2.3 同步 `rtl_allbinary` raw score 公式

原 `rtl_allbinary/binary_popcount_consensus.v`：

```text
TX = overlap
SC = overlap - mismatch
score = TX + mu * SC
```

修复后：

```text
same_zero = head_dim - q_active - k_active + overlap
TX        = (overlap + alpha0 * same_zero) / head_dim
SC        = overlap / head_dim
score     = TX + mu * SC
```

使用同样的 deployment 近似：

```text
SCORE_FRAC = 7
ALPHA0_Q8  = 5
MU_Q8      = 16
```

注意：

`rtl_allbinary` 是 token-level leaf，没有 row context，所以它仍然不实现 `center_scores`。完整 row centering 仍只在 `rtl_dc/unibin_h60_core_dc.sv` 中实现。

### 2.4 更新 `tb_allbinary` 期望值

测试样例：

```text
q_bits = 0b1011
k_bits = 0b1101
q_active = 3
k_active = 3
overlap = 2
mismatch = 2
same_zero = 28
```

新期望：

```text
tx_score = 10
sc_score = 8
fused_score = 11
```

这对应 Q7 定点 raw score。

### 2.5 清理 `unibin_h60_token_core` lint hack

原代码：

```text
unused_score_reduce = ^{tx_score_unused, sc_score_unused};
empty_token = ... | (unused_score_reduce & 1'b0)
```

修复后：

```text
tx_score_unused/sc_score_unused 显式声明为 unused wire
empty_token = ((q_bits | k_bits) == 0)
```

删除了恒 false 分支。

---

## 3. 本轮复验结果

### 3.1 `rtl_dc` directed simulation

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

当前 testbench 已包含：

1. 4-token 普通帧；
2. early `in_last` 帧；
3. `out_ready` backpressure；
4. 162-token synthetic row。

### 3.2 `rtl_dc` Verilator lint

命令：

```bash
./sim_dc/run_verilator_lint.sh
```

结果：

```text
PASS，无 warning 输出
```

### 3.3 `rtl_dc` Yosys synth/check

命令：

```bash
./sim_dc/run_yosys_synth.sh
```

结果：

```text
Found and reported 0 problems.
Number of memories: 0
Number of cells: 24313
```

### 3.4 `rtl_allbinary` 全量检查

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

剩余 warning：

```text
shiftmax_int8_unit: numerator memory is replaced with registers
```

这是原有 warning，和 Grok 清单第 8 条一致。本轮没有把 `shiftmax_int8_unit` 扩成 162-token 主线；后续建议以 `rtl_dc` 的 row Shiftmax 为主线。

### 3.5 Erie strict lint

`rtl_dc` 结果仍为：

```text
6 error(s), 2 warning(s)
```

主要原因：

1. SystemVerilog `function`；
2. 参数化 `for` loop bound；
3. Erie strict 偏 Verilog-2001 生成风格。

`rtl_allbinary/binary_popcount_consensus.v` 结果：

```text
1 error(s), 4 warning(s)
```

主要原因仍是参数化 loop bound 和 literal style。

解释：

这些不是 Verilator/Yosys 阻断项，但如果要 Erie strict 0-error，需要专门做 leaf-module/Verilog-2001 重构。

---

## 4. Grok 清单逐条状态

| 编号 | 问题 | 判断 | 本轮状态 |
|---:|---|---|---|
| 1 | `rtl_allbinary` 与 `rtl_dc` 语义分叉 | 有道理 | 已同步 token-level raw score；center 仍仅 `rtl_dc` 支持 |
| 2 | 缺 PyTorch golden vector | 有道理 | 未修，单独任务 |
| 3 | scalar K value 非 head_dim vector | 有道理 | 未修，属于接口升级 |
| 4 | TB 未覆盖 162-token window | 有道理 | 已修 |
| 5 | Erie strict lint 仍 6 error | 有道理 | 未修，需结构重构 |
| 6 | RTL lint hack 应清理 | 有道理 | 已修 `rtl_dc` 和 `unibin_h60_token_core` |
| 7 | 缺 SVA 断言 | 有道理 | 未修，建议下一步加 bind checker |
| 8 | `shiftmax_int8_unit` 综合 warning | 有道理 | 保留为 legacy 原型，主线用 `rtl_dc` |
| 9 | 无 SRAM wrapper | 有道理 | 后续 |
| 10 | 缺 descriptor/TTB shell | 有道理 | 后续 |
| 11 | 缺 Swin window 语义 | 有道理 | 后续 |
| 12 | `cfg_center_scores/cfg_alpha0` 无运行时配置 | 部分有道理 | 当前作为 compile-time parameter，后续可加 CSR |

---

## 5. 下一步建议

建议按这个顺序继续：

1. 做 PyTorch golden vector 导出和 RTL checker。
2. 增加 SVA bind checker，覆盖 ready-valid、early last、emit bound。
3. 决定是否需要 Erie strict 0-error；如果需要，拆 `rtl_dc` helper function 为 leaf modules。
4. 设计 head_dim lane streaming，把 scalar `k_value` 扩成向量/多拍 lane。
5. SRAM wrapper 和 descriptor controller。

当前状态：

```text
Grok 清单中的低风险 RTL/TB 问题已修；
模块级 SystemVerilog flow 通过；
软件 bit-accurate golden 和完整 vector datapath 仍未完成。
```

