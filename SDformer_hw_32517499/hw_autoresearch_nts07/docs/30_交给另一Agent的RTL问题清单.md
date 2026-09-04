# 交给另一 Agent 的 RTL 问题清单

日期：2026-06-23  
来源：两轮 skill 审阅 + P0 修复后独立复验

---

## P0 — 阻断「软件等价」声明

### 1. rtl_allbinary 与 rtl_dc 语义分叉（最重要）

现象：P0 只修了 rtl_dc/，rtl_allbinary/ 仍是旧公式。

| 能力 | rtl_dc | rtl_allbinary |
|------|--------|---------------|
| alpha0 * same_zero | 有 | 无 |
| head_dim 归一化 | 有 | 无 |
| center_scores | 有 | 无 |
| score 公式 | deployment 对齐 | 旧 proxy：overlap + mu*(overlap-mismatch) |

影响：sim_allbinary/run_all_checks.sh 仍 PASS，但验的是旧语义。

建议（二选一）：
- 方案 A：以 rtl_dc 为唯一软件对齐主线，文档标注 rtl_allbinary 为 legacy leaf
- 方案 B：把 rtl_dc 的 consensus_score 同步到 rtl_allbinary/binary_popcount_consensus.v，并更新 tb 期望值

相关文件：
- rtl_allbinary/binary_popcount_consensus.v
- rtl_allbinary/unibin_h60_token_core.v
- tb_allbinary/tb_unibin_h60_modules.v

---

### 2. 缺少 PyTorch golden vector 逐 token 对比

需要导出并对比（每个 token）：
- Q_bits, K_bits
- q_active, k_active, overlap, same_zero
- raw_TX, raw_SC, raw_score (Q7)
- row_mean, centered_score
- exp_numerator, row_sum, denom_shift
- gate, gated_k

建议任务：
1. 从 H60 forward 导出 1-2 个 window 的 golden JSON/CSV
2. tb_dc 或 Python checker 逐 token 对比 RTL
3. 记录已知近似：ALPHA0_Q8=5、fractional exp2 LUT、整数 row mean

---

### 3. 仍是 scalar k_value，非 head_dim 向量

软件：attn = k_orig * gate（k_orig 为 head_dim 向量）
RTL：in_k_value / out_gated_k 仅为 8-bit 标量

建议：设计 head_dim lane streaming 接口并更新 TB。

---

## P1 — 工程化

### 4. TB 未覆盖 162-token window

tb_dc 的 MAX_TOKENS=8，软件每 window 为 162 tokens。

建议：扩到 MAX_TOKENS=162，至少 1 个 synthetic directed test。

---

### 5. Erie strict lint 仍 6 error

NO_TASK_FUNCTION x5，FOR_CONST_BOUNDS x1。

建议：把 helper function 拆成 leaf module。

---

### 6. RTL lint hack 应清理

- rtl_dc/unibin_h60_core_dc.sv 第 162 行：unused_abs_delta & 1'b0
- rtl_allbinary/unibin_h60_token_core.v 第 44 行：unused_score_reduce & 1'b0

建议：删掉恒 false 分支。

---

### 7. 缺 SVA 断言

建议至少加：
- n_tokens_q <= MAX_TOKENS
- ST_EMIT 时 emit_idx_q < n_tokens_q
- early in_last 后 perf_tokens_loaded == n_tokens_q
- backpressure 下 out_gate/out_token_idx 不变

---

### 8. shiftmax_int8_unit 综合 warning

Yosys 将 numerator 展开为寄存器，扩到 162 tokens 可能面积爆炸。

建议：与 rtl_dc 统一 Shiftmax，或限制 MAX_TOKENS 并文档说明。

---

## P2 — 系统级（后续）

9. 无 SRAM wrapper（Yosys memories=0，面积为逻辑展开）
10. 缺 descriptor controller / TTB issue shell
11. 缺 Swin window pad/partition/reverse 语义
12. cfg_center_scores / cfg_alpha0 无运行时配置端口

---

## 建议执行顺序

1. 明确 rtl_allbinary vs rtl_dc 主线（方案 A 或 B）
2. PyTorch golden 导出 + RTL checker
3. 清理 lint hack + TB 扩到 162 tokens
4. head_dim vector datapath
5. Erie strict 拆 function
6. SVA + SRAM + controller（后续）

---

## 已确认无需再修

- in_last 早停协议（已修）
- center_scores（rtl_dc 已实现）
- Shiftmax ceil_log2 移位（rtl_dc 已实现）
- sim_dc/* 与 sim_allbinary/* 全套验证当前 PASS
- binary_atlif_state_unit 非软件主线（doc 25/26 已说明）