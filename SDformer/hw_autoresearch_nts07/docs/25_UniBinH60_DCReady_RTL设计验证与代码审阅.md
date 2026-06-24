# UniBin-H60 DC-Ready RTL 设计、验证与代码审阅

**日期**：2026-06-22  
**主线**：All-Binary NTS/H60  
**目标**：从模块烟测原型推进到可综合、可 lint、可仿真的 DC-ready 最小 H60 子系统。

## 1. 本轮结论

本轮已经完成一个 **DC-ready 最小 H60 子系统**：

```text
unibin_h60_core_dc
```

它不是完整芯片顶层，也不是最终 SRAM/NoC/descriptor controller，但已经具备以下特征：

1. 有同步时钟/复位；
2. 有 frame/window 级 `cfg_start`；
3. 有 ready-valid token 输入；
4. 有 ready-valid token 输出；
5. 内部保存 score、K value、K event、exp；
6. 实现 binary Q/K popcount consensus score；
7. 实现 INT8-ish Shiftmax gate scaffold；
8. 实现 gated-K 输出；
9. 输出性能计数器：loaded / empty / issued；
10. 通过 iverilog directed simulation、Verilator lint、Yosys synth/check。

因此当前状态可以描述为：

```text
可以进入 DC/Yosys 的模块级综合检查；
还不能作为最终 full accelerator 面积/功耗数。
```

## 2. 遵守的算法和数据流要求

### 2.1 all-binary 主线

当前 RTL 严格围绕 all-binary 设计：

```text
Q, K, activation event ∈ {0, 1}
```

不再保留主路径中的：

- ternary event；
- positive/negative dual rail；
- 2-bit event SRAM；
- mixed binary/ternary mode switch；
- opposite-polarity ternary compare。

### 2.2 H60/NTS 数据流

软件侧 H60 语义是：

```text
score = TX(Q, K) + μ * SC(Q, K)
gate  = Shiftmax(score)
out   = K * gate
```

硬件侧重排为更适合 binary event 的顺序：

```text
load token event
  ↓
popcount(q), popcount(k), popcount(q & k)
  ↓
mismatch = q_active + k_active - 2 * overlap
  ↓
fused_score = overlap + μ * (overlap - mismatch)
  ↓
row max / exp2 approximation / row sum
  ↓
INT8 gate
  ↓
gated-K output
```

这个重排保留 H60 的 score-gate-output 结构，但把 TX/SC 前端改成 binary overlap/count datapath，减少硬件复杂度。

### 2.3 TTB 和 skip

本轮 DC-ready core 没有实现完整外部 TTB scheduler，但接口和计数器已经支持 empty token/bundle 统计：

```text
empty = (Q | K) == 0
```

下一版可以把 `in_q_bits/in_k_bits` 的输入粒度扩成 TTB2，或在外层加 `ttb_issue_unit`：

```text
if TTB empty:
    skip H60 core issue
else:
    feed token bundle into H60 core
```

## 3. 借鉴论文和可写进 DATE 的 idea

### 3.1 FireFly-T

FireFly-T 的关键可迁移 idea 是 **binary attention engine + SRAM data manipulation**。它把 spiking transformer attention 单独做 binary engine，用 AND-PopCount 替代 dense MAC，并强调 SRAM byte-write/dataflow 对 attention 数据重排的重要性。

迁移到 UniBin-H60：

| FireFly-T | UniBin-H60 |
|---|---|
| binary attention engine | Binary H60 Consensus Engine |
| AND-PopCount | Q/K overlap popcount |
| SRAM byte-write data manipulation | 1-bit packed event SRAM / skip replay |
| dual-engine overlay | descriptor-driven H60 + event path |

DATE 可讲点：

> UniBin-H60 不是通用 binary attention overlay，而是针对 SDformerFlow 的 fixed all12 H60 encoder pattern，把 Q/K attention 前端约化为 overlap/active/mismatch popcount consensus engine。

参考：<https://arxiv.org/html/2505.12771v1>

### 3.2 Bishop / Token-Time Bundle

Bishop 的 TTB idea 是把 token-time workload 作为硬件调度单元，并根据 bundle 密度做调度。本项目 P0 profiling 显示 all-binary 的 TTB2 empty ratio 很高，尤其 S1/S2/S3。

迁移到 UniBin-H60：

| Bishop | UniBin-H60 |
|---|---|
| Token-Time Bundle | TTB1/TTB2 work issue unit |
| stratifier | empty/non-empty issue gate |
| heterogeneous cores | 第一版只做 skip，不做双核 |

DATE 可讲点：

> 我们把 TTB 简化为 H60 前端 issue gating，利用 all-binary 的高空 bundle 比例，避免复杂 dense/sparse 双核控制。

参考：<https://arxiv.org/html/2505.12281v1>

### 3.3 BESTformer

BESTformer 支持 binary event-driven transformer 的方向：1-bit 表示可降低存储/计算，但需要训练策略避免二值信息损失。

迁移到 UniBin-H60：

| BESTformer | UniBin-H60 |
|---|---|
| 1-bit event representation | 105 binary ATLIF |
| binary reduces memory/compute | 1-bit packed SRAM / binary popcount |
| binary may hurt accuracy | short fine-tune 后 AEE 1.4891 |

参考：<https://arxiv.org/html/2501.05904v1>

### 3.4 Xpikeformer / SSA

Xpikeformer/SSA 类工作强调用 binary attention 中的 AND 和加法替代乘法。UniBin-H60 采用同样的硬件方向，但保留 H60 的 Shiftmax gate 和 NTS score 结构。

参考：<https://arxiv.org/html/2408.08794v1>

## 4. RTL 文件

新增 DC-ready RTL：

```text
/root/private_data/work/SDformer/hw_autoresearch_nts07/rtl_dc/
├── unibin_h60_core_dc.sv
└── filelist.f
```

新增 testbench / scripts：

```text
/root/private_data/work/SDformer/hw_autoresearch_nts07/tb_dc/
└── tb_unibin_h60_core_dc.sv

/root/private_data/work/SDformer/hw_autoresearch_nts07/sim_dc/
├── run_iverilog_dc.sh
├── run_verilator_lint.sh
└── run_yosys_synth.sh
```

## 5. 模块接口

### 5.1 配置接口

| 信号 | 方向 | 位宽 | 说明 |
|---|---|---:|---|
| `clk_core` | in | 1 | 单时钟域 |
| `rst_n_core` | in | 1 | 同步使用的低有效复位 |
| `cfg_start` | in | 1 | 启动一个 window/token row |
| `cfg_n_tokens` | in | 8 | 当前 row token 数，最大 162 |
| `cfg_mu_q8` | in | 8 | `mu` 的 Q0.8 表示，当前部署推荐 `16=1/16` |
| `cfg_preserve_mean` | in | 1 | 是否保留 Shiftmax mean scaling |

### 5.2 输入 token stream

| 信号 | 方向 | 位宽 | 说明 |
|---|---|---:|---|
| `in_valid` | in | 1 | 输入 token 有效 |
| `in_ready` | out | 1 | core 可接收 |
| `in_last` | in | 1 | 当前 row 最后 token |
| `in_q_bits` | in | 32 | binary Q event vector |
| `in_k_bits` | in | 32 | binary K event vector |
| `in_k_value` | in | 8 signed | 简化 K carrier value |

### 5.3 输出 token stream

| 信号 | 方向 | 位宽 | 说明 |
|---|---|---:|---|
| `out_valid` | out | 1 | 输出 token 有效 |
| `out_ready` | in | 1 | 下游可接收 |
| `out_last` | out | 1 | 当前 row 最后输出 |
| `out_token_idx` | out | 8 | 输出 token index |
| `out_gate` | out | 8 | INT8 gate |
| `out_gated_k` | out | 16 signed | `K × gate` |

### 5.4 性能计数器

| 信号 | 说明 |
|---|---|
| `busy` | core 非 idle |
| `done` | 当前 row 完成 |
| `perf_tokens_loaded` | 输入 token 数 |
| `perf_empty_tokens` | `(Q|K)==0` 的空 token 数 |
| `perf_issued_tokens` | 输出 token 数 |

## 6. 验证流程

### 6.1 Directed simulation

命令：

```bash
cd /root/private_data/work/SDformer/hw_autoresearch_nts07
./sim_dc/run_iverilog_dc.sh
```

结果：

```text
PASS: unibin_h60_core_dc directed test passed
```

覆盖：

1. reset；
2. cfg_start；
3. 4-token 输入；
4. 一个 empty token；
5. score/gate/gated-K 输出；
6. out_last；
7. perf counter。

### 6.2 Verilator lint

命令：

```bash
./sim_dc/run_verilator_lint.sh
```

结果：0 输出，exit code 0。

含义：

- 无 Verilator lint error；
- 无 width warning；
- 无未使用信号 warning。

### 6.3 Yosys synth/check

命令：

```bash
./sim_dc/run_yosys_synth.sh
```

关键结果：

```text
Found and reported 0 problems.
Number of cells: 31912
Number of memories: 0
$_DFFE_PP_: 4860
$_MUX_: 7127
```

解释：

- Yosys 可综合并通过 `check`。
- 当前数组被综合成触发器和 mux，没有推断 memory macro。
- 这个结果只能作为 **logic sanity / synth-readiness**，不能作为最终 SRAM 面积。

## 7. 代码审阅结果

### Critical

无。

### Important

1. **当前 row buffer 没有 SRAM macro 化**
   - 位置：`score_mem_q`、`k_value_mem_q`、`k_event_mem_q`、`exp_mem_q`
   - 现象：Yosys 报 `Number of memories: 0`，说明被展开成 FF/mux。
   - 影响：不能用这个 Yosys cell count 当最终面积。
   - 处理：保留为 DC-ready logic prototype；下一步引入 SRAM wrapper 或 ASIC memory macro stub。

2. **Shiftmax 仍是近似 scaffold**
   - 位置：`exp2_approx_q8`、`gate_from_exp`
   - 现象：使用 power-of-two 近似和除法。
   - 影响：可综合，但后续要和 PyTorch deploy quant 做 bit-accurate 对齐。
   - 处理：当前 valid825 已证明 INT8 score/gate 近似可行；下一步把 gate reference vector 导出做 RTL bit-check。

3. **Erie Verilog-2001 lint 不通过**
   - 原因：该 helper 禁止 Verilog function；当前 RTL 是 SystemVerilog，按 `chip-design-rtl` 规则使用 function 封装 combinational datapath。
   - 影响：不影响 Verilator/Yosys/DC-ready SV 路线；如果需要 Verilog-2001 交付，需要把 function 拆成 leaf modules。

### Minor

1. `ST_FIND_MAX` 当前主要作为 pipeline boundary，row max 已在 LOAD 阶段更新。保留是为了后续 score centering / descriptor-dependent max，不是功能必需。
2. testbench 中 `$display/$finish` 位于 `always_ff`，iverilog 提示不可综合；这是 testbench，不影响 RTL。
3. 当前没有 SVA bind 文件；后续应补 ready-valid stable、eventual done、counter consistency 三类 property。

## 8. DATE 论文创新点写法

建议写成 5 个贡献：

1. **Algorithm-to-Hardware Binary Eventization**  
   通过 all-binary ATLIF fine-tune，把 SDformerFlow encoder 统一成 105-site binary event graph，避免 ternary/mixed datapath。

2. **UniBin-H60 Attention Core**  
   面向 all12 H60 encoder block 的共享 binary attention core，使用 overlap/active/mismatch popcount consensus score。

3. **TTB-Gated Work Issue**  
   基于 Token-Time Bundle 的 issue gating，把 P0 profiling 中的空 bundle 转换为可实现的 score/gated-K 跳过。

4. **INT8 Deployable Shiftmax Path**  
   H60 score/gate INT8 + `mu=1/16` valid825 几乎无损：AEE `1.4916` vs float `1.4891`。

5. **1-bit Packed Skip and Event SRAM**  
   all-binary 让 activation、Q/K tile、skip replay 都统一为 1-bit packed format，S0/S1/S2 skip 每样本约 `1.45 MB`。

## 9. 下一步

若目标是正式 DC：

1. 把内部 row buffer 改成 SRAM wrapper：
   - score SRAM；
   - K value SRAM；
   - K event SRAM；
   - exp/gate scratch SRAM。
2. 导出 PyTorch all-binary ep2 的一组 H60 token row golden vector；
3. 写 RTL bit-accurate checker，比较 fused score、gate、gated-K；
4. 增加 SVA；
5. 用 DC library 替代 Yosys generic gates；
6. 用 VCD/SAIF activity 做功耗估计。

当前可以对外表述：

```text
We implemented and verified a synthesizable SystemVerilog prototype of the
UniBin-H60 core, including ready-valid token streaming, binary popcount
consensus scoring, INT8 Shiftmax-style gating, gated-K emission, and performance
counters. The prototype passes directed simulation, Verilator lint, and Yosys
synthesis/check.
```
