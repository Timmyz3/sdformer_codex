# DC 发文标准差距与 FAPS 主线风险评估

**日期**：2026-06-28  
**对象**：AllBinary / UniBin-H60 当前 RTL 与可能切换到 FAPS allbinary 的硬件主线  
**结论级别**：架构与综合前审阅，不是 DC signoff

---

## 1. 一句话结论

现在的硬件**不能直接拿去做 DC 然后发文章**。

更准确地说：

```text
当前状态 = 可仿真、可 Verilator lint、可 Yosys generic synth 的 H60 模块级 SystemVerilog 原型
还不是 = 可投稿主结果使用的 ASIC/DC signoff accelerator
```

如果论文只想放一个“模块级 RTL 原型证明可实现”，当前结果可以作为 early prototype。

如果论文要按 DATE/ICCAD/TCAD 风格写“我们设计了一个硬件加速器，并给出面积、功耗、吞吐、能效”，当前还缺以下硬证据：

1. Synopsys DC 或等价商业综合结果；
2. 标准单元库、工艺节点、频率、SDC 约束；
3. SRAM macro 或 CACTI/SRAM compiler 口径；
4. PrimeTime PX / SAIF / VCD 活动功耗；
5. RTL 与 PyTorch attention golden vector 的 bit-accurate 或误差界验证；
6. 完整 workload 的 cycle/throughput 模型；
7. 模块级面积/功耗 breakdown；
8. 若主线换 FAPS，还缺 FAPS 的 RTL、量化、profiling、精度闭环。

---

## 2. 当前硬件到底到了哪一步

已有结果来自 `docs/32_UniBinH60_RTL_Skill流程详细审阅.md`：

| 项 | 当前状态 | 能不能支撑发文主表 |
|---|---|---|
| `iverilog` directed simulation | 通过 | 只能证明基本功能跑通 |
| Verilator lint | 通过 | 只能证明常规 RTL 风险较低 |
| Yosys synth/check | 通过，`0 problems` | 只能证明 generic 可综合 |
| Yosys cells | `24313` | 只能作为趋势，不是 ASIC 面积 |
| Yosys memories | `0` | 说明 row buffer 被寄存器/逻辑展开，不是 SRAM |
| Erie strict lint | 未通过 | 不满足更严格 Verilog-2001 handoff |
| PyTorch golden checker | 未完成 | 不能声称软件等价 |
| SRAM wrapper | 无 | 不能报真实片上存储面积/功耗 |
| Descriptor controller | 无 | 不能说是完整 accelerator |
| Full vector gated-K | 无，当前偏 scalar/模块级 | 不能覆盖完整 head_dim 输出路径 |
| DC/PT/LEC | 未跑 | 不能作为 ASIC signoff 结果 |

本机当前 PATH 下也没有发现 `dc_shell`，所以现在不能直接在本环境声称“可以跑 DC”。

---

## 3. 其他论文一般做到什么标准

### 3.1 ASIC 类论文的最低硬件评估口径

以最近 spiking transformer accelerator 论文为参考，ASIC 类工作通常会给：

| 论文类型 | 常见交付物 |
|---|---|
| ASIC accelerator | Verilog/RTL、工艺节点、DC 综合、PrimeTime/PrimeTime PX 功耗、频率、面积、SRAM 容量、吞吐、能效 |
| FPGA accelerator | Vivado 实现、板卡型号、频率、LUT/REG/BRAM/DSP、功耗、FPS/GOPS/GOPS/W |
| 架构探索/3D integration | synthesized layout / floorplan / 2D vs 3D 对比、wire/power/latency/area 改善 |
| 算法硬件协同 | 算法精度 + 硬件复杂度 + 部署定点/稀疏性统计 |

典型参考：

1. **Hardware Efficient Accelerator for Spiking Transformer with Reconfigurable Parallel Time Step Computing**  
   该工作明确写了：Verilog 实现，Synopsys Design Compiler，TSMC 28nm，PrimeTime PX 功耗，500MHz，198.46K gates，139.25KB SRAM，90.153mW，46.72 FPS，3456 GSOPS。  
   参考：<https://arxiv.org/html/2503.19643v1>

2. **FireFly-T**  
   该工作是 FPGA overlay，但评估表包含模型/数据集/精度/FPS/GOP/s/GOP/s/W/DSP efficiency/LUT/BRAM/DSP/frequency/device，并且给 resource breakdown。  
   参考：<https://arxiv.org/html/2505.12771v1>

3. **Spiking Transformer Hardware Accelerators in 3D Integration**  
   该工作关注 3D integration，但仍给出 2D/3D layout、有效频率、面积、功耗、memory access latency/power 改善。  
   参考：<https://gtcad.gatech.edu/www/papers/iccad24-boxun.pdf>

4. **Spike-driven Transformer**  
   算法论文也会明确强调 event-driven、binary spike communication、mask/addition 替代乘法、residual 重排以保持 binary spike signals。硬件论文若借用这个叙事，也必须对应到数据流和模块接口。  
   参考：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html>

### 3.2 对我们的直接要求

如果我们要达到上述标准，DATE 主表至少需要这些列：

| 表 | 必须列 |
|---|---|
| 算法表 | AEE、AAE、PE1、PE2、outlier、spikes、energy proxy、valid split |
| 量化表 | float vs INT8 score/gate vs pow2 mu，误差 |
| RTL 表 | 模块名、功能、位宽、是否可综合、是否 golden checked |
| ASIC 表 | 工艺、频率、面积、功耗、SRAM、吞吐、能效 |
| Breakdown 表 | Score engine、Shiftmax、ATLIF、event SRAM、controller、skip buffer |
| 数据流表 | 每 stage tokens/head/head_dim/window、buffer bytes、TTB skip ratio |

当前我们只有算法表、部分量化表、P0 profiling 表、Yosys generic RTL 表。ASIC 表和完整 breakdown 还没有。

---

## 4. 为什么不能现在直接 DC

### 4.1 RTL 还不是 DC handoff 形态

当前 `rtl_dc/unibin_h60_core_dc.sv` 仍包含：

1. function-heavy datapath；
2. 组合除法；
3. 内部数组 buffer；
4. row buffer 未 SRAM macro 化；
5. scalar K value/gated output；
6. control/datapath 混在同一模块；
7. 缺少 SDC；
8. 缺少 library / corner / operating condition；
9. 缺少 LEC 对照；
10. 缺少 SAIF/VCD 活动功耗口径。

这些问题不一定阻止 DC 解析，但会导致结果没有论文可信度。

### 4.2 `memories=0` 是硬伤

Yosys 报：

```text
Number of memories: 0
Number of cells: 24313
```

这说明 token row buffer、score buffer、K buffer、exp buffer 都没有作为真实 SRAM 宏来评估。  
若拿这个结果报面积，会把存储和控制的真实代价讲错。

对于我们的网络，存储不是边缘项：

| 项 | 已知 profiling |
|---|---|
| S0/S1/S2 pre-downsample skip | 1-bit packed 后约 `1.45 MB/sample` |
| S3 final retained output | 1-bit packed 后约 `0.10 MB/sample` |
| H60 row token | 162 tokens/window/head |
| Q/K activity | 很低，适合 event gating |

所以必须单独建 event SRAM / row buffer / skip SRAM 口径。

### 4.3 还没有软件等价验证

当前 RTL 近似了 all-binary H60：

```text
score = TX + mu * SC
center over token row
Shiftmax
gate * K
```

但还没有做到：

```text
PyTorch q_orig/k_orig row
→ 导出 golden score/gate/output
→ RTL 输入同一 row
→ 比较 score/gate/output 误差
```

没有这个，不能在论文中说 RTL faithfully implements the proposed attention。

---

## 5. FAPS allbinary 改线对硬件意味着什么

### 5.1 总体数据流大体不变

如果主线从 H60/NTS 改成 FAPS allbinary，外层数据流基本不变：

```text
DSEC event voxel
→ patch embedding
→ 12 个 Swin encoder block
→ 每 block 内 attention score/gate/output
→ residual ADD
→ MLP
→ residual ADD
→ S0/S1/S2 downsample
→ skip buffer
→ decoder
→ flow output
```

也就是说，以下硬件模块仍可复用：

| 模块 | H60 | FAPS | 是否复用 |
|---|---|---|---|
| window/row loader | yes | yes | 可复用 |
| packed 1-bit Q/K SRAM | yes | yes | 可复用 |
| token row controller | yes | yes | 可复用 |
| row mean/center | yes | yes | 可复用 |
| Shiftmax gate | yes | yes | 可复用 |
| gated-K output | yes | yes | 可复用 |
| TTB skip front-end | yes | yes | 可复用 |
| skip SRAM | yes | yes | 可复用 |

变化集中在 **attention block 内部的 score engine**。

### 5.2 FAPS score engine 比 H60 更复杂

代码位置：

```text
neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/bsa_attention.py
```

FAPS 主要函数：

```text
_faps_dyadic_channel_score()
_faps_sparse_k_magnitude()
_faps_flow_aligned_token_scores()
```

FAPS 做的事：

```text
q_event, k_event = sign(q_orig), sign(k_orig)
若 directional_channels_enabled:
    head_dim 分成 x/y 两半
    score_x = dyadic(Q_x, K_x)
    score_y = dyadic(Q_y, K_y)
    score = mean/sum(score_x, score_y) 或加入 |score_x-score_y| 惩罚
可选:
    score += sparse 2-bit K_mag correction
然后:
    center
    Shiftmax
    gate * K
```

对 all-binary 来说，`opposite` 基本为 0，但 FAPS 仍比 H60 多：

1. x/y half-head demux；
2. 两路 dyadic score；
3. same_nonzero / same_zero / single_active 加权；
4. 可选 `flow_disagreement_gamma`；
5. 可选 K magnitude 2-bit 旁路；
6. 若只在部分 block 使用，还会重新引入 mixed mode controller。

### 5.3 FAPS 是否更适合硬件

目前不能下结论说 FAPS 更适合硬件。  
从硬件复杂度看：

```text
TX < H60/NTS < FAPS no-Kmag < FAPS + Kmag
```

从当前证据完整度看：

```text
AllBinary H60/NTS > TX > FAPS
```

原因：

| 维度 | H60/NTS | FAPS |
|---|---|---|
| all12 full encoder | 已验证 | 需要确认最终主线 |
| valid825 | 已有 AEE `1.4891` |
| INT8 score/gate | 已验证 AEE `1.4916` | 未见完整部署验证 |
| P0 profiling | 已有 H60/TTB/ATLIF/skip | 需要补 FAPS 专项 |
| RTL | 已有 H60 prototype | 需要新写 score engine |
| DC readiness | 未完成 | 更未完成 |
| 论文故事 | 统一二值事件 + H60 consensus | 光流方向一致性更好讲，但硬件证据少 |

因此建议：

```text
论文主线暂时不要从 H60 直接切到 FAPS。
FAPS 可以作为“可插拔 score engine”或“下一代 flow-aware score plugin”。
只有当 FAPS allbinary 在 valid825 上明显优于 H60，且 no-Kmag/int8/全 all12 成立，才值得改成主线。
```

---

## 6. 如果主线可能改良，硬件架构应怎样设计

不要把硬件写死成 `H60-only core`。  
建议抽象成：

```text
UniBin Attention Row Engine
├── Row Loader / Event Buffer
├── Score Plugin
│   ├── TX score
│   ├── H60/NTS score
│   ├── FAPS score
│   └── future score
├── Center / Normalize
├── Shiftmax Gate
├── Gated-K Output
└── Perf Counter
```

这样论文里可以说：

```text
外层数据流固定为 all-binary event row processing；
不同算法只替换 score plugin；
H60 是当前 tape-in style 主实现；
FAPS 是 flow-aware score plugin，可在相同 loader/Shiftmax/gated-K 上复用。
```

### 6.1 推荐接口

Score plugin 输入：

```text
q_bits[HEAD_DIM-1:0]
k_bits[HEAD_DIM-1:0]
cfg_mode
cfg_mu
cfg_alpha0
cfg_faps_weights
cfg_enable_kmag
optional k_margin_bits
```

Score plugin 输出：

```text
score_q7[SCORE_W-1:0]
active_count
empty_token
debug counters
```

共用后端：

```text
score_q7 row[0:161]
→ row mean center
→ max / exp2 approximation
→ power-of-two Shiftmax
→ gate_q8
→ gated-K
```

### 6.2 DATE 叙事应避免的坑

不要写：

```text
我们已经有完整 ASIC accelerator。
我们可以直接报告最终面积功耗。
FAPS 比 H60 更硬件友好。
Shiftmax 实现了 token pruning。
```

可以写：

```text
我们提出一个 all-binary event row-engine，并用 H60/NTS 验证主线；
该 row-engine 允许 score plugin 替换，FAPS 只改变 score 前端；
事件稀疏、TTB skip、1-bit SRAM 是主要硬件收益；
INT8 score/gate 和 pow2 mu 证明 attention gate 可部署。
```

---

## 7. 达到发文/DC 标准的最小补齐清单

### P0：必须补，否则不能发硬件主结果

1. **PyTorch golden row export**
   - 导出 `q_bits/k_bits/k_value/score/gate/out`。
   - 至少覆盖 S0/S1/S2/S3，各取多个 window/head。

2. **RTL golden checker**
   - RTL 输入 golden row。
   - 比较 score/gate/output。
   - 给误差阈值，例如 gate LSB error、output MAE。

3. **Score plugin 化**
   - 拆出 `score_h60_q7`。
   - 若 FAPS 可能成为主线，同时写 `score_faps_q7`。
   - 后端 `center + Shiftmax + gated-K` 共用。

4. **Full vector gated-K**
   - 当前 scalar K 不够。
   - 需要至少支持 `HEAD_DIM=32` 的 lane/vector 输出，或明确采用 lane-serial schedule 并给 cycle 模型。

5. **SRAM 口径**
   - row buffer：score/k/gate 临时存储。
   - event SRAM：1-bit packed Q/K/activation。
   - skip SRAM：S0/S1/S2/S3 生命周期。
   - 若无 SRAM compiler，至少 CACTI + 明确 node 假设。

6. **cycle model**
   - 每 window/head/token 多少 cycle。
   - 每 stage/block/frame 总 cycle。
   - TTB2 skip 后 cycle saving。

7. **DC-ready RTL release**
   - 去掉 function-heavy style。
   - 去掉组合除法或改成常数 shift/multiply/LUT。
   - 明确 SDC。
   - 明确 clock/reset/IO delay。

### P1：用于把文章讲扎实

1. H60 allbinary full valid825 P0 profiling，而不仅是 valid40。
2. FAPS allbinary 的 valid825、INT8、profiling。
3. H60 vs FAPS vs TX 的 score engine area/cycle 对比。
4. TTB1/TTB2/TTB4 skip ratio 与实际 cycle saving 对齐。
5. ATLIF 105 vs 93 coverage 列表。
6. energy proxy 与 RTL activity power 的桥接说明。
7. 对比 FireFly-T / SpikeTA / Spike-IAND-Former 的表格。

---

## 8. 推荐下一步执行顺序

我建议不要马上做全 accelerator DC。按这个顺序推进：

1. 固定一个“可插拔 score plugin”的 RTL 结构。
2. 先把 H60 golden checker 做到通过。
3. 若 FAPS allbinary 结果真的更好，再加 FAPS score plugin，不改外层数据流。
4. 加 full vector 或 lane-serial gated-K 输出路径。
5. 做 cycle model 和 SRAM model。
6. 生成 DC release RTL + SDC。
7. 找到可用 DC/library 后跑：

```text
read_verilog
read_sdc
compile_ultra
report_timing
report_area
report_power
write netlist
LEC / formal equivalence
```

8. 最后再写 DATE 主硬件表。

---

## 9. 当前可投稿故事的建议定位

如果投稿时间紧，建议主线这样写：

```text
我们不是声称已经 tapeout。
我们提出一个面向 optical-flow spiking transformer 的 all-binary event attention accelerator architecture。
在软件上验证 all-binary + H60/NTS 几乎保持精度并大幅降低 spikes/energy proxy。
在硬件上给出可综合 RTL 原型、模块级仿真、定点部署、数据流与 PPA 模型。
后续或附录补 DC synthesis。
```

如果要更像硬件 DATE 正文，必须补齐 DC/PT/面积功耗表。

---

## 10. 最终判断

| 问题 | 回答 |
|---|---|
| 现在能不能直接 DC？ | 不能直接作为论文级 DC。可以尝试 DC 解析，但结果不可信，因为缺 SRAM、SDC、golden、release RTL。 |
| 现在能不能直接发文章？ | 不能按“完整 ASIC accelerator”发。可以按“架构 + RTL prototype + profiling”发，但说服力弱。 |
| FAPS allbinary 会不会推翻数据流？ | 不会。外层数据流基本不变，变的是 attention score plugin。 |
| FAPS 更硬件友好吗？ | 目前证据不足。FAPS 方向故事更强，但硬件复杂度和验证缺口都大于 H60。 |
| 最稳主线是什么？ | AllBinary H60/NTS 继续做主线，RTL 架构预留 FAPS score plugin。 |
| 下一步最该做什么？ | PyTorch golden row export + RTL checker + score plugin 化 + SRAM/cycle model。 |

