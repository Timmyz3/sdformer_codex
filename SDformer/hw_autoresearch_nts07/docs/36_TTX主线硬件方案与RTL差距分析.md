# TTX 主线硬件方案与 RTL 差距分析

**日期**：2026-07-09  
**角色**：硬件架构 / RTL / DATE 论文硬件线（只读审阅，本轮未改代码）  
**硬件主线**：**TTX allbinary**（不是 H60/NTS 全量 SC，也不是 FAPS）  
**关联文档**：`docs/34`（DC 差距）、`docs/35`（TTX 设计草案）、`docs/26`（端到端数据流）、`docs/32/29`（H60 RTL 审阅）

---

## 0. 一句话结论

| 判断 | 结论 |
|------|------|
| 算法/硬件语义是否可定义 | **可以**。TTX = all-binary ATLIF + `mode=h60` + **`bipolar_mu=0`** + **`k_magnitude_alpha=0`** + **no-carrier gated-K** |
| 是否已有可迁移 H60 RTL 壳 | **有**。`unibin_h60_core_dc.sv` 在 `cfg_mu_q8=0` 时语义上就是 TTX score |
| 是否 DC-ready / 可报 ASIC 主表 | **否**。缺 SRAM macro、SDC、DC/PT、golden bit-accurate、vector K、descriptor 顶层 |
| 是否可发 DATE 硬件故事 | **可写架构 + prototype + 协同证据**；**不可**把 Yosys cells 写成最终面积，也**不可**声称软件 bit-accurate 等价 |
| 本轮动作 | **只读审阅 + 方案与差距清单**；未改 RTL |

证据分级约定（全文统一）：

```text
[设想]     架构设想，尚无直接实验/RTL 支撑
[prof]     已有 profiling / valid825 / deploy 量化支撑
[rtl]      已有 RTL 模块与 iverilog/Verilator/Yosys 支撑
[综合/仿]  已有仿真/Yosys generic 支撑（≠ DC signoff）
[待补]     发文/DC 前必须补的验证
```

---

## 1. TTX 主线硬件语义

### 1.1 软件正式定义

仓库配置 note（TTX train / deploy）：

```text
TTX = all-binary ATLIFPSN + H60 TX-only selector, mu=0, no SC/Kmag
```

| 字段 | TTX 取值 | 来源 |
|------|----------|------|
| ATLIF `output_mode` | `binary` | train/deploy yml |
| attention `mode` | `h60`（框架复用） | yml + `bsa_attention.py` |
| `bipolar_mu` | **0.0** | yml；SC 支路乘子为零 |
| `k_magnitude_alpha` | **0.0** | yml；无 K_mag 旁路 |
| `center_scores` | `true` | yml |
| `preserve_mean` | `true` | yml |
| `alpha0` | `0.02`（RTL `ALPHA0_Q8=5` ≈ 5/256） | yml / RTL |
| carrier | **无** | h60 分支：`attn = k_orig * gate` |
| 部署量化 | score INT8 / gate INT8 已跑通 | deploy pipeline |

配置路径：

```text
训练：
  date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml
部署：
  date11_ttx_ep2_deploy_{float_ref,score_int8,score_int8_gate_int8}.yml
软件实现：
  bsa_attention.py → mode in {"h60", "tx_sc_k_mag_no_carrier_shiftmax"}
```

### 1.2 Attention block 输入 / 输出（硬件视角）

```text
输入（每个 token / 每个 head 的一条 Q–K 对）:
  q_bits[HEAD_DIM-1:0]   : 1-bit packed binary event（ATLIF 后 sign/eventize）
  k_bits[HEAD_DIM-1:0]   : 1-bit packed binary event
  k_value[DATA_W-1:0]    : threshold 域幅度（用于 gate 后恢复；可与 event 分离存储）

中间:
  tx_score  = (overlap + α₀ · same_zero) / head_dim
  sc_score  = overlap / head_dim          ← TTX 中 μ=0，硬件可不算或恒丢弃
  score     = tx_score + μ · sc_score     ← TTX: score = tx_score
  score_c   = score - mean_token(score)   ← row-wise center
  gate      = Shiftmax(score_c) · N       ← preserve_mean
  out       = k_value * gate              ← gated-K；无 carrier

输出:
  out_gate, out_gated_k（理想：head_dim vector；现状 RTL：scalar k_value）
```

**结论（语义）**

| 问题 | 答案 | 证据级 |
|------|------|--------|
| 输入是 Q/K/V 还是 Q/K/event？ | **Q/K event bits + threshold-valued K**；不是 dense V | [prof]+软件 |
| 是否仍 allbinary Q/K event？ | **是** | [prof] |
| score/gate 怎么算？ | **TX popcount → center → Shiftmax → ×K** | 软件 + [rtl] 近似 |
| 是否需要 Shiftmax？ | **需要**（TTX 未改 gate 后端） | 软件 |
| TX/SC/FAPS 哪些可去掉？ | **SC 融合、K_mag、carrier、FAPS x/y 拆分全可去掉** | 配置 μ=0/kmag=0 |
| gate 如何作用？ | **直接乘 K（no-carrier）** | 软件 h60 |
| row-wise center？ | **需要**（`center_scores=true`） | 软件 + [rtl] |
| INT8 / pow2 / LUT？ | score/gate INT8 部署已验证；RTL 用 Q7 + exp2 LUT/shift | [prof]+[rtl] |

### 1.3 与 H60 / 旧 TX / FAPS 的硬件语义差

| 维度 | **TTX（主线）** | 全量 H60/NTS | 旧 TX (H18) | FAPS |
|------|-----------------|--------------|-------------|------|
| score 前端 | **仅 TX popcount** | TX + μ·SC | TX | x/y dyadic + 可选 K_mag |
| μ 电路 | **tie-off 0 / 删除** | 需要 | 无 | 无 SC-μ |
| carrier | 无 | 无 | 有/视版本 | 无 |
| Shiftmax + gated-K | 有 | 有 | 有 | 有 |
| valid825 AEE | **1.5020** ep2 | **1.4891** | ~1.508 | 证据弱于 TTX |
| energy_uj | **20521** | 21046 | ~20011 | — |
| RTL 现状 | H60 壳 + μ=0 可覆盖 | 有原型 | 可视为 TTX 子集 | **无** |

**算法精度 [prof]**（valid825）：

| 方案 | AEE↓ | spikes | energy_uj |
|------|------|--------|-----------|
| TTX ep2 (float) | **1.5020** | 23.24G | 20521 |
| TTX score INT8 | **1.4971** | 23.24G | 20524 |
| TTX score+gate INT8 | **1.5003** | 23.25G | 20527 |
| NTS/H60 ft ep2（对照） | 1.4891 | 23.82G | 21046 |

> 部署量化几乎无损，甚至 score INT8 略优——**强烈支持** TTX 走 INT8 score/gate 硬件路径。  
> 来源：`date11_ttx_deploy_quant_full825_20260629_220531/pipeline.log`。

### 1.4 不能混淆的点

1. **TTX ≠ 旧 TX 模块名**：TTX 跑在 **h60 no-carrier 数据通路** 上，只是 **μ=0**；旧 TX 可能有 carrier 语义，论文应用 TTX 命名。  
2. **TTX ≠ 无 Shiftmax**：去掉的是 SC，不是 gate 后端。  
3. **PyTorch 12 个 attention block ≠ 12 份引擎实例**：硬件应是 **1 个 row engine + descriptor 分时**。  
4. **H60 profiling 可近似支撑 TTX 的 sparsity/TTB 叙事**，但 **TTB empty ratio 应以 TTX ckpt 再测一版** 才最干净（当前 [prof] 主要来自 NTS/H60 allbinary valid40）。

---

## 2. TTX 端到端硬件数据流

### 2.1 系统数据流（总体不变，变在 attention 内）

严格按 `docs/26` 的软件真实路径，仅把 block 内 attention 记为 **TTX**：

```text
DSEC voxel [B,10,2,H,W]
  → Patch Embed（dense/mixed，非 TTX 核心）
  → S0–S3 Spiking Swin encoder（depths 2/2/6/2 = 12 blocks）
       每 block:
         Norm → ATLIFPSN(binary eventize)
              → TTX Attn (TX score → center → Shiftmax → gated-K)
              → residual ADD
              → Norm → MLP → residual ADD
       S0/S1/S2 后 Patch Merging downsample
       pre-downsample skip → 1-bit skip buffer（DDR/cache）
  → Bottleneck MS residual
  → Decoder（encoder skip concat + pred feedback）
  → multi-scale flow head
```

**TTX 只替换** encoder attention 的 **score/gate 前端公式**；residual / MLP / downsample / skip / decoder **外层不变**。

### 2.2 推荐硬件引擎映射（复用 + descriptor，不是 module 计数）

```text
Input event / feature
  → [①] binary ATLIF encode cluster（1 物理集群，105 logical site 时分）
  → [②] 1-bit packed event SRAM
  → [③] TTB2 work-issue gate（empty bundle 不发分）
  → [④] TTX row/window loader
  → [⑤] TTX score engine（TX-only popcount）
  → [⑥] row center + Shiftmax token gate
  → [⑦] gated-K（理想 head_dim vector）
  → [⑧] residual / MLP / downsample Sparse-MAC
  → [⑨] 1-bit skip SRAM → decoder replay
  → flow
```

| 物理引擎 | 软件对应量 | 复用方式 | 证据级 |
|----------|------------|----------|--------|
| ATLIF cluster ×1 | ~105 wrapper / ~93 forward | descriptor 参数切换 | [prof]+部分 [rtl] |
| TTX row engine ×1 | 12 Swin attention | layer descriptor 分时 | [设想]+H60 shell [rtl] |
| Sparse MAC ×1 | MLP/DS/部分 conv | 分时 | [设想] |
| TTB2 issue ×1 | 每 window/head bundle | OR-empty 检测 | 单元 [rtl] + [prof] |
| Skip / event SRAM | stage skip + working set | 1-bit pack | [prof] 字节估算；无 macro |

### 2.3 存储与 traffic 故事（论文主收益）

| 项 | 数量级 | 证据级 |
|----|--------|--------|
| S0/S1/S2 pre-ds skip 1-bit | **~1.45 MB/sample** | [prof] `docs/23` |
| S3 retained 1-bit | ~0.10 MB/sample | [prof] |
| Q/K activity | 0.03%–5% 级（stage 相关） | [prof] NTS allbinary |
| TTB2 empty | S0≈28%，S1≈**74%**，S2/S3≈63–64% | [prof] NTS allbinary |
| Shiftmax top1 mass | ~0.006（接近均匀） | [prof] → **不能**讲强 token pruning |

**硬件收益应围绕**：

1. allbinary **1-bit event SRAM**（相对 FP16 16×，相对 2-bit 再减半）；  
2. **低 Q/K 活性** → event-gated popcount；  
3. **TTB work-issue gating**（不算/不读/不写，非语义 pruning）；  
4. **单 row engine + descriptor** 服务 12 block；  
5. TTX 相对 H60 **删 SC/μ**，score 前端更短；  
6. INT8 score/gate 部署几乎无损。

---

## 3. H60 旧 RTL：可复用 / 需改写

### 3.1 现有文件清单（历史原型，命名仍为 H60）

| 路径 | 角色 | 对 TTX |
|------|------|--------|
| `rtl_dc/unibin_h60_core_dc.sv` | 行级 core：LOAD→center→max→exp→EMIT | **短期主壳**：`cfg_mu_q8=0` = TTX |
| `rtl_allbinary/binary_popcount_consensus.v` | TX/SC/fused score | **复用 TX 输出**；可删 SC 面积 |
| `rtl_allbinary/shiftmax_int8_unit.v` | 组合 Shiftmax scaffold | 公式可参考；与 dc core 有两套近似 |
| `rtl_allbinary/gated_k_unit.v` | scalar gate×K | 复用；需扩到 vector |
| `rtl_allbinary/ttb_skip_unit.v` | empty bundle | 直接复用 |
| `rtl_allbinary/binary_atlif_*.v` | ATLIF 单元 scaffold | 可迁移思路 |
| `rtl_allbinary/unibin_h60_token_core.v` | score+gated_k 组合 | μ=0 即 TTX token 路径 |
| `tb_dc` / `tb_allbinary` | directed TB | 需加 TTX golden / μ=0 case |
| `sim_dc/*` `sim_allbinary/*` | iverilog / verilator / yosys | 继续用；增 TTX 脚本 |

### 3.2 迁移决策矩阵

| 模块/能力 | 迁移决策 | 说明 |
|-----------|----------|------|
| ready-valid + FSM 五态 | **复用** | IDLE/LOAD/FIND_MAX/SUM_EXP/EMIT/DONE |
| row score/k/exp buffer 框架 | **复用结构** | 内容仍需 SRAM 化 |
| popcount overlap / same_zero | **复用** | TTX 核心算术 |
| SC + μ 融合 | **删除或 tie-off** | TTX 默认 μ=0；中期硬删降面积 |
| Shiftmax exp2 + pow2 denom | **复用思路** | 需与软件/golden 对齐误差界 |
| gated-K | **复用 scalar，重写 vector** | 现状 `in_k_value` 标量是 P1 缺口 |
| TTB skip unit | **复用** | 接 issue 前端即可 |
| ATLIF units | **部分复用** | 缺 cluster/descriptor |
| FAPS x/y demux | **不实现主线** | 仅对照 plugin |
| descriptor controller / top | **新写** | 现无 |
| PyTorch golden export/checker | **新写** | 现无 |

### 3.3 H60 core 与 TTX 的精确关系

`unibin_h60_core_dc.sv` 中：

```text
score = TX + mu * SC   （consensus_score function）
```

当 `cfg_mu_q8 = 0`：

```text
mu_sc_q16 = 0 → score = TX   ≡ TTX
```

因此：

| 说法 | 是否成立 |
|------|----------|
| “H60 RTL 在 μ=0 时实现 TTX score 语义” | **成立（架构/RTL 公式级）** [rtl] |
| “已证明与 TTX PyTorch bit-accurate” | **不成立** [待补] |
| “已是 TTX 专用、面积最优的 tape-in RTL” | **不成立**（SC 逻辑仍在，命名仍是 H60） |
| “可报 DC 面积/功耗” | **不成立** |

### 3.4 RTL 质量审阅（TTX 视角，延续 docs/32）

| 级别 | 问题 | 位置 | 对 TTX/DC 影响 |
|------|------|------|----------------|
| **P0（发文等价）** | 无 PyTorch golden row export / checker | 全局 | 不能写 software-equivalent |
| **P0** | 无 TTX 专项 TB（固定 μ=0 + golden 比对） | tb_dc | 不能把 H60 默认 μ=16 当 TTX 验收 |
| **P1（DC 可信）** | 组合除法 `/ HEAD_DIM`、`/ n_tokens` | consensus_score, score_mean | 时序/PPA 差；应 shift/乘法倒数 |
| **P1** | function-heavy datapath 单模块 | core_dc | 不适合 timing closure |
| **P1** | 内部数组 → Yosys **memories=0** | score/k/exp mem | **不能报真实面积** |
| **P1** | scalar K / 无 head_dim vector | ports | 吞吐与软件不对齐 |
| **P1** | 无 SDC / 无工艺库 / 无 dc_shell | 环境 | 本机无 DC |
| **P2** | SC 逻辑对 TTX 冗余 | consensus | 面积浪费 ~10–18% 量级 [设想] |
| **P2** | `cfg_n_tokens=0`→MAX_TOKENS | core | 协议需文档化 |
| **P2** | `shiftmax_int8_unit` 与 core_dc 两套 exp 近似不一致 | rtl_allbinary vs rtl_dc | golden 前必须统一 |
| **P2** | Erie strict / Verilog-2001 handoff 未过 | docs/32 | 工艺交接风格问题 |

**已有验证状态（H60 原型，非 TTX 专用）** [综合/仿]：

| 检查 | 结果（docs/32 + sim_dc 产物） |
|------|-------------------------------|
| iverilog directed | PASS |
| Verilator lint | PASS |
| Yosys synth/check | PASS，**cells=24313，memories=0** |
| Erie strict | FAIL |
| DC / PT / SAIF | 未跑；PATH 无 `dc_shell` |
| golden bit-accurate | 未做 |

---

## 4. TTX 微架构（定稿建议）

### 4.1 命名

推荐论文/代码统一：

| 层级 | 名称 | 说明 |
|------|------|------|
| 系统 | **UniBin-TTX** | all-binary event + TTX attention ISA |
| 核心引擎 | **TTX-Row Engine** | 单实例、descriptor 复用 |
| score 前端 | **Binary TX Score** / Event-Consensus TX | 无 SC |
| 调度 | **TTB Work-Issue Gating** | empty skip |
| 存储 | **1-bit Packed Event/Skip SRAM** | 主收益 |
| 控制 | **Descriptor-Reused AllBinary Encoder** | 12 block 时分 |

**不要**再把主名字写成 UniBin-H60；H60/NTS、FAPS 仅作 **ablation / plugin 对照**。

### 4.2 TTX-Row Engine 流水

```text
                  cfg_descriptor (heads, N, α₀, preserve_mean, …)
                              │
  q_bits,k_bits,k_value ──► LOAD（逐 token ready-valid）
                              │  popcount TX only
                              │  score_mem[N], k_mem[N]
                              ▼
                         CENTER (row mean)
                              ▼
                         FIND_MAX → exp2_approx → SUM_EXP
                              ▼
                         EMIT gate + gated_k  (out_ready backpressure)
                              │
                         perf: loaded / empty / issued
```

相对 H60 的 RTL 差分：

```text
- 删除或综合常量传播消除：sc_num、mu_sc、fused 中的 SC 支路
- cfg_mu_q8 端口：TTX top 可 tie 0 或从 CSR 移除
- score_mode 枚举：0=TTX, 1=H60, 2=FAPS(reserved)
- 中期拆 leaf：score_tx_q7 | row_center | shiftmax_pow2 | gated_k_vec
```

### 4.3 Score plugin 接口（对照用，主线锁 TTX）

```text
score_mode:
  0 = TTX   : score = TX(q,k)
  1 = H60   : score = TX + μ·SC      （消融）
  2 = FAPS  : reserved               （未来/对照）

共用后端（不可随意拆掉）:
  center → Shiftmax → gated-K
```

### 4.4 周期模型骨架（需标定，现为 [设想]+粗 [prof]）

```text
cycles_row ≈ (
    N                      # load（1 token/cyc 原型）
  + N                      # center/max 扫描
  + N                      # exp/sum
  + N                      # emit（受 out_ready）
) × heads × windows × blocks × T
  × (1 - ttb2_skip)

更细：
  load 内 popcount 可 HEAD_DIM 并行 → 接近 1 cyc/token
  SC 删除后相对 H60 省 ~score 段 10–20% 逻辑 [设想]
```

**注意**：`docs/35` 中 ~185 FPS @500MHz 是 **perf model 设想**，不是 RTL 实测；论文主表前必须用 cycle-accurate 或 post-synth 标定。

### 4.5 与全芯片的边界

```text
[不在 TTX-Row 内，但 DATE 图必须画]
  Patch embed dense frontend
  Residual ADD paths
  MLP / downsample Sparse-MAC
  Skip DRAM bandwidth
  Decoder concat + pred feedback
```

只画 attention core 会被审稿人认为数据流不完整（`docs/26` 已强调）。

---

## 5. DC / 发文标准差距（按硬件论文口径）

### 5.1 总判

```text
当前 = 可仿真 + Verilator lint + Yosys generic 的 H60/TTX 兼容行级原型
不是 = DATE/ICCAD 风格 ASIC 主结果（面积/功耗/能效 signoff）
```

### 5.2 检查表

| 条目 | 状态 | 能否进主表 |
|------|------|------------|
| DC-ready 分层 RTL | 否（function + 组合除法 + 单模块） | 否 |
| SDC / 目标频率 | 无 | 否 |
| 工艺库 / corner | 无；本机无 `dc_shell` | 否 |
| SRAM macro / CACTI | 无；Yosys memories=0 | **否（硬伤）** |
| cycle model（workload） | 粗模型 [设想] | 仅附录/早期 |
| power activity (SAIF/VCD) | 无 | 否 |
| PyTorch golden checker | 无 | **否（硬伤）** |
| PPA breakdown | 无 DC 分解 | 否 |
| throughput / energy efficiency | 仅软件 energy_uj + 粗 FPS 设想 | 软件表可写，硬件能效表不可写 |
| TTX 专用命名/RTL | 设计有（docs/35），代码仍 H60 名 | 需改名或 wrapper |
| INT8 deploy 精度 | **有** TTX valid825 | **算法/部署表可写** |
| TTB / activity profiling | 有 allbinary H60 valid40 | 可写；建议 TTX 复测 |

### 5.3 论文允许写什么 / 禁止夸大

| 可以写 | 不能写 |
|--------|--------|
| UniBin-TTX 架构与 row ISA | “已完成 ASIC signoff” |
| valid825 AEE 1.502 + INT8 ≈ 1.50 | “RTL bit-accurate 等价软件” |
| 1-bit skip ~1.45MB、TTB2 高 empty | “Yosys 24313 cells = 芯片面积” |
| 可综合 SystemVerilog 原型 + 仿真通过 | “memories=0 的结果代表 SRAM 成本” |
| H60 为 SC 消融，FAPS 为未来 plugin | “FAPS/H60 仍是硬件主线” |
| 相对 H60 简化 score 前端的协同叙事 | “Shiftmax 实现了强 token pruning” |

---

## 6. TTX 最小补齐清单（发文/DC 路径）

按优先级。完成标准写清，避免“做了但不算数”。

### P0 — 没有则不能写“硬件实现了 TTX”

| # | 项 | 完成标准 | 证据级目标 |
|---|-----|----------|------------|
| 1 | **TTX PyTorch golden row export** | 从 TTX ep2 ckpt 导出多 stage/head 的 `q_bits,k_bits,k_value,score,gate,out` | [prof]→golden |
| 2 | **TTX RTL golden checker** | μ=0 固定；score/gate/out 误差界（如 gate ≤1 LSB，score ≤1 Q7 step） | [rtl]+[仿] |
| 3 | **TTX score engine RTL** | 明确 `score_mode=TTX` 或硬删 SC；推荐 `unibin_ttx_row_engine.sv` | [rtl] |
| 4 | **统一 Shiftmax/center 与软件** | 单一近似；与 golden 对齐；去掉双套不一致 | [rtl] |
| 5 | **gated 输出路径说明** | 至少：vector `HEAD_DIM=32` **或** lane-serial + 写清 cycle 模型 | [rtl] |
| 6 | **TTX 上 TTB skip profiling** | valid40/825 上 TTB1/2 empty、Q/K active（TTX ckpt） | [prof] |
| 7 | **activation/skip SRAM 字节表** | 按 TTX 路径重报 1-bit bytes（可沿用 allbinary 公式） | [prof] |

### P1 — 没有则不能写 ASIC 主表

| # | 项 | 完成标准 |
|---|-----|----------|
| 8 | cycle model | 每 stage tokens/heads/windows → Mcycles @ 目标 MHz；对齐 TTB skip |
| 9 | DC release RTL | 拆 leaf；消除组合除法；buffer→SRAM wrapper 端口 |
| 10 | SDC | 目标频率（如 500MHz）、IO、false path 草稿 |
| 11 | area/power | DC +（PT-PX 或估计）+ **SRAM macro/CACTI** 分列 |
| 12 | throughput / energy | FPS 或 GOPS/W；activity 来自 SAIF/VCD 或标定模型 |
| 13 | H60/TX/FAPS/TTX **对照表** | 算法 AEE + 硬件复杂度列（FAPS 可无 RTL，标 N/A） |

### P2 — 完整 accelerator 叙事

| # | 项 |
|---|-----|
| 14 | descriptor controller（12 layer ROM/SRAM） |
| 15 | event SRAM + skip DDR 接口 |
| 16 | ATLIF cluster 调度 |
| 17 | top-level `unibin_ttx_top` + AXI cfg |
| 18 | 与 FireFly-T / Spiking Transformer HW 的 related-work 对照表 |

### 建议实现顺序（若进入编码阶段；本轮未实施）

```text
1) export_ttx_golden.py          # 软件侧
2) tb: μ=0 + golden compare      # 先验证现有 core_dc
3) unibin_ttx_score_tx.sv        # 删 SC
4) unibin_ttx_row_engine.sv      # wrapper 重命名 + 接口清理
5) TTB issue + skip counter TB
6) 再动 descriptor / top / SRAM
```

每次 RTL 改动后至少：

```bash
./sim_dc/run_iverilog_dc.sh
./sim_dc/run_verilator_lint.sh
./sim_dc/run_yosys_synth.sh
./sim_allbinary/run_all_checks.sh
# 新增 TTX 脚本时写明路径，例如：
# ./sim_dc/run_ttx_golden.sh
```

改动前必须说明：目标、影响模块、验证命令、预期结果（项目规范）。

---

## 7. DATE 论文硬件故事与命名

### 7.1 故事主线（中文提纲）

1. **问题**：事件光流 U-Net + Swin attention 存储/算力重；ternary/mixed format 不适合统一 SRAM。  
2. **协同结果**：**全网 binary ATLIF event** + 注意力收敛为 **TTX（TX-only, μ=0, no-carrier）**，精度相对全量 H60 仅 +0.013 AEE，但去掉 SC 硅路径。  
3. **架构**：**UniBin-TTX**：1-bit event/skip SRAM + 单 **TTX-Row Engine** descriptor 复用 12 block + **TTB work-issue gating**。  
4. **不物化 N×N attention 矩阵**：行缓冲 score[N] + 流式 gate。  
5. **部署**：score/gate INT8 在 valid825 上几乎无损（1.502→1.500）。  
6. **硬件成熟度诚实表述**：可综合行级原型 + 仿真；ASIC PPA 为后续 signoff（或给 CACTI+模型表，不与 DC 混用）。

### 7.2 贡献句候选（英文，可进 Intro）

> We present **UniBin-TTX**, a hardware–software co-designed all-binary attention datapath for event-based optical flow. TTX freezes the encoder attention ISA to **TX-only popcount scoring** (no SC/μ fusion, no carrier, no K-mag), while retaining row centering, Shiftmax gating, and threshold-valued gated-K. A **single shared TTX-row engine** is scheduled across all 12 Swin blocks via descriptors; **TTB work-issue gating** and **1-bit packed event/skip SRAM** exploit measured low Q/K activity. On DSEC valid825, TTX reaches AEE 1.502 (INT8 score+gate 1.500) at 20.5 mJ/frame proxy energy, within 0.013 AEE of full H60 consensus fusion at a simpler score frontend.

### 7.3 图表清单（硬件侧）

| 图/表 | 内容 | 依赖 |
|-------|------|------|
| Fig.系统 | 五引擎 + skip DDR + decoder | 现可画 |
| Fig.TTX-Row | LOAD→CENTER→SHIFTMAX→EMIT | 现可画 |
| Fig.TTB | empty ratio 柱状图 | 建议 TTX 复测 |
| Tab.算法 | NB0 / H60 / **TTX** / 旧 TX | 已有 |
| Tab.量化 | float / score8 / gate8 | **已有** |
| Tab.RTL | 模块、位宽、sim/yosys、golden | golden 后 |
| Tab.ASIC | 工艺/MHz/面积/SRAM/mW/FPS | P1 后 |
| Tab.对照 | TTX vs H60 vs FAPS 硬件复杂度 | 现可写复杂度列 |

### 7.4 Related-work 一句话定位

| 工作 | 差异 |
|------|------|
| FireFly-T | FPGA overlay 通用 spike ViT；无本工作 U-Net skip + TTX ISA |
| Spiking Transformer HW (28nm 等) | tick-batching 等；无 allbinary 光流 skip 数据流 |
| Bishop TTB | 借 **调度**，不借改语义 pruning |
| 本项目旧 UniBin-H60 | TTX 为 **co-design 删 SC** 的硅主线，H60 退为 ablation |

---

## 8. 本轮实现与验证

### 8.1 代码修改

**无。** 本轮为只读审阅与方案/差距分析，未修改任何 RTL/TB/脚本。

### 8.2 验证命令

未重新跑仿真（以 `docs/32` 与 `sim_dc/build/yosys_unibin_h60_core_dc.rpt` 存量结果为准）：

| 命令 | 存量结果 |
|------|----------|
| `./sim_dc/run_iverilog_dc.sh` | PASS（历史） |
| `./sim_dc/run_verilator_lint.sh` | PASS（历史） |
| `./sim_dc/run_yosys_synth.sh` | PASS，cells=**24313**，memories=**0** |
| `./sim_allbinary/run_all_checks.sh` | PASS（历史） |

### 8.3 文档关系

| 文档 | 定位 |
|------|------|
| `docs/34` | DC 标准 + FAPS 风险（H60 时代） |
| `docs/35` | TTX DATE 设计草案（架构定稿向） |
| **`docs/36`（本文）** | **以 TTX 为唯一主线的硬件语义 + RTL 迁移 + 发文差距 + 最小清单**（更新部署量化事实） |

---

## 9. 给协作者的执行摘要

1. **主线锁 TTX**；H60 = 带 SC 的消融；FAPS = 未来/对照 plugin。  
2. **语义清晰**：TX popcount → center → Shiftmax → gated-K；**仍要** Shiftmax 与 center；**不要** SC/μ/carrier/K_mag。  
3. **现有 H60 RTL 是最好的起点**：`cfg_mu_q8=0` 即 TTX 公式；中期删 SC、改名 `unibin_ttx_*`、补 golden。  
4. **发文最大两块硬伤**：(a) 无 golden bit-accurate；(b) 无 SRAM macro / DC PPA。Yosys 通过 **不等于** 可发 ASIC 主表。  
5. **算法侧 TTX 已够写部署表**：float 1.502 / score8 1.497 / score+gate8 1.500（valid825）。  
6. **下一步最小动作**（需你明确授权再改代码）：golden export → μ=0 TB → TTX score 瘦身 RTL → 再谈 DC。

---

## 附录 A — 关键路径速查

```text
硬件仓：
  /root/private_data/work/SDformer/hw_autoresearch_nts07/

软件 attention：
  .../overlay/models/STSwinNet_SNN/bsa_attention.py  (h60 分支 ~1938)

TTX train：
  date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml

TTX valid825：
  .../h60_mu0_txonly_slowlr_cont_ep2_ft8_.../profile_ranking_valid825.md

TTX deploy quant：
  .../date11_ttx_deploy_quant_full825_20260629_220531/pipeline.log

RTL：
  rtl_dc/unibin_h60_core_dc.sv
  rtl_allbinary/*.v
```

## 附录 B — 证据来源索引

| 声明 | 来源 |
|------|------|
| TTX AEE 1.5020 | valid825 ranking，TTX slowlr cont ep2 |
| INT8 deploy | pipeline.log 2026-06-29 |
| TTB/QK/ATLIF/skip | `docs/23`（NTS allbinary valid40） |
| RTL sim/yosys | `docs/32` + `sim_dc/build/*.rpt` |
| DC 标准 | `docs/34` |
| 端到端数据流 | `docs/26` |
| 软件 h60 公式 | `bsa_attention.py` h60 分支 |
| RTL 公式 μ 融合 | `unibin_h60_core_dc.sv` `consensus_score` |
