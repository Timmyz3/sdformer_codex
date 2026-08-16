# H67 真实 Ordered Workload 与架构决策（2026-07-14）

**触发**：GPU 空闲后接手跑完此前被 round4 训练队列挡住的 TTB/Delta cycle-v2 ordered-trace profile。  
**状态**：H67 / H68 / TTX 各 valid **100 samples、ordered_trace=true** 已完成；HIT-Flow 审计与 DSE 已回填；全 encoder budget 脚本因 storage JSON schema 不匹配未产出（见 §6）。

---

## 1. 本轮做了什么

### 1.1 阻塞与修复

| 问题 | 处理 |
|------|------|
| 旧 watcher 死等 Round4 full30 | **绕过队列**，直接跑 H67/H68/TTX profile |
| `bsa_attention._spatial_pair_locality_stats` CUDA Long matmul | 改为 `float32` 计数再转 long（`bsa_attention.py`） |
| GPU | 空闲时启动；未杀任何训练进程 |

### 1.2 Profile 产出

| 模型 | 配置 | checkpoint | 输出目录 | 样本 |
|------|------|------------|----------|------|
| **H67** | `h67_*_rtl_exact.yml` | ep19 | `.../h67_ep19_ttb_delta_cycle_v2_profile100_20260713/` | 100 |
| **H68** | `h68_*_rtl_exact.yml` | ep19 | `.../h68_ep19_ttb_delta_cycle_v2_profile100_20260713/` | 100 |
| **TTX** | `date11full_ttx_dyadic_*` | ep2 | `.../ttx_ep2_ttb_delta_cycle_v2_profile100_20260713/` | 100 |

状态日志：

```text
neuron_experiments/.../results/ttb_cycle_profile_v2_grok_handoff_20260714_status.log
并已写入 ttb_cycle_profile_v2_after_round3_status.log：
  ALL COMPLETE TTB/DELTA CYCLE V2: handoff by grok after Long matmul fix
```

### 1.3 后处理 / DSE

| 产物 | 路径 |
|------|------|
| Ordered 架构审计 | `hw_autoresearch_nts07/results/hit_flow_ordered_profile_analysis.{json,md}` |
| GCM-P DSE H67 | `results/gcmp_h67_multicast_dse.{json,md}`（135 配置） |
| GCM-P DSE H68 | `results/gcmp_h68_multicast_dse.{json,md}` |
| 跨窗口 G 扫描 H67 | `results/gate_window_group_h67_dse.{json,md}`（675 配置） |
| 跨窗口 G 扫描 H68 | `results/gate_window_group_h68_dse.{json,md}` |

---

## 2. 核心统计（架构决策用）

以下为 **100 frames × 6720 rows/frame** 量级上的全局比例（`binary_temporal_pairs`），除非另行说明。

### 2.1 Token / Pair 稀疏（H67 主线）

| 指标 | H67 | H68 | TTX | 架构含义 |
|------|-----|-----|-----|----------|
| token K-zero（加权） | **88.68%** | 88.65% | 88.17% | **SCS 必做** |
| active entries / row | **18.34** | 18.38 | 19.16 | active bank 深度 |
| fold classes / row | **2.27** | 2.24 | 2.44 | 占用类扫描有油水 |
| TTB2 empty（加权） | **53.12%** | 53.54% | 51.96% | bundle issue 有价值 |
| pair empty | **73.99%** | 74.24% | 72.54% | **PAIR_EMPTY exact 路径** |
| pair motion_zero | **83.21%** | 83.37% | 82.37% | 可门控 Motion 支路 |
| pair update_zero (u≈0) | **74.09%** | 74.34% | 72.65% | 可选 t1 复用 / 慎做 delta |
| pair kzero_both | **83.14%** | 83.30% | 82.28% | 双时间片零 K 极高 |
| pair both_active | **5.79%** | 6.00% | 5.94% | 真 dense pair 很少 |

### 2.2 Stage 差异（H67，按 6720-row 权重）

| Stage | K-zero | active/row | fold classes | TTB2 empty |
|------:|-------:|-----------:|-------------:|-----------:|
| S0 | 80.58% | 31.46 | 2.74 | 28.68% |
| S1 | **97.76%** | **3.63** | 1.36 | **77.59%** |
| S2 | 93.34% | 10.79 | 2.33 | 64.83% |
| S3 | 84.96% | 24.36 | 2.13 | 61.43% |

**含义**：S1 极疏 → per-stage 阈值 / 更激进 SCS 合理；S0 相对最密 → dense path 不能删。

### 2.3 Projection / NMF 收益（H67）

| 指标 | 数值 | 含义 |
|------|------|------|
| baseline active K-lanes（全局） | 40,560,225 | direct 乘积次数上界 |
| final-gate 类通道项 | 7,101,034 | NMF/G1 乘积项 |
| **final-gate 乘积减少** | **82.49%** | **NMF G1 主收益证据** |
| score-class 项（H67 语义） | 18,220,365 | 比 gate-class 多 → gate 合并更狠 |
| gate 相对 score 类额外合并 | **61.0%**（审计报告） | 必须用 **final Q1.7 gate** 做目录键 |
| gate 活跃 class p95 / max | 3 / 6 | 目录 SLOTS=4 合理 |
| gate Q1.7 越界 | **0** | 定点网格健康 |

H68/TTX 上 final-gate 乘积减少分别为 **85.6% / 87.3%**，方向一致。

### 2.4 窗口组 G（审计摘要，H67）

相对 **逐 row G=1** 的额外乘积项减少（理想、仅投影后端）：

| G | 再减 | slot 利用率 |
|--:|-----:|------------:|
| 2 | 27.3% | 100% |
| 4 | 45.4% | 84.8% |
| 8 | 59.4% | 75.7% |
| 16 | 66.0% | **46.7%** |

DSE 低溢出表显示 G=2/S=8 零溢出理想加速约 5.3×（**仅投影后端、完全重叠假设**），但状态 KiB 随 G 上升。  
**门槛（docs/68）**：G>1 需同约束 **≥15% 子系统 EDP** 才晋级；否则 **锁 G=1**。

### 2.5 GCM-P / SLOTS

- `S=2`：overflow ≈ **6.1%** → 不能当无 fallback 主配置  
- `S=4`：overflow ≈ **0.014%** → **与现有 G1 RTL 默认 SLOTS=4 对齐**  
- 多播宽度 M 提高主要压 delivery p95，需与 bank 冲突一起看

---

## 3. 架构决策表（用数据勾选）

| 档 | 决策 | 依据 | 动作 |
|----|------|------|------|
| **档1 必做** | SCS + 占用类 | K-zero 88.7%，fold≈2.3 | 保持 H67 row engine 主线 |
| **档1 必做** | NMF **G=1** final-gate 目录 | 乘积 −82.5%；gate class p95=3 | **继续 G1 RTL**；SLOTS=**4** |
| **档1 必做** | PAIR_EMPTY exact | 74% pair empty | issue 合同：仅全零 pair 注 class-2 |
| **档1 推荐** | K-zero → hist，不写 active | token_kzero 88.7% | 已有 SCS commit 语义 |
| **档2 条件** | motion_zero 早到门控 | 83% motion_zero | 仅当写端 metadata 便宜时做 |
| **档2 条件** | update_zero / u=0 复用 | 74% | 先 pair co-compute，不做重 state ETCR |
| **档2 条件** | 跨窗口 G=2/4 | 理想再减 27–45% | **先 G=1 闭环**；G>1 要 EDP≥15% |
| **档3 暂缓** | 异构双路径 PHEA | both_active 仅 5.8% | 先单 dense popcount；无 w_set 交叉点净收益不上双核 |
| **档3 暂缓** | 蝶形互连 | docs/72 已倾向淘汰 | 先证明简单网络阻塞 |
| **否** | 三条 skip 当 binary | 审计：**否** | RPI 多 bit 独立域 |
| **否** | BMRF 主贡献 | 门槛不通过 | 不写 |

**一句话决策：**

> **锁：H67 + SCS + NMF(G=1,S=4) + exact pair/K-zero issue。**  
> **缓：G≥2 窗口组、双路径、ETCR。**  
> **证据已够开 RTL 深化；不够开 DATE 面积功耗主表。**

---

## 4. 与已有 RTL 的对齐

| 模块 | 与数据关系 |
|------|------------|
| `rtl_h67/*` SCS row | K-zero/占用类与 profile 一致 |
| `hitflow_nmf_g1` SLOTS=4 | 与 gate class p95≈3、S=4 低溢出一致 |
| `hitflow_g1_projection_top` | 应对 **82% 乘积减少** 做 cycle 分账验证 |
| fallback | S=4 overflow 0.014% → 可后做；S=2 必须做 |

---

## 5. 证据等级

| 声明 | 等级 |
|------|------|
| H67/H68/TTX valid100 ordered profile 数字 | **[prof] 实测** |
| final-gate 乘积 −82.5%（H67） | **[prof] 从 trace 计数** |
| G>1 理想加速 5×+ | **[模型] DSE**，非 DC/EDP |
| 系统 FPS / mW | **无** |
| storage budget 脚本 | **失败**（`h67_h68_storage_ablation.json` 无 `models` 键） |

---

## 6. 未完成

1. 修复 `model_hit_flow_full_encoder_budget.py` 的 storage schema 或适配器。  
2. 用本 profile **回放 cycle model**（`replay_ttb_dual_path_cycles.py` 等）写 SCS/NMF 对照表。  
3. G1 top 放大到 162×32 随机/trace 驱动等价。  
4. fallback 无损路径。  
5. DC / SAIF。

---

## 7. 建议执行顺序（硬件）

```text
1. 保持 G1 RTL：SLOTS=4，禁止默认 S=2 无 fallback
2. 用 H67 ordered JSON 做 projection backend cycle 分账（direct vs NMF）
3. 补 PAIR_EMPTY / motion metadata 合同到 H67 row（若尚未）
4. G≥2 仅当投影 EDP 模型 + 状态代价过 15% 门槛再开 RTL
5. 双路径 PHEA：先用 pair 事件数分布估交叉点，不过线不上
```

---

## 8. 一句话

> GPU 空闲后已跑通 **H67/H68/TTX ordered-trace profile100**，并完成 HIT-Flow 审计与 DSE：数据强烈支持 **SCS + final-gate NMF(G=1,S=4)** 作为硬件主线；**跨窗口 G 与异构双核仍是条件候选**，不能单靠理想加速写进 DATE 贡献。
