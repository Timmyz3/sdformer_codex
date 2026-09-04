# 11aa / 11aah 软硬件协同审计与软件修复路线

**日期**：2026-06-13  
**目标**：DATE 2027 — 在保持两神经元部署故事前提下，消除硅片实现难点  
**软件锚点**：11aa ep19（稀疏/能效）+ 11aah ep0（精度回调 AEE 1.516）

---

## 1. 剖面证据：11aa 的「省电」里有硬件陷阱

| 痛点 | 11aa ep19 | 10d ep29 | NB0 ep59 | 硬件影响 |
|------|-----------|----------|----------|----------|
| downsample 层 firing | **31–53%** | 25–29% | 26–29% | 2-bit 编码器满载，Sparse MAC 零跳过失效 |
| downsample spikes | **4.40G** | 2.20G | 2.28G | 占总量 15%；DATE 叙事「downsample 低成本三值」不成立 |
| synops_logic 占比 | **13.2%** | 9.7% | 5.7% | 逻辑/popcount 通路压力↑，难做 MAC 阵列复用 |
| S0/S1 Q/K 三值 + Legacy 注意力 | 全线 2-bit | 同左 | PSN 1-bit | S0/S1 不跑 H60，仍付三值带宽与编码面积 |
| 双引擎 | S0/S1 Legacy + S2/S3 H60 | 同左 | 单一路径 | 控制器 + 两套 attn 微码；sn2_q 仅 Legacy 需要 |

**结论**：11aa 的全局 firing 6.22% 很好看，但 **downsample 三值把局部热点推到 50%+**，对 TTB 空窗跳过和片上 SRAM 预算都不友好。稀疏应继续用 `sparsity_ratio` + `effective_FLOPs`，不要用 SOPs 代替。

---

## 2. 硬件难点 → 软件可改项（不改 H60 ISA）

### P1 — downsample 三值（最高优先级）

| | |
|---|---|
| **问题** | 3 层 downsample.sn 三值后 firing 翻倍；与「统一 ATLIF + ternary_en」故事冲突（额外 2-bit 侧路） |
| **软件修复** | **11aw/11ay**：恢复 downsample 为 `all_non_qk` 二值官方 ATLIF（即 11r/sn2q_binary scope） |
| **实验** | `nts11aw_hw_h60_s23_sn2qbin_w720_stdlr` full30 from NB0 |
| **DATE 表述** | 「协同搜索发现 downsample 三值不 Pareto；硅片仅 Q/K@H60 保留 2-bit」 |

### P2 — S0/S1 无效三值带宽

| | |
|---|---|
| **问题** | Legacy QKFormer 不吃 TX/SC，但 S0/S1 的 sn_q/sn_k 仍是三值 |
| **软件修复** | **11ax**：`target_paths` 仅列出 s23 八个 block 的 sn_q/sn_k；S0/S1 Q/K 落回二值 |
| **实验** | `nts11ax_hw_h60_s23_qks23_w720_stdlr` |
| **风险** | S0/S1 表达力下降；需 valid825 验证 AEE |

### P3 — 精度 −0.06 AEE（配方，非 scope）

| | |
|---|---|
| **问题** | 11aa fastlr+freeze816 vs 10d warm720+freeze1224 |
| **已有** | 11aah ep0 AEE **1.516**（−0.027 vs 11aa ep19），仍差 10d ~0.038 |
| **软件修复** | **11az**：11aah ep0 起点 + **11aw scope**（去掉 downsample 三值）finetune 5ep |
| **实验** | `nts11az_hw_h60_s23_sn2qbin_w720_stdlr_ftaah0` |

### P4 — synops_logic 偏高

| | |
|---|---|
| **问题** | 三值层增多 → TX/SC popcount 统计归入 logic |
| **软件修复** | 缩小三值范围（P1+P2）；attention 侧 `bipolar_mu`/`alpha0` 消融（phase5 11at–11av） |
| **硬件** | 已有 H60-ISA；软件侧避免扩大三值到 FFN（11u 已证伪） |

### P5 — TTB / profile 闭环未导出

| | |
|---|---|
| **问题** | 硬件 autoresearch 需要 `token_mask`，软件未自动从 valid825 产出 |
| **软件修复** | `scripts/export_token_mask_from_profile.py` → `hw_masks/nts11aa_ep19.json` |
| **证据** | 关闭 TTB skip +13.6% 能耗（doc 07 创新点五） |

### P6 — 双引擎长期方案（中期）

| | |
|---|---|
| **选项 A** | 维持现状：ENGINE_MAP 静态绑定（DATE 创新点二） |
| **选项 B** | 软件把 H60 扩到 S0/S1（全网单引擎）— 面积大，仅当 11ax 精度失败时考虑 |
| **不推荐** | FFN 全三值（11u）：effective +11%，energy +8% |

---

## 3. 推荐 DATE 2027 软件–硬件主线

```
NB0 / 11aah ep0
    ↓  software sweep (overnight)
11aw full30  ──→  valid825  ──→  若 AEE≤1.50 & sparsity≥11aa
    │                              ↓
    │                         更新 hw 锚点 ckpt
    └─ 失败则 11ax (qk_s23_only) ──→ 再 valid825
```

**论文故事**：软件 scope sweep 产出 **Pareto 前沿**（10d 精度 / 11aa 稀疏 / 11aw 硬件友好），硬件固化 **s23 H60 + 统一 ATLIF + TTB mask**。

---

## 4. 今夜已启动的自动化

| 步骤 | 脚本 / 配置 |
|------|-------------|
| 生成配置 | `make_nts11_hw_friendly_configs.py` |
| 短测 → full30 | `run_nts11_hw_friendly_autopilot.py` |
| TTB mask | `export_token_mask_from_profile.py` on 11aa/11aah profiles |
| 记录 | `RUNS.md` § NTS-11hw + `hw_autoresearch_nts07/experiments/worklog.md` |

---

## 5. 验收标准（推进 DATE）

| 指标 | 11aa ep19 | 目标 (11aw ep29) |
|------|-----------|------------------|
| AEE | 1.543 | **≤ 1.52**（先收回 aah 水平） |
| effective_G | 76.5 | **< 90**（保持对 NB0 −20%） |
| downsample max firing | **53%** | **< 30%** |
| synops_logic ratio | 13.2% | **< 10%** |
| 硬件 mJ/frame (model) | 9.2 | **≤ 10**（nts11aa_anchor.json 口径） |