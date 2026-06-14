# NTS-07b 硬件 Autoresearch 工作日志

## 会话目标

交付 DATE 2027 硬件文档包 + H60 RTL 骨架 + 自动实验循环，**全部 MD 中文**。

## 软件锚点

- **主线**：NTS-11aa ep19（AEE 1.543，firing 6.22%，energy 22.9k µJ）
- **精度回调**：11aah ft ep0（AEE 1.516）
- **DATE 硬件包**：`docs/16_hw_sw_pain_points_and_software_fixes.md`
- 四引擎异构：Scatter / Sparse MAC / H60 Binary / Dense MAC
- H60 scope **s23**：8 blocks，TX+μ·SC+Shiftmax，无 carrier

### 2026-06-13 软硬件协同发现

- 11aa **downsample.sn 三值** → firing **52.8%**（layer.2），spikes 4.4G；DATE「低成本三值」不成立
- synops_logic 占比 13.2% vs NB0 5.7% → 缩小三值范围（11aw 24层 / 11ax 16层）
- 已导出 TTB mask：`hw_masks/nts11aa_ep19_token_mask.json`
- 短测晋级 **11aw**；full30 曾因 `save_path` bug 丢 checkpoint，**13:36 已修复并重训**
- 运行目录：`results/nts11aw_hw_h60_s23_sn2qbin_w720_stdlr_bs8_20260613_133609_setsid`
- ep6 val loss **1.75**；16:44 从 ep6 resume ep7（guardian 重挂 + stale-log 修复）
- **22:24 ep22 完成**：short-valid loss **1.465**（ep19–22 稳定在 ~1.44–1.46）
- **22:32** 被 11bd GPU 抢占 SIGKILL；ep23 中断于 ~24%
- **22:34** 从 ep22 resume + guardian 重挂（poll 180s）；GPU 独占，无 nts11bd
- 剩余 ep23–29 ≈ **2.5h** + valid825 ≈ **2h**
- 目标：ep29 valid825 → AEE≤1.52、downsample firing<30%

---

## 自动实验记录（11 轮，2026-06-09）

### 第 1 轮：基线配置 — 25.94 mJ（保留）
- 锚点配置，FPS 92.8，SRAM 772KB

### 第 2 轮：关闭空窗跳过 — 29.48 mJ（丢弃）
- 能耗 +13.6%，证明 TTB skip 必要

### 第 3 轮：PE 256 路 — 12.97 mJ（保留）
- 能耗减半，FPS 101.3

### 第 4 轮：PE 64 路 — 51.88 mJ（丢弃）
- 吞吐与能效双差

### 第 5 轮：TX/SC 串行 — 25.94 mJ（丢弃）
- 与并行几乎同能耗，FPS 略降

### 第 6 轮：Window SRAM 256KB — 25.94 mJ（丢弃）
- 仅缩小 SRAM，能耗不变

### 第 7–10 轮：组合与 ep24 — 见 dashboard

### 第 11 轮：终极组合 — 12.97 mJ，388KB SRAM（最优）
- 能耗与 PE256 持平，SRAM 最小 → **Pareto 最优**

**目标达成：** 是（能耗 ≤ 22 mJ，FPS ≥ 30，SRAM ≤ 2048 KB）

---

## 关键洞察

1. 硬件故事核心是 **stage 静态绑定 + 256 路稀疏 MAC**，不是全网 reconfig
2. H60 引擎功耗占比小，但 DATE 创新点在于 **dyadic ISA 与 Shiftmax 单元**
3. 空窗跳过必须与软件 spike_profile 对齐才有真实收益
4. 终极组合适合写进论文主表

## 待办（下一轮）

- [ ] `export_hw_golden.py` 位精确验证
- [ ] Yosys 综合 ultimate 配置
- [ ] 将推荐配置写入 `nts07_controller.v` 默认参数