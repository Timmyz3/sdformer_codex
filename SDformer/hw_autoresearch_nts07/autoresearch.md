# Autoresearch：NTS-07b 硬件加速器

## 目标

在 **统一 H60 全线注意力（11bc+）** 前提下，优化异构加速器的 **有效能耗** 与 **有效周期**。

**工作负载：** `MS_SpikingformerFlowNet_en4` + **12 block H60** @ 288×384。  
**废弃：** 11aa / 07b Legacy+H60 混用模型（`hw_anchor` 仍可查旧值对比）。

## 评价指标

- **主指标**：`effective_energy_mj`（mJ/帧，越低越好）
- **次指标**：`effective_cycles`、`sram_kb`、`fps_at_500mhz`
- **约束**：`epe_drift ≤ 0.02`（golden 回归建立前固定为 0）

## DATE 目标（已达成）

| 指标 | 目标 | 最优结果 |
|------|------|----------|
| 能耗 | ≤ 22 mJ | **12.97 mJ** |
| FPS | ≥ 30 | **101**（解析模型 @ 500MHz） |
| SRAM | ≤ 2048 KB | **388 KB** |

## 如何运行

```bash
python3 scripts/run_all_experiments.py      # Segment 0：11 轮
python3 scripts/run_literature_experiments.py  # Segment 1：文献启发 7 轮
./autoresearch.sh                           # 单次基线
```

## 可修改文件

| 文件 | 作用 |
|------|------|
| `scripts/nts07_perf_model.py` | 性能/能耗解析模型 |
| `scripts/configs/*.json` | 实验参数变体 |
| `scripts/run_all_experiments.py` | 自动网格搜索 |
| `rtl/*.v` | H60 引擎与控制器 |
| `docs/04_optimization_strategies.md` | 搜索空间定义 |

## 禁止修改

- `neuron_experiments/**/overlay/**`（软件注意力语义）
- NTS-07b checkpoint 权重
- DSEC 训练脚本

## 硬约束

- `s2_only_h60`：Stage-2 必须走 H60 引擎
- 不得引入 carrier / K_mag / softmax
- SRAM 总量 < 2048 KB

## 已尝试方案（摘要）

| 方案 | 能耗 | 结论 |
|------|------|------|
| 基线 | 25.94 mJ | 锚点 |
| 关闭空窗跳过 | 29.48 mJ | 丢弃 |
| PE 256 | 12.97 mJ | 保留 |
| PE 64 | 51.88 mJ | 丢弃 |
| TX/SC 串行 | 25.94 mJ | 丢弃（并行略优） |
| **终极组合** | **12.97 mJ, 388KB** | **最优 Pareto** |

详见 `docs/10_autoresearch实验结果.md`、`docs/12_文献启发_autoresearch.md`。

## Segment 1（文献启发，在终极组合上）

| 方案 | 能耗 | 结论 |
|------|------|------|
| 锚点（终极组合） | 12.97 mJ | 基线 |
| 统一 ATLIF 编码 | 12.93 mJ | 保留（面积 −0.05 mm²） |
| Bishop TTB-2 | 12.44 mJ | 保留 |
| **统一编码 + TTB-2** | **12.40 mJ** | **Segment-1 最优** |