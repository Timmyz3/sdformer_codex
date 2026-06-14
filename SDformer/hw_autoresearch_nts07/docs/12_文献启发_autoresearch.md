# 文献启发 Autoresearch（Segment 1）

**锚点：** 终极组合 = 12.97 mJ
**最优：** 统一编码 + TTB-2 = 12.40 mJ（面积 2.80 mm²，较 Segment-0 再降 4.4%）

| # | 能耗(mJ) | FPS | SRAM | 面积(mm²) | 状态 | 描述 |
|---|---------|-----|------|-----------|------|------|
| 12 | 12.97 | 98.9 | 388 | 2.85 | keep | 终极组合（文献轮锚点） |
| 13 | 12.93 | 99.2 | 388 | 2.80 | keep | 统一 ATLIF 编码器（共享比较器） |
| 14 | 12.44 | 101.7 | 388 | 2.85 | keep | Bishop TTB depth-2 打包 |
| 15 | 12.97 | 111.1 | 388 | 2.85 | discard | FireFly 风格 popcount×64 |
| 16 | 12.93 | 99.2 | 388 | 2.80 | discard | 共享编码 16 lane 宽发射 |
| 17 | 12.40 | 102.0 | 388 | 2.80 | keep | 统一编码 + TTB-2 |
| 18 | 12.40 | 114.3 | 388 | 2.80 | discard | 文献终极组合 |

---

## 文献 → 搜索维度映射

| 论文 | 借鉴机制 | 本包参数 | Run |
|------|----------|----------|-----|
| Bishop ISCA'25 | Token-Time Bundle 空窗跳过 | `bishop_ttb_depth=2` | 14, 17 |
| FireFly-T TC'26 | 高并行 popcount | `firefly_popcount_par=64` | 15（周期↓，能耗持平→丢弃） |
| 统一编码器（本工作） | 共享比较器树 | `unified_atlif_encode=1` | 13, 17 |
| SDformerFlow | 稀疏 MAC + firing profile | `pe_mac=256`, `firing=0.079` | 锚点 |

## 推荐配置（Segment-1 Pareto）

`scripts/configs/best_config_lit.json`：

```json
{
  "skip_empty_windows": 1,
  "pe_mac": 256,
  "tx_sc_parallel": 1,
  "window_sram_kb": 256,
  "weight_buffer_kb": 128,
  "unified_atlif_encode": 1,
  "bishop_ttb_depth": 2
}
```

**解读**：
- **统一 ATLIF + TTB-2** 在不动 SRAM 的前提下再省 **0.57 mJ（−4.4%）**，与软件「双模神经元、单算子」叙事一致。
- FireFly 式 popcount×64 只换周期（FPS 111），能耗锚定 12.97 mJ，**不纳入 DATE 主配置**（面积/功耗 trade-off 可放消融）。
- 「文献终极组合」与「统一+TTB-2」能耗相同，但前者面积/控制更复杂，**Pareto 取后者**。

## 与软件实验的协同

软件若全线改为 `ATLIFTernaryPSN`（Q/K=`ternary`，其余=`binary`），硬件无需新增第三编码数据通路；仅需在 layer descriptor 写 `neuron_mode`，数据流见 `docs/11_统一ATLIF算子与数据流.md`。
