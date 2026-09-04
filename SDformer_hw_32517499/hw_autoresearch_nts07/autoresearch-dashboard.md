# Autoresearch 仪表盘：nts07_hw

**总轮次：** 11 | **最优：** 终极组合 = 12.97 mJ
**基线：** 25.94 mJ | **改善：** 50.0%
**目标：** 能耗≤22.0 mJ，FPS≥30.0，SRAM≤2048.0 KB

| # | 能耗(mJ) | FPS | SRAM | 状态 | 描述 |
|---|---------|-----|------|------|------|
| 1 | 25.94 | 92.8 | 772 | keep | 基线配置 |
| 2 | 29.48 | 86.3 | 772 | discard | 关闭空窗跳过 |
| 3 | 12.97 | 101.3 | 772 | keep | PE 256 路 |
| 4 | 51.88 | 79.5 | 772 | discard | PE 64 路 |
| 5 | 25.94 | 91.1 | 772 | discard | TX/SC 串行 |
| 6 | 25.94 | 92.8 | 516 | discard | Window SRAM 256KB |
| 7 | 12.97 | 101.3 | 772 | discard | 跳过+PE256+并行 |
| 8 | 12.97 | 101.3 | 516 | keep | 跳过+PE256+小SRAM |
| 9 | 17.29 | 98.3 | 772 | discard | 跳过+PE192 |
| 10 | 25.82 | 92.9 | 772 | discard | ep24 发放率 |
| 11 | 12.97 | 101.3 | 388 | keep | 终极组合 |

## Segment 1：文献启发（在终极组合上叠加）

- Run 12: 12.97 mJ, 99 FPS, keep — 终极组合（文献轮锚点）
- Run 13: 12.93 mJ, 99 FPS, keep — 统一 ATLIF 编码器（共享比较器）
- Run 14: 12.44 mJ, 102 FPS, keep — Bishop TTB depth-2 打包
- Run 15: 12.97 mJ, 111 FPS, discard — FireFly 风格 popcount×64
- Run 16: 12.93 mJ, 99 FPS, discard — 共享编码 16 lane 宽发射
- Run 17: 12.40 mJ, 102 FPS, keep — 统一编码 + TTB-2
- Run 18: 12.40 mJ, 114 FPS, discard — 文献终极组合
