# arXiv:2505.12771 · FireFly-T

- uid/来源: `MAIN-R060`｜arxiv_2505.12771+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: FireFly-T: High-Throughput Sparsity Exploitation for Spiking Transformer Acceleration with Dual-Engine Overlay Architecture
- 精读深度: 方法级（仅方法摘录窗口）+依据：双引擎（sparse + binary）overlay；多 lane 稀疏解码+无 bank 冲突负载均衡；binary 引擎隐式转置+LUT6 AND-PopCount；prefers 二值注意力+神经元前残差；摘录偏总览/背景

## 可继承 A
稀疏引擎吞吐向 sparsity 利用 + 二值注意力专用引擎（AND-PopCount）双 overlay——作 spiking transformer 稀疏/二值路径硬件分母（借入≠X；与已有 FireFly-S 卡区分）。

## 强对照 B
粗粒度分组稀疏（整组跳过率低）；CSR 单非零/周期细粒度；无注意力支持的纯卷积 FireFly。

## 可差分 X线索
FireFly-T overlay≠ lifting X；本地不把 FPGA DSP/LUT 节省写成 Stage B 净服务。

## 与 F1–F7 / Stage B 关系
F7 硬件稀疏执行对照；旁路主岛。不抢 Stage B。

## 不可搬用边界
与 MAIN-R059 FireFly-S/批错绑 SCNN 区分；KV260 实现≠本地 RTL；仅方法摘录窗口。

## 可复用 idea 点
- 双引擎分工（稀疏 GeMM vs 二值注意力）作路径专用化对照
- 无冲突负载均衡作不规则供数叙事
- AND-PopCount 二值注意力作乘法消除强对照
- 神经元前残差保 spike 输入效率

## 杀门建议
无法对齐 same-port 计费 → 仅文献对照；若与 FireFly-S/SCNN 错绑引用 → 废错绑。
