# arXiv:2503.19643 · Hardware Efficient Reconfigurable Time-Step Spiking Transformer

- uid/来源: `MAIN-R064`｜arxiv_2503.19643+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: Hardware Efficient Accelerator for Spiking Transformer With Reconfigurable Parallel Time Step Computing
- 精读深度: 方法级（仅方法摘录窗口）+依据：element-wise-IAND 代残差加法使全链路 spike I/O；可重构向量化数据流覆盖 3×3/1×1/MatMul；unrolled LIF 同时算多时间步、免膜 SRAM；摘录偏结果/结论，前半数据流细节可能截断

## 可继承 A
时间步展开 LIF + 可重构向量数据流 + IAND 残差保 spike-only I/O——作「时空并行供数/免膜状态」硬件分母对照（借入≠X；近 Gustav CPTB 时间并行叙事）。

## 强对照 B
逐步串行 LIF+膜 SRAM；仅 CNN 稀疏加速器；残差用实值加破坏 spike I/O。

## 可差分 X线索
unrolled LIF/IAND≠本地 lifting 执行对象 X；差分须在 same-port 下相对 ordinary dense/raw，而非搬 3456GSOPS。

## 与 F1–F7 / Stage B 关系
F5/F7 时空并行与状态驻留对照；旁路。不抢 Stage B。

## 不可搬用边界
28nm 表项/TSOPS/W≠本地 θg 合同；Spike-IAND-former 精度表勿外推光流 AEE；仅方法摘录窗口且方法段偏短。

## 可复用 idea 点
- 多时间步展开免膜 SRAM→对照有限 RF/状态存活（F5）
- IAND 残差作 spike-only 接口强约束对照
- 可重构 3×3/1×1/MatMul 同阵列作多算子分母
- 负结果只停「照搬 unrolled LIF 到 T10」

## 杀门建议
展开后端口/状态账吃掉并行收益或破坏同端口公平 → 仅文献对照，不停 Stage B。
