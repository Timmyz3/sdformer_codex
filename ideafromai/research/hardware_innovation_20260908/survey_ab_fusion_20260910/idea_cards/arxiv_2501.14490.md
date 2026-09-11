# arXiv:2501.14490 · MFPSN

- uid/来源: `MAIN-R199`｜arxiv_2501.14490+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics
- 精读深度: 方法级（仅方法摘录窗口）+依据：§4 channel-wise+膨胀/锯齿 d；PoT 量化+移位代乘；相对 PSN/sliding PSN；并行可实现；STE/round 梯度讨论

## 可继承 A
通道独立 PSN + 锯齿膨胀感受野 + PoT/移位无乘法动力学——与本地结构化 T10/lifting PSN 直接相关的神经元谱系强对照（借入并行/PoT≠X）。

## 强对照 B
通道共享 sliding PSN；固定膨胀有网格效应；满阶可学习 W∈R^{T×T}；含乘法的浮点充电。

## 可差分 X线索
MFPSN/channel-wise/PoT≠本地 lifting 结构化因子图 X；本地主张在结构化执行对象与消费者接口。

## 与 F1–F7 / Stage B 关系
直接支撑 PSN/lifting 主岛理解（算法先验）；Stage B 用 ordinary 对照时本族为强控制。不抢 Stage B 排程。

## 不可搬用边界
原作任务/实现（CUDA 等）≠ θg/非因果 T10 合同；不得把无乘法写成净服务%；仅方法摘录窗口。

## 可复用 idea 点
- channel-wise 权作「每通道时间核」对照 vs lifting 共享结构
- 锯齿膨胀作长时依赖低参技巧（借入）
- PoT+移位作普通压缩强对照，叠层须过 AEE
- 与 Fang PSN/masked/sliding 卡成谱系阅读

## 杀门建议
仅换 MFPSN/PoT 而无 lifting 结构，相对 ordinary 无净服务/精度门 → 停「当标题」，保留对照。
