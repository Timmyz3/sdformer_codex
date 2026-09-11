# arXiv:2409.04082 · SDformerFlow

- uid/来源: `MAIN-R200`｜arxiv_2409.04082+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: SDformerFlow: Spatiotemporal swin spikeformer for event-based optical flow estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：STTFlowNet（ANN swin 时空窗）与 SDformerFlow（swin spikeformer）成对；两种神经元变体；DSEC/MVSEC；首个 spikeformer 稠密光流

## 可继承 A
事件光流上的 swin 时空注意力 + 全脉冲对照网——任务极相关的算法架构底座；与本地 nts07/结构化 PSN 并列的强任务对照（借入≠X）。

## 强对照 B
相关体积/迭代去模糊 ANN；卷积-only 光流；无时空窗的普通 spikeformer 分类网；帧聚合再 ANN。

## 可差分 X线索
SDformerFlow 精度/功耗叙事≠ lifting 硬件 X；本地差分在结构化执行对象与 same-port，不在换 backbone。

## 与 F1–F7 / Stage B 关系
光流算法对照（F7 注意力旁路）；明确不抢 Stage B 主实验。

## 不可搬用边界
DSEC/MVSEC EPE≠ valid825 AEE 合同；开源训练码≠硬件分母；仅方法摘录窗口（网络超参细节截断）。

## 可复用 idea 点
- ANN swin vs 全脉冲 swin 成对消融法可复用到神经元选型
- 时空移窗注意力作「要不要注意力轴」的任务侧证据
- 相对相关体积的资源叙事支持「轻骨干」对照
- 负结果只停某 spikeformer 骨干，不影响 Stage B

## 杀门建议
本地数据上不胜生产 nts07/结构化 PSN 或破 AEE → 停换骨干布局，不杀主实验。
