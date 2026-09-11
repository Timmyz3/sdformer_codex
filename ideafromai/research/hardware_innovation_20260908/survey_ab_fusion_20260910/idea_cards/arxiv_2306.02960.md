# arXiv:2306.02960 · Best of Both Worlds

- uid/来源: `MAIN-R207`｜arxiv_2306.02960+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: Best of Both Worlds: Hybrid SNN-ANN Architecture for Event-based Optical Flow Estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：前浅 LIF + 后深 ReLU 混合；EV-FlowNet/Fire-FlowNet 骨架；消融 spiking 层数与位置（Fig.5）；自监督 photometric+smooth / DSEC 监督；相对 ConvRNN 参数更少

## 可继承 A
事件前端脉冲编码 + 后端 ANN 可训性的混合分工——光流任务架构对照；硬复位 LIF 设定明确（借入≠X）。

## 强对照 B
全 ANN；全 SNN；深层先 spiking；显式 ConvRNN 加重参。

## 可差分 X线索
混合层类型≠结构化 lifting PSN X；不把混合当本地硬件标题。

## 与 F1–F7 / Stage B 关系
光流算法对照；与 Adaptive-SpikeNet 成对读。不抢 Stage B。

## 不可搬用边界
MVSEC/DSEC AEE 数字≠valid825 合同；混合网≠nts07；仅方法摘录窗口。

## 可复用 idea 点
- 「浅层脉冲滤噪+深层 ANN」位置消融法可复用到神经元选型实验设计
- Fire-FlowNet 轻量骨架作小模型对照
- 相对 ConvRNN 的参数/能量叙事作「隐式复发够用」对照
- 负结果只停某混合深度布局

## 杀门建议
混合在本地数据上不胜纯结构化 PSN/ANN 基线或破 AEE → 停该混合布局。
