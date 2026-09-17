# arXiv:2503.10195 · ST-FlowNet

- uid/来源: `ARX-009`｜arxiv_2503.10195+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`；PDF §4 方法窗补齐）
- 题名: ST-FlowNet: An Efficient Spiking Neural Network for Event-Based Optical Flow Estimation
- 精读深度: 方法级（方法摘录窗+PDF §4 补齐）+依据：FlowNet 半金字塔+ConvGRU 时序对齐；ANN 自监督 contrast+smooth→A2S；BISNN 交叉初始化+STBP；MVSEC AEE；完整超参表可能截断

## 可继承 A
事件光流尖峰网：ConvGRU 增强编解码 + ANN 自监督再 A2S/BISNN 转尖峰——事件光流尖峰前端与训推合同对照（借入≠X）。

## 强对照 B
纯 ANN FlowNet/EV-FlowNet；无 ConvGRU 的单帧尖峰；仅经验阈值的 A2S 无再训；有监督真值流。

## 可差分 X线索
ST-FlowNet≠lifting X；同任务旁路，差分不在 MVSEC AEE%。

## 与 F1–F7 / Stage B 关系
F2相关（事件时间表示/递推）。旁路基线。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
MVSEC/ECD/HQF 指标≠valid825；仅摘录+§4；理论能耗≠芯片 mW；禁止虚报完整训练表。

## 可复用 idea 点
- ConvGRU 隐状态跨步对齐作时序光流合同
- 自监督 contrast+smooth 作缺 GT 训法旁证
- BISNN：ANN 权重初始化+STBP 免手调生物参
- 与 EVA-Flow/TMA/尖峰光流成簇 | 负结果只停 ST-FlowNet 头替换

## 杀门建议
替换无 AEE/服务增益或 ConvGRU 费用挤占 Stage B → 保持旁路。
