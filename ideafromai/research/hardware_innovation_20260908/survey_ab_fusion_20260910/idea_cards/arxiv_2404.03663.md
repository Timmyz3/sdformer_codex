# arXiv:2404.03663 · Spike-driven Transformer V2 / Meta-SpikeFormer

- uid/来源: `MAIN-R256`｜arxiv_2404.03663+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 meta Transformer；§3.2 LIF；§3.3 Conv-SNN块SepConv+ChannelConv与Transformer-SDSA金字塔；完整SDSA公式/表5可能截断

## 可继承 A
Meta-SpikeFormer：前两段Conv-SNN（SepConv token mixer）+后两段金字塔SDSA——可替换token mixer的尖峰元架构与芯片友好算子集对照（借入≠X）。

## 强对照 B
纯Transformer尖峰（无Conv茎多段）；Spike-driven Transformer仅四层Conv编码；ANN meta块直接换激活。

## 可差分 X线索
Meta块/SDSA≠lifting X；架构元设计旁路。

## 与 F1–F7 / Stage B 关系
F1/F2/F4/F7弱相关（表示、时间步、残差/块、注意力打包）。旁路基线。不抢 Stage B。f_candidates含F1,F2,F4,F7。

## 不可搬用边界
ImageNet/检测精度≠AEE；启发下一代芯片≠nts07合同；仅摘录窗。

## 可复用 idea 点
- TokenMixer可替换作算子集合同（对照芯片原语选择）
- SepConv+3x3 ChannelConv增强归纳偏置消融法
- 金字塔SDSA与V1成对读
- 负结果只停某mixer布局，不杀meta家族

## 杀门建议
挂主网无增益或块费用挤占 Stage B → 保持旁路。
