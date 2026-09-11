# arXiv:2209.15425 · Spikformer

- uid/来源: `MAIN-R254`｜arxiv_2209.15425+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Spikformer: When Spiking Neural Network Meets Transformer
- 精读深度: 方法级（仅方法摘录窗口）+依据：SPS 卷积茎→尖峰 patch；条件 RPE；SSA：Q/K/V 全尖峰、取消 softmax、直接 QKᵀV·s；多块编码器+GAP；摘录§3.1–3.3，完整附录多头细部可能截断

## 可继承 A
尖峰自注意力（无 softmax）+尖峰 patch 分裂——稀疏二进制 Q/K 上的注意力聚合对照（借入≠X）。

## 强对照 B
浮点 ViT/VSA+softmax；把 V 尖峰化仍保留浮点 Q/K；无 SPS 的平坦 flatten。

## 可差分 X线索
Spikformer/SSA≠lifting 源字 X；旁路/表示。

## 与 F1–F7 / Stage B 关系
F1/F7弱相关（尖峰表示与注意力打包）。旁路基线。不抢 Stage B。f_candidates含F1,F7。

## 不可搬用边界
CIFAR/DVS 精度≠valid825；仅摘录窗；SSA FLOPs 勿写 same-port%。

## 可复用 idea 点
- 无 softmax 的非负尖峰注意力作聚合合同
- SPS+RPE 作事件/帧统一嵌入旁证
- 与 Spiking Transformer 族成簇
- 负结果只停 SSA 替换本地出口

## 杀门建议
挂主网无增益或注意力费用挤占 Stage B → 保持旁路。
