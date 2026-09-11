# arXiv:2203.16896 · CRAFT

- uid/来源: `MAIN-R313`｜arxiv_2203.16896+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: CRAFT: Cross-Attentional Flow Transformer for Robust Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：SSTrans/Expanded Attention（多 mode MoE）+相对位置偏置；Cross-Frame Attention 以简化 EA 替换点积相关体；GMA；摘录§3.1–3.2 与评测混排，完整训练/实现表可能截断

## 可继承 A
用 Expanded Attention 做帧内语义平滑，再用多 mode Q/K 投影构造交叉帧相关体——相关体构造/注意力打包对照（借入≠X）。

## 强对照 B
RAFT 朴素点积相关；单头 MHA 无 mode 聚合；无相对位置偏置的全局自注意力。

## 可差分 X线索
CRAFT/交叉注意力≠lifting 源字 X；旁路。

## 与 F1–F7 / Stage B 关系
F7弱相关（相关体/注意力打包）；旁路基线。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
Sintel/KITTI EPE≠valid825；仅摘录窗；大位移鲁棒勿写 same-port%。

## 可复用 idea 点
- SSTrans mode 聚合作多专家特征合同
- EA 式多投影相关→多消费者匹配旁证
- 与 FlowFormer/GMA/RAFT 成簇
- 负结果只停 CRAFT 头替换

## 杀门建议
挂主网无增益或注意力费用挤占 Stage B → 保持旁路。
