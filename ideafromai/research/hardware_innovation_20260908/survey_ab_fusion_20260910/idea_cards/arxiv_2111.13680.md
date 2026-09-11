# arXiv:2111.13680 · GMFlow

- uid/来源: `MAIN-R312`｜arxiv_2111.13680+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: GMFlow: Learning Optical Flow via Global Matching
- 精读深度: 方法级（仅方法摘录窗口）+依据：自/交叉注意力增强特征；全局相关+softmax 可微匹配；流传播自注意力；权共享精修；消融全局vs局部匹配；摘录§3，完整精修栈可能截断

## 可继承 A
显式全局匹配（softmax 对应）+ Transformer 特征增强——大位移光流一次匹配强对照（借入≠X）。

## 强对照 B
RAFT 式局部 lookup 迭代；纯局部窗匹配；无交叉注意力特征。

## 可差分 X线索
全局匹配/Transformer≠lifting X；旁路不在 Sintel 排行差分。

## 与 F1–F7 / Stage B 关系
F2/F7弱相关（全局对应、特征传播）。旁路基线。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
帧光流 EPE/参数量≠AEE/same-port%；仅摘录窗。

## 可复用 idea 点
- softmax 全局对应作「一次并集」对照多迭代 lookup
- 交叉注意力贡献最大→特征质量分母
- 流传播改善未匹配像素
- 与 RAFT/FlowFormer 成簇
- 负结果只停 GMFlow 头替换

## 杀门建议
挂主网无增益或注意力费用挤占 Stage B → 保持旁路。
