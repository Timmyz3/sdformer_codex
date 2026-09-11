# arXiv:2203.16194 · FlowFormer

- uid/来源: `MAIN-R314`｜arxiv_2203.16194+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: FlowFormer: A Transformer Architecture for Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：4D cost→逐源像素 cost map 分块；潜变量摘要 tokenization；Transformer 编 cost memory；cost query 共注意力+循环精修；凸上采样；摘录§3.1–3.3，完整训练表可能截断

## 可继承 A
把全对代价体压成 cost memory 再用查询迭代取回——相关体压缩/多消费者查询对照（借入≠X）。

## 强对照 B
RAFT 原始相关金字塔+局部 lookup；无潜变量摘要的朴素 patch token；单次非迭代解码。

## 可差分 X线索
FlowFormer/cost memory≠lifting 源字 X；旁路。

## 与 F1–F7 / Stage B 关系
F7弱相关（代价体打包/查询）；旁路基线。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
Sintel/KITTI EPE≠valid825；仅摘录窗；参数/迭代勿写 same-port%。

## 可复用 idea 点
- 潜变量摘要压冗余代价→类比源 DAG 公共虚节点 F3 弱旁证
- cost query 共注意力作多像素共享读取
- 与 RAFT/GMFlow/EDCFlow 成簇
- 负结果只停 FlowFormer 头替换

## 杀门建议
挂主网无增益或注意力/记忆费用挤占 Stage B → 保持旁路。
