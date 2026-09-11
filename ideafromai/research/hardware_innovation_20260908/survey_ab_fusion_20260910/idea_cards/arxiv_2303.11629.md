# arXiv:2303.11629 · TMA

- uid/来源: `MAIN-R227`｜arxiv_2303.11629+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: TMA: Temporal Motion Aggregation for Event-based Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：事件切段+辅助参考段；时密相关体 Ci=F0Fiᵀ；线性 lookup 按时间对齐；运动模式聚合（中间特征×末特征 cross-attn）；基于 RAFT；摘录§3 Methodology，完整训练表可能截断

## 可继承 A
时密相关 + 线性 lookup 对齐 + cross-attn 聚合运动模式——事件时间维打包与多段共享读取对照（借入≠X）。

## 强对照 B
两帧事件相关（E-RAFT）；标准非时间对齐 lookup；无聚合的单相关图。

## 可差分 X线索
TMA≠lifting X；旁路/前端，相关体费用勿当净服务%。

## 与 F1–F7 / Stage B 关系
F2相关（时间切段/时密相关）。旁路基线。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
基准 EPE≠valid825；仅摘录窗；O((HW)²) 相关存储勿写 same-port%。

## 可复用 idea 点
- 事件切段作时间打包合同
- 线性 lookup 作时间对齐读取旁证
- 与 IDNet（无相关体）成簇对照费用
- 负结果只停 TMA 头替换

## 杀门建议
挂主网无增益或相关体费用挤占 Stage B → 保持旁路。
