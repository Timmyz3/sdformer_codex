# arXiv:2506.07878 · STSSM (event OF)

- uid/来源: `ARX-078`｜arxiv_2506.07878+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: Spatio-Temporal State Space Model For Efficient Event-Based Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3 事件生成↔光流；体素网格；四段 STSSM（patch→SSM/Mamba→reproj）；无 4D 相关体；flow/mask 头+凸上采样；完整损失/表可能截断

## 可继承 A
用选择性 SSM/Mamba 直接吃事件体素提时空特征、避开 4D 相关金字塔——高效事件光流编码器对照（借入≠X）。

## 强对照 B
E-RAFT/TMA 式 4D 相关体；双视图相关；纯 CNN/Transformer 编码器；非选择性 S4 无输入依赖参数。

## 可差分 X线索
STSSM≠lifting X；同任务旁路，差分不在 EPE%；相关体费用对照≠本地 same-port%。

## 与 F1–F7 / Stage B 关系
F2/F7弱相关（时间状态、特征打包）。旁路基线。不抢 Stage B。f_candidates空。

## 不可搬用边界
光流基准≠valid825；双栏 LaTeX 残片摘录须谨慎；禁止虚报完整 SOTA 表。

## 可复用 idea 点
- 单窗 ε(tR,tT) 免双体积相关作存储分母合同
- STSSM：时空 patch→SSM→回投作编码器模板
- Mamba 选择性 vs S4/S4D/S5 消融法
- 与 IDNet（无相关体）/TMA（有相关体）成簇 | 负结果只停 STSSM 头替换

## 杀门建议
替换无增益或 SSM 状态费用挤占 Stage B → 保持旁路。
