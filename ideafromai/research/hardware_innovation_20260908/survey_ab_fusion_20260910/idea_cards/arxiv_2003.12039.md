# arXiv:2003.12039 · RAFT

- uid/来源: `MAIN-R311`｜arxiv_2003.12039+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：全对相关体+金字塔池化；局部 lookup；权共享 ConvGRU 迭代更新；上下文编码器；摘录§3.1–3.3+实验消融，完整补充材料可能截断

## 可继承 A
全对相关金字塔 + 权共享循环更新算子——稠密光流迭代精修强底座（借入≠X；E-RAFT/SEA-RAFT/EDCFlow 下游簇）。

## 强对照 B
单次回归光流头；无全对相关的局部相关；无 GRU 门控的多层卷积更新；权不共享逐迭代。

## 可差分 X线索
RAFT 迭代头≠lifting 源字 X；同任务旁路，差分不在 Sintel/KITTI EPE。

## 与 F1–F7 / Stage B 关系
F7弱相关（相关体/迭代打包）；旁路基线。不抢 Stage B。与 OS SEA-RAFT 开源卡联读勿双计。

## 不可搬用边界
帧光流基准≠事件 valid825；迭代次数–精度勿写成 same-port%；仅摘录窗。

## 可复用 idea 点
- 相关金字塔+lookup 作多尺度分母
- 权共享+门控鼓励收敛→迭代预算杀门
- warm-start 接口留给 E-RAFT
- 负结果只停「RAFT 头替换本地」

## 杀门建议
挂主网无增益或迭代费用挤占 Stage B → 保持旁路。
