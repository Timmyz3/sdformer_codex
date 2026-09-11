# arXiv:1802.06898 · EV-FlowNet

- uid/来源: `MAIN-R261`｜arxiv_1802.06898+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: EV-FlowNet: Self-Supervised Optical Flow Estimation for Event-based Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：事件图=计数+最近时间戳双极性通道；时间窗归一化；自监督 photometric+smooth；U-Net/hourglass；摘录进入§III 损失/架构，完整训练表可能截断

## 可继承 A
异步事件→同步图像通道（计数+最新戳）+自监督光度合同——事件光流输入表示与无真值训练底座（借入≠X）。

## 强对照 B
仅平面拟合/局部 Lucas 事件流；需稠密真值监督；无时间窗归一化的原始戳输入。

## 可差分 X线索
事件图/自监督损失≠lifting X；同任务算法旁路。

## 与 F1–F7 / Stage B 关系
F2弱相关（时间窗打包）；旁路基线。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
MVSEC EPE≠valid825 AEE；仅摘录窗；勿把自监督增益写成 same-port%。

## 可复用 idea 点
- 计数+最新戳双通道作输入合同对照
- 时间窗归一化对齐快慢运动
- photometric+smooth 作无真值训练模板
- 与 Spike-FlowNet/E-RAFT/Adaptive-SpikeNet 成簇
- 负结果只停 EV-FlowNet 头替换

## 杀门建议
替换无增益或挤占 Stage B → 保持旁路。
