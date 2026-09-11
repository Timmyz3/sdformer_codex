# arXiv:2108.10552 · E-RAFT

- uid/来源: `MAIN-R228`｜arxiv_2108.10552+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: E-RAFT: Dense Optical Flow from Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：双连续事件体素体积作 RAFT 相关对；上下文编码器；前向扭曲/平均 splat 暖启动；提出 DSEC-Flow；摘录§3.2–4，完整训练超参可能截断；ERAFT FPGA unresolved 另卡勿双计

## 可继承 A
事件体素双体积相关 + 前向扭曲暖启动——事件稠密光流迭代底座与时序状态继承对照（借入≠X）。

## 强对照 B
单窗事件图直接回归（EV-FlowNet）；无暖启动复制上帧流；仅 MVSEC 自监督基线。

## 可差分 X线索
E-RAFT/DSEC≠lifting X；同任务旁路。

## 与 F1–F7 / Stage B 关系
F2/F7弱相关（时间体积、暖启动状态）；旁路基线。不抢 Stage B。

## 不可搬用边界
DSEC/MVSEC EPE≠valid825；暖启动≠same-port 状态合同；仅摘录窗；勿与 MAIN-R291 FPGA unresolved 双计。

## 可复用 idea 点
- 双体积相关作时序打包合同
- 前向扭曲暖启动作状态继承模板（对照 shared-Q 消融纪律）
- DSEC-Flow 位移分布作难度表
- 与 RAFT/BAT/TMA 成簇
- 负结果只停 E-RAFT 头替换

## 杀门建议
替换无增益或迭代费用挤占 Stage B → 保持旁路。
