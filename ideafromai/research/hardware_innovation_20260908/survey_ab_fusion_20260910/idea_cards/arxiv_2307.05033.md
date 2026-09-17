# arXiv:2307.05033 · EVA-Flow

- uid/来源: `MAIN-R226`｜arxiv_2307.05033+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: Towards Anytime Optical Flow Estimation with Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2 UVG串行+SMR（ConvGRU时空递推、粗到细）；§3.3仅监督末步；§3.4 RFWL；UVG构造细部可能截断

## 可继承 A
事件bin串行低延迟输入 + Spatiotemporal Motion Recurrent（金字塔warp+ConvGRU残差更新）——anytime/时间密光流与状态继承对照（借入≠X）。

## 强对照 B
整段一次出流（非anytime）；E-RAFT双体积相关无逐bin输出；线性运动假设插值中间流；无RFWL的FWL。

## 可差分 X线索
SMR/anytime≠lifting源字X；同任务旁路，差分不在DSEC EPE%。

## 与 F1–F7 / Stage B 关系
F2弱相关（时间密递推/检查点）。旁路基线。不抢 Stage B。f_candidates空。

## 不可搬用边界
DSEC/RFWL≠valid825；仅摘录窗；中间步无显式监督≠本地合同；禁止虚报已读完整UVG定义。

## 可复用 idea 点
- 逐bin输出作anytime服务合同
- 同层共享权重横时+异层竖空作递推模板
- 仅监督末步+稠密warp隐式约束中间流
- RFWL矫正FWL尺度 | 与E-RAFT/TMA/BAT成簇 | 负结果只停EVA头替换

## 杀门建议
替换无AEE/服务增益或递推费用挤占 Stage B → 保持旁路。
