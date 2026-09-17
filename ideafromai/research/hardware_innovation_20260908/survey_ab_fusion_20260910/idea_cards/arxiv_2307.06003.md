# arXiv:2307.06003 · USFlow / Dynamic Timing Representation

- uid/来源: `ARX-016`｜arxiv_2307.06003+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera
- 精读深度: 方法级（仅方法摘录窗口）+依据：§4.2 TMR多膨胀1D卷积+Layer Attention；PWC/RAFT双骨干；§4.3无监督损失入口；完整损失公式可能截断

## 可继承 A
尖峰相机流上的动态时序表示（像素共享多膨胀时域卷积+层注意力融合）+无监督光流——时间窗自适应打包与输入表示对照（借入≠X）。

## 强对照 B
固定长度尖峰窗；仅计数/最新戳事件图；纯监督真值流；无层注意力的单尺度时域卷积。

## 可差分 X线索
TMR/USFlow≠lifting X；尖峰相机前端旁路，差分不在PHM EPE。

## 与 F1–F7 / Stage B 关系
F2/F7相关（时间表示打包）。旁路基线。不抢 Stage B。f_candidates含F2,F7。

## 不可搬用边界
Spike camera/PHM≠DSEC/valid825；仅摘录窗；禁止虚报已读完整§4.3损失闭式。

## 可复用 idea 点
- 多膨胀1D时域作长短窗并集表示合同
- Layer Attention融多层作自适应时间选择旁证
- 同表示接PWC/RAFT作骨干无关检查
- 与EV-FlowNet/SCFlow成簇 | 负结果只停TMR替换入口

## 杀门建议
表示替换无增益或膨胀费用挤占 Stage B → 保持旁路。
