# arXiv:2501.11554 · Precise-event-timing egomotion

- uid/来源: `MAIN-R220`｜arxiv_2501.11554+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: Event-based vision for egomotion estimation using precise event timing
- 精读深度: 方法级（仅方法摘录窗口）+依据：全事件域流水（无帧聚合）；精确事件时序→脉冲突发编码局部光流速度；片上评测；相对 LIF 特征不足的批评；含 ITDE 等模块痕迹

## 可继承 A
保留事件精确时序的 egomotion 流水（时序→局部速度脉冲）——运动旁路/事件合同对照；反衬「聚帧再 ANN」损失（借入≠X）。

## 强对照 B
均匀时间窗聚帧+CNN/ANN VO；纯轮速/IMU；忽略亚毫秒时序的 LIF 特征。

## 可差分 X线索
Egomotion 旁路≠ r1 lifting 主岛 X；有限份额，不排 Stage B。

## 与 F1–F7 / Stage B 关系
旁路（motion）；明确不进 F1–F7 主线。不抢 Stage B。

## 不可搬用边界
专用芯片 egomotion；非端到端光流生产网；仅方法摘录窗口（结构细节不全）。

## 可复用 idea 点
- 精确时序→速度脉冲作事件输入合同对照
- 「禁聚帧」主张对齐本地是否保留异步性的产品纪律
- 与 TDE-3 旁路成对，份额有限
- 负结果只停 egomotion 旁路布局

## 杀门建议
接入主网无净 AEE/服务或挤占主岛档期 → 停旁路，不杀主实验。
