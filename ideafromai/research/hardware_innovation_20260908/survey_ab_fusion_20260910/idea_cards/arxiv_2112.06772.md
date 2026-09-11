# arXiv:2112.06772 · hARMS

- uid/来源: `MAIN-R241`｜arxiv_2112.06772+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow
- 精读深度: 方法级（仅方法摘录窗口）+依据：ARMS 多尺度邻域选窗（argmax |Un|）；事件级无监督孔径鲁棒流；FPGA/SoC 并行加速；吞吐至约 1.21 Mevent/s；摘录偏 Intro+算法背景，完整 RTL/流水细部可能截断

## 可继承 A
多尺度邻域选「孔径」+事件级局部流硬件加速——实时事件光流前端与并行事件服务对照（借入≠X）。

## 强对照 B
固定小窗 Lucas/平面拟合（孔径失败）；帧累积再算流；无硬件并行的纯软件 ARMS。

## 可差分 X线索
hARMS FPGA≠lifting 数字链 X；旁路/前端，勿搬 Mevent/s 当净服务%。

## 与 F1–F7 / Stage B 关系
运动前端旁路；弱 F2（事件服务）。不抢 Stage B。

## 不可搬用边界
FPGA 吞吐/资源≠valid825/same-port%；摘录方法细部偏背景；禁止虚报已读完整微架构。

## 可复用 idea 点
- 多尺度 argmax|Un| 作无监督窗选择合同
- 事件级流水作服务分母旁证
- 与平面拟合 FPGA unresolved 卡成簇勿双计
- 负结果只停 hARMS 前端替换

## 杀门建议
前端替换无 AEE/服务增益或挤占 Stage B → 保持旁路。
