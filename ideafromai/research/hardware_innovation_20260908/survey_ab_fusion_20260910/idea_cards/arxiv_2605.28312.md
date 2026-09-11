# arXiv:2605.28312 · EventShiftFlow

- uid/来源: `MAIN-R221`｜arxiv_2605.28312+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: EventShiftFlow: Towards Hardware-efficient FPGA-based Flow Estimation
- 精读深度: 方法级（仅方法摘录窗口）+依据：面向 FPGA 的移位式光流估计；无帧重建、无浮点、无迭代优化；密度相关参数；单轴原型、<2kB 存储；与平面拟合等对照

## 可继承 A
硬件友好的事件光流移位 datapath（确定性、低存储）——同任务域（光流）的非学习/轻量基线对照（借入≠X）。

## 强对照 B
迭代优化/平面拟合重算法；需帧重建的 DNN 光流；高资源 FPGA 光流。

## 可差分 X线索
经典移位光流≠ SNN Transformer+lifting 标题 X；可作任务域弱基线，不进主岛新颖性。

## 与 F1–F7 / Stage B 关系
旁路（同任务非 SNN 基线）。不抢 Stage B。

## 不可搬用边界
合成/真实速度误差≠AEE 合同口径；FPGA LUT/BRAM 勿写成 Stage B%；仅方法摘录窗口。

## 可复用 idea 点
- 无 FP/无迭代作「可综合光流核」下界
- <2kB 存储作极端 SWaP 对照
- 密度相关参数作鲁棒性旋钮
- 与本地学习式光流 SNN 并读时只作任务基线

## 杀门建议
对主岛主张无差分贡献 → 保持旁路基线，不进 A+B 标题。
