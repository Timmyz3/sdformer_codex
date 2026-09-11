# arXiv:2303.07169 · Dynamic Event OCC+Spiking OF ID

- uid/来源: `ARX-019`｜arxiv_2303.07169+本地excerpt（`p0_excerpt_batches_gap/gap_02.json`）
- 题名: Dynamic Event-based Optical Identification and Communication
- 精读深度: 方法级（仅方法摘录窗口）+依据：红外 beacon OCC（850nm/ESP32）；事件相机解码；动态场景 Kalman+尖峰光流辅助跟踪；摘录偏§3 静/动态评测与讨论，完整§2 算法细部可能截断

## 可继承 A
事件 OCC 识别 + 尖峰光流辅助动态跟踪——事件通信/身份前端与运动补偿旁证（借入≠X）。

## 强对照 B
纯帧 OCC；无光流辅助的盲跟踪；无 beacon 的外观识别。

## 可差分 X线索
OCC/beacon 系统≠lifting X；旁路传感，勿搬 MAR/BAR。

## 与 F1–F7 / Stage B 关系
传感/应用旁路；与光流前端弱相关。不抢 Stage B。

## 不可搬用边界
MAR/BAR/距离≠valid825；摘录偏评测禁止虚报已读完整光流网络；仿真≠芯片。

## 可复用 idea 点
- Kalman∩尖峰光流作动态身份稳定合同
- beacon 时分编码作事件服务旁证
- 与事件光流卡成簇勿当执行 X
- 负结果只停 OCC 前端替换

## 杀门建议
无主网增益或挤占 Stage B → 保持旁路。
