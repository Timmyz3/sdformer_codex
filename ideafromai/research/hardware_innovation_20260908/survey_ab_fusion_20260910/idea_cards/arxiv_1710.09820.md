# arXiv:1710.09820 · TrueNorth spiking optical flow

- uid/来源: `MAIN-R244`｜arxiv_1710.09820+本地excerpt（`p0_excerpt_batches_gap/gap_01.json`）
- 题名: Spiking Optical Flow for Event-based Sensors Using IBM's TrueNorth Neurosynaptic System
- 精读深度: 方法级（仅方法摘录窗口）+依据：Barlow–Levick DS 单元（兴奋+延时抑制方向选择）；四向组合；TrueNorth 256×256 突触核拓扑/轴突扇出；摘录混排 Intro+§II–III 图，完整映射/评测公式可能截断

## 可继承 A
事件驱动方向敏感（DS）尖峰单元+邻像素时差→速度——早期「时序相关/延时抑制」运动前端与低功耗神经形态部署对照（借入≠X）。

## 强对照 B
帧采样光流（Horn–Schunck/Lucas–Kanade）；无方向抑制的局部相关；纯软件离线事件批处理。

## 可差分 X线索
TrueNorth/DS 映射≠lifting 源字 X；同任务旁路，差分不在 AEE% 或芯片 mW。

## 与 F1–F7 / Stage B 关系
F2弱相关（事件时序打包）；旁路基线。不抢 Stage B。

## 不可搬用边界
原作 AEE≈11%/<80mW≠valid825/same-port%；TrueNorth 专用核≠nts07；仅摘录窗；禁止虚报已读完整§III 映射表。

## 可复用 idea 点
- 兴奋–抑制时差作方向门控对照
- 轴突扇出减通信体积作多消费者广播分母旁证
- 与 EV-FlowNet/Spike-FlowNet 成簇读早期→深度线
- 负结果只停「DS 单元替换本地出口」

## 杀门建议
挂主网无 AEE/服务增益或挤占 Stage B → 保持旁路。
