# arXiv:2503.12905 · UCF-Crime-DVS

- uid/来源: `ARX-008`｜arxiv_2503.12905+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`；PDF Methods 窗补齐）
- 题名: UCF-Crime-DVS: A Novel Event-Based Dataset for Video Anomaly Detection with Spiking Neural Networks
- 精读深度: 方法级（数据集+MSF 方法窗+PDF补齐）+依据：事件帧积分；MSF 多尺度时域膨胀卷积+LIF；SpikingGCN 全局时依；弱监督 VAD；完整采集协议细部可能截断

## 可继承 A
事件异常检测数据集 + 多尺度尖峰融合（局部膨胀时域 + SpikingGCN 全局）——事件时序表示与弱监督旁证（借入≠X）。

## 强对照 B
仅 RGB 帧 UCF-Crime；单尺度尖峰池化；非尖峰 Transformer 全精度注意力；无全局图的纯局部时域。

## 可差分 X线索
UCF-Crime-DVS/MSF≠lifting X；应用/数据旁路，勿搬 AUC。

## 与 F1–F7 / Stage B 关系
F1/F2弱相关（事件表示、时间多尺度）。旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
异常检测 AUC≠valid825；双栏混排摘录须 PDF 补齐；仿真/转换事件≠芯片传感。

## 可复用 idea 点
- 固定微秒窗积分事件帧作输入合同
- 金字塔时域膨胀+LIF 作多尺度局部尖峰旁证
- SpikingGCN 抓跨 clip 全局时依
- 与动作/异常事件集成簇 | 负结果只停 MSF 头/该数据集替换

## 杀门建议
替换无增益或挤占 Stage B → 保持旁路。
