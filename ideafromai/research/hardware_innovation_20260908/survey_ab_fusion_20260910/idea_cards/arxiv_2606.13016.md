# arXiv:2606.13016 · Otters++ TTFS Optical Spiking Transformer

- uid/来源: `ARX-031`｜arxiv_2606.13016+本地excerpt（`p0_excerpt_batches_gap/gap_06.json`）
- 题名: Otters++: A Time-to-first-spike Based Energy Efficient Optical Spiking Transformer
- 精读深度: 方法级（仅方法摘录窗口）+依据：光电子衰减作TTFS时间核；SNN前向/QNN反向STE；DFT层窗同步；设备噪声采样；系统能量含共享/多跳；完整光学工艺可能截断

## 可继承 A
物理TTFS光尖峰Transformer：器件衰减当时间算子 + SNN前向/等价QNN反向STE + 设备方差感知——光电子共设计与数字TTFS对照（借入≠X）。

## 强对照 B
软件显式衰减TTFS；直接对首尖微分；忽略器件噪声的后置校准；仅算子计数无数据移动的能量模型。

## 可差分 X线索
Otters++光TTFS≠lifting X；光学/训练旁路，勿搬GLUE分或层能量×当净服务%。

## 与 F1–F7 / Stage B 关系
光学模拟旁路；与F族弱挂。不抢 Stage B。f_candidates空。

## 不可搬用边界
GLUE/层能量≠valid825；仅摘录窗；模拟光突触≠数字same-port合同。

## 可复用 idea 点
- 光衰减响应代数字时间核作TTFS合同
- SNN前∥QNN反STE作可训模板
- DFT层窗同步作因果旁证
- 设备上下界采样+共享/多跳能量作鲁棒/计价边
- 负结果只停该光学TTFS挂接

## 杀门建议
器件漂移或训练仍过稀疏挤占 Stage B → 保持旁路。
