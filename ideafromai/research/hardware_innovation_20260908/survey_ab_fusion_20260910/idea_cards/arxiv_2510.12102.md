# arXiv:2510.12102 · SpikePool

- uid/来源: `MAIN-R213`｜arxiv_2510.12102+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: SpikePool: Event-driven Spiking Transformer with Pooling Attention
- 精读深度: 方法级（仅方法摘录窗口）+依据：频域 RLA 分析（SNN transformer 高通）；Pooling Attention 用 2D max pooling 替代 SSA；频域感知带通设计动机

## 可继承 A
用池化注意力替换二次 SSA，以低通池化平衡尖峰变压器固有高通——算法侧降注意力复杂度的强简化对照（借入≠X）。

## 强对照 B
标准 SSA/QKFormer 式自注意力；纯卷积无注意力；无频域动机的随意池化。

## 可差分 X线索
池化注意力≠ lifting X；事件分类/检测延迟≠本地光流 same-port。

## 与 F1–F7 / Stage B 关系
F7 弱相关（降注意力费用）；旁路。不抢 Stage B。

## 不可搬用边界
训练/推理时间 −42%/−33% 勿写成硬件净服务%；仅方法摘录窗口。

## 可复用 idea 点
- 高通偏差诊断作「为何事件任务需带通」阅读
- max-pool 注意力作 O(N²) 消除强基线
- 与 SDT/SSA 特征图对照阅读
- 负结果只停「池化注意力挂光流头」

## 杀门建议
替换注意力后 AEE/任务门不过或服务不降 → 停该替换布局。
