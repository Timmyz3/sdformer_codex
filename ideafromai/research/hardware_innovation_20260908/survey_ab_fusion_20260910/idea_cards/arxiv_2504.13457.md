# arXiv:2504.13457 · Neural Ganglion Sensors

- uid/来源: `ARX-007`｜arxiv_2504.13457+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: Neural Ganglion Sensors: Learning Task-specific Event Cameras Inspired by the Neural Circuit of the Human Retina
- 精读深度: 方法级（仅方法摘录窗口）+依据：RGC 空间核 W∗(Icurr−Imem)；可微分箱闭式；学核/正负阈/多通道 Bayer 式；退化为恒等核=传统事件；完整任务头细部可能截断

## 可继承 A
可学习视网膜神经节式事件传感（空间核+阈值）+ 可微分箱反传——任务专用事件前端与传感–任务共训对照（借入≠X）。

## 强对照 B
像素独立传统事件（W=I）；手刻 Sobel/梯度 RGC；不可微 ESIM/V2E 固定参数；仅后处理事件表示不学传感。

## 可差分 X线索
学到的事件相机≠lifting X；传感前端旁路，勿搬下游任务精度当净服务%。

## 与 F1–F7 / Stage B 关系
F1弱相关（表示/前端选择）。传感旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
下游视觉任务指标≠valid825；仅摘录窗；可微仿真≠真实传感器硅实现。

## 可复用 idea 点
- W∗ΔI+双阈作可学习事件生成合同
- 闭式可微分箱作传感–体素共训旁证
- 多通道/Bayer 式多种 RGC 并行作特征稀疏化模板
- 恒等核退化对照传统事件 | 负结果只停该前端替换

## 杀门建议
学核无增益或不可映射硅实现挤占 Stage B → 保持旁路。
