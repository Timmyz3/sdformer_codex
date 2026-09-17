# arXiv:2501.14484 · SpikePack

- uid/来源: `ARX-190`｜arxiv_2501.14484+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: SpikePack: Enhanced Information Flow in Spiking Neural Networks with High Hardware Compatibility
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2 全局膜 vgl=W S q；动态阈解码；slzip 压缩 O(1)；免 BPTT 梯度；互信息对照 LIF；完整附录推导可能截断

## 可继承 A
全时序打包为全局膜电位 + 整数 zip 压缩尖峰序列 + 并行 GeMM——SNN 信息保留与硬件友好时间折叠对照（借入≠X）。

## 强对照 B
逐步递归 LIF O(T) 串行；逐步存尖峰 O(T) 空间；每步 spike×W 欠用 GeMM；依赖 BPTT+代理梯度逐步反传。

## 可差分 X线索
SpikePack 神经元≠lifting 结构化 PSN X；表示/神经元旁路，差分不在 ImageNet%。

## 与 F1–F7 / Stage B 关系
F1/F2弱相关（尖峰表示、时间折叠）。旁路基线。不抢 Stage B。f_candidates空。

## 不可搬用边界
分类/检测精度与互信息≠valid825；仅摘录窗；O(1) 压缩勿写 same-port%；τ 量化≠本地定点声明。

## 可复用 idea 点
- vgl=W(Sq) 一次聚合作时间折叠合同
- slzip 位压缩作尖峰序列存储旁证
- 免逐步 BPTT 的直接梯度作训推简化模板
- 与 LIF 互信息对照作信息分母 | 负结果只停该神经元替换

## 杀门建议
替换无增益或打包损害时间可分性挤占 Stage B → 保持旁路。
