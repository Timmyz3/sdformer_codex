# arXiv:2307.01694 · Spike-driven Transformer

- uid/来源: `MAIN-R255`｜arxiv_2307.01694+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: Spike-driven Transformer
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 SPS+膜捷径；§3.3 SDSA-V1（Hadamard/掩码、无softmax）；§4 理论能耗；完整补充材料可能截断

## 可继承 A
膜势捷径（MS）保二进制尖峰驱动 + Spike-driven Self-Attention（列和+掩码、取消scale/softmax）——稀疏加法/地址化注意力聚合对照（借入≠X）。

## 强对照 B
SEW捷径多比特尖峰；SpikFormer浮点Q/K仍乘；Vanilla Transformer MAC+softmax；软注意力连续分值。

## 可差分 X线索
SDSA/MS≠lifting结构化PSN X；表示与注意力旁路，差分不在ImageNet%。

## 与 F1–F7 / Stage B 关系
F1/F7相关（尖峰表示、注意力打包）。旁路基线。不抢 Stage B。f_candidates含F1,F7。

## 不可搬用边界
ImageNet/CIFAR/DVS精度与理论EAC≠valid825/same-port%；仅摘录窗；SDSA能量勿外推芯片mW。

## 可复用 idea 点
- MS捷径保证后接SN仅为二进制→spike-driven加法合同
- Hadamard/掩码替换矩阵乘作硬注意力旁证
- 稀疏率R·T·FLOPs作能耗分母模板
- 与Spikformer/Meta-SpikeFormer成簇 | 负结果只停SDSA布局

## 杀门建议
挂主网无增益或注意力/掩码费用挤占 Stage B → 保持旁路。
