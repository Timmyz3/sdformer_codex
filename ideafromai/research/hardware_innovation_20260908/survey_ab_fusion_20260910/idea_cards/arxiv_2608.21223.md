# arXiv:2608.21223 · IPZO Event-Triggered Perturbation

- uid/来源: `ARX-023`｜arxiv_2608.21223+本地excerpt（p0_excerpt_batches_gap/gap_08.json）+补arxiv HTML方法（§IV）
- 题名: Event-triggered Implicit Perturbation for Zeroth-Order Fine-Tuning of Spiking Transformers
- 精读深度: 方法级（原摘录偏能效/参考文献；补HTML §IV架构）+依据：IPZO输出域隐式扰动消RMW；事件触发行缩减PGU；PGU-XOR地址分解XOR重组；争用串行累加网络；相对EPZO/PGU-Reuse

## 可继承 A
IMC上ZO微调：扰动和与加权和在输出域合并（保权驻留、消扰动RMW）+ 尖峰稀疏行缩减PGU + PGU-XOR(商/余LFSR异或)抑相关 + 争用有界串行累加——相对显式改权EPZO与直接复用相关扰动的片上学习对照（借入≠X）。

## 强对照 B
EPZO全阵列读改写扰动；每权独立RNG满配；PGU-Reuse分组共享致相关；一阶BPTT存中间态。

## 可差分 X线索
IPZO/PGU≠lifting X；片上ZO学习旁路，差分不在CIFAR/PPL或16nm面积点。

## 与 F1–F7 / Stage B 关系
加速器/片上学习旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
Spikingformer/SpikeGPT点与TSMC16nm开销≠valid825；须以HTML§IV为准；隐式扰动≠lifting源字删字合同。

## 可复用 idea 点
- 输出域合并扰动作保权驻留合同
- 事件行缩减PGU作稀疏触发维度模板
- PGU-XOR地址异或作消相关旁证
- 争用串行∥IMC流水隐藏延迟作时序边
- 负结果只停该ZO-IMC挂接

## 杀门建议
与Stage B同分母冲突或仅作微调叙事无端口差分 → 保持旁路。
