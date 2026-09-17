# arXiv:2306.15220 · S-TLLR

- uid/来源: `ARX-017`｜arxiv_2306.15220+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking Neural Networks
- 精读深度: 方法级（仅方法摘录窗口）+依据：§4.1–4.2 三因子规则、瞬时eligibility O(n)、因果+非因果项、Algo1；复发连接§4.2.1；完整训练表/附录可能截断

## 可继承 A
STDP启发时空局部三因子学习：丢弃eligibility递归→O(n)内存，兼用因果/非因果尖峰时序——在线可训SNN与时间局部梯度合同对照（借入≠X）。

## 强对照 B
BPTT全时序存状态；e-prop/OSTL等O(n^2) eligibility；纯无监督STDP；仅因果近似BPTT的OTTT类规则。

## 可差分 X线索
S-TLLR学习规则≠lifting源字/半步图X；差分只在训练期时间局部合同旁路。

## 与 F1–F7 / Stage B 关系
训练/可训性旁证；弱挂时间局部。不抢 Stage B。f_candidates空。

## 不可搬用边界
分类准确率/复杂度表≠valid825 AEE；仅摘录窗；O(n)勿写same-port%。

## 可复用 idea 点
- 瞬时eligibility（β=0）作时间局部可训内存合同
- 因果+非因果双项作时序相关对照
- 层间反传、时间局部→可DFA空间局部分拆
- 负结果只停该规则替换，不杀三因子家族

## 杀门建议
同预算不胜BPTT/OTTT，或eligibility旁路吃满RF → 停该规则作主训，保留旁证。
