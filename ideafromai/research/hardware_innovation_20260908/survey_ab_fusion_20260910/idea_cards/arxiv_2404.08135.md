# arXiv:2404.08135 · SciFlow

- uid/来源: `MAIN-R234`｜arxiv_2404.08135+本地excerpt（`p0_excerpt_batches_gap/gap_03.json`）
- 题名: SciFlow: Empowering Lightweight Optical Flow Models with Self-Cleaning Iterations
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 SCI高斯相似图拼入ConvGRU；§3.2 RFL置信加权；§3.3二者组合；挂MobileFlow等轻量骨干

## 可继承 A
自清洁迭代：用参考–扭曲特征高斯相似作SCI图指导GRU更新 + Regression Focal Loss重加权难像素——轻量迭代光流自评估/聚焦对照（借入≠X）。

## 强对照 B
无自评估的朴素迭代；仅监督L1无focal；重型RAFT堆算力换精度。

## 可差分 X线索
SCI/RFL≠lifting X；训练/推理辅助旁路。

## 与 F1–F7 / Stage B 关系
弱F2/F4（迭代自清洁/有损聚焦）。旁路基线。不抢 Stage B。f_candidates空。

## 不可搬用边界
Sintel/KITTI增益≠valid825；轻量参数量勿写same-port%；仅摘录窗。

## 可复用 idea 点
- SCI图作无GT前向自评估合同（训推皆用）
- RFL用终局置信回加权全迭代作难例聚焦
- SCI与RFL协同消融模板
- 与DFlow迭代策略成簇对照 | 负结果只停SCI头

## 杀门建议
自清洁无增益或相似图费用抵消轻量优势 → 停该头，保留轻量骨干。
