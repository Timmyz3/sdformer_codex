# arXiv:2509.18968 · Otters

- uid/来源: `ARX-048`｜arxiv_2509.18968+本地excerpt（`p0_excerpt_batches_gap/gap_05.json`）
- 题名: Otters: An Energy-Efficient SpikingTransformer via Optical Time-to-First-Spike Encoding
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 Otters光电子突触TFT衰减O(t)；动态阶梯阈θ(t)；1-bit K/V注意+Canon式PE；Prop.1 QNN→SNN无损转换；能模型细部可能截断

## 可继承 A
光TFT物理衰减+动态阶梯阈实现TTFS；1-bit K/V使TTFS注意退化为加/减——光电TTFS Transformer与无损量化转换对照（借入≠X）。

## 强对照 B
均匀时钟恒定阈；率编码SNN注意；全精度Q·K^T乘；直接SNN训练无QNN映射。

## 可差分 X线索
Otters/TTFS光电≠lifting X；光电器件旁路，勿搬能模EAnalog当净服务%。

## 与 F1–F7 / Stage B 关系
F2/F7弱相关（时间编码、注意打包）。器件/转换旁路。不抢 Stage B。f_candidates空。

## 不可搬用边界
ImageNet/能模≠valid825；仅摘录窗；TFT拟合参数≠本地事件时间合同。

## 可复用 idea 点
- 非线衰减用阶梯θ(tk)对齐量化电平作TTFS合同
- 1-bit K/V+选择性加减作注意乘积分母
- Prop.1 γ=w·α·T与θ=α(T−k)作QNN无损映射模板
- Canon式PE广播TTFS/Q作稀疏数据流旁证 | 负结果只停该光电TTFS挂接

## 杀门建议
器件不可复现或转换精度掉点挤占 Stage B → 保持旁路。
