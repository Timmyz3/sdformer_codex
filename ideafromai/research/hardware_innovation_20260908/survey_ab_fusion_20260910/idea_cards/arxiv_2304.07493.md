# arXiv:2304.07493 · OliVe

- uid/来源: `MAIN-R052`｜arxiv_2304.07493+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: OliVe: Accelerating Large Language Models via Hardware-friendly Outlier-Victim Pair Quantization
- 精读深度: 方法级（仅方法摘录窗口）+依据：Algo1 OVP 成对编码（outlier-normal→victim）、abfloat 自适应偏置离群类型、MSE 选阈值控 outlier-outlier 比、轻量 OVP decoder 挂 Tensor Core、PTQ

## 可继承 A
成对离群-牺牲品量化 + 硬件友好解码——普通压缩/异常值处理强对照；与 GOBO 并列作量化底座（借入≠X）。

## 强对照 B
逐元素离群存储无 victim；对称均匀量化；需重训的 QAT。

## 可差分 X线索
OVP≠标题 X；本地差分不在 LLM 量化格式。

## 与 F1–F7 / Stage B 关系
普通压缩对照；量化轴旁路。不抢 Stage B。

## 不可搬用边界
LLM/Transformer 权重激活分布；OVP 成对假设迁到源字可能破坏 lifting 图；仅方法摘录窗口。

## 可复用 idea 点
- 成对牺牲品策略作「保关键保留、其余降精度」对照
- 轻量 decoder 挂既有 MAC 阵列——少改数据通路的工程边界
- MSE+outlier-outlier 比率作量化杀门数字模板
- 负结果只停 OVP 网格

## 杀门建议
同精度不胜 GOBO/均匀量化或服务无增益 → 停该 OVP 布局。
