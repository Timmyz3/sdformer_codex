# arXiv:2209.09570 · FABNet

- uid/来源: `MAIN-R088`｜arxiv_2209.09570+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design
- 精读深度: 方法级（仅方法摘录窗口）+依据：统一 butterfly 稀疏同时近似 attention 与 FFN、静态细粒度规则稀疏、运行时可配置单一引擎覆盖不同 butterfly 层、与动态非结构注意力加速器对比表

## 可继承 A
静态规则结构化稀疏 + 可配置统一引擎——可作 F7「因子组对齐打包」与普通结构稀疏的硬件友好对照（借入≠X）。

## 强对照 B
仅优化 attention 或仅 FFN；动态非结构稀疏+重控制器；无统一引擎的多专用核。

## 可差分 X线索
Butterfly 替换注意力≠ lifting 结构化 T10 X；借用模式规律≠自称差分。

## 与 F1–F7 / Stage B 关系
F7 结构规律对照；旁路。不抢 Stage B。

## 不可搬用边界
Transformer/LRA；butterfly 因子图≠本地 lifting matchings；仅方法摘录窗口（硬件§IV 细节截断）；勿搬 14–23×。

## 可复用 idea 点
- 静态规则稀疏避免动态控制器开销——对照 F1 共享掩码可实现性
- 统一引擎多模式配置作 Gustav/lifting 同分母复用思路
- 同时稀疏 attention+FFN 的端到端可扩展性警告：只优一段不够
- 与 SpAtten 动态粗粒度对照，选静态 vs 动态杀门

## 杀门建议
规则 butterfly 映射到源字后精度/服务不胜 HiNM 或稠密序+同 Gustav → 停该模式。
