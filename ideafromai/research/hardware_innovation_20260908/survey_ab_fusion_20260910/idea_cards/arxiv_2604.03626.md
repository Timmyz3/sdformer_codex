# arXiv:2604.03626 · L-SPINE

- uid/来源: `MAIN-R071`｜arxiv_2604.03626+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: L-SPINE: A Low-Precision SIMD Spiking Neural Compute Engine for Resource-efficient Edge Inference
- 精读深度: 方法级（仅方法摘录窗口）+依据：低精度 SIMD 脉冲计算引擎；摘录可见 CORDIC/无乘/PWL/查找表等算术线索与可适配流水级；图多文碎，完整 ISA/数据流可能截断

## 可继承 A
面向边缘的低精度 SIMD SNN 引擎（无乘/近似算术族）——普通低精度执行对照底座（借入≠X）。

## 强对照 B
全精度浮点 SNN；纯标量非 SIMD；通用 CPU 解释执行。

## 可差分 X线索
低精度 SIMD≠结构化 T10 X；PoT/移位等作强对照时权限同等。

## 与 F1–F7 / Stage B 关系
旁路（边缘算术引擎）。不抢 Stage B。

## 不可搬用边界
摘录噪声大、方法段不完整；勿搬未核 PPA；仅方法摘录窗口。

## 可复用 idea 点
- 无乘/PWL/LUT 近似作普通压缩对照
- SIMD 打包粒度对照源字向量化
- 可适配流水级作精度-资源旋钮
- 证据不足时只保留「待补全文」标记

## 杀门建议
证据不足以支撑迁移主张 → 不升主岛；补全文前不作硬件承诺。
