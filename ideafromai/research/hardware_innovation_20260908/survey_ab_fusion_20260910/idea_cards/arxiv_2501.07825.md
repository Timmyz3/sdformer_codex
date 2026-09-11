# arXiv:2501.07825 · SMAM / Spike-Driven Transformer HW

- uid/来源: `MAIN-R063`｜arxiv_2501.07825+本地excerpt（`p0_excerpt_batches/batch_02.json`）
- 题名: An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer
- 精读深度: 方法级（仅方法摘录窗口）+依据：脉冲位置编码→地址比较消零；单/双脉冲输入；覆盖 maxpool/SDSA/linear；加/比较代乘；SPS/SDEB 核；FPGA 评测

## 可继承 A
把脉冲计算改成位置编码+地址比较以旁路零——双稀疏输入（SDSA）友好的执行编码底座；可作 Gustav/lifting 非零源字供数对照（借入≠X）。

## 强对照 B
稠密 MAC 注意力/线性；只优化卷积脉冲、忽略 SDSA 双输入；叠加式通用 SNN 加速器。

## 可差分 X线索
地址比较消零≠标题 X；须证明对 lifting 源活动降 same-port 周期且过 AEE。

## 与 F1–F7 / Stage B 关系
F2/F4/F7 数据通路对照；旁路主岛排程。不抢 Stage B。

## 不可搬用边界
Cifar-10 Spike-driven Transformer FPGA≠光流 valid825；勿搬 13.24×/1.33×；仅方法摘录窗口。

## 可复用 idea 点
- 位置编码+比较代乘作非零交汇执行模板
- 双脉冲输入 SDSA 模块边界对照多消费者并集
- 加/比较统一算子降低控制器——对照 Stage B 少改数据通路纪律
- 负结果只停该编码单元布局

## 杀门建议
编码/比较开销≥稀疏收益或精度门失败 → 停该 SMAM 布局。
