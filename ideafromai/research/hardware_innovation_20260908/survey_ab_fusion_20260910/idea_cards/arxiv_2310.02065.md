# arXiv:2310.02065 · VENOM

- uid/来源: `MAIN-R027`｜arxiv_2310.02065+本地excerpt（`p0_excerpt_batches/batch_01.json`）
- 题名: VENOM: A Vectorized N:M Format for Unleashing the Power of Sparse Tensor Cores
- 精读深度: 方法级（仅方法摘录窗口）+依据：V:N:M 向量化格式突破固定 2:4；Spatha 模板 SpMM；二阶剪枝适配高稀疏；column-loc 元数据；与块剪枝/向量剪枝/N:M 对比

## 可继承 A
半结构化 V:N:M + 二阶剪枝 + 开源内核——F1 开源强对照工具链（HiNM/2:4/VENOM）；借入格式≠X。

## 强对照 B
非结构幅值剪枝；固定 2:4 cuSparseLt；粗块剪枝伤精度。

## 可差分 X线索
VENOM 模式训练/内核≠自动 X；须同槽服务+AEE 对比。

## 与 F1–F7 / Stage B 关系
F1 强对照；Stage B 后可挂。不抢 Stage B。已有 audit 卡，本卡为 arXiv 摘录窗口复核。

## 不可搬用边界
GPU SPTC/Spatha≠本地 ASIC 源字；高稀疏比收益不可外推本地 ~50% W；仅方法摘录窗口。

## 可复用 idea 点
- V:N:M 作 HiNM/2:4 之外的结构对照谱系
- 二阶剪枝+渐进恢复流程对齐 r1 双消费者损失再谈
- column-loc/元数据开销纳入并集读费用
- 负结果只停该模式布局，不杀半结构家族

## 杀门建议
同服务不胜窄稠密/HiNM 或源字仍读满 → 停该 V:N:M 布局。
