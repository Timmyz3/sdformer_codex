# arXiv:2505.12281 · Bishop

- uid/来源: `MAIN-R062`｜arxiv_2505.12281+本地excerpt（`p0_excerpt_batches/batch_03.json`）
- 题名: Bishop: Sparsified Bundling Spiking Transformers on Heterogeneous Cores with Error-Constrained Pruning
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.2 Token-Time Bundle（TTB）打包；BSA 束级稀疏损失；§5.1 ECP 按 nab/θp 界剪 Q/K 束行；异构 sparse/dense/attention 三核

## 可继承 A
TTB 结构化打包 + BSA 束级稀疏训练 + ECP 误差约束剪 Q/K——「结构化可跳过工作单元 + 有界误差剪枝」可直接对照 F1 共同删字与 F2 组完成（借入≠X）。

## 强对照 B
逐尖峰不规则跳过；无束级损失的朴素稀疏；无误差界的注意力剪枝；单一稠密核。

## 可差分 X线索
TTB/BSA/ECP≠ lifting 源活动×双 PED 删字 X；本地主张在非因果 T10×残差链消费者，而非 SSA 二次复杂度本身。

## 与 F1–F7 / Stage B 关系
F1/F2/F7 强相关（结构化剪枝/组稀疏）；Stage B 后可挂。不抢 Stage B。

## 不可搬用边界
CIFAR/ImageNet-100 注意力图≠AEE 门；异构三核 PPA 勿搬；仅方法摘录窗口。

## 可复用 idea 点
- TTB 作「可共同调度的结构化工作单元」模板对齐 F1 广播域
- L_bsp 束级损失作训练诱导结构稀疏先验
- ECP 用 nab<θp 证 S 行上界→对照 F6 有界提前完成（借入界≠X）
- 稀/密分层异构核作 Stage B 后执行组织对照

## 杀门建议
同槽不胜 HiNM/窄稠密或破 AEE，或并集仍读满无周期余量 → 停该 TTB/ECP 挂 r1 布局。
